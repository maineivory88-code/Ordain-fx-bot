"""Tris forex agent: OANDA execution, ATR rules, OpenRouter notes, Telegram control.

Profit is taken in repeating cycles. Each time account P/L rises by another
DAILY_PROFIT_TARGET (default $20), winning trades are closed and a new scan
opens fresh trades. Hitting the target does not stop the bot.

The daily loss limit still flattens and pauses new entries until the next
New York forex day (17:00).

Automatic scans run in the Asian (19:00–04:00 ET), London (03:00–12:00 ET),
and New York (08:00–17:00 ET) sessions.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests
from fastapi import FastAPI, HTTPException, Request
from openai import OpenAI
from telegram import Bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tris")

REQUIRED_ENV = (
    "TELEGRAM_TOKEN",
    "TELEGRAM_CHAT_ID",
    "OANDA_API_KEY",
    "OANDA_ACCOUNT_ID",
    "OPENROUTER_API_KEY",
)
TELEGRAM_LIMIT = 4000
SEEN_UPDATES_LIMIT = 500
CANDLE_LOOKBACK = 300
MIN_CLOSES = 50
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    telegram_chat_id: str
    oanda_api_key: str
    oanda_account_id: str
    openrouter_api_key: str
    cron_secret: str
    public_url: str
    oanda_base: str
    oanda_env: str
    openrouter_model: str
    trade_units: Decimal
    webhook_secret: str
    breakeven_trigger_pips: Decimal = Decimal("8")
    breakeven_lock_pips: Decimal = Decimal("1")
    trail_pips: Decimal = Decimal("0")
    protect_interval: int = 60
    daily_profit_target: Decimal = Decimal("20")
    daily_loss_limit: Decimal = Decimal("20")
    scan_interval: int = 300


@dataclass(frozen=True)
class InstrumentSpec:
    name: str
    pip_location: int
    display_precision: int
    trade_units_precision: int
    minimum_trade_size: Decimal
    maximum_order_units: Decimal

    @property
    def pip(self) -> Decimal:
        return Decimal(10) ** self.pip_location

    @property
    def pair(self) -> str:
        return self.name.replace("_", "/")


@dataclass
class ScanResult:
    report: str
    executed: int = 0
    failed: int = 0
    alert: bool = False


class SkipPair(Exception):
    pass


OpinionFn = Callable[[str, dict, dict], dict | None]


# Session windows in America/New_York, end exclusive. Asian wraps past midnight.
# 19:00–04:00 Asian, 03:00–12:00 London, 08:00–17:00 New York.
SESSION_WINDOWS = (
    ("Asian", 19 * 60, 4 * 60),
    ("London", 3 * 60, 12 * 60),
    ("New York", 8 * 60, 17 * 60),
)
FOREX_DAY_OPEN_MINUTE = 17 * 60  # 17:00 New York


def ny_local(now: datetime | None = None) -> datetime:
    """Current time in America/New_York (EST fallback if the tz database is missing)."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        return now.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        return now.astimezone(timezone(timedelta(hours=-5)))


def trading_day_start(now: datetime | None = None) -> datetime:
    """Start of the current forex trading day (17:00 New York), as a UTC datetime."""
    local = ny_local(now)
    start = local.replace(hour=17, minute=0, second=0, microsecond=0)
    if local < start:
        start -= timedelta(days=1)
    return start.astimezone(timezone.utc)


def active_sessions(now: datetime | None = None) -> list[str]:
    """Session names open at `now`. Asian, London, and New York can overlap."""
    local = ny_local(now)
    minute = local.hour * 60 + local.minute
    found: list[str] = []
    for name, start, end in SESSION_WINDOWS:
        if start <= end:
            on = start <= minute < end
        else:
            on = minute >= start or minute < end
        if on:
            found.append(name)
    return found


def forex_market_open(now: datetime | None = None) -> bool:
    """False from Friday 17:00 ET through Sunday 17:00 ET."""
    local = ny_local(now)
    weekday = local.weekday()
    minute = local.hour * 60 + local.minute
    if weekday == 5:
        return False
    if weekday == 6 and minute < FOREX_DAY_OPEN_MINUTE:
        return False
    if weekday == 4 and minute >= FOREX_DAY_OPEN_MINUTE:
        return False
    return True


def next_cycle_baseline(
    baseline: Decimal, cycle_pnl: Decimal, equity_after: Decimal, target: Decimal
) -> tuple[Decimal, int]:
    """Where the next profit cycle starts after a target hit.

    Returns (new_baseline, steps). The next cycle needs a fresh `target`
    of profit above `new_baseline`. Baseline never moves backward. If account
    equity is already above the steps just consumed, the baseline sits on that
    equity so the same profit cannot fire twice.
    """
    if target <= 0 or cycle_pnl < target:
        return baseline, 0
    steps = int(cycle_pnl // target)
    if steps < 1:
        steps = 1
    booked = baseline + target * steps
    if equity_after > booked:
        booked = equity_after
    return booked, steps


def env_decimal(name: str, default: str) -> Decimal:
    raw = os.getenv(name, "").strip()
    if not raw:
        return Decimal(default)
    try:
        return Decimal(raw)
    except Exception as exc:
        raise RuntimeError(f"{name} must be a number") from exc


def load_settings() -> Settings:
    missing = [key for key in REQUIRED_ENV if not os.getenv(key, "").strip()]
    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))

    oanda_env = os.getenv("OANDA_ENV", "practice").strip().lower()
    if oanda_env not in ("practice", "live"):
        raise RuntimeError("OANDA_ENV must be 'practice' or 'live'")
    host = "api-fxpractice.oanda.com" if oanda_env == "practice" else "api-fxtrade.oanda.com"

    units_raw = os.getenv("TRADE_UNITS", "5000").strip()
    try:
        units = Decimal(units_raw)
    except Exception as exc:
        raise RuntimeError("TRADE_UNITS must be a number") from exc
    if units <= 0:
        raise RuntimeError("TRADE_UNITS must be greater than 0")

    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if secret and not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", secret):
        raise RuntimeError("TELEGRAM_WEBHOOK_SECRET must match [A-Za-z0-9_-]{1,256}")

    be_trigger = env_decimal("BREAKEVEN_TRIGGER_PIPS", "8")
    be_lock = env_decimal("BREAKEVEN_LOCK_PIPS", "1")
    trail = env_decimal("TRAIL_PIPS", "0")
    if be_lock < 0 or trail < 0 or be_trigger < 0:
        raise RuntimeError("BREAKEVEN_*/TRAIL_PIPS values cannot be negative")
    if be_trigger > 0 and be_lock >= be_trigger:
        raise RuntimeError("BREAKEVEN_LOCK_PIPS must be smaller than BREAKEVEN_TRIGGER_PIPS")
    try:
        interval = int(os.getenv("PROTECT_INTERVAL_SECONDS", "60").strip() or "60")
    except ValueError as exc:
        raise RuntimeError("PROTECT_INTERVAL_SECONDS must be an integer") from exc

    daily_target = env_decimal("DAILY_PROFIT_TARGET", "20")
    if daily_target < 0:
        raise RuntimeError("DAILY_PROFIT_TARGET cannot be negative")

    daily_loss = env_decimal("DAILY_LOSS_LIMIT", "20")
    if daily_loss < 0:
        raise RuntimeError("DAILY_LOSS_LIMIT cannot be negative")

    try:
        scan_interval = int(os.getenv("SCAN_INTERVAL_SECONDS", "300").strip() or "300")
    except ValueError as exc:
        raise RuntimeError("SCAN_INTERVAL_SECONDS must be an integer") from exc
    if scan_interval < 0:
        raise RuntimeError("SCAN_INTERVAL_SECONDS cannot be negative")

    public_url = os.getenv("PUBLIC_URL", "").strip().rstrip("/")
    return Settings(
        telegram_token=os.environ["TELEGRAM_TOKEN"].strip(),
        telegram_chat_id=os.environ["TELEGRAM_CHAT_ID"].strip(),
        oanda_api_key=os.environ["OANDA_API_KEY"].strip(),
        oanda_account_id=os.environ["OANDA_ACCOUNT_ID"].strip(),
        openrouter_api_key=os.environ["OPENROUTER_API_KEY"].strip(),
        cron_secret=os.getenv("CRON_SECRET", "").strip(),
        public_url=public_url,
        oanda_base=f"https://{host}/v3",
        oanda_env=oanda_env,
        openrouter_model=os.getenv("OPENROUTER_MODEL", "openrouter/free").strip() or "openrouter/free",
        trade_units=units,
        webhook_secret=secret,
        breakeven_trigger_pips=be_trigger,
        breakeven_lock_pips=be_lock,
        trail_pips=trail,
        protect_interval=interval,
        daily_profit_target=daily_target,
        daily_loss_limit=daily_loss,
        scan_interval=scan_interval,
    )


def webhook_url(public_url: str) -> str:
    if public_url.endswith("/telegram-webhook"):
        return public_url
    return f"{public_url}/telegram-webhook"


def parse_command(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    token = stripped.split()[0]
    return token.split("@", 1)[0].lower()


def chunk_text(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    text = text or ""
    if len(text) <= limit:
        return [text] if text else []
    chunks: list[str] = []
    rest = text
    while rest:
        if len(rest) <= limit:
            chunks
