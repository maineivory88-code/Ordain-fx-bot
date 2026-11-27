"""Tris forex bot + GBPJPY Multi-Bot System.

Preserves the legacy EMA bot (H1 trend-pullback entries, fixed-dollar risk,
broker-side stop and target) while adding three independent GBPJPY bots:
  - intraday: KJ McLaurin Daily/H4 trend, H1 liquidity sweep, structure shift, M15 FVG
  - swing:    KJ McLaurin framework with multi-session management & trailing/BE
  - scalp:    Auto-selects M5/M1 EMA momentum pullback vs Asian-range breakout/retest

Safety & defaults:
  - All bots start PAUSED (/resume required per bot).
  - GBPJPY bots enforce OANDA Practice by default; Live requires BOTH
    GJ_OANDA_ENV=live AND ALLOW_GJ_LIVE=true.
  - Risk: Intraday 0.50%, Swing 0.50%, Scalp 0.25%; combined open risk <= 1.00%;
    daily loss <= 2.00%; max 3 trades/week per bot (fail-closed).
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Any, Callable
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests
from fastapi import FastAPI, HTTPException, Request
from openai import OpenAI
from telegram import Bot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("tris")

REQUIRED_ENV = ("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "OANDA_API_KEY", "OANDA_ACCOUNT_ID", "OPENROUTER_API_KEY")
TELEGRAM_LIMIT = 4000
SEEN_UPDATES_LIMIT = 500
CANDLE_LOOKBACK = 300
MIN_CANDLES = 210
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Legacy EMA bot constants (preserved)
PAIRS = ("EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD", "EUR_JPY", "EUR_GBP")
RISK_PER_TRADE_USD = Decimal("10")
STOP_ATR = Decimal("1.5")
TARGET_ATR = Decimal("1.5")
MAX_SPREAD_ATR = Decimal("0.10")
MAX_OPEN_TRADES = 3
SCAN_INTERVAL_SECONDS = 300
HOUSEKEEPING_SECONDS = 60
DEFAULT_DAILY_LOSS_ALERT = Decimal("30")


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
    webhook_secret: str
    daily_loss_alert: Decimal
    trading_session: str


@dataclass(frozen=True)
class InstrumentSpec:
    name: str
    pip_location: int
    display_precision: int
    trade_units_precision: int
    minimum_trade_size: Decimal
    maximum_order_units: Decimal

    @property
    def quote_ccy(self) -> str:
        return self.name.split("_")[1]


@dataclass
class ScanResult:
    report: str
    executed: int = 0
    failed: int = 0
    alert: bool = False
    entered: list[str] = field(default_factory=list)


class SkipPair(Exception):
    pass


SESSION_WINDOWS_UTC = {"all": None, "asia": (0, 9), "london": (7, 16), "ny": (12, 21)}


# ---------- time and market hours ----------

def trading_day_start(now: datetime | None = None) -> datetime:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        local = now.astimezone(ZoneInfo("America/New_York"))
        start = local.replace(hour=17, minute=0, second=0, microsecond=0)
        if local < start:
            start -= timedelta(days=1)
        return start.astimezone(timezone.utc)
    except Exception:
        start = now.replace(hour=22, minute=0, second=0, microsecond=0)
        if now < start:
            start -= timedelta(days=1)
        return start


def trading_week_start(now: datetime | None = None) -> datetime:
    """Forex week starts Sunday 17:00 America/New_York."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    local = now.astimezone(ZoneInfo("America/New_York"))
    # weekday(): Monday=0 .. Sunday=6
    days_since_sun = (local.weekday() + 1) % 7
    sun = (local - timedelta(days=days_since_sun)).replace(hour=17, minute=0, second=0, microsecond=0)
    if local < sun:
        sun -= timedelta(days=7)
    return sun.astimezone(timezone.utc)


def bar_key(now: datetime, minutes: int = 60) -> str:
    utc = now.astimezone(timezone.utc)
    floored_min = (utc.minute // minutes) * minutes if minutes < 60 else 0
    if minutes >= 240:
        floored_hr = (utc.hour // (minutes // 60)) * (minutes // 60)
        return utc.replace(hour=floored_hr, minute=0, second=0, microsecond=0).isoformat()
    return utc.replace(minute=floored_min, second=0, microsecond=0).isoformat()


def market_open(now: datetime | None = None) -> bool:
    local = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    wd, hour = local.weekday(), local.hour
    if wd == 5 or (wd == 4 and hour >= 17) or (wd == 6 and hour < 17):
        return False
    return True


def session_open(now: datetime | None, session: str) -> bool:
    window = SESSION_WINDOWS_UTC.get(session)
    if window is None:
        return True
    hour = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).hour
    return window[0] <= hour < window[1]


# ---------- configuration ----------

def load_settings() -> Settings:
    missing = [key for key in REQUIRED_ENV if not os.getenv(key, "").strip()]
    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))
    oanda_env = os.getenv("OANDA_ENV", "practice").strip().lower()
    if oanda_env not in ("practice", "live"):
        raise RuntimeError("OANDA_ENV must be 'practice' or 'live'")
    host = "api-fxpractice.oanda.com" if oanda_env == "practice" else "api-fxtrade.oanda.com"
    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if secret and not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", secret):
        raise RuntimeError("TELEGRAM_WEBHOOK_SECRET must match [A-Za-z0-9_-]{1,256}")
    loss_alert_raw = os.getenv("DAILY_LOSS_LIMIT", "").strip()
    try:
        loss_alert = Decimal(loss_alert_raw) if loss_alert_raw else DEFAULT_DAILY_LOSS_ALERT
    except Exception as exc:
        raise RuntimeError("DAILY_LOSS_LIMIT must be a number") from exc
    session = os.getenv("TRADING_SESSION", "all").strip().lower()
    if session not in SESSION_WINDOWS_UTC:
        raise RuntimeError("TRADING_SESSION must be one of: " + ", ".join(SESSION_WINDOWS_UTC))
    return Settings(
        telegram_token=os.environ["TELEGRAM_TOKEN"].strip(),
        telegram_chat_id=os.environ["TELEGRAM_CHAT_ID"].strip(),
        oanda_api_key=os.environ["OANDA_API_KEY"].strip(),
        oanda_account_id=os.environ["OANDA_ACCOUNT_ID"].strip(),
        openrouter_api_key=os.environ["OPENROUTER_API_KEY"].strip(),
        cron_secret=os.getenv("CRON_SECRET", "").strip(),
        public_url=os.getenv("PUBLIC_URL", "").strip().rstrip("/"),
        oanda_base=f"https://{host}/v3",
        oanda_env=oanda_env,
        openrouter_model=os.getenv("OPENROUTER_MODEL", "openrouter/free").strip() or "openrouter/free",
        webhook_secret=secret,
        daily_loss_alert=loss_alert,
        trading_session=session,
    )


def webhook_url(public_url: str) -> str:
    return public_url if public_url.endswith("/telegram-webhook") else f"{public_url}/telegram-webhook"


# ---------- telegram helpers ----------

def parse_command(text: str) -> str:
    stripped = text.strip()
    return stripped.split()[0].split("@", 1)[0].lower() if stripped else ""


def parse_args(text: str) -> list[str]:
    parts = text.strip().split()
    return parts[1:] if len(parts) > 1 else []


def chunk_text(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    text = text or ""
    if len(text) <= limit:
        return [text] if text else []
    chunks, rest = [], text
    while rest:
        if len(rest) <= limit:
            chunks.append(rest)
            break
        cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip("\n")
    return [c for c in chunks if c]


def remember_update(seen: OrderedDict, update_id: Any) -> bool:
    if update_id is None:
        return False
    if update_id in seen:
        return True
    seen[update_id] = True
    while len(seen) > SEEN_UPDATES_LIMIT:
        seen.popitem(last=False)
    return False


# ---------- indicators and legacy EMA strategy ----------

def ema(values: list[float], period: int) -> float:
    period = min(period, len(values))
    k = 2 / (period + 1)
    seed = sum(values[:period]) / period
    for value in values[period:]:
        seed = value * k + seed * (1 - k)
    return seed


def ema_series(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    p = min(period, len(values))
    k = 2 / (p + 1)
    seed = sum(values[:p]) / p
    out = [seed] * p
    for value in values[p:]:
        seed = value * k + seed * (1 - k)
        out.append(seed)
    return out


def calculate_atr(candles: list[dict], period: int = 14) -> float:
    true_ranges = []
    for i in range(1, len(candles)):
        h, l = float(candles[i]["mid"]["h"]), float(candles[i]["mid"]["l"])
        pc = float(candles[i - 1]["mid"]["c"])
        true_ranges.append(max(h - l, abs(h - pc), abs(l - pc)))
    window = true_ranges[-period:]
    return sum(window) / len(window) if window else 0.0


def mid_price(price: dict) -> Decimal:
    return (Decimal(str(price["bids"][0]["price"])) + Decimal(str(price["asks"][0]["price"]))) / 2


def evaluate_setup(candles: list[dict], price: dict) -> tuple[str | None, str, Decimal]:
    """Legacy EMA bot setup: completed H1 candles."""
    completed = [c for c in candles if c.get("complete") and isinstance(c.get("mid"), dict)]
    if len(completed) < MIN_CANDLES:
        return None, f"only {len(completed)} completed candles", Decimal(0)
    try:
        closes = [float(c["mid"]["c"]) for c in completed]
        atr = calculate_atr(completed)
        bid = Decimal(str(price["bids"][0]["price"]))
        ask = Decimal(str(price["asks"][0]["price"]))
    except Exception:
        return None, "bad price or candle data", Decimal(0)
    if atr <= 0:
        return None, "ATR is zero", Decimal(0)
    atr_d = Decimal(str(atr))
    spread = ask - bid
    if spread > MAX_SPREAD_ATR * atr_d:
        return None, f"spread {spread} above {MAX_SPREAD_ATR} x ATR", atr_d

    ema20, ema50, ema200 = ema(closes, 20), ema(closes, 50), ema(closes, 200)
    bar = completed[-1]["mid"]
    o, h, l, c = float(bar["o"]), float(bar["h"]), float(bar["l"]), float(bar["c"])
    if ema50 > ema200 and c > ema200 and l <= ema20 and c > ema20 and c > o:
        return "BUY", "uptrend pullback to EMA20, bullish close", atr_d
    if ema50 < ema200 and c < ema200 and h >= ema20 and c < ema20 and c < o:
        return "SELL", "downtrend pullback to EMA20, bearish close", atr_d
    return None, "no pullback setup in trend", atr_d


def size_order(spec: InstrumentSpec, direction: str, price: dict, atr: Decimal, usd_per_quote: Decimal) -> dict:
    """Legacy EMA order sizing: fixed RISK_PER_TRADE_USD."""
    bid = Decimal(str(price["bids"][0]["price"]))
    ask = Decimal(str(price["asks"][0]["price"]))
    entry = ask if direction == "BUY" else bid
    quantum = Decimal(1).scaleb(-spec.display_precision)
    stop_dist = STOP_ATR * atr
    target_dist = TARGET_ATR * atr
    sign = 1 if direction == "BUY" else -1
    stop = (entry - sign * stop_dist).quantize(quantum, rounding=ROUND_HALF_UP)
    target = (entry + sign * target_dist).quantize(quantum, rounding=ROUND_HALF_UP)
    loss_per_unit_usd = abs(entry - stop) * usd_per_quote
    if loss_per_unit_usd <= 0:
        raise SkipPair("stop distance rounded to zero")
    unit_q = Decimal(1).scaleb(-spec.trade_units_precision)
    units = (RISK_PER_TRADE_USD / loss_per_unit_usd).quantize(unit_q, rounding=ROUND_DOWN)
    if units < spec.minimum_trade_size:
        needed = spec.minimum_trade_size * loss_per_unit_usd
        raise SkipPair(f"minimum size needs about ${needed:.2f} risk per trade (budget ${RISK_PER_TRADE_USD})")
    if spec.maximum_order_units > 0 and units > spec.maximum_order_units:
        units = spec.maximum_order_units
    signed_units = units * sign
    return {
        "order": {
            "units": format(signed_units, "f"),
            "instrument": spec.name,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": format(stop, "f")},
            "takeProfitOnFill": {"price": format(target, "f")},
        },
        "_entry": entry,
        "_stop": stop,
        "_target": target,
        "_risk_usd": units * loss_per_unit_usd,
    }


def usd_per_quote(spec: InstrumentSpec, prices: dict[str, dict]) -> Decimal | None:
    """USD value of one unit of the quote currency (e.g., 1 JPY = 1 / USD_JPY USD)."""
    q = spec.quote_ccy
    if q == "USD":
        return Decimal(1)
    direct = prices.get(f"{q}_USD")
    if direct:
        return mid_price(direct)
    inverse = prices.get(f"USD_{q}")
    if inverse:
        m = mid_price(inverse)
        return (Decimal(1) / m) if m > 0 else None
    return None


def interpret_order_response(res: Any) -> tuple[str, str]:
    if not isinstance(res, dict):
        return "rejected", str(res)[:500]
    if "orderFillTransaction" in res:
        return "filled", ""
    reject = res.get("orderRejectTransaction") or res.get("orderCancelTransaction") or {}
    if not isinstance(reject, dict):
        reject = {}
    return "rejected", str(reject.get("reason") or res.get("errorMessage") or res)[:500]


# ---------- OpenRouter (advisory only; never blocks or invents a trade) ----------

def extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("model JSON was not an object")
    return data


def _short_error(exc: Exception) -> str:
    return (str(exc) or exc.__class__.__name__)[:160]


def get_openrouter_opinion(client: OpenAI, model: str, pair: str, signal: dict) -> dict:
    prompt = (
        f"Review trade signal for {pair}.\nSignal: {json.dumps(signal)}\n"
        'Return JSON only: {"agree": true, "comment": "short text"}'
    )
    messages = [{"role": "system", "content": "You are a financial JSON generator."}, {"role": "user", "content": prompt}]
    last_error: Exception | None = None
    for json_mode in (True, False):
        try:
            kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": 300}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            data = extract_json(response.choices[0].message.content or "")
            return {"agree": data.get("agree", True), "comment": str(data.get("comment", "N/A"))[:500]}
        except Exception as exc:
            last_error = exc
            logger.warning("OpenRouter attempt (json_mode=%s) failed for %s: %r", json_mode, pair, exc)
    return {"agree": True, "comment": "", "error": _short_error(last_error) if last_error else "unknown"}


def select_scalp_setup_with_ai(
    client: OpenAI | None,
    model: str,
    candidates: list[dict],
    market_context: dict,
) -> tuple[dict, str]:
    """Choose between already-validated scalp setups. Never invents a trade."""
    # Sort deterministically by score descending as baseline fallback
    ranked = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)
    best = ranked[0]
    if len(ranked) == 1 or client is None:
        return best, f"Deterministic selection ({best['setup_name']}, score={best.get('score', 0):.2f})"

    prompt = (
        "Select the strongest eligible GBP/JPY scalp setup from the validated candidates below.\n"
        f"Market context: {json.dumps(market_context)}\n"
        f"Candidates: {json.dumps([{k: v for k, v in c.items() if k != 'signal_obj'} for c in ranked])}\n"
        'Return JSON only: {"selected": "<setup_name>", "reason": "short text"}'
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a disciplined trading setup selector. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        data = extract_json(resp.choices[0].message.content or "")
        chosen_name = str(data.get("selected", "")).strip()
        for c in ranked:
            if c["setup_name"] == chosen_name:
                return c, f"AI selected {chosen_name}: {str(data.get('reason', ''))[:160]}"
    except Exception as exc:
        logger.warning("Scalp AI selector failed, using deterministic best: %s", _short_error(exc))
    return best, f"Deterministic fallback ({best['setup_name']}, score={best.get('score', 0):.2f})"


def ai_note(opinion: dict | None) -> str:
    if not opinion:
        return "OpenRouter AI: Offline / Skipped"
    if opinion.get("error"):
        return f"OpenRouter AI: Offline - {opinion['error']}"
    value = opinion.get("agree", True)
    agree = not (value is False or (isinstance(value, str) and value.strip().lower() in {"false", "no", "0"}))
    return f"OpenRouter AI: {'Agree' if agree else 'Caution'} - {opinion.get('comment', 'N/A')}"


# ---------- statistics ----------

def _money(x: Decimal) -> str:
    return f"{x:+.2f}"


def compute_stats(trades: list[dict], days: int, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    wins: list[Decimal] = []
    losses: list[Decimal] = []
    flat = 0
    per_day: dict[str, list] = {}
    per_pair: dict[str, list] = {}
    for t in trades:
        try:
            closed = datetime.strptime(str(t["closeTime"])[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            net = Decimal(str(t.get("realizedPL") or "0")) + Decimal(str(t.get("financing") or "0"))
        except Exception:
            continue
        if closed < cutoff:
            continue
        if net > 0:
            wins.append(net)
        elif net < 0:
            losses.append(net)
        else:
            flat += 1
        for bucket, key in ((per_day, trading_day_start(closed).date().isoformat()),
                            (per_pair, str(t.get("instrument", "?")).replace("_", "/"))):
            row = bucket.setdefault(key, [Decimal(0), 0])
            row[0] += net
            row[1] += 1
    total = len(wins) + len(losses) + flat
    if total == 0:
        return f"STATS (last {days} days)\nNo closed trades in this period."
    gross_win = sum(wins, Decimal(0))
    gross_loss = -sum(losses, Decimal(0))
    avg_win = gross_win / len(wins) if wins else None
    avg_loss = -gross_loss / len(losses) if losses else None
    pf = (gross_win / gross_loss) if gross_loss > 0 else None
    lines = [
        f"STATS (last {days} days)",
        f"Closed trades: {total} (wins {len(wins)}, losses {len(losses)}, flat {flat})",
        f"Win rate: {len(wins) / total * 100:.1f}%",
        f"Avg win: {_money(avg_win) if avg_win is not None else 'n/a'} | "
        f"Avg loss: {_money(avg_loss) if avg_loss is not None else 'n/a'}",
        f"Profit factor: {f'{pf:.2f}' if pf is not None else 'n/a'}",
        f"Net P/L: {_money(gross_win - gross_loss)}",
        "",
        "By day (trading day):",
    ]
    for day in sorted(per_day, reverse=True):
        amt, n = per_day[day]
        lines.append(f"  {day}: {_money(amt)} ({n} trades)")
    lines += ["", "By pair:"]
    for pair in sorted(per_pair, key=lambda k: per_pair[k][0], reverse=True):
        amt, n = per_pair[pair]
        lines.append(f"  {pair}: {_money(amt)} ({n} trades)")
    return "\n".join(lines)


# ---------- OANDA ----------

class OandaBroker:
    def __init__(self, oanda_api_key: str, oanda_account_id: str, oanda_base: str, session: requests.Session | None = None):
        self.oanda_api_key = oanda_api_key
        self.oanda_account_id = oanda_account_id
        self.oanda_base = oanda_base.rstrip("/")
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {oanda_api_key}", "Content-Type": "application/json"})

    @classmethod
    def from_settings(cls, settings: Settings, session: requests.Session | None = None) -> "OandaBroker":
        return cls(settings.oanda_api_key, settings.oanda_account_id, settings.oanda_base, session=session)

    def close(self) -> None:
        self.session.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        kwargs.setdefault("timeout", 20)
        response = self.session.request(method, f"{self.oanda_base}{path}", **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"OANDA HTTP {response.status_code}: {scrub_secrets(response.text[:180])}")
        return response.json()

    def _account_path(self, suffix: str) -> str:
        return f"/accounts/{quote(self.oanda_account_id, safe='')}{suffix}"

    def instrument_specs(self, names: tuple[str, ...]) -> dict[str, InstrumentSpec]:
        data = self._request("GET", self._account_path("/instruments"))
        specs: dict[str, InstrumentSpec] = {}
        for raw in data.get("instruments") or []:
            name = str(raw.get("name") or "")
            if name in names and raw.get("type") in ("CURRENCY", "METAL"):
                specs[name] = InstrumentSpec(
                    name=name,
                    pip_location=int(raw["pipLocation"]),
                    display_precision=int(raw["displayPrecision"]),
                    trade_units_precision=int(raw.get("tradeUnitsPrecision") or 0),
                    minimum_trade_size=Decimal(str(raw.get("minimumTradeSize") or "1")),
                    maximum_order_units=Decimal(str(raw.get("maximumOrderUnits") or "0")),
                )
        return specs

    def prices(self, names: list[str]) -> dict[str, dict]:
        if not names:
            return {}
        data = self._request("GET", self._account_path("/pricing"), params={"instruments": ",".join(names)})
        return {str(p["instrument"]): p for p in (data.get("prices") or []) if isinstance(p, dict) and p.get("instrument")}

    def candles(self, name: str, granularity: str = "H1", count: int = CANDLE_LOOKBACK) -> list[dict]:
        data = self._request(
            "GET", f"/instruments/{quote(name, safe='')}/candles",
            params={"granularity": granularity, "count": count, "price": "M"},
        )
        return data.get("candles") or []

    def open_trades(self) -> list[dict]:
        data = self._request("GET", self._account_path("/openTrades"))
        return [t for t in (data.get("trades") or []) if isinstance(t, dict) and t.get("id") is not None]

    def all_trades(self, count: int = 500) -> list[dict]:
        data = self._request("GET", self._account_path("/trades"), params={"state": "ALL", "count": count})
        return [t for t in (data.get("trades") or []) if isinstance(t, dict) and t.get("id") is not None]

    def submit(self, built: dict) -> tuple[str, str]:
        response = self._request("POST", self._account_path("/orders"), json={"order": built["order"]})
        return interpret_order_response(response)

    def update_trade_sl(self, trade_id: str, stop_price: Decimal, display_precision: int) -> None:
        quantum = Decimal(1).scaleb(-display_precision)
        sl_str = format(stop_price.quantize(quantum, rounding=ROUND_HALF_UP), "f")
        self._request(
            "PUT",
            self._account_path(f"/trades/{quote(str(trade_id), safe='')}/orders"),
            json={"stopLoss": {"price": sl_str, "timeInForce": "GTC"}},
        )

    def realized_pl_since(self, since: datetime) -> Decimal:
        fmt = "%Y-%m-%dT%H:%M:%S.000000000Z"
        params = {"from": since.strftime(fmt), "to": datetime.now(timezone.utc).strftime(fmt), "type": "ORDER_FILL", "pageSize": 1000}
        index = self._request("GET", self._account_path("/transactions"), params=params)
        total = Decimal("0")
        for page_url in index.get("pages") or []:
            resp = self.session.get(page_url, timeout=20)
            if resp.status_code >= 400:
                raise RuntimeError(f"OANDA HTTP {resp.status_code}: {scrub_secrets(resp.text[:180])}")
            for tx in resp.json().get("transactions") or []:
                for key in ("pl", "financing", "commission"):
                    try:
                        total += Decimal(str(tx.get(key) or "0"))
                    except Exception:
                        pass
        return total

    def daily_pnl(self) -> tuple[Decimal, Decimal]:
        realized = self.realized_pl_since(trading_day_start())
        data = self._request("GET", self._account_path("/summary"))
        return realized, Decimal(str((data.get("account") or {}).get("unrealizedPL", "0")))

    def account_details(self) -> dict:
        return self._request("GET", self._account_path("/summary")).get("account") or {}

    def closed_trades(self, max_pages: int = 4) -> list[dict]:
        out: list[dict] = []
        before: int | None = None
        for _ in range(max_pages):
            params: dict[str, Any] = {"state": "CLOSED", "count": 500}
            if before is not None:
                params["beforeID"] = before
            page = [t for t in (self._request("GET", self._account_path("/trades"), params=params).get("trades") or [])
                    if isinstance(t, dict) and t.get("id") is not None]
            out.extend(page)
            if len(page) < 500:
                break
            before = min(int(t["id"]) for t in page)
        return out

    def close_trade(self, trade_id: str) -> None:
        self._request("PUT", self._account_path(f"/trades/{quote(str(trade_id), safe='')}/close"), json={"units": "ALL"})

    def close_losing_trades(self) -> tuple[list[str], list[str], int]:
        closed, failed, kept = [], [], 0
        for trade in self.open_trades():
            try:
                pl = Decimal(str(trade.get("unrealizedPL", "0")))
            except Exception:
                continue
            label = f"{str(trade.get('instrument', '?')).replace('_', '/')} #{trade['id']}"
            if pl >= 0:
                kept += 1
                continue
            try:
                self.close_trade(str(trade["id"]))
                closed.append(f"{label} ({format_dec(pl)})")
            except Exception as exc:
                failed.append(f"{label}: {exc}")
        return closed, failed, kept

    def account_summary(self) -> tuple[Any, Any, Any]:
        acc = self.account_details()
        return acc.get("balance", "N/A"), acc.get("unrealizedPL", "N/A"), acc.get("openTradeCount", 0)


def format_dec(value: Decimal) -> str:
    return f"{value:+.2f}"


def price_request_names(specs: dict) -> list[str]:
    return list(specs.keys())


def run_cycle(broker: OandaBroker, settings: Settings, opinion_fn: Callable[[str, dict], dict], blocked: set[str]) -> ScanResult:
    """Legacy EMA bot cycle. Unchanged."""
    try:
        specs = broker.instrument_specs(PAIRS)
        open_trades = broker.open_trades()
        prices = broker.prices(price_request_names(specs))
    except Exception as exc:
        return ScanResult(report=f"Scan failed: {exc}", failed=1, alert=True)

    open_names = {str(t.get("instrument")) for t in open_trades}
    slots = MAX_OPEN_TRADES - len(open_trades)
    reports, entered = [], []
    executed = failed = 0
    if slots <= 0:
        return ScanResult(report=f"Scan skipped: {len(open_trades)} trades open (max {MAX_OPEN_TRADES}).")

    for name in PAIRS:
        spec = specs.get(name)
        if spec is None or name in open_names or name in blocked:
            continue
        if slots <= 0:
            break
        try:
            price = prices.get(name)
            if not price or not price.get("tradeable", True):
                continue
            direction, reason, atr = evaluate_setup(broker.candles(name), price)
            if not direction:
                continue
            usd_q = usd_per_quote(spec, prices)
            if usd_q is None:
                logger.info("No USD conversion for %s, skipped", name)
                continue
            built = size_order(spec, direction, price, atr, usd_q)
            signal = {"direction": direction, "reason": reason}
            opinion = opinion_fn(spec.name.replace("_", "/"), signal)
            status, reject = broker.submit(built)
            if status == "filled":
                executed += 1
                slots -= 1
                entered.append(name)
                reports.append(
                    f"TRADE EXECUTED ({spec.name.replace('_', '/')})\n"
                    f"{direction} @ {format(built['_entry'], 'f')}\n"
                    f"SL {format(built['_stop'], 'f')} | TP {format(built['_target'], 'f')} | "
                    f"risk ~${built['_risk_usd']:.2f}\n"
                    f"Setup: {reason}\n{ai_note(opinion)}"
                )
            else:
                failed += 1
                reports.append(f"ORDER FAILED ({name}): {reject}")
        except SkipPair as exc:
            logger.info("Skipped %s: %s", name, exc)
        except Exception:
            logger.exception("Pair error for %s", name)
            failed += 1
    if not reports:
        note = "Scan complete. No setups met the rules."
        if failed:
            note += f" ({failed} pair(s) errored; check logs.)"
        return ScanResult(report=note, failed=failed)
    return ScanResult(report=f"--- SCAN ({executed} opened) ---\n\n" + "\n\n".join(reports),
                      executed=executed, failed=failed, alert=True, entered=entered)


def check_pair_text(broker: OandaBroker, name: str) -> str:
    specs = broker.instrument_specs(PAIRS)
    if name not in PAIRS:
        return f"{name} is not in the trade list: {', '.join(PAIRS)}"
    spec = specs.get(name)
    if not spec:
        return f"{name} is not offered on this account."
    prices = broker.prices(price_request_names(specs))
    price = prices.get(name)
    if not price:
        return f"No price for {name}."
    direction, reason, atr = evaluate_setup(broker.candles(name), price)
    spread = Decimal(str(price["asks"][0]["price"])) - Decimal(str(price["bids"][0]["price"]))
    lines = [f"{name.replace('_', '/')}: {direction or 'no trade'}", f"Reason: {reason}",
             f"ATR: {atr:.6f} | spread: {spread} (limit {MAX_SPREAD_ATR} x ATR)"]
    if direction:
        usd_q = usd_per_quote(spec, prices)
        try:
            built = size_order(spec, direction, price, atr, usd_q or Decimal(0))
            lines.append(f"Would size to ~${built['_risk_usd']:.2f} risk, SL {built['_stop']}, TP {built['_target']}")
        except SkipPair as exc:
            lines.append(f"Would NOT trade: {exc}")
    return "\n".join(lines)


# ---------- GBPJPY MULTI-BOT SYSTEM ----------

@dataclass(frozen=True)
class GJBotConfig:
    key: str
    display_name: str
    enabled: bool
    telegram_token: str
    telegram_chat_id: str
    oanda_account_id: str
    risk_fraction: Decimal
    weekly_trade_limit: int
    scan_interval_seconds: int
    bar_minutes: int
    config_error: str = ""
    oanda_env: str = "practice"
    oanda_base: str = "https://api-fxpractice.oanda.com/v3"


@dataclass
class GJRuntime:
    config: GJBotConfig
    bot: Bot | None = None
    broker: OandaBroker | None = None
    seen: OrderedDict = field(default_factory=OrderedDict)
    mode: str = "offline"
    username: str = ""
    last_error: str = ""
    paused: bool = True  # ALWAYS starts PAUSED; /resume required per bot
    scanning: bool = False
    entered_bar: str | None = None
    loss_alert_day: str | None = None
    poll_task: asyncio.Task | None = None
    scan_task: asyncio.Task | None = None
    housekeeping_task: asyncio.Task | None = None


GJ_BOT_DEFINITIONS = {
    "intraday": {
        "env": "ENABLE_GJ_INTRADAY",
        "token": "TELEGRAM_GJ_INTRADAY_TOKEN",
        "chat": "TELEGRAM_GJ_INTRADAY_CHAT_ID",
        "account": "OANDA_ACCOUNT_ID_GJ_INTRADAY",
        "risk": Decimal("0.0050"),
        "weekly_limit": 3,
        "scan_seconds": 300,   # 5m scan for M15 FVG
        "bar_minutes": 15,
    },
    "swing": {
        "env": "ENABLE_GJ_SWING",
        "token": "TELEGRAM_GJ_SWING_TOKEN",
        "chat": "TELEGRAM_GJ_SWING_CHAT_ID",
        "account": "OANDA_ACCOUNT_ID_GJ_SWING",
        "risk": Decimal("0.0050"),
        "weekly_limit": 3,
        "scan_seconds": 900,   # 15m scan for H1/M15 swing entries
        "bar_minutes": 60,
    },
    "scalp": {
        "env": "ENABLE_GJ_SCALP",
        "token": "TELEGRAM_GJ_SCALP_TOKEN",
        "chat": "TELEGRAM_GJ_SCALP_CHAT_ID",
        "account": "OANDA_ACCOUNT_ID_GJ_SCALP",
        "risk": Decimal("0.0025"),
        "weekly_limit": 3,
        "scan_seconds": 60,    # 1m scan for M1/M5 scalp entries
        "bar_minutes": 1,
    },
}

GJ_PORTFOLIO_OPEN_RISK_LIMIT = Decimal("0.0100")  # 1.00% combined open risk ceiling
GJ_DAILY_LOSS_LIMIT = Decimal("0.0200")           # 2.00% daily loss cap
GJ_RISK_TIMEZONE = ZoneInfo("America/New_York")
GJ_INSTRUMENT = "GBP_JPY"
GJ_CONVERSION_PAIR = "USD_JPY"
GJ_REQUIRED_INSTRUMENTS = (GJ_INSTRUMENT, GJ_CONVERSION_PAIR)
GJ_MAX_SPREAD_PIPS_KJ = Decimal("5.0")
GJ_MAX_SPREAD_PIPS_SCALP = Decimal("3.5")

_SECRET_ENV_KEYS = (
    "TELEGRAM_TOKEN",
    "TELEGRAM_GJ_INTRADAY_TOKEN",
    "TELEGRAM_GJ_SWING_TOKEN",
    "TELEGRAM_GJ_SCALP_TOKEN",
    "OANDA_API_KEY",
    "OPENROUTER_API_KEY",
    "CRON_SECRET",
    "TELEGRAM_WEBHOOK_SECRET",
)


def scrub_secrets(text: str) -> str:
    out = text or ""
    for key in _SECRET_ENV_KEYS:
        raw = os.getenv(key, "").strip()
        if len(raw) >= 6:
            out = out.replace(raw, "***")
    return out[:300]


def public_origin(public_url: str) -> str:
    base = (public_url or "").strip().rstrip("/")
    suffix = "/telegram-webhook"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    return base.rstrip("/")


def gj_webhook_url(public_url: str, key: str) -> str:
    origin = public_origin(public_url)
    return f"{origin}/gj/{key}/telegram-webhook" if origin else ""


def _parse_flag(name: str) -> tuple[bool, str]:
    raw = os.getenv(name, "false").strip().lower()
    if raw == "":
        raw = "false"
    if raw not in ("true", "false", "1", "0", "yes", "no"):
        return False, f"{name} must be true/false (got {raw!r})"
    return raw in ("true", "1", "yes"), ""


def _resolve_gj_oanda_env() -> tuple[str, str, str]:
    """Enforce Practice by default. Live requires BOTH GJ_OANDA_ENV=live AND ALLOW_GJ_LIVE=true."""
    raw_env = os.getenv("GJ_OANDA_ENV", "practice").strip().lower() or "practice"
    if raw_env not in ("practice", "live"):
        return "practice", "https://api-fxpractice.oanda.com/v3", f"GJ_OANDA_ENV must be 'practice' or 'live' (got {raw_env!r})"
    if raw_env == "live":
        allow_live, _ = _parse_flag("ALLOW_GJ_LIVE")
        if not allow_live:
            return (
                "practice",
                "https://api-fxpractice.oanda.com/v3",
                "GJ_OANDA_ENV=live blocked: ALLOW_GJ_LIVE=true is required to enable live execution",
            )
        return "live", "https://api-fxtrade.oanda.com/v3", ""
    return "practice", "https://api-fxpractice.oanda.com/v3", ""


def _with_config_error(cfg: GJBotConfig, extra: str) -> GJBotConfig:
    combined = "; ".join(part for part in (cfg.config_error, extra) if part)
    return GJBotConfig(
        key=cfg.key,
        display_name=cfg.display_name,
        enabled=cfg.enabled,
        telegram_token=cfg.telegram_token,
        telegram_chat_id=cfg.telegram_chat_id,
        oanda_account_id=cfg.oanda_account_id,
        risk_fraction=cfg.risk_fraction,
        weekly_trade_limit=cfg.weekly_trade_limit,
        scan_interval_seconds=cfg.scan_interval_seconds,
        bar_minutes=cfg.bar_minutes,
        config_error=combined,
        oanda_env=cfg.oanda_env,
        oanda_base=cfg.oanda_base,
    )


def load_gj_bot_configs() -> dict[str, GJBotConfig]:
    configs: dict[str, GJBotConfig] = {}
    shared_chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    gj_env, gj_base, env_error = _resolve_gj_oanda_env()
    for key, definition in GJ_BOT_DEFINITIONS.items():
        enabled, flag_error = _parse_flag(definition["env"])
        token = os.getenv(definition["token"], "").strip()
        account_id = os.getenv(definition["account"], "").strip()
        chat_id = os.getenv(definition["chat"], "").strip() or shared_chat
        problems: list[str] = []
        if env_error:
            problems.append(env_error)
        if flag_error:
            problems.append(flag_error)
            enabled = False
        if enabled and not token:
            problems.append(f"{definition['token']} is required because {definition['env']} is true")
        if enabled and not account_id:
            problems.append(f"{definition['account']} is required because {definition['env']} is true")
        configs[key] = GJBotConfig(
            key=key,
            display_name=f"GBPJPY {key.title()}",
            enabled=enabled,
            telegram_token=token,
            telegram_chat_id=chat_id,
            oanda_account_id=account_id,
            risk_fraction=definition["risk"],
            weekly_trade_limit=definition["weekly_limit"],
            scan_interval_seconds=definition["scan_seconds"],
            bar_minutes=definition["bar_minutes"],
            config_error="; ".join(problems),
            oanda_env=gj_env,
            oanda_base=gj_base,
        )

    main_token = os.getenv("TELEGRAM_TOKEN", "").strip()
    by_token: dict[str, list[str]] = {}
    by_account: dict[str, list[str]] = {}
    for cfg in configs.values():
        if cfg.telegram_token:
            by_token.setdefault(cfg.telegram_token, []).append(cfg.key)
        if cfg.enabled and cfg.oanda_account_id:
            by_account.setdefault(cfg.oanda_account_id, []).append(cfg.key)
    for token, keys in by_token.items():
        if len(keys) > 1:
            for key in keys:
                configs[key] = _with_config_error(configs[key], "Telegram token is shared with another GJ bot")
        elif token and token == main_token:
            key = keys[0]
            configs[key] = _with_config_error(configs[key], "Telegram token is the same as TELEGRAM_TOKEN; create a separate bot")
    for keys in by_account.values():
        if len(keys) > 1:
            for key in keys:
                configs[key] = _with_config_error(
                    configs[key], "Enabled GBPJPY bots must use distinct OANDA account IDs"
                )
    return configs


def gj_foundation_status(configs: dict[str, GJBotConfig]) -> dict[str, Any]:
    envs = {cfg.oanda_env for cfg in configs.values()}
    env_label = next(iter(envs)) if len(envs) == 1 else "practice"
    return {
        "instrument": GJ_INSTRUMENT,
        "broker_environment": env_label,
        "order_execution": "enabled_when_resumed",
        "portfolio_risk_gate": "enforced_fail_closed",
        "open_risk_limit": str(GJ_PORTFOLIO_OPEN_RISK_LIMIT),
        "daily_loss_limit": str(GJ_DAILY_LOSS_LIMIT),
        "daily_reset_timezone": "America/New_York",
        "bots": {
            key: {
                "name": cfg.display_name,
                "enabled": cfg.enabled,
                "account_configured": bool(cfg.oanda_account_id),
                "telegram_configured": bool(cfg.telegram_token),
                "chat_configured": bool(cfg.telegram_chat_id),
                "config_error": cfg.config_error,
                "risk_fraction": str(cfg.risk_fraction),
                "weekly_trade_limit": cfg.weekly_trade_limit,
                "broker_environment": cfg.oanda_env,
            }
            for key, cfg in configs.items()
        },
        "legacy_ema_shared_risk_enforcement": False,
    }


def gj_public_status() -> dict[str, Any]:
    if getattr(app.state, "gj_config_error", "") and not getattr(app.state, "gj_configs", None):
        return {"error": app.state.gj_config_error}
    configs = getattr(app.state, "gj_configs", None) or {}
    if not configs:
        try:
            configs = load_gj_bot_configs()
        except Exception as exc:
            return {"error": scrub_secrets(str(exc))}
    payload = gj_foundation_status(configs)
    runtimes = getattr(app.state, "gj_bots", None) or {}
    for key, info in payload["bots"].items():
        rt = runtimes.get(key)
        info["telegram"] = "offline"
        info["username"] = ""
        info["paused"] = True
        if rt is not None:
            info["telegram"] = rt.mode
            info["username"] = rt.username
            info["paused"] = rt.paused
            if rt.last_error:
                info["telegram_error"] = rt.last_error
    return payload


def format_gj_report() -> str:
    status = gj_public_status()
    if "bots" not in status:
        return "GJ bots: " + status.get("error", "not loaded")
    lines = [f"GBPJPY bots (OANDA {status.get('broker_environment', 'practice')}, portfolio risk gate active):"]
    for info in status["bots"].values():
        who = f" @{info['username']}" if info.get("username") else ""
        state = "PAUSED" if info.get("paused", True) else "RUNNING"
        flag = "ON" if info.get("enabled") else "OFF"
        line = f"{info['name']}{who}: {state} (flag {flag}) | Telegram {info.get('telegram', 'offline')}"
        if not info.get("telegram_configured"):
            line += ", token missing"
        if info.get("config_error"):
            line += f"\n  {info['config_error']}"
        if info.get("telegram_error"):
            line += f"\n  {info['telegram_error']}"
        lines.append(line)
    lines.append("Send /status or /resume to each bot individually.")
    return "\n".join(lines)


# ---------- GBPJPY Sizing & Portfolio Risk Gate (Fail-Closed) ----------

def gj_usd_per_jpy(prices: dict[str, dict]) -> Decimal | None:
    """Convert 1 JPY to USD using USD_JPY mid price: 1 JPY = 1 / USD_JPY USD."""
    usd_jpy = prices.get(GJ_CONVERSION_PAIR)
    if not usd_jpy:
        return None
    m = mid_price(usd_jpy)
    if m <= 0:
        return None
    return Decimal(1) / m


def gj_size_order(
    spec: InstrumentSpec,
    direction: str,
    price: dict,
    stop_price: Decimal,
    target_price: Decimal,
    equity_usd: Decimal,
    risk_fraction: Decimal,
    usd_per_jpy: Decimal,
    units_multiplier: Decimal = Decimal(1),
) -> dict:
    """Size a GBP_JPY market order so stop-out loses equity_usd * risk_fraction * units_multiplier."""
    if equity_usd <= 0:
        raise SkipPair("account equity is non-positive")
    if usd_per_jpy <= 0:
        raise SkipPair("invalid JPY-to-USD conversion rate")

    bid = Decimal(str(price["bids"][0]["price"]))
    ask = Decimal(str(price["asks"][0]["price"]))
    entry = ask if direction == "BUY" else bid

    quantum = Decimal(1).scaleb(-spec.display_precision)
    stop = stop_price.quantize(quantum, rounding=ROUND_HALF_UP)
    target = target_price.quantize(quantum, rounding=ROUND_HALF_UP)

    if direction == "BUY" and not (stop < entry < target):
        raise SkipPair(f"invalid BUY bracket SL={stop} entry={entry} TP={target}")
    if direction == "SELL" and not (target < entry < stop):
        raise SkipPair(f"invalid SELL bracket SL={stop} entry={entry} TP={target}")

    loss_per_unit_usd = abs(entry - stop) * usd_per_jpy
    if loss_per_unit_usd <= 0:
        raise SkipPair("stop distance rounded to zero")

    risk_budget_usd = equity_usd * risk_fraction * units_multiplier
    unit_q = Decimal(1).scaleb(-spec.trade_units_precision)
    units = (risk_budget_usd / loss_per_unit_usd).quantize(unit_q, rounding=ROUND_DOWN)

    if units < spec.minimum_trade_size:
        needed = spec.minimum_trade_size * loss_per_unit_usd
        raise SkipPair(f"minimum size needs ${needed:.2f} risk (budget ${risk_budget_usd:.2f})")
    if spec.maximum_order_units > 0 and units > spec.maximum_order_units:
        units = spec.maximum_order_units

    sign = 1 if direction == "BUY" else -1
    signed_units = units * sign
    return {
        "order": {
            "units": format(signed_units, "f"),
            "instrument": spec.name,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": format(stop, "f")},
            "takeProfitOnFill": {"price": format(target, "f")},
        },
        "_entry": entry,
        "_stop": stop,
        "_target": target,
        "_risk_usd": units * loss_per_unit_usd,
        "_units": units,
    }


def _parse_oanda_time(raw: str) -> datetime | None:
    try:
        return datetime.strptime(str(raw)[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def count_weekly_trades(broker: OandaBroker, now: datetime | None = None) -> int:
    week_start = trading_week_start(now)
    trades = broker.all_trades(count=500)
    count = 0
    for t in trades:
        opened = _parse_oanda_time(str(t.get("openTime") or ""))
        if opened is not None and opened >= week_start:
            count += 1
    return count


def compute_trade_open_risk_usd(trade: dict, prices: dict[str, dict]) -> Decimal | None:
    """Compute open risk in USD for an existing OANDA trade. Returns None if SL missing or unconvertible."""
    sl_order = trade.get("stopLossOrder")
    if not isinstance(sl_order, dict) or sl_order.get("price") is None:
        return None
    try:
        units_signed = Decimal(str(trade.get("currentUnits") or "0"))
        units = abs(units_signed)
        entry = Decimal(str(trade.get("price")))
        sl_price = Decimal(str(sl_order["price"]))
    except Exception:
        return None
    if units == 0:
        return Decimal(0)

    # If stop is already at/beyond breakeven in profit direction, open risk from entry is 0
    if units_signed > 0 and sl_price >= entry:
        return Decimal(0)
    if units_signed < 0 and sl_price <= entry:
        return Decimal(0)

    inst = str(trade.get("instrument") or "")
    quote_ccy = inst.split("_")[1] if "_" in inst else ""
    if quote_ccy == "USD":
        usd_q = Decimal(1)
    elif quote_ccy == "JPY":
        usd_q = gj_usd_per_jpy(prices)
    else:
        direct = prices.get(f"{quote_ccy}_USD")
        inverse = prices.get(f"USD_{quote_ccy}")
        if direct:
            usd_q = mid_price(direct)
        elif inverse and mid_price(inverse) > 0:
            usd_q = Decimal(1) / mid_price(inverse)
        else:
            usd_q = None

    if usd_q is None:
        return None
    return abs(entry - sl_price) * units * usd_q


@dataclass
class PortfolioRiskSnapshot:
    ok: bool
    reason: str
    target_equity_usd: Decimal = Decimal(0)
    combined_equity_usd: Decimal = Decimal(0)
    combined_open_risk_usd: Decimal = Decimal(0)
    combined_daily_pnl_usd: Decimal = Decimal(0)
    target_weekly_trades: int = 0

    @property
    def open_risk_fraction(self) -> Decimal:
        if self.combined_equity_usd <= 0:
            return Decimal(1)
        return self.combined_open_risk_usd / self.combined_equity_usd


def evaluate_gj_portfolio_risk(
    target_key: str,
    runtimes: dict[str, GJRuntime],
    prices: dict[str, dict],
    now: datetime | None = None,
) -> PortfolioRiskSnapshot:
    """Fail-closed portfolio risk gate across all enabled GBPJPY accounts."""
    target_rt = runtimes.get(target_key)
    if target_rt is None or target_rt.broker is None:
        return PortfolioRiskSnapshot(ok=False, reason=f"{target_key} broker not initialized")
    if target_rt.config.config_error:
        return PortfolioRiskSnapshot(ok=False, reason=f"config error: {target_rt.config.config_error}")

    enabled_rts = [rt for rt in runtimes.values() if rt.config.enabled]
    if not enabled_rts:
        return PortfolioRiskSnapshot(ok=False, reason="no enabled GBPJPY bots")

    combined_equity = Decimal(0)
    combined_open_risk = Decimal(0)
    combined_daily_pnl = Decimal(0)
    target_equity = Decimal(0)
    target_weekly = 0

    for rt in enabled_rts:
        if rt.broker is None or rt.config.config_error:
            return PortfolioRiskSnapshot(
                ok=False,
                reason=f"fail-closed: {rt.config.display_name} broker unavailable or has config error",
            )
        try:
            acc = rt.broker.account_details()
            equity = Decimal(str(acc.get("NAV") or acc.get("balance") or "0"))
            realized, unrealized = rt.broker.daily_pnl()
            daily_pnl = realized + unrealized
            open_trades = rt.broker.open_trades()
            weekly_count = count_weekly_trades(rt.broker, now=now) if rt.config.key == target_key else 0
        except Exception as exc:
            return PortfolioRiskSnapshot(
                ok=False,
                reason=f"fail-closed: could not read {rt.config.display_name} account ({scrub_secrets(_short_error(exc))})",
            )

        if equity <= 0:
            return PortfolioRiskSnapshot(ok=False, reason=f"fail-closed: {rt.config.display_name} equity <= 0")

        # Per-account daily loss cap (2.00%)
        if daily_pnl <= -(equity * GJ_DAILY_LOSS_LIMIT):
            return PortfolioRiskSnapshot(
                ok=False,
                reason=f"daily loss cap reached on {rt.config.display_name}: {daily_pnl:.2f} (limit -{equity * GJ_DAILY_LOSS_LIMIT:.2f})",
            )

        for t in open_trades:
            risk_usd = compute_trade_open_risk_usd(t, prices)
            if risk_usd is None:
                return PortfolioRiskSnapshot(
                    ok=False,
                    reason=f"fail-closed: open trade #{t.get('id')} on {rt.config.display_name} lacks verifiable stop-loss",
                )
            combined_open_risk += risk_usd

        combined_equity += equity
        combined_daily_pnl += daily_pnl
        if rt.config.key == target_key:
            target_equity = equity
            target_weekly = weekly_count
            if any(str(t.get("instrument")) == GJ_INSTRUMENT for t in open_trades):
                return PortfolioRiskSnapshot(
                    ok=False,
                    reason=f"{rt.config.display_name} already has an open {GJ_INSTRUMENT} trade",
                    target_equity_usd=target_equity,
                    combined_equity_usd=combined_equity,
                    combined_open_risk_usd=combined_open_risk,
                    combined_daily_pnl_usd=combined_daily_pnl,
                    target_weekly_trades=target_weekly,
                )

    if target_weekly >= target_rt.config.weekly_trade_limit:
        return PortfolioRiskSnapshot(
            ok=False,
            reason=f"weekly trade cap reached ({target_weekly}/{target_rt.config.weekly_trade_limit})",
            target_equity_usd=target_equity,
            combined_equity_usd=combined_equity,
            combined_open_risk_usd=combined_open_risk,
            combined_daily_pnl_usd=combined_daily_pnl,
            target_weekly_trades=target_weekly,
        )

    if combined_daily_pnl <= -(combined_equity * GJ_DAILY_LOSS_LIMIT):
        return PortfolioRiskSnapshot(
            ok=False,
            reason=f"combined daily loss cap reached: {combined_daily_pnl:.2f} (limit -{combined_equity * GJ_DAILY_LOSS_LIMIT:.2f})",
            target_equity_usd=target_equity,
            combined_equity_usd=combined_equity,
            combined_open_risk_usd=combined_open_risk,
            combined_daily_pnl_usd=combined_daily_pnl,
            target_weekly_trades=target_weekly,
        )

    proposed_risk_usd = target_equity * target_rt.config.risk_fraction
    projected_open_risk = combined_open_risk + proposed_risk_usd
    if combined_equity <= 0 or (projected_open_risk / combined_equity) > GJ_PORTFOLIO_OPEN_RISK_LIMIT:
        pct_now = (combined_open_risk / combined_equity * 100) if combined_equity > 0 else Decimal(100)
        pct_proj = (projected_open_risk / combined_equity * 100) if combined_equity > 0 else Decimal(100)
        return PortfolioRiskSnapshot(
            ok=False,
            reason=f"combined open risk would be {pct_proj:.2f}% (currently {pct_now:.2f}%, cap {GJ_PORTFOLIO_OPEN_RISK_LIMIT:.2%})",
            target_equity_usd=target_equity,
            combined_equity_usd=combined_equity,
            combined_open_risk_usd=combined_open_risk,
            combined_daily_pnl_usd=combined_daily_pnl,
            target_weekly_trades=target_weekly,
        )

    return PortfolioRiskSnapshot(
        ok=True,
        reason="ok",
        target_equity_usd=target_equity,
        combined_equity_usd=combined_equity,
        combined_open_risk_usd=combined_open_risk,
        combined_daily_pnl_usd=combined_daily_pnl,
        target_weekly_trades=target_weekly,
    )


# ---------- GBPJPY Strategies ----------

@dataclass
class GJSignal:
    direction: str  # "BUY" or "SELL"
    setup_name: str
    reason: str
    stop_price: Decimal
    target_price: Decimal
    score: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)


def _completed_candles(candles: list[dict]) -> list[dict]:
    return [c for c in candles if c.get("complete") and isinstance(c.get("mid"), dict)]


def _trend_bias(candles: list[dict]) -> str | None:
    """Determine directional bias from completed candles using EMA20/EMA50 + close."""
    if len(candles) < 55:
        return None
    closes = [float(c["mid"]["c"]) for c in candles]
    e20, e50 = ema(closes, 20), ema(closes, 50)
    last_c = closes[-1]
    if e20 > e50 and last_c > e50:
        return "BUY"
    if e20 < e50 and last_c < e50:
        return "SELL"
    return None


def evaluate_kj_mclaurin_setup(
    d_candles: list[dict],
    h4_candles: list[dict],
    h1_candles: list[dict],
    m15_candles: list[dict],
    price: dict,
    mode: str = "intraday",
) -> tuple[GJSignal | None, str]:
    """KJ McLaurin framework:
    1. Daily + H4 trend alignment
    2. H1 liquidity sweep (prior H1 swing low/high swept & reclaimed)
    3. M15 structure shift (CHoCH / BOS)
    4. M15 Fair Value Gap (FVG) confirmation
    """
    d_comp = _completed_candles(d_candles)
    h4_comp = _completed_candles(h4_candles)
    h1_comp = _completed_candles(h1_candles)
    m15_comp = _completed_candles(m15_candles)

    if min(len(d_comp), len(h4_comp), len(h1_comp), len(m15_comp)) < 60:
        return None, "insufficient completed candles across D/H4/H1/M15"

    try:
        bid = Decimal(str(price["bids"][0]["price"]))
        ask = Decimal(str(price["asks"][0]["price"]))
    except Exception:
        return None, "bad live price"

    spread_pips = (ask - bid) * Decimal("100")
    if spread_pips > GJ_MAX_SPREAD_PIPS_KJ:
        return None, f"spread {spread_pips:.1f} pips exceeds {GJ_MAX_SPREAD_PIPS_KJ} pips"

    d_bias = _trend_bias(d_comp)
    h4_bias = _trend_bias(h4_comp)
    if not d_bias or d_bias != h4_bias:
        return None, f"D/H4 trend not aligned (D={d_bias}, H4={h4_bias})"

    direction = d_bias
    m15_atr = calculate_atr(m15_comp, 14)
    if m15_atr <= 0:
        return None, "M15 ATR is zero"

    # 2. H1 liquidity sweep within last 6 H1 bars against prior H1 swing [-30:-6]
    prior_h1 = h1_comp[-30:-6]
    recent_h1 = h1_comp[-6:]
    sweep_extreme: float | None = None
    if direction == "BUY":
        prior_low = min(float(c["mid"]["l"]) for c in prior_h1)
        for c in recent_h1:
            l, cl = float(c["mid"]["l"]), float(c["mid"]["c"])
            if l < prior_low and cl > prior_low:
                sweep_extreme = min(sweep_extreme, l) if sweep_extreme is not None else l
        if sweep_extreme is None:
            return None, f"no bullish H1 liquidity sweep below {prior_low:.3f}"
    else:
        prior_high = max(float(c["mid"]["h"]) for c in prior_h1)
        for c in recent_h1:
            h, cl = float(c["mid"]["h"]), float(c["mid"]["c"])
            if h > prior_high and cl < prior_high:
                sweep_extreme = max(sweep_extreme, h) if sweep_extreme is not None else h
        if sweep_extreme is None:
            return None, f"no bearish H1 liquidity sweep above {prior_high:.3f}"

    # 3. M15 structure shift in last 16 M15 bars
    m15_window = m15_comp[-24:]
    latest_m15 = m15_comp[-1]["mid"]
    latest_o, latest_h, latest_l, latest_c = (
        float(latest_m15["o"]), float(latest_m15["h"]), float(latest_m15["l"]), float(latest_m15["c"])
    )
    if direction == "BUY":
        ref_high = max(float(c["mid"]["h"]) for c in m15_window[:-6])
        shifted = any(float(c["mid"]["c"]) > ref_high for c in m15_window[-6:])
        if not shifted:
            return None, f"no M15 bullish structure shift above {ref_high:.3f}"
    else:
        ref_low = min(float(c["mid"]["l"]) for c in m15_window[:-6])
        shifted = any(float(c["mid"]["c"]) < ref_low for c in m15_window[-6:])
        if not shifted:
            return None, f"no M15 bearish structure shift below {ref_low:.3f}"

    # 4. M15 Fair Value Gap (FVG) confirmation in recent 10 bars
    fvg_zone: tuple[float, float] | None = None
    for i in range(len(m15_window) - 10, len(m15_window) - 1):
        if i < 2:
            continue
        c0 = m15_window[i - 2]["mid"]
        c2 = m15_window[i]["mid"]
        if direction == "BUY" and float(c2["l"]) > float(c0["h"]):
            fvg_zone = (float(c0["h"]), float(c2["l"]))
        elif direction == "SELL" and float(c2["h"]) < float(c0["l"]):
            fvg_zone = (float(c2["h"]), float(c0["l"]))

    if fvg_zone is None:
        return None, "no active M15 FVG in trade direction"

    fvg_low, fvg_high = fvg_zone
    if direction == "BUY":
        if not (latest_l <= fvg_high and latest_c >= fvg_low and latest_c > latest_o):
            return None, f"price has not confirmed M15 bullish FVG [{fvg_low:.3f}, {fvg_high:.3f}]"
    else:
        if not (latest_h >= fvg_low and latest_c <= fvg_high and latest_c < latest_o):
            return None, f"price has not confirmed M15 bearish FVG [{fvg_low:.3f}, {fvg_high:.3f}]"

    entry = ask if direction == "BUY" else bid
    buffer = Decimal(str(m15_atr * 0.25))
    sweep_d = Decimal(str(sweep_extreme))
    if direction == "BUY":
        stop = min(sweep_d - buffer, entry - Decimal(str(m15_atr * 1.0)))
        risk_dist = entry - stop
        rr = Decimal("2.0") if mode == "intraday" else Decimal("3.0")
        target = entry + risk_dist * rr
    else:
        stop = max(sweep_d + buffer, entry + Decimal(str(m15_atr * 1.0)))
        risk_dist = stop - entry
        rr = Decimal("2.0") if mode == "intraday" else Decimal("3.0")
        target = entry - risk_dist * rr

    if risk_dist <= 0:
        return None, "non-positive stop distance"

    return GJSignal(
        direction=direction,
        setup_name=f"KJ_{mode.upper()}",
        reason=f"D/H4 {direction}, H1 sweep @ {sweep_extreme:.3f}, M15 shift + FVG [{fvg_low:.3f}-{fvg_high:.3f}]",
        stop_price=stop,
        target_price=target,
        score=1.0,
        details={"rr": float(rr), "spread_pips": float(spread_pips), "m15_atr": m15_atr},
    ), "ok"


def _compute_asian_range(m5_candles: list[dict], now: datetime) -> tuple[float, float, float] | None:
    """Return (asian_high, asian_low, width_pips) for today's 00:00-07:00 UTC window."""
    today = now.astimezone(timezone.utc).date()
    asian_bars = []
    for c in m5_candles:
        t = _parse_oanda_time(str(c.get("time") or ""))
        if t is not None and t.date() == today and 0 <= t.hour < 7:
            asian_bars.append(c)
    if len(asian_bars) < 12:
        return None
    hi = max(float(c["mid"]["h"]) for c in asian_bars)
    lo = min(float(c["mid"]["l"]) for c in asian_bars)
    width_pips = (hi - lo) * 100.0
    return hi, lo, width_pips


def evaluate_scalp_setups(
    m5_candles: list[dict],
    m1_candles: list[dict],
    price: dict,
    now: datetime | None = None,
) -> tuple[list[GJSignal], dict[str, Any], str]:
    """Evaluate both scalp setups deterministically:
    1. M5 EMA20/50 momentum + M1 EMA20 pullback & engulfing
    2. Asian-range breakout / retest
    Returns (eligible_signals, market_context, diagnostic_reason).
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    m5_comp = _completed_candles(m5_candles)
    m1_comp = _completed_candles(m1_candles)
    if len(m5_comp) < 60 or len(m1_comp) < 30:
        return [], {}, "insufficient M5/M1 completed candles"

    try:
        bid = Decimal(str(price["bids"][0]["price"]))
        ask = Decimal(str(price["asks"][0]["price"]))
    except Exception:
        return [], {}, "bad live price"

    spread_pips = (ask - bid) * Decimal("100")
    m5_atr = calculate_atr(m5_comp, 14)
    if m5_atr <= 0 or (m5_atr * 100.0) < 3.0:
        return [], {}, f"M5 ATR too low ({m5_atr * 100.0:.1f} pips)"
    if spread_pips > GJ_MAX_SPREAD_PIPS_SCALP or (ask - bid) > Decimal(str(m5_atr * 0.25)):
        return [], {}, f"spread {spread_pips:.1f} pips too wide for M5 ATR {m5_atr * 100.0:.1f} pips"

    m5_closes = [float(c["mid"]["c"]) for c in m5_comp]
    m1_closes = [float(c["mid"]["c"]) for c in m1_comp]
    m5_e20, m5_e50 = ema(m5_closes, 20), ema(m5_closes, 50)
    m1_e20 = ema(m1_closes, 20)
    trend_strength = abs(m5_e20 - m5_e50) / m5_atr

    asian = _compute_asian_range(m5_comp, now)
    asian_quality = "unavailable"
    if asian is not None:
        _, _, w_pips = asian
        asian_quality = "good" if 15.0 <= w_pips <= 75.0 else "poor"

    ctx = {
        "utc_hour": now.hour,
        "spread_pips": round(float(spread_pips), 2),
        "m5_atr_pips": round(m5_atr * 100.0, 2),
        "trend_strength_atr": round(trend_strength, 2),
        "asian_range_quality": asian_quality,
    }

    candidates: list[GJSignal] = []

    # Setup 1: M5 EMA20/50 momentum + M1 EMA20 pullback & engulfing
    m1_prev = m1_comp[-2]["mid"]
    m1_curr = m1_comp[-1]["mid"]
    po, pc = float(m1_prev["o"]), float(m1_prev["c"])
    co, ch, cl, cc = float(m1_curr["o"]), float(m1_curr["h"]), float(m1_curr["l"]), float(m1_curr["c"])

    if trend_strength >= 0.25:
        if m5_e20 > m5_e50 and m5_closes[-1] > m5_e20:
            dipped = float(m1_prev["l"]) <= m1_e20 or cl <= m1_e20
            bull_engulf = cc > co and po > pc and cc >= po and co <= pc and cc > m1_e20
            if dipped and bull_engulf:
                entry = ask
                stop = min(Decimal(str(cl)), Decimal(str(m1_prev["l"]))) - Decimal(str(m5_atr * 0.25))
                risk_dist = entry - stop
                if risk_dist > 0:
                    candidates.append(GJSignal(
                        direction="BUY",
                        setup_name="EMA_PULLBACK",
                        reason=f"M5 EMA20>50 ({trend_strength:.2f}x ATR) + M1 EMA20 bullish engulfing",
                        stop_price=stop,
                        target_price=entry + risk_dist * Decimal("1.8"),
                        score=min(2.0, trend_strength) + 0.5,
                        details=ctx,
                    ))
        elif m5_e20 < m5_e50 and m5_closes[-1] < m5_e20:
            rallied = float(m1_prev["h"]) >= m1_e20 or ch >= m1_e20
            bear_engulf = cc < co and po < pc and cc <= po and co >= pc and cc < m1_e20
            if rallied and bear_engulf:
                entry = bid
                stop = max(Decimal(str(ch)), Decimal(str(m1_prev["h"]))) + Decimal(str(m5_atr * 0.25))
                risk_dist = stop - entry
                if risk_dist > 0:
                    candidates.append(GJSignal(
                        direction="SELL",
                        setup_name="EMA_PULLBACK",
                        reason=f"M5 EMA20<50 ({trend_strength:.2f}x ATR) + M1 EMA20 bearish engulfing",
                        stop_price=stop,
                        target_price=entry - risk_dist * Decimal("1.8"),
                        score=min(2.0, trend_strength) + 0.5,
                        details=ctx,
                    ))

    # Setup 2: Asian-range breakout / retest (after 07:00 UTC)
    if now.hour >= 7 and asian is not None and asian_quality == "good":
        a_hi, a_lo, w_pips = asian
        recent_m5 = m5_comp[-8:-1]
        last_m5 = m5_comp[-1]["mid"]
        lo5, lh5, ll5, lc5 = float(last_m5["o"]), float(last_m5["h"]), float(last_m5["l"]), float(last_m5["c"])
        retest_buf = m5_atr * 0.35

        broke_up = any(float(c["mid"]["c"]) > a_hi for c in recent_m5)
        if broke_up and ll5 <= a_hi + retest_buf and lc5 > a_hi and lc5 > lo5:
            entry = ask
            stop = Decimal(str(min(ll5, a_hi) - m5_atr * 0.5))
            risk_dist = entry - stop
            if risk_dist > 0:
                candidates.append(GJSignal(
                    direction="BUY",
                    setup_name="ASIAN_BREAKOUT_RETEST",
                    reason=f"Asian range ({w_pips:.1f} pips) bullish breakout & retest of {a_hi:.3f}",
                    stop_price=stop,
                    target_price=entry + risk_dist * Decimal("1.8"),
                    score=1.5 if 7 <= now.hour <= 16 else 1.0,
                    details=ctx,
                ))

        broke_dn = any(float(c["mid"]["c"]) < a_lo for c in recent_m5)
        if broke_dn and lh5 >= a_lo - retest_buf and lc5 < a_lo and lc5 < lo5:
            entry = bid
            stop = Decimal(str(max(lh5, a_lo) + m5_atr * 0.5))
            risk_dist = stop - entry
            if risk_dist > 0:
                candidates.append(GJSignal(
                    direction="SELL",
                    setup_name="ASIAN_BREAKOUT_RETEST",
                    reason=f"Asian range ({w_pips:.1f} pips) bearish breakout & retest of {a_lo:.3f}",
                    stop_price=stop,
                    target_price=entry - risk_dist * Decimal("1.8"),
                    score=1.5 if 7 <= now.hour <= 16 else 1.0,
                    details=ctx,
                ))

    if not candidates:
        return [], ctx, "neither EMA pullback nor Asian-range breakout/retest met conditions"
    return candidates, ctx, "ok"


# ---------- GBPJPY Execution & Multi-Session Management ----------

def _select_gj_signal(
    key: str,
    broker: OandaBroker,
    price: dict,
    ai_client: OpenAI | None,
    ai_model: str,
    now: datetime | None = None,
) -> tuple[GJSignal | None, str, str]:
    """Return (signal or None, selection_note, diagnostic_reason)."""
    if key in ("intraday", "swing"):
        d_c = broker.candles(GJ_INSTRUMENT, "D", 120)
        h4_c = broker.candles(GJ_INSTRUMENT, "H4", 120)
        h1_c = broker.candles(GJ_INSTRUMENT, "H1", 120)
        m15_c = broker.candles(GJ_INSTRUMENT, "M15", 120)
        sig, reason = evaluate_kj_mclaurin_setup(d_c, h4_c, h1_c, m15_c, price, mode=key)
        return sig, "", reason

    m5_c = broker.candles(GJ_INSTRUMENT, "M5", 180)
    m1_c = broker.candles(GJ_INSTRUMENT, "M1", 120)
    candidates, ctx, reason = evaluate_scalp_setups(m5_c, m1_c, price, now=now)
    if not candidates:
        return None, "", reason
    cand_dicts = [
        {
            "setup_name": c.setup_name,
            "direction": c.direction,
            "reason": c.reason,
            "score": c.score,
            "signal_obj": c,
        }
        for c in candidates
    ]
    chosen, sel_note = select_scalp_setup_with_ai(ai_client, ai_model, cand_dicts, ctx)
    return chosen["signal_obj"], sel_note, "ok"


def run_gj_cycle(key: str, runtimes: dict[str, GJRuntime], opinion_fn: Callable[[str, dict], dict], ai_client: OpenAI | None, ai_model: str) -> ScanResult:
    rt = runtimes.get(key)
    if rt is None or rt.broker is None:
        return ScanResult(report=f"{key} broker offline.", failed=1, alert=True)
    broker = rt.broker
    try:
        specs = broker.instrument_specs(GJ_REQUIRED_INSTRUMENTS)
        spec = specs.get(GJ_INSTRUMENT)
        if spec is None:
            return ScanResult(report=f"{GJ_INSTRUMENT} is not offered on this account.", failed=1, alert=True)
        prices = broker.prices(list(GJ_REQUIRED_INSTRUMENTS))
        price = prices.get(GJ_INSTRUMENT)
        if not price or not price.get("tradeable", True):
            return ScanResult(report="GBP/JPY is not currently tradeable.")
        usd_jpy_rate = gj_usd_per_jpy(prices)
        if usd_jpy_rate is None:
            return ScanResult(report="Scan failed: missing USD_JPY conversion price.", failed=1, alert=True)

        # 1. Fail-closed portfolio risk gate
        risk_snap = evaluate_gj_portfolio_risk(key, runtimes, prices)
        if not risk_snap.ok:
            return ScanResult(report=f"Blocked by risk gate: {risk_snap.reason}")

        # 2. Evaluate strategy
        signal, sel_note, diag = _select_gj_signal(key, broker, price, ai_client, ai_model)
        if signal is None:
            return ScanResult(report=f"Scan complete. No setup met rules ({diag}).")

        # 3. Position sizing (JPY -> USD)
        built = gj_size_order(
            spec=spec,
            direction=signal.direction,
            price=price,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            equity_usd=risk_snap.target_equity_usd,
            risk_fraction=rt.config.risk_fraction,
            usd_per_jpy=usd_jpy_rate,
        )

        # 4. Advisory AI note (cannot block or invent a trade)
        opinion = opinion_fn("GBP/JPY", {"direction": signal.direction, "setup": signal.setup_name, "reason": signal.reason})
        status, reject = broker.submit(built)
        if status == "filled":
            msg = (
                f"TRADE EXECUTED ({rt.config.display_name} - GBP/JPY)\n"
                f"{signal.direction} {built['_units']} units @ {format(built['_entry'], 'f')}\n"
                f"SL {format(built['_stop'], 'f')} | TP {format(built['_target'], 'f')} | "
                f"risk ~${built['_risk_usd']:.2f} ({rt.config.risk_fraction:.2%})\n"
                f"Setup: {signal.reason}\n"
            )
            if sel_note:
                msg += f"Selector: {sel_note}\n"
            msg += ai_note(opinion)
            return ScanResult(report=msg, executed=1, alert=True, entered=[GJ_INSTRUMENT])
        return ScanResult(report=f"ORDER FAILED ({rt.config.display_name}): {reject}", failed=1, alert=True)
    except SkipPair as exc:
        return ScanResult(report=f"Skipped GBP/JPY: {exc}")
    except Exception as exc:
        logger.exception("GJ %s scan error", key)
        return ScanResult(report=f"Scan error: {scrub_secrets(_short_error(exc))}", failed=1, alert=True)


def check_gj_pair_text(key: str, runtimes: dict[str, GJRuntime], ai_client: OpenAI | None, ai_model: str) -> str:
    rt = runtimes.get(key)
    if rt is None or rt.broker is None:
        return f"{key} broker is not initialized."
    broker = rt.broker
    specs = broker.instrument_specs(GJ_REQUIRED_INSTRUMENTS)
    spec = specs.get(GJ_INSTRUMENT)
    if not spec:
        return f"{GJ_INSTRUMENT} is not offered on this account."
    prices = broker.prices(list(GJ_REQUIRED_INSTRUMENTS))
    price = prices.get(GJ_INSTRUMENT)
    if not price:
        return "No live price for GBP/JPY."
    usd_jpy_rate = gj_usd_per_jpy(prices)
    risk_snap = evaluate_gj_portfolio_risk(key, runtimes, prices)
    signal, sel_note, diag = _select_gj_signal(key, broker, price, ai_client, ai_model)
    spread = Decimal(str(price["asks"][0]["price"])) - Decimal(str(price["bids"][0]["price"]))
    lines = [
        f"{rt.config.display_name} — GBP/JPY ({rt.config.oanda_env})",
        f"Risk gate: {'PASS' if risk_snap.ok else 'BLOCKED — ' + risk_snap.reason}",
        f"Weekly trades: {risk_snap.target_weekly_trades}/{rt.config.weekly_trade_limit} | Spread: {spread * 100:.1f} pips",
        f"Signal: {signal.direction + ' (' + signal.setup_name + ')' if signal else 'none'}",
        f"Reason: {signal.reason if signal else diag}",
    ]
    if sel_note:
        lines.append(f"Selector: {sel_note}")
    if signal and usd_jpy_rate is not None and risk_snap.target_equity_usd > 0:
        try:
            built = gj_size_order(
                spec, signal.direction, price, signal.stop_price, signal.target_price,
                risk_snap.target_equity_usd, rt.config.risk_fraction, usd_jpy_rate,
            )
            lines.append(
                f"Would size {built['_units']} units (~${built['_risk_usd']:.2f} / {rt.config.risk_fraction:.2%}), "
                f"SL {built['_stop']}, TP {built['_target']}"
            )
        except SkipPair as exc:
            lines.append(f"Would NOT size: {exc}")
    return "\n".join(lines)


def manage_swing_positions(rt: GJRuntime) -> str | None:
    """Multi-session management for Swing: move SL to breakeven once >= +1.5R."""
    if rt.broker is None or rt.config.key != "swing":
        return None
    trades = rt.broker.open_trades()
    if not trades:
        return None
    prices = rt.broker.prices([GJ_INSTRUMENT])
    price = prices.get(GJ_INSTRUMENT)
    if not price:
        return None
    specs = rt.broker.instrument_specs((GJ_INSTRUMENT,))
    spec = specs.get(GJ_INSTRUMENT)
    if not spec:
        return None

    bid = Decimal(str(price["bids"][0]["price"]))
    ask = Decimal(str(price["asks"][0]["price"]))
    moved: list[str] = []
    for t in trades:
        if str(t.get("instrument")) != GJ_INSTRUMENT:
            continue
        sl_order = t.get("stopLossOrder")
        if not isinstance(sl_order, dict) or sl_order.get("price") is None:
            continue
        units = Decimal(str(t.get("currentUnits") or "0"))
        entry = Decimal(str(t.get("price")))
        sl = Decimal(str(sl_order["price"]))
        if units > 0 and sl < entry:
            risk = entry - sl
            if bid >= entry + risk * Decimal("1.5"):
                be = entry + Decimal("0.05")
                rt.broker.update_trade_sl(str(t["id"]), be, spec.display_precision)
                moved.append(f"#{t['id']} SL -> breakeven {be:.3f}")
        elif units < 0 and sl > entry:
            risk = sl - entry
            if ask <= entry - risk * Decimal("1.5"):
                be = entry - Decimal("0.05")
                rt.broker.update_trade_sl(str(t["id"]), be, spec.display_precision)
                moved.append(f"#{t['id']} SL -> breakeven {be:.3f}")
    return ("Swing management: " + ", ".join(moved)) if moved else None


# ---------- Telegram Transport & Webhook Helpers ----------

def make_bot(token: str) -> Bot:
    try:
        from telegram.request import HTTPXRequest
        request = HTTPXRequest(connect_timeout=10.0, read_timeout=30.0, write_timeout=10.0, pool_timeout=5.0)
        return Bot(token=token, request=request)
    except Exception as exc:
        logger.warning("Using default Telegram client: %s", exc.__class__.__name__)
        return Bot(token=token)


def webhook_secret_from_env() -> str:
    settings = getattr(app.state, "settings", None)
    secret = settings.webhook_secret if settings is not None else os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if secret and not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", secret):
        return ""
    return secret


def webhook_secret_ok(request: Request, secret: str) -> bool:
    if not secret:
        return True
    received = request.headers.get("x-telegram-bot-api-secret-token", "")
    return bool(hmac.compare_digest(received, secret))


async def bind_telegram(bot: Bot, hook_url: str, secret: str) -> tuple[str, str]:
    try:
        await bot.initialize()
    except Exception as exc:
        return "offline", scrub_secrets(_short_error(exc))
    webhook_error = ""
    if hook_url.startswith("https://"):
        try:
            kwargs: dict[str, Any] = {"allowed_updates": ["message"]}
            if secret:
                kwargs["secret_token"] = secret
            await bot.set_webhook(url=hook_url, **kwargs)
            return "webhook", ""
        except Exception as exc:
            webhook_error = scrub_secrets(_short_error(exc))
            logger.warning("Webhook rejected (%s); falling back to polling", webhook_error)
    try:
        await bot.delete_webhook(drop_pending_updates=False)
    except Exception as exc:
        return "offline", scrub_secrets(_short_error(exc))
    if webhook_error:
        return "polling", f"Webhook failed ({webhook_error}). Using polling instead."
    return "polling", ""


async def poll_loop(bot: Bot, seen: OrderedDict, on_update) -> None:
    offset = None
    while True:
        try:
            updates = await bot.get_updates(offset=offset, timeout=20, allowed_updates=["message"])
            for update in updates:
                offset = update.update_id + 1
                data = update.to_dict()
                if remember_update(seen, data.get("update_id")):
                    continue
                try:
                    await on_update(data)
                except Exception:
                    logger.exception("Telegram update handler failed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if exc.__class__.__name__ in {"TimedOut", "NetworkError"}:
                continue
            logger.warning("Telegram poll error: %s", scrub_secrets(_short_error(exc)))
            await asyncio.sleep(5)


async def gj_say(key: str, text: str, chat_id: str | None = None) -> bool:
    runtime = app.state.gj_bots.get(key) if getattr(app.state, "gj_bots", None) else None
    if runtime is None or runtime.bot is None:
        return False
    target = chat_id or runtime.config.telegram_chat_id
    if not target:
        return False
    sent = False
    for part in chunk_text(text):
        try:
            await runtime.bot.send_message(chat_id=target, text=part)
            sent = True
        except Exception as exc:
            runtime.last_error = scrub_secrets(_short_error(exc))
            logger.warning("GJ %s send failed: %s", key, runtime.last_error)
            return False
        await asyncio.sleep(0.3)
    return sent


def gj_help(key: str) -> str:
    cfg = app.state.gj_bots[key].config
    strat_desc = {
        "intraday": "KJ McLaurin (D/H4 trend, H1 sweep, shift, M15 FVG; 2R target)",
        "swing": "KJ McLaurin (D/H4 trend, H1 sweep, shift, M15 FVG; 3R + multi-session BE)",
        "scalp": "Auto-select: M5/M1 EMA20/50 pullback vs Asian-range breakout/retest",
    }[key]
    return (
        f"{cfg.display_name} ({cfg.oanda_env})\n"
        f"Strategy: {strat_desc}\n\n"
        "/status - balance, open trades, risk gate, paused/running\n"
        "/resume - start automatic trading for this bot\n"
        "/pause - stop new entries (exits stay on OANDA)\n"
        "/trade - scan GBP/JPY now\n"
        "/check - what the strategy & risk gate see for GBP/JPY\n"
        "/closelosers - close all losing trades on this account\n"
        "/stats [days] - win rate, avg win/loss, P/L by day\n"
        "/today - today's P/L on this account\n"
        "/instruments - check GBP/JPY and USD/JPY availability\n"
        "/ai - test OpenRouter connection\n"
        "/gj - status of all 3 GBPJPY bots\n"
        "/id - this chat's id\n"
        "/ping - health check"
    )


async def run_gj_scan(key: str) -> ScanResult | None:
    rt = app.state.gj_bots.get(key)
    if rt is None or rt.broker is None:
        return None
    now = datetime.now(timezone.utc)
    if not market_open(now):
        return ScanResult(report="Forex market is closed (weekend). No new entries.")
    if rt.scanning:
        return None
    bkey = bar_key(now, rt.config.bar_minutes)
    if rt.entered_bar == bkey:
        return ScanResult(report=f"Already entered on current {rt.config.bar_minutes}m bar.")
    rt.scanning = True
    try:
        opinion_fn = app.state.opinion_fn or (lambda p, s: {"agree": True, "comment": "AI offline"})
        model = app.state.settings.openrouter_model if app.state.settings else os.getenv("OPENROUTER_MODEL", "openrouter/free")
        result = await asyncio.to_thread(
            run_gj_cycle, key, app.state.gj_bots, opinion_fn, app.state.ai_client, model
        )
    finally:
        rt.scanning = False
    if result.executed > 0:
        rt.entered_bar = bkey
    return result


async def gj_scan_loop(key: str) -> None:
    rt = app.state.gj_bots[key]
    while True:
        await asyncio.sleep(rt.config.scan_interval_seconds)
        if not rt.config.enabled or rt.config.config_error or rt.paused or rt.broker is None:
            continue
        try:
            result = await run_gj_scan(key)
            if result and result.alert:
                await gj_say(key, result.report)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("GJ %s scan loop error", key)


async def gj_housekeeping_loop(key: str) -> None:
    rt = app.state.gj_bots[key]
    while True:
        await asyncio.sleep(HOUSEKEEPING_SECONDS)
        if not rt.config.enabled or rt.broker is None:
            continue
        try:
            today = trading_day_start().isoformat()
            if rt.loss_alert_day != today:
                acc = await asyncio.to_thread(rt.broker.account_details)
                equity = Decimal(str(acc.get("NAV") or acc.get("balance") or "0"))
                realized, unrealized = await asyncio.to_thread(rt.broker.daily_pnl)
                total = realized + unrealized
                if equity > 0 and total <= -(equity * GJ_DAILY_LOSS_LIMIT):
                    rt.loss_alert_day = today
                    rt.paused = True
                    await gj_say(
                        key,
                        f"DAILY LOSS CAP HIT ({rt.config.display_name}): {total:.2f} "
                        f"(limit -{equity * GJ_DAILY_LOSS_LIMIT:.2f}). Bot auto-PAUSED.",
                    )
            if key == "swing" and not rt.paused:
                note = await asyncio.to_thread(manage_swing_positions, rt)
                if note:
                    await gj_say(key, note)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("GJ %s housekeeping error", key)


async def handle_gj_command(key: str, cmd: str, args: list[str] | None = None) -> None:
    args = args or []
    runtime = app.state.gj_bots[key]
    cfg = runtime.config
    definition = GJ_BOT_DEFINITIONS[key]
    try:
        if cmd in ("/start", "/help"):
            await gj_say(key, gj_help(key))
        elif cmd == "/ping":
            await gj_say(key, f"pong ({runtime.mode})")
        elif cmd == "/id":
            await gj_say(key, f"This chat's ID is: {cfg.telegram_chat_id or 'not locked'}")
        elif cmd == "/gj":
            await gj_say(key, format_gj_report())
        elif cmd == "/status":
            state = "PAUSED" if runtime.paused else "RUNNING"
            lines = [
                cfg.display_name + (f" (@{runtime.username})" if runtime.username else ""),
                f"State: {state} | Env: {cfg.oanda_env} | Flag {definition['env']}: {'ON' if cfg.enabled else 'OFF'}",
                f"Telegram: {runtime.mode} | Risk: {cfg.risk_fraction:.2%} | Weekly cap: {cfg.weekly_trade_limit}",
            ]
            if cfg.config_error:
                lines.append("Config error: " + cfg.config_error)
            if runtime.last_error:
                lines.append("Note: " + runtime.last_error)
            if runtime.broker is not None:
                bal, pnl, count = await asyncio.to_thread(runtime.broker.account_summary)
                weekly = await asyncio.to_thread(count_weekly_trades, runtime.broker)
                prices = await asyncio.to_thread(runtime.broker.prices, list(GJ_REQUIRED_INSTRUMENTS))
                snap = await asyncio.to_thread(evaluate_gj_portfolio_risk, key, app.state.gj_bots, prices)
                lines += [
                    f"Balance: {bal} | Unrealized: {pnl} | Open trades: {count}",
                    f"Weekly trades: {weekly}/{cfg.weekly_trade_limit}",
                    f"Portfolio gate: {'PASS' if snap.ok else 'BLOCKED (' + snap.reason + ')'}",
                ]
            else:
                lines.append(f"Account not connected (set {definition['account']} and OANDA_API_KEY).")
            await gj_say(key, "\n".join(lines))
        elif cmd == "/resume":
            if not cfg.enabled:
                await gj_say(key, f"Cannot resume: {definition['env']} is not enabled.")
                return
            if cfg.config_error:
                await gj_say(key, f"Cannot resume due to config error: {cfg.config_error}")
                return
            runtime.paused = False
            await gj_say(
                key,
                f"{cfg.display_name} RESUMED ({cfg.oanda_env}). "
                f"Scans every {cfg.scan_interval_seconds}s with portfolio risk gate active.",
            )
        elif cmd == "/pause":
            runtime.paused = True
            await gj_say(key, f"{cfg.display_name} PAUSED. No new entries; open trades keep OANDA SL/TP.")
        elif cmd == "/trade":
            if not cfg.enabled or cfg.config_error:
                await gj_say(key, f"Bot not active: {cfg.config_error or definition['env'] + ' is OFF'}")
                return
            if runtime.paused:
                await gj_say(key, "Trading is paused. Send /resume first.")
                return
            await gj_say(key, f"Manual scan triggered for {cfg.display_name}...")
            result = await run_gj_scan(key)
            await gj_say(key, "A scan is already running." if result is None else result.report)
        elif cmd == "/check":
            if runtime.broker is None:
                await gj_say(key, "Broker not connected.")
                return
            model = app.state.settings.openrouter_model if app.state.settings else os.getenv("OPENROUTER_MODEL", "openrouter/free")
            text = await asyncio.to_thread(check_gj_pair_text, key, app.state.gj_bots, app.state.ai_client, model)
            await gj_say(key, text)
        elif cmd == "/instruments":
            if runtime.broker is None:
                await gj_say(key, "Broker not connected.")
                return
            specs = await asyncio.to_thread(runtime.broker.instrument_specs, GJ_REQUIRED_INSTRUMENTS)
            await gj_say(
                key,
                "Required instruments on this account:\n" + "\n".join(sorted(specs))
                + ("\nMissing: " + ", ".join(p for p in GJ_REQUIRED_INSTRUMENTS if p not in specs)
                   if any(p not in specs for p in GJ_REQUIRED_INSTRUMENTS) else ""),
            )
        elif cmd == "/today":
            if runtime.broker is None:
                await gj_say(key, "Broker not connected.")
                return
            realized, unrealized = await asyncio.to_thread(runtime.broker.daily_pnl)
            await gj_say(key, f"TODAY'S P/L ({cfg.display_name})\nRealized: {realized:.2f}\nUnrealized: {unrealized:.2f}\nTotal: {realized + unrealized:.2f}")
        elif cmd == "/stats":
            if runtime.broker is None:
                await gj_say(key, "Broker not connected.")
                return
            usage = "Usage: /stats or /stats 30 (days, 1 to 365)"
            try:
                days = int(args[0]) if args else 7
            except ValueError:
                await gj_say(key, usage)
                return
            if not 1 <= days <= 365:
                await gj_say(key, usage)
                return
            trades = await asyncio.to_thread(runtime.broker.closed_trades)
            report = compute_stats(trades, days)
            if len(trades) >= 2000:
                report += "\n\nNote: only the 2,000 most recent closed trades were read."
            await gj_say(key, report)
        elif cmd in ("/closelosers", "/closelosses"):
            if runtime.broker is None:
                await gj_say(key, "Broker not connected.")
                return
            await gj_say(key, f"Closing all losing trades on {cfg.display_name}...")
            closed, failed, kept = await asyncio.to_thread(runtime.broker.close_losing_trades)
            lines = [f"CLOSED {len(closed)} losing trade(s); kept {kept} winner(s)."] + closed
            if failed:
                lines += ["FAILED:"] + failed
            await gj_say(key, "\n".join(lines))
        elif cmd == "/ai":
            client = app.state.ai_client
            if client is None:
                await gj_say(key, "OpenRouter client is not loaded. Check OPENROUTER_API_KEY in Render.")
            else:
                model = app.state.settings.openrouter_model if app.state.settings else os.getenv("OPENROUTER_MODEL", "openrouter/free")
                try:
                    resp = await asyncio.to_thread(
                        client.chat.completions.create,
                        model=model,
                        messages=[{"role": "user", "content": "Reply with the single word OK."}],
                        max_tokens=10,
                    )
                    await gj_say(key, f"OpenRouter ONLINE\nModel: {model}\nReply: {(resp.choices[0].message.content or '').strip()[:100]}")
                except Exception as exc:
                    await gj_say(key, f"OpenRouter OFFLINE\nModel: {model}\nError: {_short_error(exc)}")
        else:
            await gj_say(key, f"Unknown command: {cmd}\nSend /help")
    except Exception as exc:
        logger.exception("GJ %s command failed", key)
        await gj_say(key, "Error: " + scrub_secrets(_short_error(exc)))


async def handle_gj_update(key: str, data: dict) -> None:
    runtime = app.state.gj_bots.get(key) if getattr(app.state, "gj_bots", None) else None
    if runtime is None or runtime.bot is None:
        return
    msg = data.get("message")
    if not isinstance(msg, dict) or "text" not in msg:
        return
    sender = str((msg.get("chat") or {}).get("id"))
    text = str(msg.get("text") or "")
    cmd = parse_command(text)
    args = parse_args(text)
    allowed = runtime.config.telegram_chat_id
    if allowed and sender != allowed:
        logger.warning("GJ %s rejected chat %s", key, sender)
        if cmd.startswith("/"):
            chat_env = GJ_BOT_DEFINITIONS[key]["chat"]
            await gj_say(
                key,
                f"This chat's ID is: {sender}\nThis bot only answers {allowed}.\n"
                f"To use this chat, set {chat_env} to {sender} and redeploy.",
                chat_id=sender,
            )
        return
    if cmd == "/id":
        await gj_say(key, f"This chat's ID is: {sender}", chat_id=sender)
        if not allowed:
            await gj_say(
                key,
                f"Chat is not locked yet. Set {GJ_BOT_DEFINITIONS[key]['chat']}={sender} "
                "(or TELEGRAM_CHAT_ID) so nobody else can use this bot.",
                chat_id=sender,
            )
        return
    await handle_gj_command(key, cmd, args)


async def start_gj_bots() -> None:
    try:
        configs = load_gj_bot_configs()
    except Exception as exc:
        app.state.gj_config_error = scrub_secrets(str(exc))
        app.state.gj_configs = {}
        app.state.gj_bots = {}
        logger.exception("GJ config load failed")
        return
    app.state.gj_configs = configs
    app.state.gj_bots = {}
    public_url = os.getenv("PUBLIC_URL", "").strip()
    settings = app.state.settings
    if settings is not None and settings.public_url:
        public_url = settings.public_url
    secret = webhook_secret_from_env()
    oanda_api_key = os.getenv("OANDA_API_KEY", "").strip()

    for key, cfg in configs.items():
        broker = OandaBroker(oanda_api_key, cfg.oanda_account_id, cfg.oanda_base) if (oanda_api_key and cfg.oanda_account_id) else None
        runtime = GJRuntime(config=cfg, broker=broker, paused=True)
        app.state.gj_bots[key] = runtime
        if cfg.config_error:
            logger.error("GJ %s config: %s", key, cfg.config_error)
        if not cfg.telegram_token:
            runtime.last_error = f"Set {GJ_BOT_DEFINITIONS[key]['token']} so this bot can reply on Telegram."
            logger.warning("GJ %s has no Telegram token", key)
            continue
        bot = make_bot(cfg.telegram_token)
        runtime.bot = bot
        try:
            mode, note = await bind_telegram(bot, gj_webhook_url(public_url, key), secret)
            runtime.mode = mode
            if note:
                runtime.last_error = note
            if mode == "offline":
                logger.error("GJ %s Telegram offline: %s", key, note)
                continue
            try:
                me = await bot.get_me()
                runtime.username = getattr(me, "username", "") or ""
            except Exception as exc:
                runtime.last_error = scrub_secrets(_short_error(exc))
                logger.warning("GJ %s get_me failed: %s", key, runtime.last_error)
            logger.info("GJ %s Telegram %s as @%s", key, mode, runtime.username)
            if mode == "polling":
                runtime.poll_task = asyncio.create_task(poll_loop(bot, runtime.seen, lambda data, key=key: handle_gj_update(key, data)))
            if cfg.enabled and not cfg.config_error and broker is not None:
                runtime.scan_task = asyncio.create_task(gj_scan_loop(key))
                runtime.housekeeping_task = asyncio.create_task(gj_housekeeping_loop(key))
            if cfg.telegram_chat_id:
                await gj_say(key, f"{cfg.display_name} is online ({mode}, PAUSED, {cfg.oanda_env}).\nSend /status or /resume")
        except Exception as exc:
            runtime.mode = "offline"
            runtime.last_error = scrub_secrets(_short_error(exc))
            logger.exception("GJ %s Telegram setup failed", key)


async def stop_gj_bots() -> None:
    runtimes = list((getattr(app.state, "gj_bots", None) or {}).values())
    tasks = []
    for runtime in runtimes:
        for t in (runtime.poll_task, runtime.scan_task, runtime.housekeeping_task):
            if t is not None:
                tasks.append(t)
    for task in tasks:
        task.cancel()
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("GJ task shutdown failed")
    for runtime in runtimes:
        if runtime.broker is not None:
            runtime.broker.close()
        if runtime.bot is not None:
            try:
                await runtime.bot.shutdown()
            except Exception:
                logger.exception("GJ %s shutdown failed", runtime.config.key)


# ---------- app ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks: list[asyncio.Task] = []
    settings = None
    try:
        settings = load_settings()
    except RuntimeError as exc:
        logger.error("Configuration error: %s", exc)
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        app.state.ai_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_key, timeout=30)
        model = settings.openrouter_model if settings else (os.getenv("OPENROUTER_MODEL", "openrouter/free").strip() or "openrouter/free")
        app.state.opinion_fn = lambda pair, signal: get_openrouter_opinion(app.state.ai_client, model, pair, signal)

    if settings is not None:
        bot = make_bot(settings.telegram_token)
        broker = OandaBroker.from_settings(settings)
        app.state.settings = settings
        app.state.bot = bot
        app.state.broker = broker
        try:
            mode, note = await bind_telegram(
                bot,
                webhook_url(settings.public_url) if settings.public_url else "",
                settings.webhook_secret,
            )
            app.state.main_telegram_mode = mode
            if note:
                logger.warning("Main Telegram: %s", note)
            if mode == "polling":
                tasks.append(asyncio.create_task(poll_loop(bot, app.state.seen_updates, dispatch_main_update)))
            elif mode == "offline":
                logger.error("Main Telegram offline: %s", note)
        except Exception:
            logger.exception("Main Telegram setup failed")
        tasks.append(asyncio.create_task(scan_loop()))
        tasks.append(asyncio.create_task(housekeeping_loop()))
    try:
        await start_gj_bots()
    except Exception:
        logger.exception("GJ Telegram setup failed")
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        if app.state.bot is not None:
            try:
                await app.state.bot.shutdown()
            except Exception:
                logger.exception("Main Telegram shutdown failed")
        if app.state.broker is not None:
            app.state.broker.close()
        await stop_gj_bots()


app = FastAPI(lifespan=lifespan)
app.state.settings = None
app.state.bot = None
app.state.broker = None
app.state.ai_client = None
app.state.opinion_fn = None
app.state.paused = True
app.state.scanning = False
app.state.seen_updates = OrderedDict()
app.state.tasks = set()
app.state.entered_bar = {}
app.state.loss_alert_day = None
app.state.main_telegram_mode = "offline"
app.state.gj_configs = {}
app.state.gj_bots = {}
app.state.gj_config_error = ""


async def say(text: str, chat_id: str | None = None) -> None:
    bot, settings = app.state.bot, app.state.settings
    if not bot or not settings:
        return
    for part in chunk_text(text):
        try:
            await bot.send_message(chat_id=chat_id or settings.telegram_chat_id, text=part)
        except Exception:
            logger.exception("Failed to send Telegram message")
            return
        await asyncio.sleep(0.3)


async def housekeeping_loop() -> None:
    while True:
        await asyncio.sleep(HOUSEKEEPING_SECONDS)
        if not app.state.settings:
            continue
        try:
            msg = await check_loss_alert()
            if msg:
                await say(msg)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Housekeeping loop error")


async def scan_loop() -> None:
    while True:
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)
        if not app.state.settings or app.state.paused:
            continue
        try:
            result = await run_scan()
            if result and result.alert:
                await say(result.report)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Scan loop error")


async def check_loss_alert() -> str | None:
    settings = app.state.settings
    today = trading_day_start().isoformat()
    if app.state.loss_alert_day == today:
        return None
    realized, unrealized = await asyncio.to_thread(app.state.broker.daily_pnl)
    total = realized + unrealized
    if total > -settings.daily_loss_alert:
        return None
    app.state.loss_alert_day = today
    return (f"LOSS ALERT: today's P/L {total:.2f} (alert level -{settings.daily_loss_alert}). "
            "The bot keeps running. Send /pause to stop new trades, or /closelosers.")


async def run_scan() -> ScanResult | None:
    now = datetime.now(timezone.utc)
    if not market_open(now):
        return ScanResult(report="Forex market is closed (weekend). No new entries.")
    settings = app.state.settings
    if not session_open(now, settings.trading_session):
        return ScanResult(report=f"Outside the '{settings.trading_session}' session window. No new entries.")
    if app.state.scanning:
        return None
    bar = bar_key(now)
    blocked = {name for name, key in app.state.entered_bar.items() if key == bar}
    app.state.scanning = True
    try:
        result = await asyncio.to_thread(run_cycle, app.state.broker, settings, app.state.opinion_fn, blocked)
    finally:
        app.state.scanning = False
    for name in result.entered:
        app.state.entered_bar[name] = bar
    return result


async def handle_command(cmd: str, args: list[str] | None = None) -> None:
    args = args or []
    try:
        if cmd in ("/start", "/help"):
            await say(
                "Tris Bot (H1 pullback strategy)\n\n"
                "/status - balance, open trades, paused or running\n"
                "/resume - start automatic trading\n"
                "/pause - stop new entries (exits stay on OANDA)\n"
                "/trade - scan now\n"
                "/check EUR_USD - what the rules see for one pair\n"
                "/closelosers - close all losing trades now\n"
                "/stats [days] - win rate, avg win/loss, P/L by day\n"
                "/today - today's P/L\n"
                "/instruments [filter] - tradable instruments\n"
                "/ai - test OpenRouter connection\n"
                "/gj - whether the 3 GBPJPY Telegram bots are online\n"
                "/id - this chat's id"
            )
        elif cmd == "/status":
            bal, pnl, count = await asyncio.to_thread(app.state.broker.account_summary)
            state = "PAUSED" if app.state.paused else "RUNNING"
            await say(f"STATUS ({state}, {app.state.settings.oanda_env})\nBalance: {bal}\nUnrealized: {pnl}\n"
                      f"Open trades: {count} (max {MAX_OPEN_TRADES})\nRisk per trade: ${RISK_PER_TRADE_USD}")
        elif cmd == "/resume":
            app.state.paused = False
            await say("Trading resumed. Automatic scans run every 5 minutes.")
        elif cmd == "/pause":
            app.state.paused = True
            await say("Trading paused. No new entries. Open trades keep their stop and target on OANDA.")
        elif cmd == "/check":
            name = (args[0] if args else "EUR_USD").upper().replace("/", "_")
            text = await asyncio.to_thread(check_pair_text, app.state.broker, name)
            await say(text)
        elif cmd == "/instruments":
            specs = await asyncio.to_thread(app.state.broker.instrument_specs, PAIRS)
            await say("Configured pairs on this account:\n" + "\n".join(sorted(specs)) +
                      "\nNot offered: " + ", ".join(p for p in PAIRS if p not in specs))
        elif cmd == "/trade":
            if app.state.paused:
                await say("Trading is paused. Send /resume first.")
                return
            await say("Manual scan triggered...")
            result = await run_scan()
            await say("A scan is already running." if result is None else result.report)
        elif cmd == "/today":
            realized, unrealized = await asyncio.to_thread(app.state.broker.daily_pnl)
            await say(f"TODAY'S P/L\nRealized: {realized:.2f}\nUnrealized: {unrealized:.2f}\n"
                      f"Total: {realized + unrealized:.2f}")
        elif cmd == "/stats":
            usage = "Usage: /stats or /stats 30 (days, 1 to 365)"
            try:
                days = int(args[0]) if args else 7
            except ValueError:
                await say(usage)
                return
            if not 1 <= days <= 365:
                await say(usage)
                return
            trades = await asyncio.to_thread(app.state.broker.closed_trades)
            report = compute_stats(trades, days)
            if len(trades) >= 2000:
                report += "\n\nNote: only the 2,000 most recent closed trades were read."
            await say(report)
        elif cmd in ("/closelosers", "/closelosses"):
            await say("Closing all losing trades...")
            closed, failed, kept = await asyncio.to_thread(app.state.broker.close_losing_trades)
            lines = [f"CLOSED {len(closed)} losing trade(s); kept {kept} winner(s)."]
            lines += closed
            if failed:
                lines += ["FAILED:"] + failed
            await say("\n".join(lines))
        elif cmd == "/gj":
            await say(format_gj_report())
        elif cmd == "/id":
            await say(f"This chat's ID is: {app.state.settings.telegram_chat_id}")
        elif cmd == "/ai":
            client = app.state.ai_client
            if client is None:
                await say("OpenRouter client is not loaded. Check OPENROUTER_API_KEY in Render.")
            else:
                model = app.state.settings.openrouter_model
                try:
                    resp = await asyncio.to_thread(
                        client.chat.completions.create, model=model,
                        messages=[{"role": "user", "content": "Reply with the single word OK."}], max_tokens=10)
                    await say(f"OpenRouter ONLINE\nModel: {model}\nReply: {(resp.choices[0].message.content or '').strip()[:100]}")
                except Exception as exc:
                    await say(f"OpenRouter OFFLINE\nModel: {model}\nError: {_short_error(exc)}")
        else:
            await say(f"Unknown command: {cmd}\nSend /help for the list.")
    except Exception as exc:
        logger.exception("Error executing command handler")
        await say(f"Error: {exc}")


@app.get("/")
def home():
    return {
        "status": "Active",
        "configured": app.state.settings is not None,
        "telegram": app.state.main_telegram_mode,
        "gj": gj_public_status(),
    }


@app.get("/gj/status")
def gj_status():
    return gj_public_status()


async def dispatch_main_update(data: dict) -> str:
    settings = app.state.settings
    if not settings or app.state.bot is None:
        return "not configured"
    msg = data.get("message")
    if not isinstance(msg, dict) or "text" not in msg:
        return "ignored"
    sender_chat_id = str((msg.get("chat") or {}).get("id"))
    text = str(msg.get("text") or "")
    cmd = parse_command(text)
    if sender_chat_id != settings.telegram_chat_id:
        logger.warning("UNAUTHORIZED chat rejected: %s", sender_chat_id)
        if cmd == "/id":
            await say(f"This chat's ID is: {sender_chat_id}", chat_id=sender_chat_id)
        return "unauthorized"
    task = asyncio.create_task(handle_command(cmd, parse_args(text)))
    app.state.tasks.add(task)
    task.add_done_callback(app.state.tasks.discard)
    return "ok"


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    settings = app.state.settings
    if not settings:
        return {"status": "not configured"}
    if not webhook_secret_ok(request, settings.webhook_secret):
        return {"status": "unauthorized"}
    try:
        data = await request.json()
    except Exception:
        return {"status": "bad json"}
    if remember_update(app.state.seen_updates, data.get("update_id")):
        return {"status": "duplicate"}
    return {"status": await dispatch_main_update(data)}


@app.post("/gj/{bot_key}/telegram-webhook")
async def gj_telegram_webhook(bot_key: str, request: Request):
    if bot_key not in GJ_BOT_DEFINITIONS:
        return {"status": "unknown bot"}
    if not webhook_secret_ok(request, webhook_secret_from_env()):
        return {"status": "unauthorized"}
    runtime = app.state.gj_bots.get(bot_key) if app.state.gj_bots else None
    if runtime is None or runtime.bot is None:
        return {"status": "offline"}
    try:
        data = await request.json()
    except Exception:
        return {"status": "bad json"}
    if remember_update(runtime.seen, data.get("update_id")):
        return {"status": "duplicate"}
    task = asyncio.create_task(handle_gj_update(bot_key, data))
    app.state.tasks.add(task)
    task.add_done_callback(app.state.tasks.discard)
    return {"status": "ok"}


@app.get("/run-agent-check")
@app.post("/run-agent-check")
async def run_agent_check(request: Request, key: str = ""):
    settings = app.state.settings
    if not settings or not settings.cron_secret:
        raise HTTPException(status_code=403, detail="CRON_SECRET not configured")
    supplied = key or request.headers.get("x-cron-key", "")
    if not hmac.compare_digest(supplied, settings.cron_secret):
        raise HTTPException(status_code=403)
    if app.state.paused:
        return {"status": "paused"}
    result = await run_scan()
    if result is None:
        return {"status": "busy"}
    if result.alert:
        await say(result.report)
    return {"status": "ok", "detail": result.report}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
