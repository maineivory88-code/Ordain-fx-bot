"""Tris forex agent: OANDA execution, ATR rules, OpenRouter notes, Telegram control.

Exit rules (applied to every open trade on the account, checked every few seconds):
  - unrealized P/L >= TAKE_PROFIT_USD ($3.00)          -> close (take profit)
  - unrealized P/L <= -CUT_LOSS_USD (-$1.50)            -> close (cut loss)
  - once a trade reaches FLOOR_ARM_USD ($2.00) it is armed; if it later falls below $2.00 -> close
Entries: ATR/EMA rules on every tradable currency pair and XAU_USD, scanned on a timer.
Each pair can open at most one new trade per H1 bar. No broker take profit is set.
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
TRADABLE_TYPES = ("CURRENCY", "METAL")
GOLD = "XAU_USD"

# Exit rules, in USD of unrealized P/L per trade
TAKE_PROFIT_USD = Decimal("3.00")
FLOOR_ARM_USD = Decimal("2.00")
CUT_LOSS_USD = Decimal("1.50")

EXIT_CHECK_SECONDS = 5
HOUSEKEEPING_SECONDS = 60
SCAN_INTERVAL_SECONDS = 300


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
    cycle_target: Decimal
    daily_loss_limit: Decimal
    xau_risk_usd: Decimal
    gold_sl_atr: Decimal
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
    entered: list[str] = field(default_factory=list)
    gold_status: str = ""


class SkipPair(Exception):
    pass


OpinionFn = Callable[[str, dict, dict], dict]


# ---------- time, sessions, market hours ----------

def trading_day_start(now: datetime | None = None) -> datetime:
    """Start of the current forex trading day (17:00 New York), as a UTC datetime."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        local = now.astimezone(ZoneInfo("America/New_York"))
        start = local.replace(hour=17, minute=0, second=0, microsecond=0)
        if local < start:
            start -= timedelta(days=1)
        return start.astimezone(timezone.utc)
    except Exception:  # tz database missing: approximate with 22:00 UTC
        start = now.replace(hour=22, minute=0, second=0, microsecond=0)
        if now < start:
            start -= timedelta(days=1)
        return start


def bar_key(now: datetime) -> str:
    """Identifier of the current H1 bar."""
    return now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()


SESSION_WINDOWS_UTC = {
    "all": None,        # 24h
    "asia": (0, 9),     # roughly 00:00-09:00 UTC
    "london": (7, 16),  # roughly 07:00-16:00 UTC
    "ny": (12, 21),     # roughly 12:00-21:00 UTC
}


def market_open(now: datetime | None = None) -> bool:
    """Forex trades Sunday 17:00 to Friday 17:00 New York time."""
    local = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    wd, hour = local.weekday(), local.hour
    if wd == 5:
        return False
    if wd == 4 and hour >= 17:
        return False
    if wd == 6 and hour < 17:
        return False
    return True


def session_open(now: datetime | None, session: str) -> bool:
    window = SESSION_WINDOWS_UTC.get(session)
    if window is None:
        return True
    hour = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).hour
    return window[0] <= hour < window[1]


# ---------- configuration ----------

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

    units = env_decimal("TRADE_UNITS", "5000")
    if units <= 0:
        raise RuntimeError("TRADE_UNITS must be greater than 0")

    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if secret and not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", secret):
        raise RuntimeError("TELEGRAM_WEBHOOK_SECRET must match [A-Za-z0-9_-]{1,256}")

    cycle = env_decimal("CYCLE_PROFIT_TARGET", "20")
    loss_limit = env_decimal("DAILY_LOSS_LIMIT", "150")
    xau_risk = env_decimal("XAU_RISK_USD", "20")
    gold_sl = env_decimal("GOLD_SL_ATR", "1.5")
    if min(cycle, loss_limit) < 0 or xau_risk <= 0 or gold_sl <= 0:
        raise RuntimeError("Profit/loss/risk settings have invalid values")

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
        trade_units=units,
        webhook_secret=secret,
        cycle_target=cycle,
        daily_loss_limit=loss_limit,
        xau_risk_usd=xau_risk,
        gold_sl_atr=gold_sl,
        trading_session=session,
    )


def webhook_url(public_url: str) -> str:
    if public_url.endswith("/telegram-webhook"):
        return public_url
    return f"{public_url}/telegram-webhook"


# ---------- telegram helpers ----------

def parse_command(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped.split()[0].split("@", 1)[0].lower()


def parse_args(text: str) -> list[str]:
    parts = text.strip().split()
    return parts[1:] if len(parts) > 1 else []


def chunk_text(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    text = text or ""
    if len(text) <= limit:
        return [text] if text else []
    chunks: list[str] = []
    rest = text
    while rest:
        if len(rest) <= limit:
            chunks.append(rest)
            break
        cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip("\n")
    return [chunk for chunk in chunks if chunk]


def remember_update(seen: OrderedDict, update_id: Any) -> bool:
    if update_id is None:
        return False
    if update_id in seen:
        return True
    seen[update_id] = True
    while len(seen) > SEEN_UPDATES_LIMIT:
        seen.popitem(last=False)
    return False


# ---------- indicators and entry rules ----------

def ema(values: list[float], period: int) -> float:
    if not values:
        raise ValueError("EMA requires at least one value")
    period = min(period, len(values))
    k = 2 / (period + 1)
    seed = sum(values[:period]) / period
    for value in values[period:]:
        seed = value * k + seed * (1 - k)
    return seed


def calculate_atr(candles: list[dict], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    true_ranges: list[float] = []
    for index in range(1, len(candles)):
        high = float(candles[index]["mid"]["h"])
        low = float(candles[index]["mid"]["l"])
        prev_close = float(candles[index - 1]["mid"]["c"])
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    window = true_ranges[-period:]
    return sum(window) / len(window) if window else 0.0


def evaluate_rules(md: dict) -> dict:
    closes = md.get("last_closes") or []
    if len(closes) < 4:
        return {"should_trade": False, "reason": "not enough completed closes"}
    atr = float(md.get("atr") or 0)
    if atr <= 0:
        return {"should_trade": False, "reason": "ATR is zero"}

    ema50 = float(md["ema50"])
    ema200 = float(md.get("ema200") or ema50)
    last_close = float(closes[-1])
    prev_close = float(closes[-2])
    momentum = last_close - float(closes[-4])
    min_distance = 0.5 * atr

    above = last_close > ema50 + min_distance
    below = last_close < ema50 - min_distance
    momentum_up = last_close > prev_close and momentum > min_distance
    momentum_down = last_close < prev_close and momentum < -min_distance

    # Trade only with the bigger trend: EMA50/EMA200 aligned, price on the same side of EMA200.
    bullish = (above or momentum_up) and last_close > ema200 and ema50 > ema200
    bearish = (below or momentum_down) and last_close < ema200 and ema50 < ema200
    if bullish and bearish:
        return {"should_trade": False, "reason": "conflicting signals"}
    if bullish:
        return {"should_trade": True, "direction": "BUY", "sl_pips": 15.0, "reason": f"bullish expansion, ATR {atr:.5f}"}
    if bearish:
        return {"should_trade": True, "direction": "SELL", "sl_pips": 15.0, "reason": f"bearish expansion, ATR {atr:.5f}"}
    return {"should_trade": False, "reason": "no setup (needs ATR expansion and EMA50/EMA200 alignment)"}


def build_market_snapshot(candles: list[dict], price: dict) -> dict | None:
    completed = [c for c in candles if c.get("complete") and isinstance(c.get("mid"), dict)]
    if len(completed) < MIN_CLOSES:
        return None
    try:
        closes = [float(c["mid"]["c"]) for c in completed]
        atr = calculate_atr(completed)
        bid = Decimal(str(price["bids"][0]["price"]))
        ask = Decimal(str(price["asks"][0]["price"]))
    except Exception:
        return None
    if atr <= 0 or bid <= 0 or ask <= 0 or ask < bid:
        return None
    return {
        "bid": format_decimal(bid),
        "ask": format_decimal(ask),
        "tradeable": bool(price.get("tradeable", True)),
        "ema50": ema(closes, 50),
        "ema200": ema(closes, 200),
        "last_closes": closes[-10:],
        "atr": atr,
    }


def price_quantum(precision: int) -> Decimal:
    return Decimal(1).scaleb(-precision)


def format_decimal(value: Decimal) -> str:
    return format(value, "f")


def order_units(units: Decimal, direction: str, spec: InstrumentSpec) -> str:
    signed = units if direction == "BUY" else -units
    sized = signed.quantize(price_quantum(spec.trade_units_precision), rounding=ROUND_HALF_UP)
    if sized == 0 or abs(sized) < spec.minimum_trade_size:
        raise SkipPair(f"size {sized} below minimum {spec.minimum_trade_size}")
    if spec.maximum_order_units > 0 and abs(sized) > spec.maximum_order_units:
        raise SkipPair(f"size {sized} above maximum {spec.maximum_order_units}")
    return format_decimal(sized)


def build_order(spec: InstrumentSpec, direction: str, units: Decimal, sl_pips: float, md: dict, settings: Settings) -> dict:
    """Market order with a hard stop loss and NO take profit (exits are handled by the bot)."""
    entry = Decimal(md["ask" if direction == "BUY" else "bid"])
    quantum = price_quantum(spec.display_precision)
    if spec.name == GOLD:
        atr = Decimal(str(md["atr"]))
        sl_dist = (settings.gold_sl_atr * atr).quantize(quantum, rounding=ROUND_HALF_UP)
        if sl_dist <= 0:
            raise SkipPair("gold stop distance rounded to zero")
        oz = (settings.xau_risk_usd / sl_dist).quantize(price_quantum(spec.trade_units_precision), rounding=ROUND_DOWN)
        # Never size below one ounce. If the ATR stop is wider than the risk budget, the
        # trade risks more than XAU_RISK_USD, and the scan report says so.
        minimum = max(spec.minimum_trade_size, Decimal(1).scaleb(-spec.trade_units_precision))
        units_str = order_units(max(oz, minimum), direction, spec)
    else:
        sl_dist = (Decimal(str(sl_pips)) * spec.pip).quantize(quantum, rounding=ROUND_HALF_UP)
        if sl_dist <= 0:
            raise SkipPair("stop distance rounded to zero")
        units_str = order_units(units, direction, spec)
    stop = (entry - sl_dist if direction == "BUY" else entry + sl_dist).quantize(quantum, rounding=ROUND_HALF_UP)
    return {
        "order": {
            "units": units_str,
            "instrument": spec.name,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": format_decimal(stop)},
        },
        "_entry": entry,
        "_stop": stop,
    }


def interpret_order_response(res: Any) -> tuple[str, str]:
    if not isinstance(res, dict):
        return "rejected", str(res)[:500]
    if "orderFillTransaction" in res:
        return "filled", ""
    reject = res.get("orderRejectTransaction") or res.get("orderCancelTransaction") or {}
    if not isinstance(reject, dict):
        reject = {}
    return "rejected", str(reject.get("reason") or res.get("errorMessage") or res)[:500]


# ---------- OpenRouter (advisory only; never blocks a trade) ----------

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


def explicit_disagree(opinion: dict | None) -> bool:
    if not opinion or "agree" not in opinion:
        return False
    value = opinion.get("agree")
    if isinstance(value, str):
        return value.strip().lower() in {"false", "no", "0"}
    return value is False


def _short_error(exc: Exception) -> str:
    return (str(exc) or exc.__class__.__name__)[:160]


def get_openrouter_opinion(client: OpenAI, model: str, pair: str, md: dict, rule_signal: dict) -> dict:
    prompt = (
        f"Review trade signal for {pair}.\nSignal: {json.dumps(rule_signal)}\nData: {json.dumps(md)}\n"
        'Return JSON only: {"agree": true, "comment": "short text"}'
    )
    messages = [
        {"role": "system", "content": "You are a financial JSON generator."},
        {"role": "user", "content": prompt},
    ]
    last_error: Exception | None = None
    for json_mode in (True, False):  # some models reject JSON mode; retry as plain text
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


def ai_note(opinion: dict | None) -> str:
    if not opinion:
        return "OpenRouter AI: Offline / Skipped"
    if opinion.get("error"):
        return f"OpenRouter AI: Offline - {opinion['error']}"
    agree = not explicit_disagree(opinion)
    return f"OpenRouter AI: {'Agree' if agree else 'Caution'} - {opinion.get('comment', 'N/A')}"


# ---------- statistics ----------

def _money(x: Decimal) -> str:
    return f"{x:+.2f}"


def compute_stats(trades: list[dict], days: int, now: datetime | None = None) -> str:
    """Summarize closed trades from the last `days` days. Net = realized P/L + financing."""
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
    ratio = (avg_win / -avg_loss) if (avg_win is not None and avg_loss is not None) else None
    pf = (gross_win / gross_loss) if gross_loss > 0 else None
    lines = [
        f"STATS (last {days} days)",
        f"Closed trades: {total} (wins {len(wins)}, losses {len(losses)}, flat {flat})",
        f"Win rate: {len(wins) / total * 100:.1f}%",
        f"Avg win: {_money(avg_win) if avg_win is not None else 'n/a'} | "
        f"Avg loss: {_money(avg_loss) if avg_loss is not None else 'n/a'}",
        f"Avg win / avg loss: {f'{ratio:.2f}' if ratio is not None else 'n/a'}",
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
    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {settings.oanda_api_key}", "Content-Type": "application/json"}
        )

    def close(self) -> None:
        self.session.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        url = f"{self.settings.oanda_base}{path}"
        kwargs.setdefault("timeout", 20)
        response = self.session.request(method, url, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"OANDA HTTP {response.status_code}: {response.text[:180]}")
        return response.json()

    def _account_path(self, suffix: str) -> str:
        return f"/accounts/{quote(self.settings.oanda_account_id, safe='')}{suffix}"

    def list_instruments(self) -> list[InstrumentSpec]:
        """Every currency pair and metal this account can trade."""
        data = self._request("GET", self._account_path("/instruments"))
        specs = []
        for raw in data.get("instruments") or []:
            name = str(raw.get("name") or "")
            if raw.get("type") not in TRADABLE_TYPES or not re.fullmatch(r"[A-Z]{3}_[A-Z]{3}", name):
                continue
            specs.append(
                InstrumentSpec(
                    name=name,
                    pip_location=int(raw["pipLocation"]),
                    display_precision=int(raw["displayPrecision"]),
                    trade_units_precision=int(raw.get("tradeUnitsPrecision") or 0),
                    minimum_trade_size=Decimal(str(raw.get("minimumTradeSize") or "1")),
                    maximum_order_units=Decimal(str(raw.get("maximumOrderUnits") or "0")),
                )
            )
        return specs

    def open_trades(self) -> list[dict]:
        data = self._request("GET", self._account_path("/openTrades"))
        return [t for t in (data.get("trades") or []) if isinstance(t, dict) and t.get("id") is not None]

    def open_instruments(self) -> set[str]:
        return {str(t.get("instrument")) for t in self.open_trades()}

    def market_data(self, spec: InstrumentSpec) -> dict | None:
        candles = self._request(
            "GET",
            f"/instruments/{quote(spec.name, safe='')}/candles",
            params={"granularity": "H1", "count": CANDLE_LOOKBACK, "price": "M"},
        )
        pricing = self._request("GET", self._account_path("/pricing"), params={"instruments": spec.name})
        prices = pricing.get("prices") or []
        if not prices:
            return None
        return build_market_snapshot(candles.get("candles") or [], prices[0])

    def realized_pl_since(self, since: datetime) -> Decimal:
        fmt = "%Y-%m-%dT%H:%M:%S.000000000Z"
        params = {
            "from": since.strftime(fmt),
            "to": datetime.now(timezone.utc).strftime(fmt),
            "type": "ORDER_FILL",
            "pageSize": 1000,
        }
        index = self._request("GET", self._account_path("/transactions"), params=params)
        total = Decimal("0")
        for page_url in index.get("pages") or []:
            resp = self.session.get(page_url, timeout=20)
            if resp.status_code >= 400:
                raise RuntimeError(f"OANDA HTTP {resp.status_code}: {resp.text[:180]}")
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
        unrealized = Decimal(str((data.get("account") or {}).get("unrealizedPL", "0")))
        return realized, unrealized

    def closed_trades(self, max_pages: int = 4) -> list[dict]:
        out: list[dict] = []
        before: int | None = None
        for _ in range(max_pages):
            params: dict[str, Any] = {"state": "CLOSED", "count": 500}
            if before is not None:
                params["beforeID"] = before
            data = self._request("GET", self._account_path("/trades"), params=params)
            page = [t for t in (data.get("trades") or []) if isinstance(t, dict) and t.get("id") is not None]
            out.extend(page)
            if len(page) < 500:
                break
            before = min(int(t["id"]) for t in page)
        return out

    def close_trade(self, trade_id: str) -> None:
        self._request(
            "PUT",
            self._account_path(f"/trades/{quote(str(trade_id), safe='')}/close"),
            json={"units": "ALL"},
        )

    def manage_exits(self, armed: set[str]) -> tuple[list[str], list[str]]:
        """Apply the take-profit, cut-loss and $2 floor rules to every open trade.
        `armed` (trade ids that have reached FLOOR_ARM_USD) is updated in place."""
        closed: list[str] = []
        failed: list[str] = []
        trades = self.open_trades()
        armed.intersection_update({str(t["id"]) for t in trades})
        for trade in trades:
            tid = str(trade["id"])
            instrument = str(trade.get("instrument", "?"))
            label = f"{instrument.replace('_', '/')} #{tid}"
            try:
                pl = Decimal(str(trade.get("unrealizedPL", "0")))
            except Exception:
                continue

            reason = None
            if pl >= TAKE_PROFIT_USD:
                reason = f"take profit at ${TAKE_PROFIT_USD}"
            elif pl <= -CUT_LOSS_USD:
                reason = f"cut loss at -${CUT_LOSS_USD}"
            elif pl >= FLOOR_ARM_USD:
                armed.add(tid)
            elif tid in armed:
                reason = f"profit fell below ${FLOOR_ARM_USD}"

            if reason is None:
                continue
            try:
                self.close_trade(tid)
                armed.discard(tid)
                closed.append(f"{label} ({format_decimal(pl)}): {reason}")
            except Exception as exc:
                failed.append(f"{label}: {exc}")
        return closed, failed

    def close_winning_trades(self, min_profit: Decimal | None = None) -> tuple[list[tuple[str, str]], list[str], int]:
        closed: list[tuple[str, str]] = []
        failed: list[str] = []
        kept = 0
        for trade in self.open_trades():
            try:
                pl = Decimal(str(trade.get("unrealizedPL", "0")))
            except Exception:
                continue
            instrument = str(trade.get("instrument", "?"))
            label = f"{instrument.replace('_', '/')} #{trade.get('id')}"
            if pl <= 0 or (min_profit is not None and pl < min_profit):
                kept += 1
                continue
            try:
                self.close_trade(str(trade["id"]))
                closed.append((instrument, f"{label} (+{format_decimal(pl)})"))
            except Exception as exc:
                failed.append(f"{label}: {exc}")
        return closed, failed, kept

    def close_losing_trades(self) -> tuple[list[str], list[str], int]:
        closed: list[str] = []
        failed: list[str] = []
        kept = 0
        for trade in self.open_trades():
            try:
                pl = Decimal(str(trade.get("unrealizedPL", "0")))
            except Exception:
                continue
            label = f"{str(trade.get('instrument', '?')).replace('_', '/')} #{trade.get('id')}"
            if pl >= 0:
                kept += 1
                continue
            try:
                self.close_trade(str(trade["id"]))
                closed.append(f"{label} ({format_decimal(pl)})")
            except Exception as exc:
                failed.append(f"{label}: {exc}")
        return closed, failed, kept

    def account_summary(self) -> tuple[Any, Any, Any]:
        acc = self._request("GET", self._account_path("/summary")).get("account") or {}
        return acc.get("balance", "N/A"), acc.get("unrealizedPL", "N/A"), acc.get("openTradeCount", 0)

    def submit(self, spec: InstrumentSpec, direction: str, units: Decimal, sl_pips: float, md: dict):
        built = build_order(spec, direction, units, sl_pips, md, self.settings)
        entry, stop = built.pop("_entry"), built.pop("_stop")
        response = self._request("POST", self._account_path("/orders"), json=built)
        status, reason = interpret_order_response(response)
        return status, entry, stop, reason


def run_cycle(broker: Any, units: Decimal, opinion_fn: OpinionFn, blocked: set[str] | None = None) -> ScanResult:
    try:
        instruments = broker.list_instruments()
        open_now = broker.open_instruments()
    except Exception as exc:
        return ScanResult(report=f"Scan failed: {exc}", failed=1, alert=True)

    reports: list[str] = []
    entered: list[str] = []
    executed = failed = 0
    gold_status = "XAU_USD not offered on this account" if all(i.name != GOLD for i in instruments) else ""

    for spec in instruments:
        is_gold = spec.name == GOLD
        if spec.name in open_now:
            if is_gold:
                gold_status = "XAU_USD: already has an open trade"
            continue
        if blocked and spec.name in blocked:
            if is_gold:
                gold_status = "XAU_USD: already entered this hour"
            continue
        try:
            md = broker.market_data(spec)
            if not md:
                if is_gold:
                    gold_status = "XAU_USD: no price data returned"
                continue
            if not md.get("tradeable"):
                if is_gold:
                    gold_status = "XAU_USD: not tradeable right now (market closed or halted)"
                continue
            signal = evaluate_rules(md)
            if not signal.get("should_trade"):
                if is_gold:
                    gold_status = f"XAU_USD: {signal['reason']}"
                continue
            opinion = opinion_fn(spec.pair, md, signal)
            status, entry, stop, reason = broker.submit(spec, signal["direction"], units, float(signal["sl_pips"]), md)
            if status == "filled":
                executed += 1
                entered.append(spec.name)
                reports.append(
                    f"TRADE EXECUTED ({spec.pair})\n"
                    f"Direction: {signal['direction']} @ {format_decimal(entry)}\n"
                    f"SL: {format_decimal(stop)} | exits: +${TAKE_PROFIT_USD} take profit, "
                    f"-${CUT_LOSS_USD} cut loss, ${FLOOR_ARM_USD} floor\n"
                    f"{ai_note(opinion)}"
                )
                if is_gold:
                    gold_status = f"XAU_USD: opened {signal['direction']}"
            else:
                failed += 1
                reports.append(f"ORDER FAILED ({spec.pair}): {reason}")
                if is_gold:
                    gold_status = f"XAU_USD: order rejected: {reason}"
        except SkipPair as exc:
            if is_gold:
                gold_status = f"XAU_USD: skipped, {exc}"
            logger.info("Skipped %s: %s", spec.name, exc)
        except Exception as exc:
            logger.exception("Pair scan error for %s", spec.name)
            failed += 1
            if is_gold:
                gold_status = f"XAU_USD: error {_short_error(exc)}"

    if not reports:
        note = "Scan complete. No eligible setups found."
        if failed:
            note += f" ({failed} pair(s) errored; check logs.)"
        return ScanResult(report=note, failed=failed, alert=False, gold_status=gold_status)
    return ScanResult(
        report=f"--- SCAN COMPLETED ({executed} opened) ---\n\n" + "\n\n".join(reports),
        executed=executed,
        failed=failed,
        alert=True,
        entered=entered,
        gold_status=gold_status,
    )


# ---------- app ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        settings = load_settings()
    except RuntimeError as exc:
        logger.error("Configuration error: %s", exc)
        yield
        return

    bot = Bot(token=settings.telegram_token)
    broker = OandaBroker(settings)
    ai_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=settings.openrouter_api_key, timeout=30)
    app.state.settings = settings
    app.state.bot = bot
    app.state.broker = broker
    app.state.ai_client = ai_client
    app.state.opinion_fn = lambda p, m, s: get_openrouter_opinion(ai_client, settings.openrouter_model, p, m, s)

    await bot.initialize()
    if settings.public_url:
        try:
            kwargs: dict[str, Any] = {"allowed_updates": ["message"]}
            if settings.webhook_secret:
                kwargs["secret_token"] = settings.webhook_secret
            await bot.set_webhook(url=webhook_url(settings.public_url), **kwargs)
            logger.info("Webhook registered to %s", webhook_url(settings.public_url))
        except Exception:
            logger.exception("Failed to set webhook")
    else:
        logger.warning("PUBLIC_URL is not set: Telegram webhook NOT registered")

    tasks = [
        asyncio.create_task(exit_loop()),
        asyncio.create_task(housekeeping_loop()),
        asyncio.create_task(scan_loop()),
    ]
    yield
    for task in tasks:
        task.cancel()
    await bot.shutdown()
    broker.close()


app = FastAPI(lifespan=lifespan)
app.state.settings = None
app.state.bot = None
app.state.broker = None
app.state.ai_client = None
app.state.opinion_fn = None
app.state.paused = False
app.state.scanning = False
app.state.seen_updates = OrderedDict()
app.state.tasks = set()
app.state.armed = set()                 # trade ids that have reached FLOOR_ARM_USD
app.state.entered_bar = {}              # pair -> H1 bar of its last entry
app.state.milestone_day = None
app.state.milestone = 0
app.state.loss_alert_day = None
app.state.last_gold_status = "no scan yet"


async def say(text: str, chat_id: str | None = None) -> None:
    bot = app.state.bot
    settings = app.state.settings
    if not bot or not settings:
        return
    target = chat_id or settings.telegram_chat_id
    for part in chunk_text(text):
        try:
            await bot.send_message(chat_id=target, text=part)
        except Exception:
            logger.exception("Failed to send Telegram message")
            return
        await asyncio.sleep(0.3)


async def exit_loop() -> None:
    """Fast loop: take profit at $3, cut losses at -$1.50, and close armed trades that fall below $2."""
    while True:
        await asyncio.sleep(EXIT_CHECK_SECONDS)
        if not app.state.settings:
            continue
        try:
            closed, failed = await asyncio.to_thread(app.state.broker.manage_exits, app.state.armed)
            if closed or failed:
                lines = [f"EXIT: closed {len(closed)} trade(s)."] + closed
                if failed:
                    lines += ["FAILED:"] + failed
                await say("\n".join(lines))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Exit loop error")


async def housekeeping_loop() -> None:
    """Slow loop: daily loss alert and profit-step banking."""
    while True:
        await asyncio.sleep(HOUSEKEEPING_SECONDS)
        if not app.state.settings:
            continue
        try:
            loss_msg = await check_loss_limit()
            if loss_msg:
                await say(loss_msg)
            step_msg = await bank_milestones()
            if step_msg:
                await say(step_msg)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Housekeeping loop error")


async def scan_loop() -> None:
    """Automatic entries on a timer, unless paused."""
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


async def check_loss_limit() -> str | None:
    """Alert once per trading day when total P/L reaches -daily_loss_limit. Never closes trades."""
    settings = app.state.settings
    if not settings or settings.daily_loss_limit <= 0:
        return None
    today = trading_day_start().isoformat()
    if app.state.loss_alert_day == today:
        return None
    realized, unrealized = await asyncio.to_thread(app.state.broker.daily_pnl)
    total = realized + unrealized
    if total > -settings.daily_loss_limit:
        return None
    app.state.loss_alert_day = today
    return (
        f"LOSS ALERT: total P/L {total:.2f} (alert level -{settings.daily_loss_limit}). "
        "Trading continues. Send /closelosers to close losing trades."
    )


async def bank_milestones() -> str | None:
    """Each time today's realized profit crosses another cycle_target step, close winners of at
    least $2.00 (never below the floor). Trading continues."""
    settings = app.state.settings
    if not settings or settings.cycle_target <= 0:
        return None
    today = trading_day_start().isoformat()
    realized, _ = await asyncio.to_thread(app.state.broker.daily_pnl)
    milestone = int(realized // settings.cycle_target)
    if app.state.milestone_day != today:
        app.state.milestone_day = today
        app.state.milestone = milestone
        return None
    if milestone <= app.state.milestone:
        return None
    closed, failed, kept = await asyncio.to_thread(app.state.broker.close_winning_trades, FLOOR_ARM_USD)
    if failed:
        return "PROFIT STEP reached but some closes failed:\n" + "\n".join(failed)
    app.state.milestone = milestone
    return (
        f"PROFIT STEP: realized today {realized:.2f} (step ${settings.cycle_target}). "
        f"Closed {len(closed)} winner(s) of at least ${FLOOR_ARM_USD}; {kept} left open."
    )


async def run_scan(skip: set[str] | None = None) -> ScanResult | None:
    """One scan in a worker thread. Returns None if a scan is already running.
    Each pair can be entered at most once per H1 bar."""
    try:
        loss_msg = await check_loss_limit()
        if loss_msg:
            await say(loss_msg)
    except Exception:
        logger.exception("Loss limit check failed")

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
    if skip:
        blocked |= skip
    app.state.scanning = True
    try:
        result = await asyncio.to_thread(run_cycle, app.state.broker, settings.trade_units, app.state.opinion_fn, blocked)
    finally:
        app.state.scanning = False
    for name in result.entered:
        app.state.entered_bar[name] = bar
    app.state.last_gold_status = result.gold_status or "no gold status"
    if result.gold_status and not result.alert:
        result.report += f"\n{result.gold_status}"
    return result


async def handle_command(cmd: str, args: list[str] | None = None) -> None:
    args = args or []
    try:
        if cmd in ("/start", "/help"):
            await say(
                "Tris Bot Active (ATR + OpenRouter)\n\n"
                "/status - balance and open trades\n"
                "/trade - scan all pairs now\n"
                "/target - today's P/L and profit step\n"
                "/takeprofit - close all winners now, then rescan\n"
                "/closelosers - close all losing trades now\n"
                "/instruments [filter] - list tradable instruments, e.g. /instruments XAU\n"
                "/stats [days] - win rate, avg win/loss, P/L by day\n"
                "/debug - dry run EUR/USD\n"
                "/ai - test OpenRouter connection\n"
                "/pause - stop automatic and manual entries\n"
                "/resume - resume entries\n\n"
                f"Exits are automatic: take profit at ${TAKE_PROFIT_USD}, cut loss at -${CUT_LOSS_USD}, "
                f"and close if a trade that reached ${FLOOR_ARM_USD} falls back below it."
            )
        elif cmd == "/status":
            bal, pnl, count = await asyncio.to_thread(app.state.broker.account_summary)
            state = "PAUSED" if app.state.paused else "RUNNING"
            await say(f"STATUS ({state}, {app.state.settings.oanda_env})\nBalance: {bal}\nPnL: {pnl}\n"
                      f"Open Trades: {count}\nLast gold check: {app.state.last_gold_status}")
        elif cmd == "/instruments":
            instruments = await asyncio.to_thread(app.state.broker.list_instruments)
            names = sorted(i.name for i in instruments)
            needle = args[0].upper().replace("/", "_") if args else ""
            matches = [n for n in names if needle in n]
            body = "\n".join(matches[:80]) if matches else f"No instrument matches '{needle}'."
            await say(f"{len(names)} tradable instruments on this account.\n{body}")
        elif cmd == "/debug":
            await say("Running debug test on EUR/USD...")
            instruments = await asyncio.to_thread(app.state.broker.list_instruments)
            spec = next((i for i in instruments if i.name == "EUR_USD"), None)
            md = await asyncio.to_thread(app.state.broker.market_data, spec) if spec else None
            if not md:
                await say("Debug Error: Could not fetch EUR_USD data.")
            else:
                signal = evaluate_rules(md)
                await say(f"DEBUG EUR/USD:\nClose: {md['last_closes'][-1]}\nATR: {md['atr']:.5f}\nSignal: {json.dumps(signal)}")
        elif cmd == "/target":
            realized, unrealized = await asyncio.to_thread(app.state.broker.daily_pnl)
            st_ = app.state.settings
            if st_.cycle_target > 0:
                next_step = f"${(int(realized // st_.cycle_target) + 1) * st_.cycle_target}"
            else:
                next_step = "off"
            await say(
                f"TODAY'S P/L\nRealized: {realized:.2f}\nUnrealized: {unrealized:.2f}\n"
                f"Total: {realized + unrealized:.2f}\nNext profit step (realized): {next_step}\n"
                f"Session: {st_.trading_session}\nLoss alert level: -{st_.daily_loss_limit}"
            )
        elif cmd in ("/takeprofit", "/takeprofits"):
            await say("Taking profit on all winning trades...")
            closed, failed, kept = await asyncio.to_thread(app.state.broker.close_winning_trades)
            lines = [f"TOOK PROFIT on {len(closed)} trade(s); {kept} trade(s) not in profit left open."]
            if closed:
                lines.append("\n".join(label for _, label in closed))
            if failed:
                lines.append("FAILED:\n" + "\n".join(failed))
            await say("\n".join(lines))
            if not closed:
                return
            if app.state.paused:
                await say("Trading is paused (/pause). Send /resume so the bot can open new trades.")
                return
            await say("Scanning for new setups...")
            result = await run_scan(skip={name for name, _ in closed})
            await say("A scan is already running." if result is None else result.report)
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
        elif cmd == "/ai":
            client = app.state.ai_client
            if client is None:
                await say("OpenRouter client is not loaded. Check OPENROUTER_API_KEY in Render.")
            else:
                model = app.state.settings.openrouter_model
                try:
                    resp = await asyncio.to_thread(
                        client.chat.completions.create,
                        model=model,
                        messages=[{"role": "user", "content": "Reply with the single word OK."}],
                        max_tokens=10,
                    )
                    text = (resp.choices[0].message.content or "").strip()
                    await say(f"OpenRouter ONLINE\nModel: {model}\nReply: {text[:100]}")
                except Exception as exc:
                    await say(f"OpenRouter OFFLINE\nModel: {model}\nError: {_short_error(exc)}\n"
                              "If it says 401, the key is wrong. If 404 or model not found, set OPENROUTER_MODEL in Render.")
        elif cmd in ("/closelosers", "/closelosses"):
            await say("Closing all losing trades...")
            closed, failed, kept = await asyncio.to_thread(app.state.broker.close_losing_trades)
            lines = [f"CLOSED {len(closed)} losing trade(s); kept {kept} winner(s)."]
            if closed:
                lines.append("\n".join(closed))
            if failed:
                lines.append("FAILED:\n" + "\n".join(failed))
            await say("\n".join(lines))
        elif cmd == "/pause":
            app.state.paused = True
            await say("Trading paused. Automatic and manual entries are off. Exits stay active.")
        elif cmd == "/resume":
            app.state.paused = False
            await say("Trading resumed.")
        elif cmd == "/trade":
            if app.state.paused:
                await say("Trading is paused. Send /resume first.")
                return
            await say("Manual scan triggered...")
            result = await run_scan()
            await say("A scan is already running." if result is None else result.report)
        else:
            await say(f"Unknown command: {cmd}\nSend /help for the list.")
    except Exception as exc:
        logger.exception("Error executing command handler")
        await say(f"Error: {exc}")


@app.get("/")
def home():
    return {"status": "Active", "configured": app.state.settings is not None}


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    settings = app.state.settings
    if not settings:
        logger.warning("Webhook received but settings are not loaded (check env vars)")
        return {"status": "not configured"}

    if settings.webhook_secret:
        received = request.headers.get("x-telegram-bot-api-secret-token", "")
        if not hmac.compare_digest(received, settings.webhook_secret):
            logger.warning("Webhook secret token mismatch")
            return {"status": "unauthorized"}

    try:
        data = await request.json()
    except Exception:
        return {"status": "bad json"}

    if remember_update(app.state.seen_updates, data.get("update_id")):
        return {"status": "duplicate"}

    msg = data.get("message")
    if not isinstance(msg, dict) or "text" not in msg:
        return {"status": "ignored"}

    sender_chat_id = str((msg.get("chat") or {}).get("id"))
    text = str(msg.get("text") or "")
    cmd = parse_command(text)

    if sender_chat_id != settings.telegram_chat_id:
        logger.warning("UNAUTHORIZED chat rejected: %s", sender_chat_id)
        if cmd == "/id":
            await say(f"This chat's ID is: {sender_chat_id}", chat_id=sender_chat_id)
        return {"status": "unauthorized"}

    logger.info("Command received: %s", cmd)
    task = asyncio.create_task(handle_command(cmd, parse_args(text)))
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
