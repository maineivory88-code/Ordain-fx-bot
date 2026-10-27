"""Tris forex agent: OANDA execution, ATR rules, OpenRouter notes, Telegram control."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
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
    cycle_target: Decimal = Decimal("20")
    daily_loss_limit: Decimal = Decimal("150")
    trade_instruments: frozenset = frozenset(("EUR_USD", "GBP_USD", "GBP_JPY", "XAU_USD", "USD_JPY"))
    xau_risk_usd: Decimal = Decimal("20")
    gold_sl_atr: Decimal = Decimal("1.5")
    gold_tp_atr: Decimal = Decimal("3")
    trading_session: str = "all"


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


SESSION_WINDOWS_UTC = {
    "all": None,          # 24h, all sessions
    "asia": (0, 9),       # Tokyo/Sydney, roughly 00:00-09:00 UTC
    "london": (7, 16),    # roughly 07:00-16:00 UTC
    "ny": (12, 21),       # roughly 12:00-21:00 UTC
}


def market_open(now: datetime | None = None) -> bool:
    """Forex trades Sunday 17:00 to Friday 17:00 New York time."""
    local = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    wd, hour = local.weekday(), local.hour
    if wd == 5:  # Saturday
        return False
    if wd == 4 and hour >= 17:  # Friday evening
        return False
    if wd == 6 and hour < 17:  # Sunday before the open
        return False
    return True


def session_open(now: datetime | None, session: str) -> bool:
    window = SESSION_WINDOWS_UTC.get(session)
    if window is None:
        return True
    hour = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).hour
    return window[0] <= hour < window[1]


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

    cycle = env_decimal("CYCLE_PROFIT_TARGET", "20")
    if cycle < 0:
        raise RuntimeError("CYCLE_PROFIT_TARGET cannot be negative")
    loss_limit = env_decimal("DAILY_LOSS_LIMIT", "150")
    if loss_limit < 0:
        raise RuntimeError("DAILY_LOSS_LIMIT cannot be negative (use 0 to turn it off)")
    raw_instruments = os.getenv("TRADE_INSTRUMENTS", "").strip()
    if raw_instruments:
        instruments = frozenset(x.strip().upper() for x in raw_instruments.split(",") if x.strip())
    else:
        instruments = frozenset(("EUR_USD", "GBP_USD", "GBP_JPY", "XAU_USD", "USD_JPY"))
    if not all(re.fullmatch(r"[A-Z]{3}_[A-Z]{3}", i) for i in instruments):
        raise RuntimeError("TRADE_INSTRUMENTS must look like EUR_USD,GBP_USD,...")
    xau_risk = env_decimal("XAU_RISK_USD", "20")
    gold_sl = env_decimal("GOLD_SL_ATR", "1.5")
    gold_tp = env_decimal("GOLD_TP_ATR", "3")
    if xau_risk <= 0 or gold_sl <= 0 or gold_tp <= 0:
        raise RuntimeError("XAU_RISK_USD, GOLD_SL_ATR and GOLD_TP_ATR must be greater than 0")
    session = os.getenv("TRADING_SESSION", "all").strip().lower()
    if session not in SESSION_WINDOWS_UTC:
        raise RuntimeError("TRADING_SESSION must be one of: " + ", ".join(SESSION_WINDOWS_UTC))

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
        cycle_target=cycle,
        daily_loss_limit=loss_limit,
        trade_instruments=instruments,
        xau_risk_usd=xau_risk,
        gold_sl_atr=gold_sl,
        gold_tp_atr=gold_tp,
        trading_session=session,
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
            chunks.append(rest)
            break
        cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip("\n")
    return [chunk for chunk in chunks if chunk]


def remember_update(seen: OrderedDict, update_id: Any) -> bool:
    """Return True if this update was already processed."""
    if update_id is None:
        return False
    if update_id in seen:
        return True
    seen[update_id] = True
    while len(seen) > SEEN_UPDATES_LIMIT:
        seen.popitem(last=False)
    return False


def ema(values: list[float], period: int) -> float:
    if not values:
        raise ValueError("EMA requires at least one value")
    period = min(period, len(values))
    if period <= 0:
        raise ValueError("EMA period must be positive")
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
    if not window:
        return 0.0
    return sum(window) / len(window)


def evaluate_rules(md: dict, require_trend: bool = True) -> dict:
    closes = md.get("last_closes") or []
    if len(closes) < 4:
        return {"should_trade": False, "reason": "Not enough completed closes."}

    atr = float(md.get("atr") or 0)
    if atr <= 0:
        return {"should_trade": False, "reason": "ATR is zero; skipping flat or invalid data."}

    ema50 = float(md["ema50"])
    last_close = float(closes[-1])
    prev_close = float(closes[-2])
    three_bars_ago = float(closes[-4])
    momentum = last_close - three_bars_ago
    min_distance = 0.5 * atr

    above = last_close > ema50 + min_distance
    below = last_close < ema50 - min_distance
    momentum_up = last_close > prev_close and momentum > min_distance
    momentum_down = last_close < prev_close and momentum < -min_distance

    bullish = above or momentum_up
    bearish = below or momentum_down
    if require_trend:
        # Only trade with the bigger trend: EMA50/EMA200 aligned and price on the right side.
        ema200 = float(md.get("ema200") or ema50)
        bullish = bullish and last_close > ema200 and ema50 > ema200
        bearish = bearish and last_close < ema200 and ema50 < ema200
    # Conflicting signals (e.g. above EMA but momentum falling hard) -> stand aside.
    if bullish and bearish:
        return {"should_trade": False, "reason": "Conflicting trend and momentum signals."}

    if bullish:
        return {
            "should_trade": True,
            "direction": "BUY",
            "sl_pips": 15.0,
            "tp_pips": 30.0,
            "reason": f"Bullish expansion (ATR {atr:.5f})",
        }
    if bearish:
        return {
            "should_trade": True,
            "direction": "SELL",
            "sl_pips": 15.0,
            "tp_pips": 30.0,
            "reason": f"Bearish expansion (ATR {atr:.5f})",
        }
    return {"should_trade": False, "reason": "No qualified setup (needs ATR expansion + EMA50/EMA200 trend alignment)."}


def price_quantum(precision: int) -> Decimal:
    return Decimal(1).scaleb(-precision)


def format_decimal(value: Decimal) -> str:
    return format(value, "f")


def protective_prices(
    entry: Decimal,
    direction: str,
    sl_pips: Decimal,
    tp_pips: Decimal,
    pip: Decimal,
    precision: int,
) -> tuple[Decimal, Decimal]:
    quantum = price_quantum(precision)
    sl_distance = (sl_pips * pip).quantize(quantum, rounding=ROUND_HALF_UP)
    tp_distance = (tp_pips * pip).quantize(quantum, rounding=ROUND_HALF_UP)
    if sl_distance <= 0 or tp_distance <= 0:
        raise ValueError("SL/TP distance rounded to zero")
    if direction == "BUY":
        stop = (entry - sl_distance).quantize(quantum, rounding=ROUND_HALF_UP)
        target = (entry + tp_distance).quantize(quantum, rounding=ROUND_HALF_UP)
        return stop, target
    if direction == "SELL":
        stop = (entry + sl_distance).quantize(quantum, rounding=ROUND_HALF_UP)
        target = (entry - tp_distance).quantize(quantum, rounding=ROUND_HALF_UP)
        return stop, target
    raise ValueError(f"Unknown direction: {direction}")


def order_units(units: Decimal, direction: str, spec: InstrumentSpec) -> str:
    signed = units if direction == "BUY" else -units
    quantum = price_quantum(spec.trade_units_precision)
    sized = signed.quantize(quantum, rounding=ROUND_HALF_UP)
    if sized == 0 or abs(sized) < spec.minimum_trade_size:
        raise SkipPair(f"units below minimum for {spec.name}")
    if spec.maximum_order_units > 0 and abs(sized) > spec.maximum_order_units:
        raise SkipPair(f"units exceed maximum for {spec.name}")
    return format_decimal(sized)


def build_order(
    spec: InstrumentSpec,
    direction: str,
    units: Decimal,
    sl_pips: float,
    tp_pips: float,
    md: dict,
    settings: Settings,
) -> dict:
    entry_key = "ask" if direction == "BUY" else "bid"
    entry = Decimal(md[entry_key])
    quantum = price_quantum(spec.display_precision)
    if spec.name == "XAU_USD":
        # Gold: stops and targets are multiples of ATR (in dollars), and size is set so
        # a stop-out costs about XAU_RISK_USD.
        atr = Decimal(str(md["atr"]))
        sl_dist = (settings.gold_sl_atr * atr).quantize(quantum, rounding=ROUND_HALF_UP)
        tp_dist = (settings.gold_tp_atr * atr).quantize(quantum, rounding=ROUND_HALF_UP)
        if sl_dist <= 0 or tp_dist <= 0:
            raise SkipPair("gold ATR distance rounded to zero")
        oz = (settings.xau_risk_usd / sl_dist).quantize(price_quantum(spec.trade_units_precision), rounding=ROUND_DOWN)
        units_str = order_units(oz, direction, spec)  # raises SkipPair if below minimum
    else:
        sl_dist = (Decimal(str(sl_pips)) * spec.pip).quantize(quantum, rounding=ROUND_HALF_UP)
        tp_dist = (Decimal(str(tp_pips)) * spec.pip).quantize(quantum, rounding=ROUND_HALF_UP)
        if sl_dist <= 0 or tp_dist <= 0:
            raise ValueError("SL/TP distance rounded to zero")
        units_str = order_units(units, direction, spec)
    if direction == "BUY":
        stop, target = entry - sl_dist, entry + tp_dist
    else:
        stop, target = entry + sl_dist, entry - tp_dist
    stop = stop.quantize(quantum, rounding=ROUND_HALF_UP)
    target = target.quantize(quantum, rounding=ROUND_HALF_UP)
    return {
        "order": {
            "units": units_str,
            "instrument": spec.name,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": format_decimal(stop)},
            "takeProfitOnFill": {"price": format_decimal(target)},
        },
        "_entry": entry,
        "_stop": stop,
        "_target": target,
    }


def interpret_order_response(res: Any) -> tuple[str, str]:
    if not isinstance(res, dict):
        return "rejected", str(res)[:500]
    if "orderFillTransaction" in res:
        return "filled", ""
    reject = res.get("orderRejectTransaction") or res.get("orderCancelTransaction") or {}
    if not isinstance(reject, dict):
        reject = {}
    reason = reject.get("reason") or res.get("errorMessage") or str(res)
    return "rejected", str(reason)[:500]


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


def build_market_snapshot(spec: InstrumentSpec, candles: list[dict], price: dict) -> dict | None:
    completed = [c for c in candles if c.get("complete") and isinstance(c.get("mid"), dict)]
    if len(completed) < MIN_CLOSES:
        return None
    try:
        closes = [float(c["mid"]["c"]) for c in completed]
        atr = calculate_atr(completed)
    except Exception:
        return None
    if atr <= 0:
        return None

    bids = price.get("bids") or []
    asks = price.get("asks") or []
    if not bids or not asks:
        return None
    try:
        bid = Decimal(str(bids[0]["price"]))
        ask = Decimal(str(asks[0]["price"]))
    except Exception:
        return None
    if bid <= 0 or ask <= 0 or ask < bid:
        return None

    return {
        "bid": format_decimal(bid),
        "ask": format_decimal(ask),
        "tradeable": bool(price.get("tradeable", True)),
        "ema50": ema(closes, 50),
        "ema200": ema(closes, 200),
        "last_closes": closes[-10:],
        "atr": atr,
        "display_precision": spec.display_precision,
    }


def _short_error(exc: Exception) -> str:
    text = str(exc) or exc.__class__.__name__
    return text[:160]


def get_openrouter_opinion(client: OpenAI, model: str, pair: str, md: dict, rule_signal: dict) -> dict | None:
    prompt = (
        f"Review trade signal for {pair}.\nSignal: {json.dumps(rule_signal)}\nData: {json.dumps(md)}\n"
        'Return JSON only: {"agree": true, "comment": "short text"}'
    )
    messages = [
        {"role": "system", "content": "You are a financial JSON generator."},
        {"role": "user", "content": prompt},
    ]
    last_error: Exception | None = None
    # Some OpenRouter models reject JSON mode; if so, retry as plain text and parse the JSON.
    for json_mode in (True, False):
        try:
            kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": 300}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            data = extract_json(content)
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


class OandaBroker:
    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {settings.oanda_api_key}",
                "Content-Type": "application/json",
            }
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
        account = quote(self.settings.oanda_account_id, safe="")
        return f"/accounts/{account}{suffix}"

    def list_instruments(self) -> list[InstrumentSpec]:
        data = self._request("GET", self._account_path("/instruments"))
        specs = []
        for raw in data.get("instruments") or []:
            name = str(raw.get("name") or "")
            if raw.get("type") in ("CURRENCY", "METAL") and name in self.settings.trade_instruments:
                specs.append(
                    InstrumentSpec(
                        name=str(raw["name"]),
                        pip_location=int(raw["pipLocation"]),
                        display_precision=int(raw["displayPrecision"]),
                        trade_units_precision=int(raw.get("tradeUnitsPrecision") or 0),
                        minimum_trade_size=Decimal(str(raw.get("minimumTradeSize") or "1")),
                        maximum_order_units=Decimal(str(raw.get("maximumOrderUnits") or "0")),
                    )
                )
        return specs

    def open_instruments(self) -> set[str]:
        data = self._request("GET", self._account_path("/openTrades"))
        return {str(t["instrument"]) for t in (data.get("trades") or []) if isinstance(t, dict)}

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
        return build_market_snapshot(spec, candles.get("candles") or [], prices[0])

    def realized_pl_since(self, since: datetime) -> Decimal:
        """Sum realized P/L (incl. financing and commission) from fills since `since`."""
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
        """Return (realized today, unrealized now) in account currency."""
        realized = self.realized_pl_since(trading_day_start())
        data = self._request("GET", self._account_path("/summary"))
        unrealized = Decimal(str((data.get("account") or {}).get("unrealizedPL", "0")))
        return realized, unrealized

    def close_winning_trades(self) -> tuple[list[tuple[str, str]], list[str], int]:
        """Close every trade currently in profit. Returns ([(instrument, label)], failed, kept)."""
        data = self._request("GET", self._account_path("/openTrades"))
        closed: list[tuple[str, str]] = []
        failed: list[str] = []
        kept = 0
        for trade in data.get("trades") or []:
            if not isinstance(trade, dict):
                continue
            try:
                pl = Decimal(str(trade.get("unrealizedPL", "0")))
            except Exception:
                continue
            trade_id = str(trade.get("id"))
            instrument = str(trade.get("instrument", "?"))
            label = f"{instrument.replace('_', '/')} #{trade_id}"
            if pl <= 0:
                kept += 1
                continue
            try:
                self._request(
                    "PUT",
                    self._account_path(f"/trades/{quote(trade_id, safe='')}/close"),
                    json={"units": "ALL"},
                )
                closed.append((instrument, f"{label} (+{format_decimal(pl)})"))
            except Exception as exc:
                failed.append(f"{label}: {exc}")
        return closed, failed, kept

    def close_all_trades(self) -> tuple[list[str], list[str]]:
        data = self._request("GET", self._account_path("/openTrades"))
        closed: list[str] = []
        failed: list[str] = []
        for trade in data.get("trades") or []:
            if not isinstance(trade, dict):
                continue
            trade_id = str(trade.get("id"))
            label = f"{str(trade.get('instrument', '?')).replace('_', '/')} #{trade_id}"
            try:
                self._request(
                    "PUT",
                    self._account_path(f"/trades/{quote(trade_id, safe='')}/close"),
                    json={"units": "ALL"},
                )
                closed.append(label)
            except Exception as exc:
                failed.append(f"{label}: {exc}")
        return closed, failed

    def protect_trades(self) -> list[str]:
        """Move stops to break-even (+lock) once a trade is in profit; optional trailing stop."""
        s = self.settings
        if s.breakeven_trigger_pips <= 0:
            return []
        data = self._request("GET", self._account_path("/openTrades"))
        trades = [t for t in (data.get("trades") or []) if isinstance(t, dict)]
        if not trades:
            return []
        specs = {sp.name: sp for sp in self.list_instruments()}
        names = sorted({str(t["instrument"]) for t in trades})
        pricing = self._request("GET", self._account_path("/pricing"), params={"instruments": ",".join(names)})
        quotes: dict[str, tuple[Decimal, Decimal]] = {}
        for pr in pricing.get("prices") or []:
            try:
                quotes[str(pr["instrument"])] = (
                    Decimal(str(pr["bids"][0]["price"])),
                    Decimal(str(pr["asks"][0]["price"])),
                )
            except Exception:
                continue

        notes: list[str] = []
        for t in trades:
            try:
                name = str(t["instrument"])
                spec, quote_pair = specs.get(name), quotes.get(name)
                if not spec or not quote_pair:
                    continue
                bid, ask = quote_pair
                entry = Decimal(str(t["price"]))
                is_long = Decimal(str(t["currentUnits"])) > 0
                pip = spec.pip
                move = (bid - entry) if is_long else (entry - ask)
                profit_pips = move / pip
                is_gold = name == "XAU_USD"
                quantum = price_quantum(spec.display_precision)
                if is_gold:
                    # Gold: break-even once profit reaches 1R (the stop distance), locking one tick.
                    sl_raw = t.get("stopLossOrder")
                    if not isinstance(sl_raw, dict) or not sl_raw.get("price"):
                        continue
                    risk = abs(entry - Decimal(str(sl_raw["price"])))
                    if risk <= 0 or move < risk:
                        continue
                    lock = pip
                else:
                    if profit_pips < s.breakeven_trigger_pips:
                        continue
                    lock = s.breakeven_lock_pips * pip
                raw_sl = entry + lock if is_long else entry - lock
                target_sl = raw_sl.quantize(quantum, rounding=ROUND_HALF_UP)
                current_sl = None
                sl_order = t.get("stopLossOrder")
                if isinstance(sl_order, dict) and sl_order.get("price"):
                    current_sl = Decimal(str(sl_order["price"]))
                improves = current_sl is None or (target_sl > current_sl if is_long else target_sl < current_sl)

                body: dict[str, Any] = {}
                if improves:
                    body["stopLoss"] = {"price": format_decimal(target_sl)}
                if s.trail_pips > 0 and not is_gold and not t.get("trailingStopLossOrder") and profit_pips > s.trail_pips:
                    dist = (s.trail_pips * pip).quantize(quantum, rounding=ROUND_HALF_UP)
                    body["trailingStopLoss"] = {"distance": format_decimal(dist)}
                if not body:
                    continue

                trade_id = quote(str(t["id"]), safe="")
                self._request("PUT", self._account_path(f"/trades/{trade_id}/orders"), json=body)
                parts = []
                if "stopLoss" in body:
                    parts.append(f"SL -> {format_decimal(target_sl)}")
                if "trailingStopLoss" in body:
                    parts.append(f"trail {s.trail_pips} pips")
                notes.append(f"PROTECTED {spec.pair} #{t['id']} (+{profit_pips:.1f} pips): " + ", ".join(parts))
            except Exception as exc:
                logger.warning("protect_trades error for %s: %r", t.get("instrument"), exc)
        return notes

    def close_losing_trades(self) -> tuple[list[str], list[str], int]:
        """Close every open trade with negative unrealized P/L. Returns (closed, failed, kept)."""
        data = self._request("GET", self._account_path("/openTrades"))
        closed: list[str] = []
        failed: list[str] = []
        kept = 0
        for trade in data.get("trades") or []:
            if not isinstance(trade, dict):
                continue
            try:
                pl = Decimal(str(trade.get("unrealizedPL", "0")))
            except Exception:
                continue
            trade_id = str(trade.get("id"))
            label = f"{str(trade.get('instrument', '?')).replace('_', '/')} #{trade_id}"
            if pl >= 0:
                kept += 1
                continue
            try:
                self._request(
                    "PUT",
                    self._account_path(f"/trades/{quote(trade_id, safe='')}/close"),
                    json={"units": "ALL"},
                )
                closed.append(f"{label} ({format_decimal(pl)})")
            except Exception as exc:
                failed.append(f"{label}: {exc}")
        return closed, failed, kept

    def account_summary(self) -> tuple[Any, Any, Any]:
        data = self._request("GET", self._account_path("/summary"))
        acc = data.get("account") or {}
        return acc.get("balance", "N/A"), acc.get("unrealizedPL", "N/A"), acc.get("openTradeCount", 0)

    def submit(self, spec: InstrumentSpec, direction: str, units: Decimal, sl_pips: float, tp_pips: float, md: dict):
        built = build_order(spec, direction, units, sl_pips, tp_pips, md, self.settings)
        entry, stop, target = built.pop("_entry"), built.pop("_stop"), built.pop("_target")
        response = self._request("POST", self._account_path("/orders"), json=built)
        status, reason = interpret_order_response(response)
        return status, entry, stop, target, reason


def run_cycle(broker: Any, units: Decimal, opinion_fn: OpinionFn, skip_instruments: set[str] | None = None) -> ScanResult:
    try:
        instruments = broker.list_instruments()
        open_instruments = broker.open_instruments()
    except Exception as exc:
        return ScanResult(report=f"Scan failed: {exc}", failed=1, alert=True)

    reports: list[str] = []
    executed, failed = 0, 0
    for spec in instruments:
        if spec.name in open_instruments or (skip_instruments and spec.name in skip_instruments):
            continue
        try:
            md = broker.market_data(spec)
            if not md or not md.get("tradeable"):
                continue
            signal = evaluate_rules(md)
            if not signal.get("should_trade"):
                continue
            opinion = opinion_fn(spec.pair, md, signal)
            status, entry, stop, target, reason = broker.submit(
                spec, signal["direction"], units, float(signal["sl_pips"]), float(signal["tp_pips"]), md
            )
            if status == "filled":
                executed += 1
                reports.append(
                    f"TRADE EXECUTED ({spec.pair})\n"
                    f"Direction: {signal['direction']} @ {format_decimal(entry)}\n"
                    f"SL: {format_decimal(stop)} | TP: {format_decimal(target)}\n"
                    f"{ai_note(opinion)}"
                )
            else:
                failed += 1
                reports.append(f"ORDER FAILED ({spec.pair}): {reason}")
        except SkipPair as exc:
            logger.info("Skipped %s: %s", spec.name, exc)
        except Exception:
            logger.exception("Pair scan error for %s", spec.name)
            failed += 1

    if not reports:
        note = "Scan complete. No eligible setups found."
        if failed:
            note += f" ({failed} pair(s) errored; check logs.)"
        return ScanResult(report=note, executed=0, failed=failed, alert=False)
    return ScanResult(
        report=f"--- SCAN COMPLETED ({executed} opened) ---\n\n" + "\n\n".join(reports),
        executed=executed,
        failed=failed,
        alert=True,
    )


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
        logger.warning("PUBLIC_URL is not set: Telegram webhook NOT registered, bot will not receive commands")
    protect_task = None
    if settings.protect_interval > 0 and (settings.breakeven_trigger_pips > 0 or settings.cycle_target > 0):
        protect_task = asyncio.create_task(protect_loop())
    yield
    if protect_task:
        protect_task.cancel()
    await bot.shutdown()
    broker.close()


app = FastAPI(lifespan=lifespan)
# Defaults so handlers never hit AttributeError if config failed to load.
app.state.settings = None
app.state.bot = None
app.state.broker = None
app.state.opinion_fn = None
app.state.paused = False
app.state.scanning = False
app.state.seen_updates = OrderedDict()
app.state.milestone_day = None
app.state.ai_client = None
app.state.milestone = 0
app.state.loss_alert_day = None  # trading day on which the loss alert was sent
app.state.tasks = set()


async def protect_loop() -> None:
    """Background loop: lock in profits (break-even / trailing stop) on open trades."""
    while True:
        await asyncio.sleep(app.state.settings.protect_interval)
        try:
            loss_msg = await check_loss_limit()
            if loss_msg:
                await say(loss_msg)
            step_msg = await bank_milestones()
            if step_msg:
                await say(step_msg)
            notes = await asyncio.to_thread(app.state.broker.protect_trades)
            if notes:
                await say("\n".join(notes))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Trade protection loop error")


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


async def check_loss_limit() -> str | None:
    """Alert once per trading day when total P/L (realized + open) reaches -daily_loss_limit.
    Does not close trades or stop trading; use /closelosers to close losing trades."""
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
    """Each time today's realized profit crosses another cycle_target step, take profit on
    all winning trades. Trading keeps going; the next step is the next cycle_target."""
    settings = app.state.settings
    if not settings or settings.cycle_target <= 0:
        return None
    today = trading_day_start().isoformat()
    realized, _ = await asyncio.to_thread(app.state.broker.daily_pnl)
    milestone = int(realized // settings.cycle_target)
    if app.state.milestone_day != today:  # new trading day, or first check after a restart
        app.state.milestone_day = today
        app.state.milestone = milestone
        return None
    if milestone <= app.state.milestone:
        return None
    closed, failed, kept = await asyncio.to_thread(app.state.broker.close_winning_trades)
    if failed:  # do not advance; retry on the next check
        return "PROFIT STEP reached but some closes failed:\n" + "\n".join(failed)
    app.state.milestone = milestone
    return (
        f"PROFIT STEP: realized today {realized:.2f} (step ${settings.cycle_target}). "
        f"Took profit on {len(closed)} winning trade(s); {kept} not in profit left open. Trading continues."
    )


async def run_scan(skip: set[str] | None = None) -> ScanResult | None:
    """Run one scan in a worker thread. Returns None if a scan is already running."""
    try:
        loss_msg = await check_loss_limit()
        if loss_msg:
            await say(loss_msg)
    except Exception:
        logger.exception("Loss limit check failed")
    now = datetime.now(timezone.utc)
    if not market_open(now):
        return ScanResult(report="Forex market is closed (weekend). No new entries.")
    if not session_open(now, app.state.settings.trading_session):
        return ScanResult(
            report=f"Outside the '{app.state.settings.trading_session}' session window. No new entries this time."
        )
    if app.state.scanning:
        return None
    app.state.scanning = True
    try:
        return await asyncio.to_thread(
            run_cycle, app.state.broker, app.state.settings.trade_units, app.state.opinion_fn, skip
        )
    finally:
        app.state.scanning = False


async def handle_command(cmd: str) -> None:
    try:
        if cmd in ("/start", "/help"):
            await say(
                "Tris Bot Active (ATR + OpenRouter)\n\n"
                "/status - state\n/debug - dry run EUR/USD\n/trade - scan & execute\n"
                "/target - today's P/L and next profit step\n/takeprofit - close all winners, then rescan\n/ai - test OpenRouter connection\n/protect - lock in profits now\n/closelosers - close all losing trades\n/pause - pause\n/resume - resume"
            )
        elif cmd == "/status":
            bal, pnl, count = await asyncio.to_thread(app.state.broker.account_summary)
            state = "PAUSED" if app.state.paused else "RUNNING"
            await say(f"STATUS ({state}, {app.state.settings.oanda_env})\nBalance: {bal}\nPnL: {pnl}\nOpen Trades: {count}")
        elif cmd == "/debug":
            await say("Running debug test on EUR/USD...")
            instruments = await asyncio.to_thread(app.state.broker.list_instruments)
            spec = next(
                (i for i in instruments if i.name == "EUR_USD"),
                InstrumentSpec("EUR_USD", -4, 5, 0, Decimal("1"), Decimal("0")),
            )
            md = await asyncio.to_thread(app.state.broker.market_data, spec)
            if not md:
                await say("Debug Error: Could not fetch EUR_USD data.")
            else:
                signal = evaluate_rules(md)
                await say(f"DEBUG EUR/USD:\nClose: {md['last_closes'][-1]}\nATR: {md['atr']:.5f}\nSignal: {json.dumps(signal)}")
        elif cmd == "/target":
            realized, unrealized = await asyncio.to_thread(app.state.broker.daily_pnl)
            st_ = app.state.settings
            await say(
                f"TODAY'S P/L\nRealized: {realized:.2f}\nUnrealized: {unrealized:.2f}\n"
                f"Total: {realized + unrealized:.2f}\n"
                f"Profit step: every ${st_.cycle_target} realized (next at ${(int(realized // st_.cycle_target) + 1) * st_.cycle_target})\n"
                f"Session: {st_.trading_session}\n"
                f"Loss alert level: -{st_.daily_loss_limit} "
                + ("(alert sent today, trading continues)" if app.state.loss_alert_day == trading_day_start().isoformat() else "(not reached)")
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
            await say("Scanning for new setups (skipping the pairs just closed)...")
            result = await run_scan(skip={name for name, _ in closed})
            await say("A scan is already running." if result is None else result.report)
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
        elif cmd == "/protect":
            notes = await asyncio.to_thread(app.state.broker.protect_trades)
            await say("\n".join(notes) if notes else "No trades needed protection right now.")
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
            await say("Trading paused.")
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
    cmd = parse_command(str(msg.get("text") or ""))

    if sender_chat_id != settings.telegram_chat_id:
        logger.warning(
            "UNAUTHORIZED chat rejected: incoming chat id [%s] != TELEGRAM_CHAT_ID [%s]",
            sender_chat_id,
            settings.telegram_chat_id,
        )
        if cmd == "/id":  # lets you discover the right chat id to put in TELEGRAM_CHAT_ID
            await say(f"This chat's ID is: {sender_chat_id}", chat_id=sender_chat_id)
        return {"status": "unauthorized"}

    logger.info("Command received: %s", cmd)

    # Reply to Telegram immediately; long work (scans) runs in the background so
    # Telegram doesn't time out and retry the webhook.
    task = asyncio.create_task(handle_command(cmd))
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
    try:
        notes = await asyncio.to_thread(app.state.broker.protect_trades)
        if notes:
            await say("\n".join(notes))
    except Exception:
        logger.exception("Trade protection failed")
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
