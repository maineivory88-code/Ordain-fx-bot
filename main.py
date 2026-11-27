"""Tris forex bot: H1 trend-pullback entries, fixed-dollar risk, broker-side stop and target.

Strategy (per pair, on closed H1 candles):
  BUY : EMA50 > EMA200 and close > EMA200, bar dipped to EMA20 and closed back above it bullish.
  SELL: mirror image.
  Stop = 1.5 x ATR, target = 1.5 x ATR (1:1). Spread must be below 10% of ATR.
Sizing: units chosen so a stop-out loses RISK_PER_TRADE_USD (default $10).
Exits: stop loss and take profit are set on the OANDA order, so they work without the bot running.
Safety: starts PAUSED, max open trades, one entry per pair per H1 bar, daily loss alert only.
No guarantee of profit; no backtest has been run on this exact strategy yet.
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

# Strategy and risk settings (code constants; Render's TRADE_UNITS and TRADE_INSTRUMENTS are ignored)
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


def bar_key(now: datetime) -> str:
    return now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()


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


# ---------- indicators and strategy ----------

def ema(values: list[float], period: int) -> float:
    period = min(period, len(values))
    k = 2 / (period + 1)
    seed = sum(values[:period]) / period
    for value in values[period:]:
        seed = value * k + seed * (1 - k)
    return seed


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
    """Return (direction or None, reason, atr). Uses only completed candles."""
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
    """Build an order whose stop-out loses about RISK_PER_TRADE_USD."""
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
    """USD value of one unit of the quote currency, from prices already fetched."""
    q = spec.quote_ccy
    if q == "USD":
        return Decimal(1)
    direct = prices.get(f"{q}_USD")
    if direct:
        return mid_price(direct)
    inverse = prices.get(f"USD_{q}")
    if inverse:
        return Decimal(1) / mid_price(inverse)
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
    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {settings.oanda_api_key}", "Content-Type": "application/json"})

    def close(self) -> None:
        self.session.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        kwargs.setdefault("timeout", 20)
        response = self.session.request(method, f"{self.settings.oanda_base}{path}", **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"OANDA HTTP {response.status_code}: {response.text[:180]}")
        return response.json()

    def _account_path(self, suffix: str) -> str:
        return f"/accounts/{quote(self.settings.oanda_account_id, safe='')}{suffix}"

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

    def candles(self, name: str) -> list[dict]:
        data = self._request(
            "GET", f"/instruments/{quote(name, safe='')}/candles",
            params={"granularity": "H1", "count": CANDLE_LOOKBACK, "price": "M"},
        )
        return data.get("candles") or []

    def open_trades(self) -> list[dict]:
        data = self._request("GET", self._account_path("/openTrades"))
        return [t for t in (data.get("trades") or []) if isinstance(t, dict) and t.get("id") is not None]

    def submit(self, built: dict) -> tuple[str, str]:
        response = self._request("POST", self._account_path("/orders"), json={"order": built["order"]})
        return interpret_order_response(response)

    def realized_pl_since(self, since: datetime) -> Decimal:
        fmt = "%Y-%m-%dT%H:%M:%S.000000000Z"
        params = {"from": since.strftime(fmt), "to": datetime.now(timezone.utc).strftime(fmt), "type": "ORDER_FILL", "pageSize": 1000}
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
        return realized, Decimal(str((data.get("account") or {}).get("unrealizedPL", "0")))

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
        acc = self._request("GET", self._account_path("/summary")).get("account") or {}
        return acc.get("balance", "N/A"), acc.get("unrealizedPL", "N/A"), acc.get("openTradeCount", 0)


def format_dec(value: Decimal) -> str:
    return f"{value:+.2f}"


def price_request_names(specs: dict) -> list[str]:
    """Only instruments OANDA actually lists. The configured pairs already include every
    USD conversion needed for sizing (USD_JPY, USD_CHF, USD_CAD, GBP_USD), so no extra
    names are requested. Requesting a non-existent name such as JPY_USD makes OANDA reject the call."""
    return list(specs.keys())


def run_cycle(broker: OandaBroker, settings: Settings, opinion_fn: Callable[[str, dict], dict], blocked: set[str]) -> ScanResult:
    """Scan every configured pair once. Opens at most MAX_OPEN_TRADES in total."""
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
        except Exception as exc:
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
    """Explain what the rules see for one pair, without trading."""
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


# ---------- GBPJPY MULTI-BOT FOUNDATION ----------
# Practice only. These three bots do not submit orders.
# Each bot has its own Telegram token, so each one needs its own webhook
# (or its own polling loop). They previously never connected, which is why
# messages to them got no reply.

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
    config_error: str = ""
    oanda_base: str = "https://api-fxpractice.oanda.com/v3"


@dataclass
class GJRuntime:
    config: GJBotConfig
    bot: Bot | None = None
    seen: OrderedDict = field(default_factory=OrderedDict)
    mode: str = "offline"
    username: str = ""
    last_error: str = ""
    poll_task: asyncio.Task | None = None


GJ_BOT_DEFINITIONS = {
    "intraday": {
        "env": "ENABLE_GJ_INTRADAY",
        "token": "TELEGRAM_GJ_INTRADAY_TOKEN",
        "chat": "TELEGRAM_GJ_INTRADAY_CHAT_ID",
        "account": "OANDA_ACCOUNT_ID_GJ_INTRADAY",
        "risk": Decimal("0.0050"),
        "weekly_limit": 3,
    },
    "swing": {
        "env": "ENABLE_GJ_SWING",
        "token": "TELEGRAM_GJ_SWING_TOKEN",
        "chat": "TELEGRAM_GJ_SWING_CHAT_ID",
        "account": "OANDA_ACCOUNT_ID_GJ_SWING",
        "risk": Decimal("0.0050"),
        "weekly_limit": 3,
    },
    "scalp": {
        "env": "ENABLE_GJ_SCALP",
        "token": "TELEGRAM_GJ_SCALP_TOKEN",
        "chat": "TELEGRAM_GJ_SCALP_CHAT_ID",
        "account": "OANDA_ACCOUNT_ID_GJ_SCALP",
        "risk": Decimal("0.0025"),
        "weekly_limit": 3,
    },
}

# Portfolio-wide limits agreed for the GBPJPY system.
GJ_PORTFOLIO_OPEN_RISK_LIMIT = Decimal("0.0100")
GJ_DAILY_LOSS_LIMIT = Decimal("0.0200")

GJ_RISK_TIMEZONE = ZoneInfo("America/New_York")
GJ_INSTRUMENT = "GBP_JPY"
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
    """Telegram error strings include the bot token inside the request URL."""
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
        config_error=combined,
        oanda_base=cfg.oanda_base,
    )


def load_gj_bot_configs() -> dict[str, GJBotConfig]:
    """Load the three GBPJPY bots. A bad flag or missing secret marks that bot; it does not stop the others."""
    configs: dict[str, GJBotConfig] = {}
    shared_chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    for key, definition in GJ_BOT_DEFINITIONS.items():
        enabled, flag_error = _parse_flag(definition["env"])
        token = os.getenv(definition["token"], "").strip()
        account_id = os.getenv(definition["account"], "").strip()
        chat_id = os.getenv(definition["chat"], "").strip() or shared_chat
        problems: list[str] = []
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
            config_error="; ".join(problems),
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
                    configs[key], "Enabled GBPJPY bots must use distinct OANDA Practice account IDs"
                )
    return configs


def gj_foundation_status(configs: dict[str, GJBotConfig]) -> dict[str, Any]:
    """Configuration snapshot. Does not include tokens or account ids."""
    return {
        "instrument": GJ_INSTRUMENT,
        "broker_environment": "practice",
        "order_execution": "not_implemented",
        "portfolio_risk_gate": "not_implemented_fail_closed",
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
                "broker_environment": "practice",
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
        if rt is not None:
            info["telegram"] = rt.mode
            info["username"] = rt.username
            if rt.last_error:
                info["telegram_error"] = rt.last_error
    return payload


def format_gj_report() -> str:
    status = gj_public_status()
    if "bots" not in status:
        return "GJ bots: " + status.get("error", "not loaded")
    lines = ["GBPJPY bots (OANDA practice, no orders are sent):"]
    for info in status["bots"].values():
        who = f" @{info['username']}" if info.get("username") else ""
        line = f"{info['name']}{who}: Telegram {info.get('telegram', 'offline')}"
        if not info.get("telegram_configured"):
            line += ", token missing"
        if info.get("config_error"):
            line += f"\n  {info['config_error']}"
        if info.get("telegram_error"):
            line += f"\n  {info['telegram_error']}"
        lines.append(line)
    lines.append("Message each bot and send /start. From this bot, /gj shows this again.")
    return "\n".join(lines)


def make_bot(token: str) -> Bot:
    """Long polling needs a read timeout longer than Telegram's poll window."""
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
    """Register an https webhook, or fall back to polling so the bot still replies."""
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


def gj_practice_summary(api_key: str, account_id: str) -> str:
    """Read-only practice balance. Never calls the live OANDA host and never places an order."""
    tail = account_id[-4:] if account_id else "????"
    if not api_key:
        return f"Practice account …{tail}: OANDA_API_KEY is missing, so the balance cannot be read."
    url = f"https://api-fxpractice.oanda.com/v3/accounts/{quote(account_id, safe='')}/summary"
    try:
        response = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
    except Exception as exc:
        return f"Practice account …{tail}: request failed ({scrub_secrets(_short_error(exc))})"
    if response.status_code >= 400:
        return f"Practice account …{tail}: OANDA HTTP {response.status_code}: {scrub_secrets(response.text[:180])}"
    account = (response.json() or {}).get("account") or {}
    return (
        f"Practice account …{tail}\n"
        f"Balance: {account.get('balance', 'N/A')}\n"
        f"Unrealized: {account.get('unrealizedPL', 'N/A')}\n"
        f"Open trades: {account.get('openTradeCount', 0)}"
    )


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
    definition = GJ_BOT_DEFINITIONS[key]
    cfg = app.state.gj_bots[key].config
    return (
        f"{cfg.display_name}\n\n"
        "This bot replies here and can read its OANDA practice account. It does not place orders.\n\n"
        "/status - connection, flags, practice balance\n"
        "/ping - check that this process can reply\n"
        "/id - this chat's id\n\n"
        f"Token env: {definition['token']}\n"
        f"Chat env: {definition['chat']} (falls back to TELEGRAM_CHAT_ID)\n"
        f"Account env: {definition['account']}\n"
        f"Trading flag: {definition['env']} (still does not send orders)"
    )


async def handle_gj_command(key: str, cmd: str) -> None:
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
        elif cmd == "/status":
            lines = [
                cfg.display_name + (f" (@{runtime.username})" if runtime.username else ""),
                f"Telegram: {runtime.mode}",
                f"Trading flag {definition['env']}: {'ON' if cfg.enabled else 'OFF'}",
                "Orders: none. Execution is not turned on; this bot will not buy or sell.",
                f"Risk budget: {cfg.risk_fraction:.2%} of equity | weekly cap: {cfg.weekly_trade_limit} (not enforced yet)",
                f"Chat lock: {cfg.telegram_chat_id or 'NOT SET — set ' + definition['chat']}",
            ]
            if cfg.config_error:
                lines.append("Config: " + cfg.config_error)
            if runtime.last_error:
                lines.append("Note: " + runtime.last_error)
            if cfg.oanda_account_id:
                summary = await asyncio.to_thread(
                    gj_practice_summary, os.getenv("OANDA_API_KEY", "").strip(), cfg.oanda_account_id
                )
                lines.append(summary)
            else:
                lines.append(f"Account: set {definition['account']}")
            await gj_say(key, "\n".join(lines))
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
    await handle_gj_command(key, cmd)


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
    for key, cfg in configs.items():
        runtime = GJRuntime(config=cfg)
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
            if cfg.telegram_chat_id:
                await gj_say(key, f"{cfg.display_name} is online ({mode}).\nSend /status")
        except Exception as exc:
            runtime.mode = "offline"
            runtime.last_error = scrub_secrets(_short_error(exc))
            logger.exception("GJ %s Telegram setup failed", key)


async def stop_gj_bots() -> None:
    runtimes = list((getattr(app.state, "gj_bots", None) or {}).values())
    tasks = [runtime.poll_task for runtime in runtimes if runtime.poll_task is not None]
    for task in tasks:
        task.cancel()
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("GJ poll task shutdown failed")
    for runtime in runtimes:
        if runtime.bot is not None:
            try:
                await runtime.bot.shutdown()
            except Exception:
                logger.exception("GJ %s shutdown failed", runtime.config.key)

# ---------- app ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # The legacy bot and the three GJ bots start independently. A missing
    # OpenRouter key must not silence the GJ bots, and a bad GJ token must
    # not stop the legacy bot.
    tasks: list[asyncio.Task] = []
    settings = None
    try:
        settings = load_settings()
    except RuntimeError as exc:
        logger.error("Configuration error: %s", exc)
    if settings is not None:
        bot = make_bot(settings.telegram_token)
        broker = OandaBroker(settings)
        ai_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=settings.openrouter_api_key, timeout=30)
        app.state.settings = settings
        app.state.bot = bot
        app.state.broker = broker
        app.state.ai_client = ai_client
        app.state.opinion_fn = lambda pair, signal: get_openrouter_opinion(ai_client, settings.openrouter_model, pair, signal)
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
app.state.paused = True                  # starts PAUSED: send /resume to begin trading
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
    """Alert once per trading day when today's total P/L reaches -DAILY_LOSS_LIMIT. Never closes trades."""
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
    """One scan in a worker thread. Returns None if a scan is already running.
    Each pair can be entered at most once per H1 bar."""
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