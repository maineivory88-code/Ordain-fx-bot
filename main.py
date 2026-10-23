"""Tris forex agent: OANDA execution, ATR rules, OpenRouter notes, Telegram control.

Required environment:
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, OANDA_API_KEY, OANDA_ACCOUNT_ID, OPENROUTER_API_KEY

Optional environment:
  CRON_SECRET                 Required for /run-agent-check. If unset, that route stays closed.
  PUBLIC_URL                  Public https origin used to register the Telegram webhook
  OANDA_ENV                   practice (default) or live
  OPENROUTER_MODEL            default openrouter/free
  TRADE_UNITS                 default 1000
  TELEGRAM_WEBHOOK_SECRET     [A-Za-z0-9_-]{1,256}, sent back as a webhook header
  PORT                        default 8080
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable
from urllib.parse import quote

import requests
from fastapi import FastAPI, HTTPException, Request
from openai import OpenAI
from telegram import Bot

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


OpinionFn = Callable[[str, dict, dict], dict | None]


def load_settings() -> Settings:
    missing = [key for key in REQUIRED_ENV if not os.getenv(key, "").strip()]
    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))

    oanda_env = os.getenv("OANDA_ENV", "practice").strip().lower()
    if oanda_env not in ("practice", "live"):
        raise RuntimeError("OANDA_ENV must be 'practice' or 'live'")
    host = "api-fxpractice.oanda.com" if oanda_env == "practice" else "api-fxtrade.oanda.com"

    units_raw = os.getenv("TRADE_UNITS", "1000").strip()
    try:
        units = Decimal(units_raw)
    except Exception as exc:
        raise RuntimeError("TRADE_UNITS must be a number") from exc
    if units <= 0:
        raise RuntimeError("TRADE_UNITS must be greater than 0")

    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if secret and not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", secret):
        raise RuntimeError("TELEGRAM_WEBHOOK_SECRET must match [A-Za-z0-9_-]{1,256}")

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
    """Return True if this Telegram update was already handled."""
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


def evaluate_rules(md: dict) -> dict:
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

    if above or momentum_up:
        reasons = []
        if above:
            reasons.append("close is more than 0.5 ATR above EMA50")
        if momentum_up:
            reasons.append("upward momentum exceeds 0.5 ATR")
        return {
            "should_trade": True,
            "direction": "BUY",
            "sl_pips": 15.0,
            "tp_pips": 30.0,
            "reason": f"Bullish expansion: {'; '.join(reasons)} (ATR {atr:.5f})",
        }
    if below or momentum_down:
        reasons = []
        if below:
            reasons.append("close is more than 0.5 ATR below EMA50")
        if momentum_down:
            reasons.append("downward momentum exceeds 0.5 ATR")
        return {
            "should_trade": True,
            "direction": "SELL",
            "sl_pips": 15.0,
            "tp_pips": 30.0,
            "reason": f"Bearish expansion: {'; '.join(reasons)} (ATR {atr:.5f})",
        }
    return {
        "should_trade": False,
        "reason": "No expansion: inside the 0.5 ATR band and momentum is below the threshold.",
    }


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
        raise ValueError("SL/TP distance rounded to zero; instrument pip size is wrong")
    if direction == "BUY":
        stop = (entry - sl_distance).quantize(quantum, rounding=ROUND_HALF_UP)
        target = (entry + tp_distance).quantize(quantum, rounding=ROUND_HALF_UP)
        if not stop < entry < target:
            raise ValueError("BUY stop/target ended up on the wrong side of entry")
        return stop, target
    if direction == "SELL":
        stop = (entry + sl_distance).quantize(quantum, rounding=ROUND_HALF_UP)
        target = (entry - tp_distance).quantize(quantum, rounding=ROUND_HALF_UP)
        if not target < entry < stop:
            raise ValueError("SELL stop/target ended up on the wrong side of entry")
        return stop, target
    raise ValueError(f"Unknown direction: {direction}")


def order_units(units: Decimal, direction: str, spec: InstrumentSpec) -> str:
    signed = units if direction == "BUY" else -units
    quantum = price_quantum(spec.trade_units_precision)
    sized = signed.quantize(quantum, rounding=ROUND_HALF_UP)
    if sized == 0 or abs(sized) < spec.minimum_trade_size:
        raise SkipPair(
            f"units {format_decimal(abs(sized))} are below the minimum "
            f"{format_decimal(spec.minimum_trade_size)} for {spec.name}"
        )
    if spec.maximum_order_units > 0 and abs(sized) > spec.maximum_order_units:
        raise SkipPair(
            f"units {format_decimal(abs(sized))} exceed the maximum "
            f"{format_decimal(spec.maximum_order_units)} for {spec.name}"
        )
    return format_decimal(sized)


def build_order(spec: InstrumentSpec, direction: str, units: Decimal, sl_pips: float, tp_pips: float, md: dict) -> dict:
    entry_key = "ask" if direction == "BUY" else "bid"
    entry = Decimal(md[entry_key])
    stop, target = protective_prices(
        entry,
        direction,
        Decimal(str(sl_pips)),
        Decimal(str(tp_pips)),
        spec.pip,
        spec.display_precision,
    )
    spread = Decimal(md["ask"]) - Decimal(md["bid"])
    sl_distance = (Decimal(str(sl_pips)) * spec.pip).quantize(
        price_quantum(spec.display_precision), rounding=ROUND_HALF_UP
    )
    if spread < 0:
        raise ValueError(f"crossed quote on {spec.name}")
    if spread > sl_distance:
        raise SkipPair(
            f"spread {format_decimal(spread)} is wider than the {sl_pips:g}-pip stop on {spec.name}"
        )
    return {
        "order": {
            "units": order_units(units, direction, spec),
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
    reason_text = str(reason)[:500]
    if reason_text == "CLIENT_ORDER_ID_ALREADY_EXISTS":
        return "duplicate", reason_text
    return "rejected", reason_text


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
    """True only when the model explicitly says agree is false. Missing AI fails open."""
    if not opinion or "agree" not in opinion:
        return False
    value = opinion.get("agree")
    if isinstance(value, str):
        return value.strip().lower() in {"false", "no", "0"}
    return value is False


def build_market_snapshot(spec: InstrumentSpec, candles: list[dict], price: dict) -> dict | None:
    completed = [candle for candle in candles if candle.get("complete") and isinstance(candle.get("mid"), dict)]
    if len(completed) < MIN_CLOSES:
        return None
    try:
        closes = [float(candle["mid"]["c"]) for candle in completed]
        atr = calculate_atr(completed)
    except (KeyError, TypeError, ValueError):
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
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return None
    if bid <= 0 or ask <= 0 or ask < bid:
        return None

    status = str(price.get("status") or "tradeable").lower()
    tradeable_flag = price.get("tradeable", status == "tradeable")
    tradeable = bool(tradeable_flag) and status == "tradeable"
    return {
        "bid": format_decimal(bid),
        "ask": format_decimal(ask),
        "tradeable": tradeable,
        "ema50": ema(closes, 50),
        "ema200": ema(closes, 200),
        "last_closes": closes[-10:],
        "atr": atr,
        "spread": format_decimal(ask - bid),
        "candle_time": str(completed[-1].get("time") or ""),
        "pip": format_decimal(spec.pip),
        "display_precision": spec.display_precision,
    }


class SkipPair(Exception):
    """Pair was not sent because the quote or size is not tradable."""


def get_openrouter_opinion(client: OpenAI, model: str, pair: str, md: dict, rule_signal: dict) -> dict | None:
    snapshot = {
        "bid": md.get("bid"),
        "ask": md.get("ask"),
        "tradeable": md.get("tradeable"),
        "ema50": md.get("ema50"),
        "ema200": md.get("ema200"),
        "last_closes": md.get("last_closes"),
        "atr": md.get("atr"),
        "spread": md.get("spread"),
    }
    prompt = (
        f"You are a forex risk analyst reviewing a rule-based trade signal for {pair}.\n"
        f"Rule Signal: {json.dumps(rule_signal)}\n"
        f"Market Data: {json.dumps(snapshot)}\n"
        "Provide a brief second opinion. Return JSON only in this exact format:\n"
        '{"agree": true, "comment": "short analysis or warning"}'
    )
    attempts: tuple[dict, ...] = ({"response_format": {"type": "json_object"}}, {})
    last_error: Exception | None = None
    for extra in attempts:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a financial JSON generator."},
                    {"role": "user", "content": prompt},
                ],
                **extra,
            )
            content = response.choices[0].message.content or ""
            data = extract_json(content)
            comment = str(data.get("comment", "N/A"))[:500]
            return {"agree": data.get("agree", True), "comment": comment}
        except Exception as exc:
            last_error = exc
    logger.warning("OpenRouter second opinion skipped for %s: %r", pair, last_error)
    return None


def ai_note(opinion: dict | None) -> str:
    if not opinion:
        return "OpenRouter AI: Offline / Skipped"
    agree = not explicit_disagree(opinion)
    comment = opinion.get("comment") or "N/A"
    return f"OpenRouter AI: {'Agree' if agree else 'Caution'} - {comment}"


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
        last_error = "unknown error"
        for attempt in range(3):
            try:
                response = self.session.request(method, url, **kwargs)
            except requests.RequestException as exc:
                last_error = str(exc)
                time.sleep(0.4 * (attempt + 1))
                continue
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}: {response.text[:180]}"
                time.sleep(0.4 * (attempt + 1))
                continue
            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError(f"OANDA returned non-JSON (HTTP {response.status_code})") from exc
            if response.status_code >= 400:
                message = data.get("errorMessage") if isinstance(data, dict) else data
                raise RuntimeError(f"OANDA HTTP {response.status_code}: {message}")
            if not isinstance(data, dict):
                raise RuntimeError("OANDA response was not a JSON object")
            return data
        raise RuntimeError(f"OANDA request failed after retries: {last_error}")

    def _account_path(self, suffix: str) -> str:
        account = quote(self.settings.oanda_account_id, safe="")
        return f"/accounts/{account}{suffix}"

    def list_instruments(self) -> list[InstrumentSpec]:
        data = self._request("GET", self._account_path("/instruments"))
        specs: list[InstrumentSpec] = []
        for raw in data.get("instruments") or []:
            if raw.get("type") != "CURRENCY" or not raw.get("name"):
                continue
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
        return {
            str(trade["instrument"])
            for trade in data.get("trades") or []
            if isinstance(trade, dict) and trade.get("instrument")
        }

    def market_data(self, spec: InstrumentSpec) -> dict | None:
        candles = self._request(
            "GET",
            f"/instruments/{quote(spec.name, safe='')}/candles",
            params={"granularity": "H1", "count": CANDLE_LOOKBACK, "price": "M"},
        )
        pricing = self._request(
            "GET",
            self._account_path("/pricing"),
            params={"instruments": spec.name},
        )
        prices = pricing.get("prices") or []
        if not prices:
            return None
        return build_market_snapshot(spec, candles.get("candles") or [], prices[0])

    def account_summary(self) -> tuple[Any, Any, Any]:
        data = self._request("GET", self._account_path("/summary"))
        account = data.get("account") or {}
        return (
            account.get("balance", "N/A"),
            account.get("unrealizedPL", "N/A"),
            account.get("openTradeCount", 0),
        )

    def submit(self, spec: InstrumentSpec, direction: str, units: Decimal, sl_pips: float, tp_pips: float, md: dict) -> tuple[str, Decimal, Decimal, Decimal, str]:
        built = build_order(spec, direction, units, sl_pips, tp_pips, md)
        entry, stop, target = built.pop("_entry"), built.pop("_stop"), built.pop("_target")
        response = self._request("POST", self._account_path("/orders"), json=built)
        status, reason = interpret_order_response(response)
        return status, entry, stop, target, reason


def run_cycle(broker: Any, units: Decimal, opinion_fn: OpinionFn) -> ScanResult:
    try:
        instruments = broker.list_instruments()
        open_instruments = broker.open_instruments()
    except Exception as exc:
        logger.exception("scan setup failed")
        report = f"Scan failed before any orders: {exc}"
        return ScanResult(report=report, failed=1, alert=True)

    if not instruments:
        report = (
            "No currency instruments returned. Check the API key, account id, "
            "and OANDA_ENV (practice vs live)."
        )
        return ScanResult(report=report, failed=1, alert=True)

    reports: list[str] = []
    errors: list[str] = []
    executed = 0
    failed = 0

    for spec in instruments:
        if spec.name in open_instruments:
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
                spec,
                signal["direction"],
                units,
                float(signal["sl_pips"]),
                float(signal["tp_pips"]),
                md,
            )
            if status == "filled":
                executed += 1
                reports.append(
                    f"TRADE EXECUTED ({spec.pair})\n"
                    f"Direction: {signal['direction']} @ {format_decimal(entry)}\n"
                    f"SL {format_decimal(stop)} | TP {format_decimal(target)}\n"
                    f"Reason: {signal['reason']}\n"
                    f"{ai_note(opinion)}"
                )
            elif status == "duplicate":
                logger.info("duplicate order skipped for %s", spec.name)
            else:
                failed += 1
                reports.append(f"ORDER FAILED ({spec.pair})\nReason: {reason}")
        except SkipPair as exc:
            reports.append(f"SKIPPED ({spec.pair})\n{exc}")
        except Exception as exc:
            logger.exception("pair scan failed for %s", spec.name)
            errors.append(f"{spec.pair}: {exc}")

    if errors:
        failed += len(errors)
        preview = "\n".join(errors[:20])
        reports.append(f"DATA ERRORS ({len(errors)})\n{preview}")

    if not reports:
        return ScanResult(
            report="Scan complete. No eligible trade setups across currency pairs right now.",
            executed=0,
            failed=0,
            alert=False,
        )

    summary = f"--- SCAN COMPLETED ({executed} trades opened) ---\n\n" + "\n\n".join(reports)
    return ScanResult(report=summary, executed=executed, failed=failed, alert=executed > 0 or failed > 0)


def make_opinion_fn(settings: Settings) -> OpinionFn:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.openrouter_api_key, timeout=15)

    def opinion(pair: str, md: dict, rule_signal: dict) -> dict | None:
        return get_openrouter_opinion(client, settings.openrouter_model, pair, md, rule_signal)

    return opinion


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        settings = load_settings()
    except RuntimeError as exc:
        logger.error("%s", exc)
        yield
        return

    bot = Bot(token=settings.telegram_token)
    broker = OandaBroker(settings)
    app.state.settings = settings
    app.state.bot = bot
    app.state.broker = broker
    app.state.opinion_fn = make_opinion_fn(settings)
    await bot.initialize()
    try:
        if settings.public_url:
            kwargs: dict[str, Any] = {"allowed_updates": ["message"]}
            if settings.webhook_secret:
                kwargs["secret_token"] = settings.webhook_secret
            try:
                await bot.set_webhook(url=webhook_url(settings.public_url), **kwargs)
            except Exception:
                logger.exception("Telegram set_webhook failed")
        yield
    finally:
        await bot.shutdown()
        broker.close()
        app.state.bot = None
        app.state.broker = None


app = FastAPI(lifespan=lifespan)
app.state.bot = None
app.state.settings = None
app.state.broker = None
app.state.opinion_fn = None
app.state.scanning = False
app.state.paused = False
app.state.seen_updates = OrderedDict()
app.state.scan_task = None


def begin_scan() -> bool:
    if app.state.scanning:
        return False
    app.state.scanning = True
    return True


async def say(text: str) -> None:
    bot = app.state.bot
    settings = app.state.settings
    if bot is None or settings is None:
        raise RuntimeError("Bot is not configured. Set the required environment variables and restart.")
    parts = chunk_text(text)
    for index, part in enumerate(parts):
        await bot.send_message(chat_id=settings.telegram_chat_id, text=part)
        if index != len(parts) - 1:
            await asyncio.sleep(0.4)


async def guarded_scan(always_notify: bool) -> dict:
    try:
        settings = app.state.settings
        result = await asyncio.to_thread(
            run_cycle, app.state.broker, settings.trade_units, app.state.opinion_fn
        )
    except Exception as exc:
        logger.exception("scan failed")
        message = f"Scan error: {exc}"[:500]
        try:
            await say(message)
        except Exception:
            logger.exception("failed to report scan error")
        return {"status": "error", "detail": message}
    else:
        try:
            if always_notify or result.alert:
                await say(result.report)
        except Exception:
            logger.exception("failed to deliver scan report")
        return {
            "status": "scan complete",
            "executed": result.executed,
            "failed": result.failed,
            "detail": result.report,
        }
    finally:
        app.state.scanning = False


@app.get("/")
def home():
    settings = app.state.settings
    body = {"status": "Tris OpenRouter ATR Agent is active!"}
    if settings is None:
        body["configured"] = False
    else:
        body["configured"] = True
        body["oanda_env"] = settings.oanda_env
        body["paused"] = bool(app.state.paused)
    return body


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    settings = app.state.settings
    if settings is None:
        return {"status": "not configured"}
    if settings.webhook_secret:
        header = request.headers.get("x-telegram-bot-api-secret-token", "")
        if header != settings.webhook_secret:
            logger.warning("webhook secret mismatch")
            return {"status": "unauthorized"}

    try:
        data = await request.json()
    except Exception:
        return {"status": "bad json"}
    if not isinstance(data, dict):
        return {"status": "ignored"}
    if remember_update(app.state.seen_updates, data.get("update_id")):
        return {"status": "duplicate"}

    msg = data.get("message")
    if not isinstance(msg, dict) or "text" not in msg:
        return {"status": "ignored"}
    chat = msg.get("chat") or {}
    if str(chat.get("id")) != settings.telegram_chat_id:
        return {"status": "unauthorized"}

    cmd = parse_command(str(msg.get("text") or ""))
    try:
        if cmd in ("/start", "/help"):
            await say(
                "Tris Bot Active (ATR + OpenRouter Multi-Pair)\n\n"
                "/status - view account balance & state\n"
                "/debug - run indicator & rule test on EUR/USD (dry run)\n"
                "/trade - scan all available OANDA pairs & execute\n"
                "/pause - pause automated cron trading\n"
                "/resume - resume automated cron trading"
            )
        elif cmd == "/status":
            balance, pnl, count = await asyncio.to_thread(app.state.broker.account_summary)
            await say(
                "TRIS STATUS\n"
                f"Bot State: {'PAUSED' if app.state.paused else 'ACTIVE'}\n"
                f"OANDA: {settings.oanda_env}\n"
                f"Balance: {balance}\n"
                f"Unrealized PnL: {pnl}\n"
                f"Open Trades: {count}"
            )
        elif cmd == "/debug":
            await say("Running debug test on EUR/USD...")
            spec = InstrumentSpec("EUR_USD", -4, 5, 0, Decimal("1"), Decimal("0"))
            instruments = await asyncio.to_thread(app.state.broker.list_instruments)
            known = next((item for item in instruments if item.name == "EUR_USD"), spec)
            md = await asyncio.to_thread(app.state.broker.market_data, known)
            if not md:
                await say("Debug Error: Unable to fetch EUR/USD market data.")
            else:
                signal = evaluate_rules(md)
                opinion = await asyncio.to_thread(app.state.opinion_fn, "EUR/USD", md, signal)
                precision = int(md.get("display_precision") or 5)
                await say(
                    "DEBUG REPORT (EUR/USD)\n"
                    f"Last Close: {md['last_closes'][-1]:.{precision}f}\n"
                    f"EMA50: {md['ema50']:.{precision}f}\n"
                    f"EMA200: {md['ema200']:.{precision}f}\n"
                    f"ATR(14): {md['atr']:.{precision}f}\n"
                    f"Bid/Ask: {md['bid']} / {md['ask']}\n"
                    f"Rule Result: {json.dumps(signal, indent=2)}\n"
                    f"OpenRouter AI: {json.dumps(opinion) if opinion else 'Offline/Skipped'}"
                )
        elif cmd == "/pause":
            app.state.paused = True
            await say("Automated trading PAUSED.")
        elif cmd == "/resume":
            app.state.paused = False
            await say("Automated trading RESUMED.")
        elif cmd == "/trade":
            # /pause only stops cron. A manual /trade still runs.
            if not begin_scan():
                await say("A scan is already running.")
            else:
                try:
                    await say("Scanning all available OANDA currency pairs with ATR filter...")
                except Exception:
                    logger.exception("scan acknowledgement failed")
                finally:
                    # CancelledError is a BaseException on Python 3.9+, so start the scan
                    # from finally or a dropped webhook ack leaves the bot stuck "scanning".
                    app.state.scan_task = asyncio.create_task(guarded_scan(True))
        else:
            await say(f"Unknown command: {cmd}\nType /help for available commands.")
    except Exception as exc:
        logger.exception("handler error")
        try:
            await say(f"Error handling command: {type(exc).__name__}: {exc}"[:500])
        except Exception:
            logger.exception("failed to report handler error")
    return {"status": "ok"}


@app.get("/run-agent-check")
@app.post("/run-agent-check")
async def run_agent_check(request: Request, key: str = ""):
    settings = app.state.settings
    if settings is None:
        raise HTTPException(status_code=503, detail="Bot is not configured")
    header_key = request.headers.get("x-cron-key", "")
    if not settings.cron_secret or (key != settings.cron_secret and header_key != settings.cron_secret):
        raise HTTPException(status_code=403, detail="Forbidden")
    if app.state.paused:
        return {"status": "paused"}
    if not begin_scan():
        return {"status": "already running"}
    return await guarded_scan(False)


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
