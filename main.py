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
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable
from urllib.parse import quote

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

    bullish = above or momentum_up
    bearish = below or momentum_down
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
    return {"should_trade": False, "reason": "No expansion: inside 0.5 ATR band."}


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


def get_openrouter_opinion(client: OpenAI, model: str, pair: str, md: dict, rule_signal: dict) -> dict | None:
    prompt = (
        f"Review trade signal for {pair}.\nSignal: {json.dumps(rule_signal)}\nData: {json.dumps(md)}\n"
        'Return JSON only: {"agree": true, "comment": "short text"}'
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial JSON generator."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        data = extract_json(content)
        return {"agree": data.get("agree", True), "comment": str(data.get("comment", "N/A"))[:500]}
    except Exception as exc:
        logger.warning("OpenRouter opinion skipped for %s: %r", pair, exc)
        return None


def ai_note(opinion: dict | None) -> str:
    if not opinion:
        return "OpenRouter AI: Offline / Skipped"
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
            if raw.get("type") == "CURRENCY" and raw.get("name"):
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

    def account_summary(self) -> tuple[Any, Any, Any]:
        data = self._request("GET", self._account_path("/summary"))
        acc = data.get("account") or {}
        return acc.get("balance", "N/A"), acc.get("unrealizedPL", "N/A"), acc.get("openTradeCount", 0)

    def submit(self, spec: InstrumentSpec, direction: str, units: Decimal, sl_pips: float, tp_pips: float, md: dict):
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
        return ScanResult(report=f"Scan failed: {exc}", failed=1, alert=True)

    reports: list[str] = []
    executed, failed = 0, 0
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
    yield
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
app.state.tasks = set()


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


async def run_scan() -> ScanResult | None:
    """Run one scan in a worker thread. Returns None if a scan is already running."""
    if app.state.scanning:
        return None
    app.state.scanning = True
    try:
        return await asyncio.to_thread(
            run_cycle, app.state.broker, app.state.settings.trade_units, app.state.opinion_fn
        )
    finally:
        app.state.scanning = False


async def handle_command(cmd: str) -> None:
    try:
        if cmd in ("/start", "/help"):
            await say(
                "Tris Bot Active (ATR + OpenRouter)\n\n"
                "/status - state\n/debug - dry run EUR/USD\n/trade - scan & execute\n"
                "/pause - pause\n/resume - resume"
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

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))c
