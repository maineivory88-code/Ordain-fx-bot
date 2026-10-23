import asyncio
import json
import os
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI
from telegram import Bot

# ---------- Environment Variables ----------
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"].strip()
OANDA_API_KEY = os.environ["OANDA_API_KEY"]
OANDA_ACCOUNT_ID = os.environ["OANDA_ACCOUNT_ID"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
PUBLIC_URL = os.getenv("PUBLIC_URL", "").strip().rstrip("/")
CRON_SECRET = os.getenv("CRON_SECRET", "")

OANDA = "https://api-fxpractice.oanda.com/v3"
HEADERS = {"Authorization": f"Bearer {OANDA_API_KEY}", "Content-Type": "application/json"}

bot = Bot(token=TELEGRAM_TOKEN)

# Initialize OpenRouter Client via OpenAI compatibility layer
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
BOT_PAUSED = False


@asynccontextmanager
async def lifespan(app):
  if PUBLIC_URL:
    await bot.set_webhook(url=f"{PUBLIC_URL}/telegram-webhook")
  yield

app = FastAPI(lifespan=lifespan)


async def say(text):
  # Send plain text so special characters in AI or market output never break Telegram parsing
  await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)


# ---------- OANDA API & ATR Helpers ----------
def pip_size(instr): return 0.01 if "JPY" in instr else 0.0001
def decimals(instr): return 3 if "JPY" in instr else 5


def ema(values, period):
  k = 2 / (period + 1)
  e = sum(values[:period]) / period
  for v in values[period:]:
    e = v * k + e * (1 - k)
  return e


def calculate_atr(candles, period=14):
  """Calculates Average True Range (ATR) to filter out low-volatility chop."""
  if len(candles) < period + 1:
    return 0.0010
  tr_list = []
  for i in range(1, len(candles)):
    high = float(candles[i]["mid"]["h"])
    low = float(candles[i]["mid"]["l"])
    prev_close = float(candles[i - 1]["mid"]["c"])
    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
    tr_list.append(tr)
  return sum(tr_list[-period:]) / period


def get_all_tradeable_instruments():
  r = requests.get(
      f"{OANDA}/accounts/{OANDA_ACCOUNT_ID}/instruments",
      headers=HEADERS,
      timeout=15,
  ).json()
  instruments = r.get("instruments", [])
  return [i["name"] for i in instruments if i.get("type") == "CURRENCY"]


def get_market_data(instr):
  r = requests.get(
      f"{OANDA}/instruments/{instr}/candles",
      headers=HEADERS,
      params={"granularity": "H1", "count": 300, "price": "M"},
      timeout=15,
  ).json()
  candles = r.get("candles", [])
  completed_candles = [c for c in candles if c.get("complete", False)]
  if len(completed_candles) < 50:
    return None

  closes = [float(c["mid"]["c"]) for c in completed_candles]
  atr = calculate_atr(completed_candles)

  p = requests.get(
      f"{OANDA}/accounts/{OANDA_ACCOUNT_ID}/pricing",
      headers=HEADERS, params={"instruments": instr}, timeout=15,
  ).json().get("prices", [{}])[0]

  return {
      "bid": float(p.get("bids", [{}])[0].get("price", 0)),
      "ask": float(p.get("asks", [{}])[0].get("price", 0)),
      "tradeable": p.get("tradeable", False),
      "ema50": ema(closes, 50),
      "ema200": ema(closes, 200) if len(closes) >= 200 else ema(closes, len(closes)),
      "last_closes": closes[-10:],
      "atr": atr,
  }


def open_trade_count(instr):
  r = requests.get(f"{OANDA}/accounts/{OANDA_ACCOUNT_ID}/openTrades",
                   headers=HEADERS, timeout=15).json()
  return sum(1 for t in r.get("trades", []) if t["instrument"] == instr)


def get_account_summary():
  a = requests.get(f"{OANDA}/accounts/{OANDA_ACCOUNT_ID}/summary",
                   headers=HEADERS, timeout=15).json().get("account", {})
  return a.get("balance", "N/A"), a.get("unrealizedPL", "N/A"), a.get("openTradeCount", 0)


# ---------- Loosened Rule Engine + ATR Volatility Filter ----------
def evaluate_rules(md):
  ema50 = md["ema50"]
  last_close = md["last_closes"][-1]
  prev_close = md["last_closes"][-2]
  three_bars_ago = md["last_closes"][-4]
  atr = md.get("atr", 0.0010)

  momentum = last_close - three_bars_ago
  min_distance = 0.5 * atr

  if (last_close > ema50 + min_distance) or (last_close > prev_close and momentum > min_distance):
    return {
        "should_trade": True,
        "direction": "BUY",
        "sl_pips": 15.0,
        "tp_pips": 30.0,
        "reason": f"Bullish expansion: Distance > 0.5 ATR ({atr:.5f})"
    }
  elif (last_close < ema50 - min_distance) or (last_close < prev_close and momentum < -min_distance):
    return {
        "should_trade": True,
        "direction": "SELL",
        "sl_pips": 15.0,
        "tp_pips": 30.0,
        "reason": f"Bearish expansion: Distance > 0.5 ATR ({atr:.5f})"
    }

  return {"should_trade": False, "reason": "Market in low-volatility squeeze (below ATR threshold)."}


# ---------- OpenRouter AI Integration ----------
def get_openrouter_opinion(pair, md, rule_signal):
  try:
    prompt = f"""
You are a forex risk analyst reviewing a rule-based trade signal for {pair}.
Rule Signal: {json.dumps(rule_signal)}
Market Data: {json.dumps(md)}
Provide a brief second opinion. Return JSON only in this exact format:
{{"agree": true/false, "comment": "short analysis or warning"}}
"""
    model_choice = os.getenv("OPENROUTER_MODEL", "openrouter/free")
    
    response = openrouter_client.chat.completions.create(
        model=model_choice,
        messages=[
            {"role": "system", "content": "You are a financial JSON generator."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        timeout=10
    )
    data = json.loads(response.choices[0].message.content)
    return data
  except Exception as e:
    print(f"OpenRouter second opinion skipped for {pair}: {e!r}")
    return None


def place_order(instr, direction, units, sl_pips, tp_pips, md):
  d, pip = decimals(instr), pip_size(instr)
  entry = md["ask"] if direction == "BUY" else md["bid"]
  sign = 1 if direction == "BUY" else -1
  sl = round(entry - sign * sl_pips * pip, d)
  tp = round(entry + sign * tp_pips * pip, d)
  body = {"order": {
      "units": str(units * sign),
      "instrument": instr,
      "timeInForce": "FOK",
      "type": "MARKET",
      "positionFill": "DEFAULT",
      "stopLossOnFill": {"price": f"{sl:.{d}f}"},
      "takeProfitOnFill": {"price": f"{tp:.{d}f}"},
  }}
  res = requests.post(f"{OANDA}/accounts/{OANDA_ACCOUNT_ID}/orders",
                      headers=HEADERS, json=body, timeout=15).json()
  filled = "orderFillTransaction" in res
  return filled, entry, sl, tp, res


def run_cycle_all_pairs(units=1000):
  instruments = get_all_tradeable_instruments()
  reports = []
  executed_count = 0

  for instr in instruments:
    pair_display = instr.replace("_", "/")
    
    if open_trade_count(instr) > 0:
      continue

    md = get_market_data(instr)
    if not md or not md["tradeable"]:
      continue

    sig = evaluate_rules(md)
    if not sig.get("should_trade"):
      continue

    opinion = get_openrouter_opinion(pair_display, md, sig)
    ai_note = "OpenRouter AI: Offline / Skipped"
    if opinion:
      agree = opinion.get("agree", True)
      comment = opinion.get("comment", "N/A")
      ai_note = f"OpenRouter AI: {'Agree' if agree else 'Caution'} - {comment}"

    ok, entry, sl, tp, res = place_order(
        instr, sig["direction"], units, float(sig["sl_pips"]), float(sig["tp_pips"]), md)
    
    if ok:
      executed_count += 1
      reports.append(
          f"TRADE EXECUTED ({pair_display})\n"
          f"Direction: {sig['direction']} @ {entry}\n"
          f"SL {sl} | TP {tp}\nReason: {sig['reason']}\n{ai_note}"
      )
    else:
      reject = (res.get("orderRejectTransaction") or res.get("orderCancelTransaction") or {})
      reports.append(f"ORDER FAILED ({pair_display})\nReason: {reject.get('reason') or res}")

  if not reports:
    return "Scan complete. No eligible trade setups across currency pairs right now."

  return f"--- SCAN COMPLETED ({executed_count} trades opened) ---\n\n" + "\n\n".join(reports)


# ---------- FastAPI Routes & Telegram Commands ----------
@app.get("/")
def home():
  return {"status": "Tris OpenRouter ATR Agent is active!"}


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
  global BOT_PAUSED
  data = await request.json()
  msg = data.get("message")
  if not msg or "text" not in msg:
    return {"status": "ignored"}

  if str(msg["chat"]["id"]) != TELEGRAM_CHAT_ID:
    return {"status": "unauthorized"}

  cmd = msg["text"].strip().split("@")[0].lower()

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
      bal, pnl, n = await asyncio.to_thread(get_account_summary)
      await say(
          f"TRIS STATUS\n"
          f"Bot State: {'PAUSED' if BOT_PAUSED else 'ACTIVE'}\n"
          f"Balance: {bal}\n"
          f"Unrealized PnL: {pnl}\n"
          f"Open Trades: {n}"
      )
    elif cmd == "/debug":
      await say("Running debug test on EUR/USD...")
      md = await asyncio.to_thread(get_market_data, "EUR_USD")
      if not md:
        await say("Debug Error: Unable to fetch EUR/USD market data.")
      else:
        sig = evaluate_rules(md)
        opinion = await asyncio.to_thread(get_openrouter_opinion, "EUR/USD", md, sig)
        debug_msg = (
            f"DEBUG REPORT (EUR/USD)\n"
            f"Last Close: {md['last_closes'][-1]:.5f}\n"
            f"EMA50: {md['ema50']:.5f}\n"
            f"EMA200: {md['ema200']:.5f}\n"
            f"ATR(14): {md['atr']:.5f}\n"
            f"Rule Result: {json.dumps(sig, indent=2)}\n"
            f"OpenRouter AI: {json.dumps(opinion) if opinion else 'Offline/Skipped'}"
        )
        await say(debug_msg)
    elif cmd == "/pause":
      BOT_PAUSED = True
      await say("Automated trading PAUSED.")
    elif cmd == "/resume":
      BOT_PAUSED = False
      await say("Automated trading RESUMED.")
    elif cmd == "/trade":
      await say("Scanning all available OANDA currency pairs with ATR filter...")
      report = await asyncio.to_thread(run_cycle_all_pairs)
      await say(report)
    else:
      await say(f"Unknown command: {cmd}\nType /help for available commands.")
  except Exception as e:
    print("Handler error:", repr(e))
    await say(f"Error handling command: {e!r}")

  return {"status": "ok"}   # Always return 200 OK so Telegram doesn't retry


@app.get("/run-agent-check")
async def run_agent_check(key: str = ""):
  if CRON_SECRET and key != CRON_SECRET:
    raise HTTPException(status_code=403)
  if BOT_PAUSED:
    return {"status": "paused"}
  try:
    report = await asyncio.to_thread(run_cycle_all_pairs)
  except Exception as e:
    await say(f"Cron error: {e!r}")
    return {"status": "error", "detail": str(e)}

  if "EXECUTED" in report:
    await say(report)
  return {"status": "scan complete", "detail": report}


if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
