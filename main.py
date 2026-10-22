import asyncio
import json
import os
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, Request, HTTPException
from google import genai
from telegram import Bot

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"].strip()
OANDA_API_KEY = os.environ["OANDA_API_KEY"]
OANDA_ACCOUNT_ID = os.environ["OANDA_ACCOUNT_ID"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
PUBLIC_URL = os.getenv("PUBLIC_URL")          # e.g. https://your-app.onrender.com
CRON_SECRET = os.getenv("CRON_SECRET", "")    # protects /run-agent-check

OANDA = "https://api-fxpractice.oanda.com/v3"
HEADERS = {"Authorization": f"Bearer {OANDA_API_KEY}", "Content-Type": "application/json"}

bot = Bot(token=TELEGRAM_TOKEN)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
BOT_PAUSED = False


@asynccontextmanager
async def lifespan(app):
  if PUBLIC_URL:
    await bot.set_webhook(url=f"{PUBLIC_URL}/telegram-webhook")
  yield

app = FastAPI(lifespan=lifespan)


async def say(text):
  # Plain text: no Markdown, so Gemini output can't break the message
  await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)


# ---------- OANDA helpers (blocking; run via asyncio.to_thread) ----------
def pip_size(instr): return 0.01 if "JPY" in instr else 0.0001
def decimals(instr): return 3 if "JPY" in instr else 5


def ema(values, period):
  k = 2 / (period + 1)
  e = sum(values[:period]) / period
  for v in values[period:]:
    e = v * k + e * (1 - k)
  return e


def get_market_data(instr):
  r = requests.get(
      f"{OANDA}/instruments/{instr}/candles",
      headers=HEADERS,
      params={"granularity": "H1", "count": 300, "price": "M"},
      timeout=15,
  ).json()
  closes = [float(c["mid"]["c"]) for c in r["candles"] if c["complete"]]
  p = requests.get(
      f"{OANDA}/accounts/{OANDA_ACCOUNT_ID}/pricing",
      headers=HEADERS, params={"instruments": instr}, timeout=15,
  ).json()["prices"][0]
  return {
      "bid": float(p["bids"][0]["price"]),
      "ask": float(p["asks"][0]["price"]),
      "tradeable": p.get("tradeable", False),
      "ema50": ema(closes, 50),
      "ema200": ema(closes, 200),
      "last_closes": closes[-10:],
  }


def open_trade_count(instr):
  r = requests.get(f"{OANDA}/accounts/{OANDA_ACCOUNT_ID}/openTrades",
                   headers=HEADERS, timeout=15).json()
  return sum(1 for t in r.get("trades", []) if t["instrument"] == instr)


def get_account_summary():
  a = requests.get(f"{OANDA}/accounts/{OANDA_ACCOUNT_ID}/summary",
                   headers=HEADERS, timeout=15).json().get("account", {})
  return a.get("balance", "N/A"), a.get("unrealizedPL", "N/A"), a.get("openTradeCount", 0)


def analyze(pair, md):
  prompt = f"""
You are a forex analyst. Using ONLY this live data for {pair}, decide whether
to trade (EMA50/EMA200 trend + recent momentum). Do not invent prices.
Data: {json.dumps(md)}
Return JSON only:
{{"should_trade": bool, "direction": "BUY" or "SELL",
  "sl_pips": number, "tp_pips": number, "reason": "short text"}}
"""
  resp = gemini_client.models.generate_content(
      model="gemini-2.5-flash", contents=prompt,
      config={"response_mime_type": "application/json"},
  )
  data = json.loads(resp.text)
  return data[0] if isinstance(data, list) else data


def place_order(pair, direction, units, sl_pips, tp_pips, md):
  instr = pair.replace("/", "_")
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


def run_cycle(pair="EUR/USD", units=1000):
  """Returns a plain-text report."""
  instr = pair.replace("/", "_")
  md = get_market_data(instr)
  if not md["tradeable"]:
    return "Market is closed / instrument not tradeable right now."
  if open_trade_count(instr) > 0:
    return f"Already have an open {pair} trade; skipping."
  sig = analyze(pair, md)
  if not sig.get("should_trade"):
    return "No high-probability setup found at this time."
  ok, entry, sl, tp, res = place_order(
      pair, sig["direction"], units, float(sig["sl_pips"]), float(sig["tp_pips"]), md)
  if ok:
    return (f"TRADE EXECUTED (OANDA DEMO)\n{pair} {sig['direction']} @ {entry}\n"
            f"SL {sl} | TP {tp}\nReason: {sig['reason']}")
  reject = (res.get("orderRejectTransaction") or res.get("orderCancelTransaction") or {})
  return f"ORDER NOT FILLED\n{reject.get('reason') or res.get('errorMessage') or res}"


# ---------- Routes ----------
@app.get("/")
def home():
  return {"status": "Tris Agent is active!"}


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
  global BOT_PAUSED
  data = await request.json()
  msg = data.get("message")
  if not msg or "text" not in msg:
    return {"status": "ignored"}

  if str(msg["chat"]["id"]) != TELEGRAM_CHAT_ID:
    print(f"Unauthorized chat id: {msg['chat']['id']} (expected {TELEGRAM_CHAT_ID})")
    return {"status": "unauthorized"}

  cmd = msg["text"].strip().split("@")[0].lower()

  try:
    if cmd in ("/start", "/help"):
      await say("Tris Bot Active\n\n/status - balance & bot state\n"
                "/trade - force analysis & trade\n/pause - pause cron trading\n"
                "/resume - resume cron trading")
    elif cmd == "/status":
      bal, pnl, n = await asyncio.to_thread(get_account_summary)
      await say(f"TRIS STATUS\nBot: {'PAUSED' if BOT_PAUSED else 'ACTIVE'}\n"
                f"Balance: {bal}\nUnrealized PnL: {pnl}\nOpen trades: {n}")
    elif cmd == "/pause":
      BOT_PAUSED = True
      await say("Automated trading PAUSED.")
    elif cmd == "/resume":
      BOT_PAUSED = False
      await say("Automated trading RESUMED.")
    elif cmd == "/trade":
      await say("Running analysis...")
      await say(await asyncio.to_thread(run_cycle, "EUR/USD"))
  except Exception as e:
    print("Handler error:", repr(e))
    await say(f"Error: {e!r}")

  return {"status": "ok"}   # always 200 so Telegram doesn't retry


@app.get("/run-agent-check")
async def run_agent_check(key: str = ""):
  if CRON_SECRET and key != CRON_SECRET:
    raise HTTPException(status_code=403)
  if BOT_PAUSED:
    return {"status": "paused"}
  try:
    report = await asyncio.to_thread(run_cycle, "EUR/USD")
  except Exception as e:
    await say(f"Cron error: {e!r}")
    return {"status": "error", "detail": str(e)}
  if "EXECUTED" in report or "NOT FILLED" in report:
    await say(report)
  return {"status": report}


if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
