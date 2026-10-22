import json
import os
from fastapi import FastAPI, Request
from google import genai
import requests
from telegram import Bot

# Environment Variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

bot = Bot(token=TELEGRAM_TOKEN)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
app = FastAPI()

# Global state flag for pausing/resuming trading
BOT_PAUSED = False


def execute_oanda_trade(pair, direction, units, stop_loss, take_profit):
  """Executes trade directly on OANDA v20 Demo Account"""
  url = f"https://api-fxpractice.oanda.com/v3/accounts/{OANDA_ACCOUNT_ID}/orders"
  headers = {
      "Authorization": f"Bearer {OANDA_API_KEY}",
      "Content-Type": "application/json",
  }
  order_data = {
      "order": {
          "units": str(units if direction == "BUY" else -units),
          "instrument": pair.replace("/", "_"),
          "timeInForce": "FOK",
          "type": "MARKET",
          "positionFill": "DEFAULT",
          "stopLossOnFill": {"price": str(stop_loss)},
          "takeProfitOnFill": {"price": str(take_profit)},
      }
  }
  response = requests.post(url, json=order_data, headers=headers)
  return response.json()


def get_oanda_account_summary():
  """Fetches live balance and open position summary from OANDA"""
  url = f"https://api-fxpractice.oanda.com/v3/accounts/{OANDA_ACCOUNT_ID}/summary"
  headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
  res = requests.get(url, headers=headers).json()
  account = res.get("account", {})
  return {
      "balance": account.get("balance", "N/A"),
      "unrealized_pnl": account.get("unrealizedPL", "N/A"),
      "open_trades": account.get("openTradeCount", 0),
  }


def analyze_and_execute(pair="EUR/USD"):
  """Core Ref Wayne + Dapo Willis strategy engine powered by Gemini"""
  prompt = f"""
    You are Tris, an elite Forex trading agent. Combine Ref Wayne's momentum logic with Dapo Willis's risk rules.
    Analyze current setup parameters for {pair}.
    Provide a calculated high-confluence trade setup based on EMA 50/200 and institutional order blocks.
    Return strictly a JSON object formatted like:
    {{
      "should_trade": true,
      "pair": "{pair}",
      "direction": "BUY",
      "entry": "1.0850",
      "sl": "1.0820",
      "tp": "1.0910",
      "units": 1000,
      "reason": "Ref Wayne Trend Crossover + Dapo Willis Support Bounce"
    }}
    """
  response = gemini_client.models.generate_content(
      model="gemini-2.5-flash",
      contents=prompt,
      config={"response_mime_type": "application/json"},
  )
  return json.loads(response.text)


@app.get("/")
def home():
  return {"status": "Tris Agent is active!"}


# --- TELEGRAM WEBHOOK HANDLER ---
@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
  global BOT_PAUSED
  data = await request.json()

  if "message" not in data or "text" not in data["message"]:
    return {"status": "ignored"}

  chat_id = str(data["message"]["chat"]["id"])
  text = data["message"]["text"].strip()

  # Security check: only respond to your personal Chat ID
  if chat_id != TELEGRAM_CHAT_ID:
    return {"status": "unauthorized"}

  # Command 1: /start or /help
  if text in ["/start", "/help"]:
    msg = (
        "*Tris Bot Active*\n\n"
        "Available Commands:\n"
        "• `/status` - Check OANDA balance & bot state\n"
        "• `/trade` - Force an instant market analysis & trade\n"
        "• `/pause` - Pause automated trading checks\n"
        "• `/resume` - Resume automated trading checks\n"
    )
    await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")

  # Command 2: /status
  elif text == "/status":
    summary = get_oanda_account_summary()
    status_str = "PAUSED" if BOT_PAUSED else "ACTIVE"
    msg = (
        f"*TRIS ACCOUNT STATUS*\n"
        f"-----------------------------\n"
        f"Bot State: {status_str}\n"
        f"Balance: ${summary['balance']}\n"
        f"Unrealized PnL: ${summary['unrealized_pnl']}\n"
        f"Open Trades: {summary['open_trades']}"
    )
    await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")

  # Command 3: /pause
  elif text == "/pause":
    BOT_PAUSED = True
    await bot.send_message(
        chat_id=chat_id, text="*Automated trading has been PAUSED.*"
    )

  # Command 4: /resume
  elif text == "/resume":
    BOT_PAUSED = False
    await bot.send_message(
        chat_id=chat_id, text="*Automated trading RESUMED.*"
    )

  # Command 5: /trade
  elif text == "/trade":
    await bot.send_message(
        chat_id=chat_id, text="*Running Ref Wayne & Dapo Willis analysis...*"
    )
    result = analyze_and_execute("EUR/USD")

    if result.get("should_trade"):
      oanda_res = execute_oanda_trade(
          result["pair"],
          result["direction"],
          result["units"],
          result["sl"],
          result["tp"],
      )
      msg = (
          f"*TRIS SIGNAL EXECUTED*\n"
          f"-----------------------------------\n"
          f"Pair: {result['pair']}\n"
          f"Direction: {result['direction']}\n"
          f"Entry: {result['entry']}\n"
          f"SL: {result['sl']} | TP: {result['tp']}\n"
          f"Strategy: {result['reason']}\n"
          f"Status: ORDER PLACED ON OANDA DEMO"
      )
    else:
      msg = "No high-probability setup found at this time."

    await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")

  return {"status": "ok"}


# --- AUTOMATED CRON ENDPOINT ---
@app.get("/run-agent-check")
async def run_agent_analysis():
  global BOT_PAUSED
  if BOT_PAUSED:
    return {"status": "Bot is currently paused."}

  result = analyze_and_execute("EUR/USD")
  if result.get("should_trade"):
    execute_oanda_trade(
        result["pair"],
        result["direction"],
        result["units"],
        result["sl"],
        result["tp"],
    )
    msg = (
        f"*TRIS SIGNAL DETECTED*\n"
        f"-----------------------------------\n"
        f"Pair: {result['pair']} | Dir: {result['direction']}\n"
        f"SL: {result['sl']} | TP: {result['tp']}\n"
        f"Strategy: {result['reason']}\n"
        f"Status: EXECUTED ON OANDA DEMO"
    )
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown"
    )
    return {"status": "Trade executed", "details": result}

  return {"status": "No trade setup found."}


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8080)
