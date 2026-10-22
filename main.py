import json
import os
from fastapi import FastAPI
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
# Initialize Google Gemini Client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
app = FastAPI()


@app.get("/")
def home():
  return {"status": "Ordain FX Core Agent (Gemini Powered) is live 24/7!"}


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


@app.get("/run-agent-check")
async def run_agent_analysis():
  """Triggered by UptimeRobot or cron to let the Gemini AI agent analyze and trade"""

  prompt = """
    You are Ordain FX Core, an elite Forex trading agent. Combine Ref Wayne's momentum logic with Dapo Willis's risk rules.
    Analyze the current market state for EUR/USD. 
    Provide a calculated high-confluence trade setup based on EMA 50/200 and institutional order blocks.
    You must output your response strictly as a JSON object with the following fields:
    {
      "should_trade": true,
      "pair": "EUR/USD",
      "direction": "BUY",
      "entry": "1.0850",
      "sl": "1.0820",
      "tp": "1.0910",
      "units": 1000,
      "reason": "Ref Wayne Trend Crossover + Dapo Willis Support Bounce"
    }
    """

  # Request JSON-formatted evaluation from Gemini
  response = gemini_client.models.generate_content(
      model="gemini-2.5-flash",
      contents=prompt,
      config={"response_mime_type": "application/json"},
  )

  result = json.loads(response.text)

  if result.get("should_trade"):
    # Execute on OANDA Demo
    oanda_res = execute_oanda_trade(
        result["pair"],
        result["direction"],
        result["units"],
        result["sl"],
        result["tp"],
    )

    # Send Telegram Alert
    msg = (
        f"🚨 ORDAIN FX SIGNAL DETECTED (GEMINI) 🚨\n"
        f"-----------------------------------\n"
        f"Pair: {result['pair']}\n"
        f"Direction: {result['direction']}\n"
        f"Entry: {result['entry']}\n"
        f"Stop Loss: {result['sl']}\n"
        f"Take Profit: {result['tp']}\n"
        f"Strategy: {result['reason']}\n"
        f"Status: EXECUTED ON OANDA DEMO"
    )
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
    return {"status": "Trade executed", "details": result, "oanda": oanda_res}

  return {"status": "No high-probability setup found at this time."}


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8080)
