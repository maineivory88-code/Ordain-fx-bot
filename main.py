import os
import requests
from fastapi import FastAPI
from telegram import Bot

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")

bot = Bot(token=TELEGRAM_TOKEN)
app = FastAPI()

@app.get("/")
def home():
    return {"status": "Ordain FX Bot is active and running!"}

def execute_oanda_trade(pair, direction, units, stop_loss, take_profit):
    url = f"https://api-fxpractice.oanda.com/v3/accounts/{OANDA_ACCOUNT_ID}/orders"
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Content-Type": "application/json"
    }
    order_data = {
        "order": {
            "units": str(units if direction == "BUY" else -units),
            "instrument": pair.replace("/", "_"),
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": str(stop_loss)},
            "takeProfitOnFill": {"price": str(take_profit)}
        }
    }
    response = requests.post(url, json=order_data, headers=headers)
    return response.json()

@app.post("/ordain-webhook")
async def receive_signal_from_ordain(data: dict):
    pair = data.get("pair")
    direction = data.get("direction")
    entry = data.get("entry")
    sl = data.get("sl")
    tp = data.get("tp")
    units = data.get("units", 1000)

    # Execute on OANDA Demo
    oanda_res = execute_oanda_trade(pair, direction, units, sl, tp)

    # Format Telegram Notification
    msg = (
        f"🚨 ORDAIN FX SIGNAL DETECTED 🚨\n"
        f"-----------------------------------\n"
        f"Pair: {pair}\n"
        f"Direction: {direction}\n"
        f"Entry: {entry}\n"
        f"Stop Loss: {sl}\n"
        f"Take Profit: {tp}\n"
        f"Strategy: Ref Wayne Trend + Dapo Willis Zone\n"
        f"Status: EXECUTED ON OANDA DEMO"
    )
    
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
    return {"status": "success", "oanda_response": oanda_res}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
