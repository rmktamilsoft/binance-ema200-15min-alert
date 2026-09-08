import os
import time
import requests
import pandas as pd

BINANCE_URL = "https://api.binance.com"

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

INTERVAL = "15m"
EMA_PERIOD = 200
REQUIRED_CANDLES = 4

# --------------------------------------------------
# Binance API
# --------------------------------------------------

def get_usdt_symbols():
    url = f"{BINANCE_URL}/api/v3/exchangeInfo"
    data = requests.get(url, timeout=20).json()

    symbols = []

    for s in data["symbols"]:
        if (
            s["status"] == "TRADING"
            and s["quoteAsset"] == "USDT"
            and s["isSpotTradingAllowed"]
        ):
            symbols.append(s["symbol"])

    return symbols


def get_klines(symbol, limit=250):
    url = f"{BINANCE_URL}/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": limit,
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    return response.json()


# --------------------------------------------------
# EMA calculation
# --------------------------------------------------

def check_signal(symbol):
    candles = get_klines(symbol)

    if len(candles) < EMA_PERIOD + REQUIRED_CANDLES:
        return None

    # Binance kline format:
    # [
    #   open_time,
    #   open,
    #   high,
    #   low,
    #   close,
    #   volume,
    #   close_time,
    #   ...
    # ]

    df = pd.DataFrame(
        candles,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_base",
            "taker_quote",
            "ignore",
        ],
    )

    df["close"] = pd.to_numeric(df["close"])

    # ----------------------------------------------
    # Remove currently forming candle
    # ----------------------------------------------

    current_time = int(time.time() * 1000)

    df = df[df["close_time"] < current_time].copy()

    if len(df) < EMA_PERIOD + REQUIRED_CANDLES:
        return None

    # ----------------------------------------------
    # EMA 200 using CLOSE price
    # ----------------------------------------------

    df["ema200"] = df["close"].ewm(
        span=EMA_PERIOD,
        adjust=False
    ).mean()

    # Last 4 CLOSED candles
    last_four = df.iloc[-REQUIRED_CANDLES:]

    # Every close must be above its corresponding EMA200
    condition = (last_four["close"] > last_four["ema200"]).all()

    if not condition:
        return None

    last = last_four.iloc[-1]

    return {
        "symbol": symbol,
        "price": float(last["close"]),
        "ema200": float(last["ema200"]),
        "candles": [
            float(x) for x in last_four["close"].tolist()
        ],
        "candle_time": int(last["close_time"]),
    }


# --------------------------------------------------
# Telegram
# --------------------------------------------------

def send_telegram(signal):
    symbol = signal["symbol"]

    candles = signal["candles"]
    price = signal["price"]
    ema200 = signal["ema200"]

    message = (
        "🚨 EMA 200 SIGNAL\n\n"
        f"Symbol: {symbol}\n"
        f"Timeframe: {INTERVAL}\n\n"
        "4 Consecutive CLOSED Candles Above EMA 200 ✅\n\n"
        f"Candle 1: {candles[0]:.8f}\n"
        f"Candle 2: {candles[1]:.8f}\n"
        f"Candle 3: {candles[2]:.8f}\n"
        f"Candle 4: {candles[3]:.8f}\n\n"
        f"EMA 200: {ema200:.8f}\n"
        f"Last Close: {price:.8f}"
    )

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
    }

    response = requests.post(
        url,
        json=payload,
        timeout=20,
    )

    response.raise_for_status()


# --------------------------------------------------
# Main scanner
# --------------------------------------------------

def main():
    print("Starting Binance EMA200 scanner...")

    symbols = get_usdt_symbols()

    print(f"Found {len(symbols)} USDT pairs")

    signals = []

    for index, symbol in enumerate(symbols, start=1):

        try:
            signal = check_signal(symbol)

            if signal:
                signals.append(signal)

                print(
                    f"🚨 SIGNAL: {symbol} "
                    f"Price={signal['price']} "
                    f"EMA200={signal['ema200']}"
                )

            else:
                print(
                    f"[{index}/{len(symbols)}] "
                    f"{symbol} - no signal"
                )

        except Exception as e:
            print(f"{symbol} - ERROR: {e}")

        # Small delay to reduce API pressure
        time.sleep(0.05)

    # Send alerts
    for signal in signals:
        try:
            send_telegram(signal)
            print(f"Telegram sent: {signal['symbol']}")
        except Exception as e:
            print(
                f"Telegram error for "
                f"{signal['symbol']}: {e}"
            )

    print("Scan completed.")


if __name__ == "__main__":
    main()
