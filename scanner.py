import os
import time
import json
import requests
import pandas as pd

BINANCE_URL = "https://api.binance.com"

# ============================================================
# TELEGRAM CONFIG
# ============================================================

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

# Example GitHub Secret:
# 123456789,987654321,555666777
TELEGRAM_CHAT_IDS = [
    chat_id.strip()
    for chat_id in os.environ["TELEGRAM_CHAT_IDS"].split(",")
    if chat_id.strip()
]

# Delay between Telegram messages
TELEGRAM_DELAY = 1.2


# ============================================================
# SCANNER CONFIG
# ============================================================

INTERVAL = "15m"
EMA_PERIOD = 200
REQUIRED_CANDLES = 4

STATE_FILE = "alert_state.json"


# ============================================================
# STATE
# ============================================================

def load_state():
    if not os.path.exists(STATE_FILE):
        return {}

    try:
        with open(STATE_FILE, "r") as file:
            return json.load(file)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w") as file:
        json.dump(state, file, indent=2)


# ============================================================
# BINANCE
# ============================================================

def get_usdt_symbols():

    url = f"{BINANCE_URL}/api/v3/exchangeInfo"

    response = requests.get(
        url,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    symbols = []

    for symbol_info in data["symbols"]:

        if (
            symbol_info["status"] == "TRADING"
            and symbol_info["quoteAsset"] == "USDT"
            and symbol_info["isSpotTradingAllowed"]
        ):
            symbols.append(symbol_info["symbol"])

    return symbols


def get_klines(symbol, limit=250):

    url = f"{BINANCE_URL}/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": limit
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# EMA 200 SIGNAL
# ============================================================

def check_signal(symbol):

    candles = get_klines(symbol)

    if len(candles) < EMA_PERIOD + REQUIRED_CANDLES:
        return None

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
            "ignore"
        ]
    )

    df["close"] = pd.to_numeric(df["close"])

    # --------------------------------------------------------
    # Remove current unfinished candle
    # --------------------------------------------------------

    current_time = int(time.time() * 1000)

    df = df[
        df["close_time"] < current_time
    ].copy()

    if len(df) < EMA_PERIOD + REQUIRED_CANDLES:
        return None

    # --------------------------------------------------------
    # EMA 200 using CLOSE price
    # --------------------------------------------------------

    df["ema200"] = df["close"].ewm(
        span=EMA_PERIOD,
        adjust=False
    ).mean()

    # Last 4 CLOSED candles
    last_four = df.iloc[-REQUIRED_CANDLES:]

    # --------------------------------------------------------
    # CONDITION
    #
    # All 4 candle closes must be above
    # their corresponding EMA 200
    # --------------------------------------------------------

    condition = (
        last_four["close"]
        > last_four["ema200"]
    ).all()

    if not condition:
        return None

    last = last_four.iloc[-1]

    trigger_id = str(
        int(last["close_time"])
    )

    return {
        "symbol": symbol,

        "price": float(
            last["close"]
        ),

        "ema200": float(
            last["ema200"]
        ),

        "candles": [
            float(value)
            for value in last_four["close"]
        ],

        "ema_values": [
            float(value)
            for value in last_four["ema200"]
        ],

        "trigger_id": trigger_id
    }


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def create_message(signal):

    symbol = signal["symbol"]

    candles = signal["candles"]
    ema_values = signal["ema_values"]

    price = signal["price"]
    ema200 = signal["ema200"]

    message = (
        "🚨 EMA 200 ALERT\n"
        "━━━━━━━━━━━━━━━━━━\n\n"

        f"🪙 Symbol: {symbol}\n"
        "⏱ Timeframe: 15M\n\n"

        "✅ 4 CLOSED CANDLES ABOVE EMA 200\n\n"

        f"1️⃣ Close: {candles[0]:.8f}\n"
        f"   EMA:   {ema_values[0]:.8f}\n\n"

        f"2️⃣ Close: {candles[1]:.8f}\n"
        f"   EMA:   {ema_values[1]:.8f}\n\n"

        f"3️⃣ Close: {candles[2]:.8f}\n"
        f"   EMA:   {ema_values[2]:.8f}\n\n"

        f"4️⃣ Close: {candles[3]:.8f}\n"
        f"   EMA:   {ema_values[3]:.8f}\n\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"📊 EMA 200: {ema200:.8f}\n"
        f"💰 Last Close: {price:.8f}\n"

        "━━━━━━━━━━━━━━━━━━"
    )

    return message


# ============================================================
# SEND MESSAGE TO ONE CHAT
# ============================================================

def send_to_chat(chat_id, message):

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": chat_id,
        "text": message
    }

    response = requests.post(
        url,
        json=payload,
        timeout=20
    )

    response.raise_for_status()


# ============================================================
# SEND TO ALL CHAT IDS
# ============================================================

def send_telegram_to_all(signal):

    message = create_message(signal)

    for index, chat_id in enumerate(
        TELEGRAM_CHAT_IDS,
        start=1
    ):

        try:

            send_to_chat(
                chat_id,
                message
            )

            print(
                f"📨 {signal['symbol']} "
                f"sent to chat {index}/"
                f"{len(TELEGRAM_CHAT_IDS)}"
            )

        except Exception as error:

            print(
                f"❌ Telegram failed for "
                f"chat {chat_id}: {error}"
            )

        # ----------------------------------------------------
        # 1.2 second delay between chat messages
        # ----------------------------------------------------

        if index < len(TELEGRAM_CHAT_IDS):
            time.sleep(TELEGRAM_DELAY)


# ============================================================
# MAIN
# ============================================================

def main():

    print("==========================================")
    print(" Binance EMA 200 Telegram Scanner")
    print("==========================================")
    print(f"Timeframe       : {INTERVAL}")
    print(f"EMA             : {EMA_PERIOD}")
    print(f"Required candles: {REQUIRED_CANDLES}")
    print(
        f"Telegram chats  : "
        f"{len(TELEGRAM_CHAT_IDS)}"
    )
    print("==========================================")

    state = load_state()

    symbols = get_usdt_symbols()

    print(
        f"Found {len(symbols)} USDT pairs"
    )

    new_signals = []

    # ========================================================
    # SCAN ALL SYMBOLS
    # ========================================================

    for index, symbol in enumerate(
        symbols,
        start=1
    ):

        try:

            signal = check_signal(symbol)

            if signal:

                trigger_id = signal["trigger_id"]

                previous_trigger = state.get(
                    symbol
                )

                # --------------------------------------------
                # Duplicate protection
                # --------------------------------------------

                if previous_trigger == trigger_id:

                    print(
                        f"[{index}/{len(symbols)}] "
                        f"{symbol} - already alerted"
                    )

                    continue

                new_signals.append(signal)

                print(
                    f"[{index}/{len(symbols)}] "
                    f"🚨 NEW SIGNAL: {symbol}"
                )

            else:

                # Condition broken → reset
                if symbol in state:
                    del state[symbol]

                print(
                    f"[{index}/{len(symbols)}] "
                    f"{symbol} - no signal"
                )

        except Exception as error:

            print(
                f"{symbol} - ERROR: {error}"
            )

        # Small Binance API delay
        time.sleep(0.05)

    # ========================================================
    # SEND NEW SIGNALS
    # ========================================================

    for signal in new_signals:

        try:

            send_telegram_to_all(
                signal
            )

            # Save trigger only after
            # Telegram sending is attempted
            state[
                signal["symbol"]
            ] = signal["trigger_id"]

            save_state(state)

        except Exception as error:

            print(
                f"Signal send error "
                f"for {signal['symbol']}: {error}"
            )

    # Always save state
    save_state(state)

    print("==========================================")
    print("Scan completed")
    print(
        f"New signals: "
        f"{len(new_signals)}"
    )
    print("==========================================")


if __name__ == "__main__":
    main()
