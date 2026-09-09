import os
import time
import json
import requests
import pandas as pd

BINANCE_URL = "https://data-api.binance.vision"

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

TELEGRAM_CHAT_IDS = [
    chat_id.strip()
    for chat_id in os.environ["TELEGRAM_CHAT_IDS"].split(",")
    if chat_id.strip()
]

TELEGRAM_DELAY = 1.2

# =========================
# STRATEGY SETTINGS
# =========================

INTERVAL = "15m"

EMA_PERIOD = 200

REQUIRED_CANDLES = 4

MAX_PRICE = 5.0

STRUCTURE_LENGTH = 5

CHOCH_LOOKBACK_CANDLES = 32   # 8 hours = 32 x 15m

STATE_FILE = "alert_state.json"


# =========================
# STATE
# =========================

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


# =========================
# BINANCE
# =========================

def get_usdt_symbols():
    url = f"{BINANCE_URL}/api/v3/exchangeInfo"

    response = requests.get(url, timeout=20)
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


# =========================
# DATAFRAME
# =========================

def create_dataframe(candles):

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

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column])

    # Remove currently forming candle
    current_time = int(time.time() * 1000)

    df = df[
        df["close_time"] < current_time
    ].copy()

    df.reset_index(drop=True, inplace=True)

    return df


# =========================
# EMA
# =========================

def calculate_ema(df):

    df["ema200"] = (
        df["close"]
        .ewm(
            span=EMA_PERIOD,
            adjust=False
        )
        .mean()
    )

    return df


# =========================
# STRUCTURE DETECTION
# =========================

def find_pivot_high(df, index, length):
    """
    A pivot high is confirmed when the candle 'length'
    candles ago is higher than the previous 'length' highs.

    This follows the structure logic used by the reference
    indicator where:
        high[size] > highest(size)
    """

    pivot_index = index - length

    if pivot_index < length:
        return None

    if pivot_index >= len(df):
        return None

    pivot_high = df.iloc[pivot_index]["high"]

    left_highs = df.iloc[
        pivot_index - length:pivot_index
    ]["high"]

    right_highs = df.iloc[
        pivot_index + 1:index + 1
    ]["high"]

    if len(left_highs) < length:
        return None

    if len(right_highs) < length:
        return None

    if pivot_high > left_highs.max() and pivot_high > right_highs.max():

        return {
            "index": pivot_index,
            "price": float(pivot_high)
        }

    return None


def find_pivot_low(df, index, length):

    pivot_index = index - length

    if pivot_index < length:
        return None

    if pivot_index >= len(df):
        return None

    pivot_low = df.iloc[pivot_index]["low"]

    left_lows = df.iloc[
        pivot_index - length:pivot_index
    ]["low"]

    right_lows = df.iloc[
        pivot_index + 1:index + 1
    ]["low"]

    if len(left_lows) < length:
        return None

    if len(right_lows) < length:
        return None

    if pivot_low < left_lows.min() and pivot_low < right_lows.min():

        return {
            "index": pivot_index,
            "price": float(pivot_low)
        }

    return None


# =========================
# BULLISH CHOCH
# =========================

def detect_bullish_choch(df):

    """
    Detect bullish structure change.

    Logic:

    1. Maintain bearish/bullish structure state.
    2. When price closes below the active pivot low:
       structure becomes bearish.
    3. While bearish, if price CLOSE crosses above
       the active pivot high:
       bullish CHOCH is detected.
    4. Return the CHOCH candle index.

    Only closed candles are used.
    """

    if len(df) < 100:
        return None

    trend = None

    active_high = None
    active_low = None

    choch_index = None

    previous_close = None

    for i in range(
        STRUCTURE_LENGTH * 2,
        len(df)
    ):

        # ---------------------------------
        # Update confirmed pivot high
        # ---------------------------------

        pivot_high = find_pivot_high(
            df,
            i,
            STRUCTURE_LENGTH
        )

        if pivot_high is not None:

            active_high = pivot_high

        # ---------------------------------
        # Update confirmed pivot low
        # ---------------------------------

        pivot_low = find_pivot_low(
            df,
            i,
            STRUCTURE_LENGTH
        )

        if pivot_low is not None:

            active_low = pivot_low

        current_close = float(
            df.iloc[i]["close"]
        )

        # ---------------------------------
        # Bearish structure break
        # ---------------------------------

        if (
            active_low is not None
            and previous_close is not None
            and previous_close >= active_low["price"]
            and current_close < active_low["price"]
        ):

            trend = "BEARISH"

        # ---------------------------------
        # Bullish CHOCH
        # ---------------------------------

        if (
            trend == "BEARISH"
            and active_high is not None
            and previous_close is not None
            and previous_close <= active_high["price"]
            and current_close > active_high["price"]
        ):

            choch_index = i

            trend = "BULLISH"

            # Keep searching so the latest valid CHOCH
            # inside the lookback window can be used.

        previous_close = current_close

    return choch_index


# =========================
# SIGNAL CHECK
# =========================

def check_signal(symbol):

    candles = get_klines(
        symbol,
        limit=250
    )

    if len(candles) < EMA_PERIOD + 50:
        return None

    df = create_dataframe(candles)

    if len(df) < EMA_PERIOD + CHOCH_LOOKBACK_CANDLES:
        return None

    df = calculate_ema(df)

    # =================================
    # PRICE FILTER
    # =================================

    last_close = float(
        df.iloc[-1]["close"]
    )

    if last_close > MAX_PRICE:
        return None

    # =================================
    # LAST 4 CLOSED CANDLES
    # ABOVE EMA200
    # =================================

    last_four = df.iloc[
        -REQUIRED_CANDLES:
    ].copy()

    if len(last_four) != REQUIRED_CANDLES:
        return None

    above_ema = (
        last_four["close"]
        > last_four["ema200"]
    ).all()

    if not above_ema:
        return None

    # =================================
    # CHOCH LOOKBACK
    #
    # Previous 8 hours
    # =================================

    choch_search_start = max(
        EMA_PERIOD,
        len(df) - CHOCH_LOOKBACK_CANDLES - REQUIRED_CANDLES
    )

    choch_df = df.iloc[
        :len(df) - REQUIRED_CANDLES
    ].copy()

    # Only search recent 8-hour area
    if len(choch_df) > CHOCH_LOOKBACK_CANDLES + 100:
        choch_df = choch_df.iloc[
            -(
                CHOCH_LOOKBACK_CANDLES + 100
            ):
        ].copy()

    choch_df.reset_index(
        drop=True,
        inplace=True
    )

    choch_index = detect_bullish_choch(
        choch_df
    )

    if choch_index is None:
        return None

    choch_candle = choch_df.iloc[
        choch_index
    ]

    choch_close = float(
        choch_candle["close"]
    )

    choch_ema = float(
        choch_candle["ema200"]
    )

    # =================================
    # CHOCH MUST BE BELOW EMA200
    # =================================

    if choch_close >= choch_ema:
        return None

    # =================================
    # CHOCH MUST HAPPEN BEFORE
    # THE 4-CANDLE EMA RECLAIM
    # =================================

    last_four_start_time = int(
        last_four.iloc[0]["open_time"]
    )

    choch_time = int(
        choch_candle["open_time"]
    )

    if choch_time >= last_four_start_time:
        return None

    # =================================
    # SIGNAL
    # =================================

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

        "choch_close": choch_close,

        "choch_ema": choch_ema,

        "trigger_id": trigger_id
    }


# =========================
# TELEGRAM MESSAGE
# =========================

def format_price(value):

    if value >= 1:
        return f"{value:.4f}"

    if value >= 0.01:
        return f"{value:.6f}"

    if value >= 0.0001:
        return f"{value:.8f}"

    return f"{value:.12f}"


def create_message(signal):

    symbol = signal["symbol"]

    candles = signal["candles"]

    ema_values = signal["ema_values"]

    price = signal["price"]

    ema200 = signal["ema200"]

    choch_close = signal["choch_close"]

    choch_ema = signal["choch_ema"]

    message = (
        "🚨 EMA 200 ALERT\n"
        "━━━━━━━━━━━━━━━━━━\n\n"

        f"🪙 Symbol: {symbol}\n"
        "⏱ Timeframe: 15M\n\n"

        f"💵 Price: ${format_price(price)}\n"
        f"📊 EMA 200: ${format_price(ema200)}\n\n"

        "🔹 Structure Filter\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "✅ Bullish CHOCH detected\n"
        "✅ CHOCH below EMA 200\n"
        f"   CHOCH Close: ${format_price(choch_close)}\n"
        f"   CHOCH EMA:   ${format_price(choch_ema)}\n\n"

        "🔹 EMA Reclaim\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "✅ 4 CLOSED candles above EMA 200\n\n"

        f"1️⃣ Close: ${format_price(candles[0])}\n"
        f"   EMA:   ${format_price(ema_values[0])}\n\n"

        f"2️⃣ Close: ${format_price(candles[1])}\n"
        f"   EMA:   ${format_price(ema_values[1])}\n\n"

        f"3️⃣ Close: ${format_price(candles[2])}\n"
        f"   EMA:   ${format_price(ema_values[2])}\n\n"

        f"4️⃣ Close: ${format_price(candles[3])}\n"
        f"   EMA:   ${format_price(ema_values[3])}\n\n"

        "━━━━━━━━━━━━━━━━━━\n"
        "🔎 Filters Passed\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "• Price <= $5\n"
        "• CHOCH within previous 8H\n"
        "• CHOCH below EMA 200\n"
        "• 4 closes above EMA 200\n"
        "━━━━━━━━━━━━━━━━━━"
    )

    return message


# =========================
# TELEGRAM
# =========================

def send_to_chat(chat_id, message):

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
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
                f"📨 {signal['symbol']} sent to "
                f"chat {index}/"
                f"{len(TELEGRAM_CHAT_IDS)}"
            )

        except Exception as error:

            print(
                f"❌ Telegram failed for "
                f"chat {chat_id}: {error}"
            )

        if index < len(
            TELEGRAM_CHAT_IDS
        ):
            time.sleep(
                TELEGRAM_DELAY
            )


# =========================
# MAIN
# =========================

def main():

    print(
        "=========================================="
    )

    print(
        " Binance EMA 200 Telegram Scanner"
    )

    print(
        "=========================================="
    )

    print(
        f"Timeframe        : {INTERVAL}"
    )

    print(
        f"EMA              : {EMA_PERIOD}"
    )

    print(
        f"Structure length : {STRUCTURE_LENGTH}"
    )

    print(
        f"Price filter     : <= ${MAX_PRICE}"
    )

    print(
        f"CHOCH lookback   : {CHOCH_LOOKBACK_CANDLES} candles"
    )

    print(
        f"Required candles : {REQUIRED_CANDLES}"
    )

    print(
        f"Telegram chats   : {len(TELEGRAM_CHAT_IDS)}"
    )

    print(
        "=========================================="
    )

    state = load_state()

    symbols = get_usdt_symbols()

    print(
        f"Found {len(symbols)} USDT pairs"
    )

    new_signals = []

    for index, symbol in enumerate(
        symbols,
        start=1
    ):

        try:

            signal = check_signal(
                symbol
            )

            if signal:

                trigger_id = (
                    signal["trigger_id"]
                )

                previous_trigger = (
                    state.get(symbol)
                )

                if (
                    previous_trigger
                    == trigger_id
                ):

                    print(
                        f"[{index}/{len(symbols)}] "
                        f"{symbol} - already alerted"
                    )

                    continue

                new_signals.append(
                    signal
                )

                print(
                    f"[{index}/{len(symbols)}] "
                    f"🚨 NEW SIGNAL: {symbol}"
                )

            else:

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

        time.sleep(0.05)

    # =================================
    # SEND ALERTS
    # =================================

    for signal in new_signals:

        try:

            send_telegram_to_all(
                signal
            )

            state[
                signal["symbol"]
            ] = signal["trigger_id"]

            save_state(state)

        except Exception as error:

            print(
                f"Signal send error for "
                f"{signal['symbol']}: {error}"
            )

    save_state(state)

    print(
        "=========================================="
    )

    print(
        "Scan completed"
    )

    print(
        f"New signals: {len(new_signals)}"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()
