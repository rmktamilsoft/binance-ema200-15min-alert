import os
import time
import json
import requests
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed


BINANCE_URL = "https://data-api.binance.vision"

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

TELEGRAM_CHAT_IDS = [
    chat_id.strip()
    for chat_id in os.environ["TELEGRAM_CHAT_IDS"].split(",")
    if chat_id.strip()
]

TELEGRAM_DELAY = 1.2


# =========================================================
# STRATEGY SETTINGS
# =========================================================

INTERVAL = "15m"

EMA_PERIOD = 200

REQUIRED_CANDLES = 4

MAX_PRICE = 5.0

STRUCTURE_LENGTH = 5

CHOCH_LOOKBACK_CANDLES = 32

# Parallel Binance requests
MAX_WORKERS = 10

# State file
STATE_FILE = "alert_state.json"


# =========================================================
# STATE
# =========================================================

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
        json.dump(
            state,
            file,
            indent=2
        )


# =========================================================
# BINANCE
# =========================================================

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

            symbols.append(
                symbol_info["symbol"]
            )

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


# =========================================================
# DATAFRAME
# =========================================================

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

        df[column] = pd.to_numeric(
            df[column]
        )

    # Remove currently forming candle
    current_time = int(
        time.time() * 1000
    )

    df = df[
        df["close_time"] < current_time
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df


# =========================================================
# EMA
# =========================================================

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


# =========================================================
# PIVOTS
# =========================================================

def find_pivot_high(
    df,
    index,
    length
):

    pivot_index = index - length

    if pivot_index < length:
        return None

    if pivot_index >= len(df):
        return None

    pivot_high = float(
        df.iloc[pivot_index]["high"]
    )

    left_highs = df.iloc[
        pivot_index - length:pivot_index
    ]["high"]

    right_highs = df.iloc[
        pivot_index + 1:index + 1
    ]["high"]

    if (
        len(left_highs) < length
        or len(right_highs) < length
    ):
        return None

    if (
        pivot_high > left_highs.max()
        and pivot_high > right_highs.max()
    ):

        return {
            "index": pivot_index,
            "price": pivot_high
        }

    return None


def find_pivot_low(
    df,
    index,
    length
):

    pivot_index = index - length

    if pivot_index < length:
        return None

    if pivot_index >= len(df):
        return None

    pivot_low = float(
        df.iloc[pivot_index]["low"]
    )

    left_lows = df.iloc[
        pivot_index - length:pivot_index
    ]["low"]

    right_lows = df.iloc[
        pivot_index + 1:index + 1
    ]["low"]

    if (
        len(left_lows) < length
        or len(right_lows) < length
    ):
        return None

    if (
        pivot_low < left_lows.min()
        and pivot_low < right_lows.min()
    ):

        return {
            "index": pivot_index,
            "price": pivot_low
        }

    return None


# =========================================================
# BULLISH STRUCTURE CHANGE
# =========================================================

def detect_bullish_choch(df):

    if len(df) < 100:
        return None

    trend = None

    active_high = None
    active_low = None

    latest_choch = None

    previous_close = None

    for i in range(
        STRUCTURE_LENGTH * 2,
        len(df)
    ):

        # -----------------------------------------
        # New confirmed pivot high
        # -----------------------------------------

        pivot_high = find_pivot_high(
            df,
            i,
            STRUCTURE_LENGTH
        )

        if pivot_high is not None:

            active_high = pivot_high

        # -----------------------------------------
        # New confirmed pivot low
        # -----------------------------------------

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

        # -----------------------------------------
        # Bearish structure
        # -----------------------------------------

        if (
            active_low is not None
            and previous_close is not None
            and previous_close >= active_low["price"]
            and current_close < active_low["price"]
        ):

            trend = "BEARISH"

        # -----------------------------------------
        # Bullish structure change
        # -----------------------------------------

        if (
            trend == "BEARISH"
            and active_high is not None
            and previous_close is not None
            and previous_close <= active_high["price"]
            and current_close > active_high["price"]
        ):

            latest_choch = i

            trend = "BULLISH"

        previous_close = current_close

    return latest_choch


# =========================================================
# SIGNAL
# =========================================================

def check_signal(symbol):

    try:

        candles = get_klines(
            symbol,
            limit=250
        )

        if len(candles) < 230:
            return None

        df = create_dataframe(
            candles
        )

        if len(df) < 230:
            return None

        df = calculate_ema(df)

        # =================================================
        # PRICE FILTER
        # =================================================

        current_price = float(
            df.iloc[-1]["close"]
        )

        if current_price > MAX_PRICE:
            return None

        # =================================================
        # LAST 4 CLOSED CANDLES
        # ABOVE EMA200
        # =================================================

        last_four = df.iloc[
            -REQUIRED_CANDLES:
        ].copy()

        if len(last_four) != 4:
            return None

        if not (
            last_four["close"]
            > last_four["ema200"]
        ).all():

            return None

        # =================================================
        # PREVIOUS 8 HOURS
        # =================================================

        search_end = (
            len(df)
            - REQUIRED_CANDLES
        )

        search_start = max(
            0,
            search_end
            - CHOCH_LOOKBACK_CANDLES
        )

        recent_df = df.iloc[
            search_start:search_end
        ].copy()

        recent_df.reset_index(
            drop=True,
            inplace=True
        )

        if len(recent_df) < 15:
            return None

        # =================================================
        # FIND STRUCTURE CHANGE
        # =================================================

        choch_index = detect_bullish_choch(
            recent_df
        )

        if choch_index is None:
            return None

        choch_candle = recent_df.iloc[
            choch_index
        ]

        choch_close = float(
            choch_candle["close"]
        )

        choch_ema = float(
            choch_candle["ema200"]
        )

        # =================================================
        # STRUCTURE CHANGE MUST BE BELOW EMA200
        # =================================================

        if choch_close >= choch_ema:
            return None

        # =================================================
        # STRUCTURE CHANGE MUST HAPPEN
        # BEFORE THE 4 CANDLE RECLAIM
        # =================================================

        first_reclaim_time = int(
            last_four.iloc[0]["open_time"]
        )

        choch_time = int(
            choch_candle["open_time"]
        )

        if choch_time >= first_reclaim_time:
            return None

        # =================================================
        # SIGNAL
        # =================================================

        last = last_four.iloc[-1]

        return {
            "symbol": symbol,

            "price": float(
                last["close"]
            ),

            "ema200": float(
                last["ema200"]
            ),

            "candles": [
                float(x)
                for x in last_four["close"]
            ],

            "ema_values": [
                float(x)
                for x in last_four["ema200"]
            ],

            "choch_close": choch_close,

            "choch_ema": choch_ema,

            "choch_time": choch_time,

            "trigger_time": int(
                last["close_time"]
            )
        }

    except Exception as error:

        print(
            f"{symbol} ERROR: {error}"
        )

        return None


# =========================================================
# PRICE FORMAT
# =========================================================

def format_price(value):

    if value >= 1:
        return f"{value:.4f}"

    if value >= 0.01:
        return f"{value:.6f}"

    if value >= 0.0001:
        return f"{value:.8f}"

    return f"{value:.12f}"


# =========================================================
# TELEGRAM MESSAGE
# =========================================================

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
        "⏱ Timeframe: 15M\n"
        f"💵 Price: ${format_price(price)}\n\n"

        "🔹 Structure\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "✅ Bullish CHOCH\n"
        "✅ CHOCH below EMA 200\n"
        f"   Close: ${format_price(choch_close)}\n"
        f"   EMA:   ${format_price(choch_ema)}\n\n"

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
        "• Structure change within 8H\n"
        "• Structure change below EMA200\n"
        "• 4 closed candles above EMA200\n"
        "━━━━━━━━━━━━━━━━━━"
    )

    return message


# =========================================================
# TELEGRAM SEND
# =========================================================

def send_to_chat(
    chat_id,
    message
):

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

    message = create_message(
        signal
    )

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
                f"sent {index}/"
                f"{len(TELEGRAM_CHAT_IDS)}"
            )

        except Exception as error:

            print(
                f"❌ Telegram failed "
                f"for {chat_id}: {error}"
            )

        if index < len(
            TELEGRAM_CHAT_IDS
        ):

            time.sleep(
                TELEGRAM_DELAY
            )


# =========================================================
# MAIN
# =========================================================

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
        f"Timeframe       : {INTERVAL}"
    )

    print(
        f"EMA             : {EMA_PERIOD}"
    )

    print(
        f"Structure       : {STRUCTURE_LENGTH}"
    )

    print(
        f"Price filter    : <= ${MAX_PRICE}"
    )

    print(
        f"Lookback        : {CHOCH_LOOKBACK_CANDLES} candles"
    )

    print(
        f"Workers         : {MAX_WORKERS}"
    )

    print(
        f"Telegram chats  : {len(TELEGRAM_CHAT_IDS)}"
    )

    print(
        "=========================================="
    )

    state = load_state()

    symbols = get_usdt_symbols()

    print(
        f"Found {len(symbols)} USDT pairs"
    )

    signals = []

    # =====================================================
    # PARALLEL SCAN
    # =====================================================

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_map = {
            executor.submit(
                check_signal,
                symbol
            ): symbol
            for symbol in symbols
        }

        completed = 0

        for future in as_completed(
            future_map
        ):

            symbol = future_map[
                future
            ]

            completed += 1

            try:

                signal = future.result()

                if signal is None:

                    print(
                        f"[{completed}/{len(symbols)}] "
                        f"{symbol} - no signal"
                    )

                    continue

                print(
                    f"[{completed}/{len(symbols)}] "
                    f"🚨 SETUP: {symbol}"
                )

                signals.append(
                    signal
                )

            except Exception as error:

                print(
                    f"{symbol} ERROR: {error}"
                )

    # =====================================================
    # PROCESS SIGNALS
    # =====================================================

    new_alerts = 0

    for signal in signals:

        symbol = signal["symbol"]

        # -------------------------------------------------
        # SETUP ID
        #
        # CHOCH time identifies the setup.
        # -------------------------------------------------

        setup_id = str(
            signal["choch_time"]
        )

        previous_state = state.get(
            symbol
        )

        # =================================================
        # ALREADY ALERTED FOR THIS SETUP
        # =================================================

        if (
            previous_state is not None
            and previous_state.get("setup_id")
            == setup_id
        ):

            print(
                f"{symbol} - "
                f"already alerted for this setup"
            )

            continue

        # =================================================
        # NEW ALERT
        # =================================================

        try:

            send_telegram_to_all(
                signal
            )

            state[symbol] = {
                "setup_id": setup_id,
                "alert_time": int(
                    time.time()
                )
            }

            save_state(
                state
            )

            new_alerts += 1

            print(
                f"🚨 ALERT SENT: {symbol}"
            )

        except Exception as error:

            print(
                f"Signal send error "
                f"for {symbol}: {error}"
            )

    save_state(
        state
    )

    print(
        "=========================================="
    )

    print(
        "Scan completed"
    )

    print(
        f"Setups found : {len(signals)}"
    )

    print(
        f"New alerts   : {new_alerts}"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()
