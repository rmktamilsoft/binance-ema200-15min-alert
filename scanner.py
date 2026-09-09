import os
import time
import json
import requests
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed


# =========================================================
# CONFIG
# =========================================================

BINANCE_URL = "https://data-api.binance.vision"

INTERVAL = "15m"
EMA_PERIOD = 200
REQUIRED_CANDLES = 4

MAX_PRICE = 5.0

STRUCTURE_LENGTH = 5
CHOCH_LOOKBACK_CANDLES = 32

MAX_WORKERS = 10

STATE_FILE = "alert_state.json"

TELEGRAM_DELAY = 1.2


# =========================================================
# TELEGRAM
# =========================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

TELEGRAM_CHAT_IDS = [
    x.strip()
    for x in os.getenv("TELEGRAM_CHAT_IDS", "").split(",")
    if x.strip()
]


# =========================================================
# BINANCE
# =========================================================

def get_usdt_symbols():

    url = f"{BINANCE_URL}/api/v3/exchangeInfo"

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    data = response.json()

    symbols = []

    for item in data["symbols"]:

        if (
            item["status"] == "TRADING"
            and item["quoteAsset"] == "USDT"
            and item["isSpotTradingAllowed"]
        ):
            symbols.append(item["symbol"])

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

def create_dataframe(klines):

    columns = [
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

    df = pd.DataFrame(klines, columns=columns)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col])

    df["open_time"] = pd.to_numeric(df["open_time"])
    df["close_time"] = pd.to_numeric(df["close_time"])

    return df


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
# STRUCTURE
# =========================================================

def find_pivot_high(df, index, length):

    pivot_index = index - length

    if pivot_index < length:
        return None

    if pivot_index + length >= len(df):
        return None

    pivot_high = df.iloc[pivot_index]["high"]

    left_highs = df.iloc[
        pivot_index - length:pivot_index
    ]["high"]

    right_highs = df.iloc[
        pivot_index + 1:pivot_index + length + 1
    ]["high"]

    if pivot_high > left_highs.max() and pivot_high >= right_highs.max():
        return pivot_high

    return None


def find_pivot_low(df, index, length):

    pivot_index = index - length

    if pivot_index < length:
        return None

    if pivot_index + length >= len(df):
        return None

    pivot_low = df.iloc[pivot_index]["low"]

    left_lows = df.iloc[
        pivot_index - length:pivot_index
    ]["low"]

    right_lows = df.iloc[
        pivot_index + 1:pivot_index + length + 1
    ]["low"]

    if pivot_low < left_lows.min() and pivot_low <= right_lows.min():
        return pivot_low

    return None


def detect_bullish_choch(df):

    trend = None

    active_high = None
    active_low = None

    choch_events = []

    for i in range(len(df)):

        row = df.iloc[i]

        pivot_high = find_pivot_high(
            df,
            i,
            STRUCTURE_LENGTH
        )

        pivot_low = find_pivot_low(
            df,
            i,
            STRUCTURE_LENGTH
        )

        if pivot_high is not None:
            active_high = pivot_high

        if pivot_low is not None:
            active_low = pivot_low

        # ---------------------------------------------
        # Bearish structure break
        # ---------------------------------------------

        if (
            active_low is not None
            and row["close"] < active_low
        ):

            trend = "BEARISH"

        # ---------------------------------------------
        # Bullish structure break
        # ---------------------------------------------

        if (
            active_high is not None
            and row["close"] > active_high
        ):

            if trend == "BEARISH":

                choch_events.append({
                    "index": i,
                    "close": row["close"],
                    "ema200": row["ema200"],
                    "close_time": int(row["close_time"])
                })

            trend = "BULLISH"

    if not choch_events:
        return None

    return choch_events[-1]


# =========================================================
# SIGNAL CHECK
# =========================================================

def check_signal(symbol):

    try:

        # -------------------------------------------------
        # Get candles
        # -------------------------------------------------

        klines = get_klines(symbol, limit=250)

        df = create_dataframe(klines)

        # Remove currently forming candle
        current_time = int(time.time() * 1000)

        df = df[
            df["close_time"] < current_time
        ].copy()

        if len(df) < EMA_PERIOD + REQUIRED_CANDLES:

            return {
                "symbol": symbol,
                "status": "NOT_ENOUGH_CANDLES",
                "message": f"only {len(df)} closed candles"
            }

        # -------------------------------------------------
        # EMA
        # -------------------------------------------------

        df = calculate_ema(df)

        # -------------------------------------------------
        # Price filter
        # -------------------------------------------------

        price = float(df.iloc[-1]["close"])

        if price > MAX_PRICE:

            return {
                "symbol": symbol,
                "status": "PRICE_FILTER",
                "message": f"price ${price:.6f} > ${MAX_PRICE}"
            }

        # -------------------------------------------------
        # Last 4 candles
        # -------------------------------------------------

        last4 = df.tail(REQUIRED_CANDLES)

        above_ema = (
            last4["close"] > last4["ema200"]
        )

        above_count = int(above_ema.sum())

        if above_count < REQUIRED_CANDLES:

            return {
                "symbol": symbol,
                "status": "EMA_FILTER",
                "message": (
                    f"{above_count}/{REQUIRED_CANDLES} "
                    f"closed candles above EMA200 | "
                    f"price=${price:.6f} "
                    f"EMA=${df.iloc[-1]['ema200']:.6f}"
                )
            }

        # -------------------------------------------------
        # Structure lookback
        # -------------------------------------------------

        structure_start = max(
            0,
            len(df) - CHOCH_LOOKBACK_CANDLES - 100
        )

        recent_df = df.iloc[
            structure_start:
        ].copy()

        choch = detect_bullish_choch(
            recent_df
        )

        if choch is None:

            return {
                "symbol": symbol,
                "status": "NO_CHOCH",
                "message": (
                    f"last {CHOCH_LOOKBACK_CANDLES} candles "
                    f"no bullish structure change"
                )
            }

        # -------------------------------------------------
        # CHOCH must be inside previous 8h
        # -------------------------------------------------

        first_signal_index = len(df) - REQUIRED_CANDLES

        choch_global_time = choch["close_time"]

        first_signal_time = int(
            df.iloc[first_signal_index]["close_time"]
        )

        if choch_global_time >= first_signal_time:

            return {
                "symbol": symbol,
                "status": "CHOCH_TOO_RECENT",
                "message": (
                    f"CHOCH happened after/between "
                    f"the 4 EMA candles"
                )
            }

        # -------------------------------------------------
        # CHOCH candle must close below EMA200
        # -------------------------------------------------

        choch_close = float(
            choch["close"]
        )

        choch_ema = float(
            choch["ema200"]
        )

        if choch_close >= choch_ema:

            return {
                "symbol": symbol,
                "status": "CHOCH_ABOVE_EMA",
                "message": (
                    f"CHOCH close=${choch_close:.6f} "
                    f">= EMA=${choch_ema:.6f}"
                )
            }

        # -------------------------------------------------
        # SIGNAL FOUND
        # -------------------------------------------------

        return {
            "symbol": symbol,
            "status": "SIGNAL",

            "price": price,
            "ema200": float(df.iloc[-1]["ema200"]),

            "closes": [
                float(x)
                for x in last4["close"]
            ],

            "emas": [
                float(x)
                for x in last4["ema200"]
            ],

            "choch_close": choch_close,
            "choch_ema": choch_ema,

            "choch_time": choch_global_time,

            "trigger_time": int(
                df.iloc[-1]["close_time"]
            )
        }

    except Exception as e:

        return {
            "symbol": symbol,
            "status": "ERROR",
            "message": str(e)
        }


# =========================================================
# TELEGRAM
# =========================================================

def format_price(price):

    if price >= 1:
        return f"{price:.4f}"

    if price >= 0.01:
        return f"{price:.6f}"

    if price >= 0.0001:
        return f"{price:.8f}"

    return f"{price:.10f}"


def send_telegram(message):

    if not TELEGRAM_BOT_TOKEN:

        print("Telegram bot token missing")

        return False

    if not TELEGRAM_CHAT_IDS:

        print("Telegram chat IDs missing")

        return False

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    success = True

    for chat_id in TELEGRAM_CHAT_IDS:

        try:

            response = requests.post(
                url,
                data={
                    "chat_id": chat_id,
                    "text": message
                },
                timeout=20
            )

            if response.ok:

                print(
                    f"Telegram sent -> {chat_id}"
                )

            else:

                print(
                    f"Telegram FAILED -> "
                    f"{chat_id} | "
                    f"{response.text}"
                )

                success = False

        except Exception as e:

            print(
                f"Telegram ERROR -> "
                f"{chat_id} | {e}"
            )

            success = False

        time.sleep(TELEGRAM_DELAY)

    return success


# =========================================================
# ALERT MESSAGE
# =========================================================

def create_alert_message(signal):

    symbol = signal["symbol"]

    price = format_price(
        signal["price"]
    )

    ema = format_price(
        signal["ema200"]
    )

    choch_close = format_price(
        signal["choch_close"]
    )

    choch_ema = format_price(
        signal["choch_ema"]
    )

    return (
        "🚨 EMA200 SETUP\n\n"
        f"Symbol: {symbol}\n"
        f"Price: ${price}\n"
        f"EMA200: ${ema}\n\n"
        "✅ 4 CLOSED candles above EMA200\n"
        "✅ Bullish structure change found\n"
        f"Structure close: ${choch_close}\n"
        f"Structure EMA200: ${choch_ema}\n"
        "✅ Structure close below EMA200\n\n"
        f"Timeframe: {INTERVAL}"
    )


# =========================================================
# STATE
# =========================================================

def load_state():

    if not os.path.exists(STATE_FILE):
        return {}

    try:

        with open(
            STATE_FILE,
            "r"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


def save_state(state):

    with open(
        STATE_FILE,
        "w"
    ) as f:

        json.dump(
            state,
            f,
            indent=2
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 50)
    print("Binance EMA 200 Telegram Scanner - DEBUG")
    print("=" * 50)

    print(f"Timeframe        : {INTERVAL}")
    print(f"EMA              : {EMA_PERIOD}")
    print(f"Structure        : {STRUCTURE_LENGTH}")
    print(f"Price filter     : <= ${MAX_PRICE}")
    print(f"Lookback         : {CHOCH_LOOKBACK_CANDLES} candles")
    print(f"Workers          : {MAX_WORKERS}")
    print(f"Telegram chats   : {len(TELEGRAM_CHAT_IDS)}")

    print("=" * 50)

    symbols = get_usdt_symbols()

    print(
        f"Found {len(symbols)} USDT pairs"
    )

    print("=" * 50)

    state = load_state()

    results = []

    # -----------------------------------------------------
    # Parallel scan
    # -----------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                check_signal,
                symbol
            ): symbol
            for symbol in symbols
        }

        completed = 0

        for future in as_completed(futures):

            symbol = futures[future]

            completed += 1

            result = future.result()

            results.append(result)

            status = result["status"]

            # =================================================
            # DEBUG LOG
            # =================================================

            if status == "SIGNAL":

                print(
                    f"[{completed}/{len(symbols)}] "
                    f"{symbol} -> 🚨 SIGNAL | "
                    f"Price=${format_price(result['price'])} | "
                    f"EMA=${format_price(result['ema200'])} | "
                    f"CHOCH=${format_price(result['choch_close'])}"
                )

            elif status == "PRICE_FILTER":

                print(
                    f"[{completed}/{len(symbols)}] "
                    f"{symbol} -> ❌ PRICE | "
                    f"{result['message']}"
                )

            elif status == "EMA_FILTER":

                print(
                    f"[{completed}/{len(symbols)}] "
                    f"{symbol} -> ❌ EMA | "
                    f"{result['message']}"
                )

            elif status == "NO_CHOCH":

                print(
                    f"[{completed}/{len(symbols)}] "
                    f"{symbol} -> ❌ STRUCTURE | "
                    f"{result['message']}"
                )

            elif status == "CHOCH_ABOVE_EMA":

                print(
                    f"[{completed}/{len(symbols)}] "
                    f"{symbol} -> ❌ CHOCH EMA | "
                    f"{result['message']}"
                )

            elif status == "CHOCH_TOO_RECENT":

                print(
                    f"[{completed}/{len(symbols)}] "
                    f"{symbol} -> ❌ CHOCH TIMING | "
                    f"{result['message']}"
                )

            elif status == "NOT_ENOUGH_CANDLES":

                print(
                    f"[{completed}/{len(symbols)}] "
                    f"{symbol} -> ❌ DATA | "
                    f"{result['message']}"
                )

            elif status == "ERROR":

                print(
                    f"[{completed}/{len(symbols)}] "
                    f"{symbol} -> ⚠️ ERROR | "
                    f"{result['message']}"
                )

    # =====================================================
    # PROCESS SIGNALS
    # =====================================================

    signals = [
        r
        for r in results
        if r["status"] == "SIGNAL"
    ]

    print("=" * 50)
    print(
        f"Setups found: {len(signals)}"
    )

    new_alerts = 0

    # =====================================================
    # SEND ALERTS
    # =====================================================

    for signal in signals:

        symbol = signal["symbol"]

        setup_id = str(
            signal["choch_time"]
        )

        previous_state = state.get(
            symbol
        )

        if (
            previous_state is not None
            and previous_state.get("setup_id")
            == setup_id
        ):

            print(
                f"{symbol} -> 🔁 DUPLICATE "
                f"(already alerted)"
            )

            continue

        message = create_alert_message(
            signal
        )

        print(
            f"{symbol} -> 📲 Sending Telegram..."
        )

        sent = send_telegram(
            message
        )

        if sent:

            state[symbol] = {
                "setup_id": setup_id,
                "alert_time": int(
                    time.time()
                )
            }

            new_alerts += 1

    # =====================================================
    # SAVE STATE
    # =====================================================

    save_state(state)

    print("=" * 50)
    print(
        f"Setups found : {len(signals)}"
    )
    print(
        f"New alerts   : {new_alerts}"
    )
    print("=" * 50)

    print("Scan completed")


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    main()
