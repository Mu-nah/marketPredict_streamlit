import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import urllib3
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv
load_dotenv()

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Streamlit UI Configuration
st.set_page_config(page_title="📊 Price Predictor", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f9;
    }
    .block-container {
        padding: 2rem 2rem 2rem 2rem;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# API KEYS
TD_API_KEY = os.getenv("TD_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ----------- Data Fetching ------------
def fetch_td_data(symbol, interval, outputsize="5000"):
    url = f"https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TD_API_KEY,
        "format": "JSON"
    }
    r = requests.get(url, params=params, verify=False)
    data = r.json()
    if "values" not in data:
        st.error(f"Error from Twelve Data: {data.get('message', 'No data')}")
        return pd.DataFrame()
    df = pd.DataFrame(data["values"])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df.astype(float).sort_index()

def fetch_current_price(symbol):
    symbol_map = {
        "BTC/USD": "BTC/USD",
        "ETH/USD": "ETH/USD",
        "XAU/USD": "XAU/USD",
        "EUR/USD": "EUR/USD"
    }
    api_symbol = symbol_map.get(symbol.upper(), symbol)
    url = f"https://api.twelvedata.com/price?symbol={api_symbol}&apikey={TD_API_KEY}"
    try:
        r = requests.get(url, verify=False)
        price_data = r.json()
        return float(price_data['price'])
    except:
        st.warning("⚠ Failed to fetch current price. Using latest from data.")
        return None

# ---------- TA Features -------------
def get_ta(df):
    df['Return'] = df['close'].pct_change()
    df['MA7'] = df['close'].rolling(7).mean()
    df['MA21'] = df['close'].rolling(21).mean()
    df['STD21'] = df['close'].rolling(21).std()
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    return df

# ----------- Load All Timeframes -----------
def load_all_timeframes(symbol):
    df_30m = fetch_td_data(symbol, "30min")
    df_1h  = fetch_td_data(symbol, "1h")
    df_4h  = fetch_td_data(symbol, "4h")

    df_30m = get_ta(df_30m)
    df_1h = get_ta(df_1h).add_suffix('_1H')
    df_4h = get_ta(df_4h).add_suffix('_4H')

    for df in [df_30m, df_1h, df_4h]:
        df['Time'] = pd.to_datetime(df.index).floor('30min')

    df = df_30m.merge(df_1h, on='Time', how='inner').merge(df_4h, on='Time', how='inner')
    df.set_index('Time', inplace=True)
    df['Target'] = df['close'].shift(-1)
    df['NextOpen'] = df['open'].shift(-1)
    df['NextClose'] = df['close'].shift(-1)
    df['Target_2'] = df['close'].shift(-2)
    df['Target_3'] = df['close'].shift(-3)
    df.dropna(inplace=True)
    return df

# ----------- Sentiment --------------
def fetch_news_sentiment(symbol):
    keywords = {
        "BTC/USD": "bitcoin+OR+crypto+OR+btc",
        "ETH/USD": "ethereum+OR+eth+OR+crypto",
        "XAU/USD": "gold+OR+xau",
        "EUR/USD": "euro+OR+eur+OR+forex"
    }
    query = keywords.get(symbol.upper(), "finance")
    url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&q={query}&language=en&category=business"
    r = requests.get(url)
    data = r.json()

    analyzer = SentimentIntensityAnalyzer()
    scores = []
    if "results" in data:
        for article in data['results']:
            title = article.get("title", "")
            if title:
                score = analyzer.polarity_scores(title)['compound']
                scores.append(score)
    return sum(scores)/len(scores) if scores else 0.0

# ----------- Telegram Integration ------------
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("❌ Failed to send Telegram message:", e)

# ----------- Train Model ------------
def train_model(df, symbol, model_type):
    df['Sentiment'] = fetch_news_sentiment(symbol)
    features = [c for c in df.columns if 'open' in c or 'close' in c or 'Return' in c or 'MA' in c or 'STD' in c or 'Hour' in c or 'DayOfWeek' in c]
    features.append('Sentiment')

    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    if model_type == "XGBoost":
        model = XGBRegressor(n_estimators=100, learning_rate=0.05)
    elif model_type == "LightGBM":
        model = LGBMRegressor(n_estimators=100, learning_rate=0.05)
    else:
        model = RandomForestRegressor(n_estimators=100)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    test_direction = np.sign(y_test.values - X_test['close'].values)
    pred_direction = np.sign(y_pred - X_test['close'].values)
    confidence = (test_direction == pred_direction).mean()

    return model, X, y, mae, confidence

# ----------- Backtest -------------
def backtest(model, X, y, sl_pct=0.005, use_open_sl=False, use_tp=False, tp_pct=0.01, capital=10):
    df_bt = X.copy()
    df_bt['TruePrice'] = y
    df_bt['Prediction'] = model.predict(X)
    df_bt['Signal'] = np.where(df_bt['Prediction'] > df_bt['close'], 1, -1)
    df_bt['NextOpen'] = df_bt['close'].shift(-1)
    df_bt['NextClose'] = df_bt['close'].shift(-1)
    df_bt.dropna(inplace=True)

    results = []
    for i in range(len(df_bt) - 1):
        entry = df_bt.iloc[i]
        next_close = df_bt.iloc[i + 1]['NextClose']
        direction = entry['Signal']
        entry_price = entry['close']

        sl = df_bt.iloc[i + 1]['NextOpen'] if use_open_sl else entry_price * (1 - sl_pct if direction == 1 else 1 + sl_pct)
        tp = entry_price * (1 + tp_pct if direction == 1 else 1 - tp_pct)

        hit_sl = (direction == 1 and next_close <= sl) or (direction == -1 and next_close >= sl)
        hit_tp = (direction == 1 and next_close >= tp) or (direction == -1 and next_close <= tp)

        if use_tp and hit_tp:
            pnl = abs(entry_price * tp_pct)
        elif hit_sl:
            pnl = -abs(entry_price * sl_pct)
        else:
            pnl = next_close - entry_price if direction == 1 else entry_price - next_close
        pnl = (pnl / entry_price) * capital
        results.append({'Time': entry.name, 'Pnl': pnl, 'Entry': entry_price, 'NextClose': next_close, 'Direction': direction})

    df_result = pd.DataFrame(results).set_index('Time')
    df_result['Equity'] = df_result['Pnl'].cumsum()
    win_rate = (df_result['Pnl'] > 0).mean()
    return df_result, win_rate

# ----------- Streamlit App --------------
st.title("📈 AI Price Prediction & Backtesting")

with st.sidebar:
    st.header("⚙ Strategy Settings")
    pair = st.selectbox("Select Symbol", ["BTC/USD", "ETH/USD", "XAU/USD", "EUR/USD"])
    model_type = st.selectbox("Model", ["XGBoost", "LightGBM", "RandomForest"])
    sl_mode = st.radio("Stop Loss Mode", ["% SL", "Candle-Open SL"])
    use_tp = st.checkbox("Enable Take-Profit")
    tp_pct = st.slider("Take Profit %", min_value=0.1, max_value=5.0, value=1.0, step=0.1) / 100 if use_tp else 0.0
    sl_pct = st.slider("Stop Loss %", min_value=0.1, max_value=0.5, value=0.5, step=0.1) / 100
    capital = st.number_input("Capital ($)", min_value=1.0, max_value=100000.0, value=10.0, step=1.0)
    run = st.button("🔁 Retrain & Predict")

if run:
    with st.spinner("Training model & running backtest..."):
        df = load_all_timeframes(pair)
        model, X, y, mae, confidence = train_model(df, pair, model_type)
        sentiment = fetch_news_sentiment(pair)
        prediction = model.predict(X.tail(1))[0]
        current_price = fetch_current_price(pair) or X['close'].iloc[-1]
        stop_loss = current_price * (1 - sl_pct) if prediction > current_price else current_price * (1 + sl_pct)

        if prediction > current_price * 1.002:
            signal = "Strong Buy"
            signal_color = "green"
        elif prediction < current_price * 0.998:
            signal = "Strong Sell"
            signal_color = "red"
        else:
            signal = "Hold"
            signal_color = "gray"

        df_bt, win_rate = backtest(model, X, y, sl_pct=sl_pct, use_open_sl=(sl_mode != "% SL"), use_tp=use_tp, tp_pct=tp_pct, capital=capital)

        message = f"""
🚀 *AI Market Prediction Alert*

🪙 *Symbol:* `{pair}`
💰 *Live Price:* `${current_price:,.2f}`
🎯 *Predicted 30-min Close:* `${prediction:,.2f}`
🛑 *Stop Loss:* `${stop_loss:,.2f}`

📊 *Signal:* *{signal}* {"🟢" if signal == "Strong Buy" else "🔴" if signal == "Strong Sell" else "⚪"}

🔐 *Confidence:* `{confidence*100:.2f}%`
🧠 *Sentiment Score:* `{sentiment:.2f}`
📏 *MAE:* `${mae:.2f}`

⏱ *Generated at:* `{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}`
"""

        send_telegram_message(message)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Prediction")
        st.metric("Live Price", f"${current_price:,.2f}")
        st.metric("Next 30-min Prediction", f"${prediction:,.2f}")
        st.metric("SL Level", f"${stop_loss:,.2f}")
        st.metric("Confidence", f"{confidence*100:.2f}%")
        st.metric("Sentiment", f"{sentiment:.2f}")
        st.metric("MAE", f"${mae:.2f}")
        st.markdown(f"<div style='color:{signal_color};font-size:24px;'>⚑ {signal} Signal</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("📉 Backtest")
        st.metric("Win Rate", f"{win_rate*100:.2f}%")
        st.metric("Return", f"${df_bt['Equity'].iloc[-1]:,.2f}")
        st.line_chart(df_bt['Equity'], use_container_width=True)
        with st.expander("📋 Backtest Log"):
            st.dataframe(df_bt[['Entry', 'NextClose', 'Direction', 'Pnl']].tail(20))

    st.success("Analysis complete ✅")
else:
    st.info("Configure settings and click Retrain & Predict to begin.")
