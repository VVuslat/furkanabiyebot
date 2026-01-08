from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request


@dataclass
class StrategyConfig:
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    atr_period: int = 14
    lookahead_bars: int = 12
    tp_multiplier: float = 1.2
    sl_multiplier: float = 0.8


@dataclass
class SignalResult:
    action: Literal["BUY", "SELL", "HOLD"]
    reason: str
    tp_price: float
    sl_price: float
    leverage: float
    expected_value: float
    win_rate: float
    markov_up_prob: float
    markov_down_prob: float


app = Flask(__name__)


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/signal")
def api_signal():
    symbol = request.args.get("symbol", "BTC")
    quote = request.args.get("quote", "USDT")
    interval = request.args.get("interval", "5m")
    leverage = float(request.args.get("leverage", "1"))
    config = StrategyConfig()

    data = fetch_market_data(f"{symbol}-{quote}", interval)
    if data.empty:
        return jsonify({"error": "No market data returned."}), 400

    enriched = add_indicators(data, config)
    result = evaluate_signal(enriched, config, leverage)

    latest = enriched.iloc[-1]
    response = {
        "symbol": f"{symbol}-{quote}",
        "interval": interval,
        "timestamp": latest.name.isoformat(),
        "price": round(float(latest["close"]), 6),
        "ema_fast": round(float(latest["ema_fast"]), 6),
        "ema_slow": round(float(latest["ema_slow"]), 6),
        "rsi": round(float(latest["rsi"]), 4),
        "atr": round(float(latest["atr"]), 6),
        "action": result.action,
        "reason": result.reason,
        "tp_price": round(result.tp_price, 6),
        "sl_price": round(result.sl_price, 6),
        "leverage": result.leverage,
        "expected_value": round(result.expected_value, 4),
        "win_rate": round(result.win_rate, 4),
        "markov": {
            "up_prob": round(result.markov_up_prob, 4),
            "down_prob": round(result.markov_down_prob, 4),
        },
    }
    return jsonify(response)


def fetch_market_data(symbol: str, interval: str) -> pd.DataFrame:
    period = "7d" if interval in {"1m", "2m", "5m", "15m"} else "60d"
    ticker = yf.Ticker(symbol)
    history = ticker.history(period=period, interval=interval)
    if history.empty:
        return pd.DataFrame()

    history = history.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return history[["open", "high", "low", "close", "volume"]].dropna()


def add_indicators(data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    df = data.copy()
    df["ema_fast"] = df["close"].ewm(span=config.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=config.ema_slow, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=config.rsi_period).mean()
    avg_loss = loss.rolling(window=config.rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=config.atr_period).mean()
    return df.dropna()


def evaluate_signal(
    data: pd.DataFrame, config: StrategyConfig, leverage: float
) -> SignalResult:
    latest = data.iloc[-1]
    price = float(latest["close"])

    ema_trend = latest["ema_fast"] > latest["ema_slow"]
    rsi = float(latest["rsi"])

    if ema_trend and rsi < 70:
        action = "BUY"
        reason = "EMA trend up and RSI below overbought zone"
        direction = 1
    elif (not ema_trend) and rsi > 30:
        action = "SELL"
        reason = "EMA trend down and RSI above oversold zone"
        direction = -1
    else:
        action = "HOLD"
        reason = "No clear trend confirmation"
        direction = 0

    atr = float(latest["atr"])
    tp_distance = atr * config.tp_multiplier
    sl_distance = atr * config.sl_multiplier

    tp_price = price + direction * tp_distance if direction else price
    sl_price = price - direction * sl_distance if direction else price

    expected_value, win_rate = backtest_expectancy(data, config, tp_distance, sl_distance)
    markov_up, markov_down = markov_probabilities(data)

    return SignalResult(
        action=action,
        reason=reason,
        tp_price=tp_price,
        sl_price=sl_price,
        leverage=leverage,
        expected_value=expected_value * leverage,
        win_rate=win_rate,
        markov_up_prob=markov_up,
        markov_down_prob=markov_down,
    )


def backtest_expectancy(
    data: pd.DataFrame,
    config: StrategyConfig,
    tp_distance: float,
    sl_distance: float,
) -> tuple[float, float]:
    if len(data) < config.lookahead_bars + 1:
        return 0.0, 0.0

    wins = 0
    losses = 0
    total = 0

    for idx in range(len(data) - config.lookahead_bars - 1):
        row = data.iloc[idx]
        ema_trend = row["ema_fast"] > row["ema_slow"]
        rsi = float(row["rsi"])
        if ema_trend and rsi < 70:
            direction = 1
        elif (not ema_trend) and rsi > 30:
            direction = -1
        else:
            continue

        entry = float(row["close"])
        tp = entry + direction * tp_distance
        sl = entry - direction * sl_distance

        window = data.iloc[idx + 1 : idx + 1 + config.lookahead_bars]
        hit_tp = False
        hit_sl = False

        for _, future in window.iterrows():
            high = float(future["high"])
            low = float(future["low"])
            if direction == 1:
                if high >= tp:
                    hit_tp = True
                    break
                if low <= sl:
                    hit_sl = True
                    break
            else:
                if low <= tp:
                    hit_tp = True
                    break
                if high >= sl:
                    hit_sl = True
                    break

        total += 1
        if hit_tp and not hit_sl:
            wins += 1
        elif hit_sl and not hit_tp:
            losses += 1

    if total == 0:
        return 0.0, 0.0

    win_rate = wins / total
    loss_rate = losses / total
    expectancy = win_rate * (tp_distance) - loss_rate * (sl_distance)
    return expectancy, win_rate


def markov_probabilities(data: pd.DataFrame) -> tuple[float, float]:
    returns = data["close"].pct_change().dropna()
    if returns.empty:
        return 0.5, 0.5

    states = returns.apply(lambda x: "up" if x >= 0 else "down")
    transitions = {
        "up": {"up": 0, "down": 0},
        "down": {"up": 0, "down": 0},
    }

    for prev, curr in zip(states[:-1], states[1:]):
        transitions[prev][curr] += 1

    up_total = transitions["up"]["up"] + transitions["up"]["down"]
    down_total = transitions["down"]["up"] + transitions["down"]["down"]

    up_prob = transitions["up"]["up"] / up_total if up_total else 0.5
    down_prob = transitions["down"]["down"] / down_total if down_total else 0.5
    return up_prob, down_prob


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
