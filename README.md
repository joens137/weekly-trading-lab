# Weekly Trading Lab — Research Edition

This is a replacement version of your local Streamlit stock research application.
It compares **nine** virtual ETF strategies with a $100 (adjustable) simulated starting balance each, reports prediction accuracy, and journals new forecasts for subsequent scoring.

## Install on Windows
1. Have Python 3.13 x64 installed.
2. Extract this ZIP; open `weekly_trading_lab_research`.
3. Double-click `run_windows.bat`.
4. Allow it to install Python packages. Browser opens with the Streamlit dashboard.
5. First use with ML on takes longer because the models are trained separately at each historical date. Subsequent runs are cached unless parameters/data change.

An internet connection is required to download Yahoo Finance adjusted historical OHLC prices.

## Carry over your previous saved predictions
Copy `saved_forecasts.csv` from the old trading lab folder into this new folder before running; **back up the original first**. The old file is not included or overwritten in this archive. Some legacy prediction rows can be scored automatically, but their earlier close-to-close assumptions differ from the new open-to-open research methodology. Keep old and new results labeled separately when comparing.

## Evaluation features
- Chronological walk-forward refitting; no future outcome labels used for training.
- Model forecasts are five-session **next-open to later-open** returns.
- Nine comparison strategies including two model-based methods and four momentum/defensive ideas.
- Five ETFs: SPY, QQQ, IWM, GLD, TLT; adjusted Open and Close from yfinance.
- Simulated one-sided traded-dollar turnover fee from allocation changes.
- 60% development / 20% validation / 20% holdout **of evaluated weeks**.
- Ranks a candidate on validation, then compares its later holdout performance against SPY.
- Prediction mean absolute error, always-up / zero baselines, win rate, max drawdown, approximated Sharpe, profit factor and performance charts.
- New predictions are saved locally to `saved_forecasts.csv` and scored once enough daily sessions arrive.

## Important limitations
This is educational research, not investment advice. Historical results cannot guarantee future results. The current model uses only price-derived features, so don't expect reliable one-week predictive skill. Daily next-open fills and transaction costs are approximations, and this model does not trade automatically. A repeatedly viewed holdout stops being fully independent if you tune the model based on it.

## Files
`app.py`: dashboard. `engine.py`: calculations. `requirements.txt`: Python dependencies. `run_windows.bat`: Windows launcher.
