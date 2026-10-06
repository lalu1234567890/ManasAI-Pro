# ManasAI Pro — Educational InvestingPro-style AI Research Engine

This is an educational, from-scratch quantitative research prototype inspired by public descriptions of InvestingPro/ProPicks/Vision AI.

It is NOT InvestingPro's proprietary model and it is not financial advice.

## What it does
- Downloads historical OHLCV data through yfinance.
- Calculates trend, momentum, volatility, volume and relative-strength features.
- Produces an explainable 0–100 AI-style score.
- Trains a Random Forest classifier on historical features to estimate the probability of positive forward returns.
- Ranks a user-defined universe of stocks.
- Backtests a monthly top-N strategy against an equal-weight portfolio.
- Provides an AI-style technical research report with support/resistance and risk levels.
- Supports CSV upload for chart/data experiments.

## Run
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

For educational use, start with Nifty 50 symbols in `config.py`.

## Important
The ML result is a research signal, not a prediction guarantee. Real deployment requires point-in-time fundamentals, corporate actions, survivorship-bias controls, transaction costs, slippage, taxes, walk-forward validation, and licensed market data.
