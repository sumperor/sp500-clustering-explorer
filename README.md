# S&P 500 Stock Clustering Explorer

Built on the Imperial DSA-with-Python Homework 3 dataset (real, not synthetic):
- 496 S&P 500 tickers with names and GICS sectors
- Daily close prices for all 252 trading days of 2015

## What it does
- Overview: sector breakdown, normalised price paths by sector
- Returns & Volatility: best/worst yearly movers, highest volatility
- Correlations: pairwise return correlation heatmap, most/least correlated peers for any ticker
- Clustering: runs the HW3 single-linkage clustering algorithm (sort correlation edges,
  merge the top k via union-find) live, with a slider for k, and plots normalised prices
  for any resulting cluster

## Run it
```
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501
