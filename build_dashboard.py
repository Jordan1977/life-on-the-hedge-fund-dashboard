#!/usr/bin/env python3
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import yfinance as yf

# ============================================================
# GLOBAL CONFIG
# ============================================================
ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DATA_DIR = ROOT / "data"
DOCS.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

CFG = {
    "portfolio_name": "Life on the Hedge Fund",
    "benchmark": "QQQ",
    "inception": "2025-03-06",
    "risk_free_rate": 0.0450,
    "output_html": DOCS / "index.html",
    "output_snapshot": DATA_DIR / "dashboard_snapshot.json",
}

PALETTE = {
    "bg": "#06080d",
    "panel": "#0b1018",
    "card": "#0d1420",
    "border": "#1d2a3f",
    "text": "#dde7f3",
    "muted": "#91a4bf",
    "dim": "#5d708d",
    "green": "#21d07a",
    "red": "#f45b69",
    "blue": "#4d8dff",
}

# ============================================================
# HELPERS
# ============================================================
def fmt_currency(x: float, decimals: int = 0) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.{decimals}f}"

def fmt_pct(x: float, decimals: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.{decimals}f}%"

def pnl_color(x: float) -> str:
    return PALETTE["green"] if x >= 0 else PALETTE["red"]

# ============================================================
# DATA INGESTION
# ============================================================
def load_holdings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df["quantity"] = df["quantity"].astype(float)
    df["buy_price"] = df["buy_price"].astype(float)
    df["cost_basis"] = df["quantity"] * df["buy_price"]
    return df

def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        frames = []
        for t in tickers:
            if (t, "Close") in raw.columns:
                frames.append(raw[(t, "Close")].rename(t))
        close = pd.concat(frames, axis=1)
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index().ffill().dropna(how="all")
    return close

# ============================================================
# ANALYTICS
# ============================================================
def build_market_frame(prices: pd.DataFrame, holdings: pd.DataFrame) -> Dict:
    px = prices.copy()
    positions_mv = pd.DataFrame(index=px.index)
    
    for _, row in holdings.iterrows():
        positions_mv[row["ticker"]] = px[row["ticker"]] * row["quantity"]
    
    portfolio_nav = positions_mv.sum(axis=1)
    benchmark = CFG["benchmark"]
    
    init_nav = holdings["cost_basis"].sum()
    qqq_units = init_nav / px[benchmark].dropna().iloc[0]
    qqq_nav = px[benchmark] * qqq_units
    
    returns = portfolio_nav.pct_change().fillna(0)
    qqq_returns = qqq_nav.pct_change().fillna(0)
    
    cum = (1 + returns).cumprod()
    qqq_cum = (1 + qqq_returns).cumprod()
    
    return {
        "prices": px,
        "positions_mv": positions_mv,
        "portfolio_nav": portfolio_nav,
        "benchmark_nav": qqq_nav,
        "returns": returns,
        "benchmark_returns": qqq_returns,
        "base100_portfolio": cum * 100,
        "base100_benchmark": qqq_cum * 100,
        "drawdown": (portfolio_nav / portfolio_nav.cummax() - 1) * 100,
    }

def compute_metrics(frame: Dict, holdings: pd.DataFrame) -> Dict:
    r = frame["returns"]
    b = frame["benchmark_returns"]
    n = len(r)
    rf = CFG["risk_free_rate"] / 252
    
    total_return = float(frame["portfolio_nav"].iloc[-1] / frame["portfolio_nav"].iloc[0] - 1)
    ann_return = (1 + total_return) ** (252 / n) - 1 if n > 0 else 0
    
    vol = float(r.std() * math.sqrt(252))
    sharpe = float((ann_return - CFG["risk_free_rate"]) / vol) if vol else 0
    beta = float(r.cov(b) / b.var()) if b.var() else 0
    
    return {
        "current_nav": float(frame["portfolio_nav"].iloc[-1]),
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "beta": beta,
        "correlation": float(r.corr(b)),
        "max_drawdown": float((frame["portfolio_nav"] / frame["portfolio_nav"].cummax() - 1).min()),
    }

def compute_positions(frame: Dict, holdings: pd.DataFrame) -> pd.DataFrame:
    px = frame["prices"]
    nav = frame["portfolio_nav"].iloc[-1]
    rows = []
    
    for _, row in holdings.iterrows():
        ticker = row["ticker"]
        ser = px[ticker].dropna()
        latest_price = float(ser.iloc[-1])
        quantity = float(row["quantity"])
        market_value = latest_price * quantity
        cost_basis = float(row["cost_basis"])
        pnl = market_value - cost_basis
        ret = pnl / cost_basis if cost_basis else 0
        weight = market_value / nav
        
        rows.append({
            "ticker": ticker,
            "name": row["name"],
            "quantity": quantity,
            "buy_price": float(row["buy_price"]),
            "latest_price": latest_price,
            "market_value": market_value,
            "pnl": pnl,
            "return": ret,
            "weight": weight,
            "sector": row["sector"],
            "risk_bucket": row["risk_bucket"],
        })
    
    return pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)

# ============================================================
# HTML GENERATION
# ============================================================
def generate_html(holdings: pd.DataFrame, metrics: Dict, positions: pd.DataFrame, frame: Dict) -> str:
    updated_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    init_aum = holdings["cost_basis"].sum()
    
    positions_rows = ""
    for _, r in positions.iterrows():
        positions_rows += f"""
        <tr>
          <td class="mono">{r['ticker']}</td>
          <td>{r['name']}</td>
          <td class="num">{r['quantity']:,.0f}</td>
          <td class="num">${r['buy_price']:.2f}</td>
          <td class="num">${r['latest_price']:.2f}</td>
          <td class="num">${r['market_value']:,.0f}</td>
          <td class="num" style="color:{pnl_color(r['pnl'])}">${r['pnl']:,.0f}</td>
          <td class="num" style="color:{pnl_color(r['return'])}">{fmt_pct(r['return']*100)}</td>
          <td class="num">{r['weight']*100:.1f}%</td>
        </tr>
        """
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Life on the Hedge Fund — Portfolio Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />
<style>
:root {{
  --bg: {PALETTE['bg']};
  --panel: {PALETTE['panel']};
  --card: {PALETTE['card']};
  --border: {PALETTE['border']};
  --text: {PALETTE['text']};
  --muted: {PALETTE['muted']};
  --dim: {PALETTE['dim']};
  --green: {PALETTE['green']};
  --red: {PALETTE['red']};
  --blue: {PALETTE['blue']};
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: linear-gradient(180deg, #04070c 0%, #070b12 100%); color: var(--text); font-family: Inter, sans-serif; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 40px 20px; }}
h1 {{ font-size: 32px; letter-spacing: -.03em; margin: 0 0 10px; }}
.subtitle {{ color: var(--muted); margin-bottom: 30px; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
.kpi-card {{ background: rgba(13,20,32,.9); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
.kpi-label {{ font-size: 12px; text-transform: uppercase; color: var(--dim); margin-bottom: 10px; }}
.kpi-value {{ font-size: 28px; font-weight: 800; }}
.section {{ background: rgba(13,20,32,.8); border: 1px solid var(--border); border-radius: 12px; padding: 25px; margin-bottom: 20px; }}
.section h2 {{ margin: 0 0 20px; font-size: 20px; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{ text-align: left; padding: 12px; background: rgba(17,24,38,.5); border-bottom: 1px solid var(--border); font-size: 12px; color: var(--dim); text-transform: uppercase; }}
td {{ padding: 12px; border-bottom: 1px solid rgba(29,42,63,.5); }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; font-family: "JetBrains Mono", monospace; }}
.mono {{ font-family: "JetBrains Mono", monospace; }}
.footer {{ color: var(--muted); font-size: 12px; margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border); }}
</style>
</head>
<body>
<div class="container">
  <h1>Life on the Hedge Fund</h1>
  <p class="subtitle">Institutional Portfolio Analytics Terminal</p>
  
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Current NAV</div>
      <div class="kpi-value">{fmt_currency(metrics['current_nav'], 0)}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Total Return</div>
      <div class="kpi-value" style="color: {pnl_color(metrics['total_return'])}">{fmt_pct(metrics['total_return']*100)}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Annualized Return</div>
      <div class="kpi-value">{fmt_pct(metrics['annualized_return']*100)}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Volatility</div>
      <div class="kpi-value">{fmt_pct(metrics['annualized_volatility']*100)}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Sharpe Ratio</div>
      <div class="kpi-value">{metrics['sharpe_ratio']:.2f}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Beta vs QQQ</div>
      <div class="kpi-value">{metrics['beta']:.2f}x</div>
    </div>
  </div>
  
  <div class="section">
    <h2>Holdings Monitor</h2>
    <table>
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Name</th>
          <th>Qty</th>
          <th>Buy Price</th>
          <th>Current</th>
          <th>Market Value</th>
          <th>P&L</th>
          <th>Return</th>
          <th>Weight</th>
        </tr>
      </thead>
      <tbody>
        {positions_rows}
      </tbody>
    </table>
  </div>
  
  <div class="footer">
    Life on the Hedge Fund · Trinity College Dublin · Last updated: {updated_ts}
  </div>
</div>
</body>
</html>"""

# ============================================================
# MAIN
# ============================================================
def main():
    print("Building dashboard...")
    
    try:
        holdings = load_holdings(ROOT / "holdings.csv")
        print(f"Loaded {len(holdings)} holdings")
        
        tickers = holdings["ticker"].tolist() + ["QQQ"]
        prices = download_prices(tickers, CFG["inception"], pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
        print(f"Downloaded prices for {len(tickers)} tickers")
        
        frame = build_market_frame(prices, holdings)
        metrics = compute_metrics(frame, holdings)
        positions = compute_positions(frame, holdings)
        
        html = generate_html(holdings, metrics, positions, frame)
        CFG["output_html"].write_text(html, encoding="utf-8")
        print(f"Dashboard written to {CFG['output_html']}")
        
        snapshot = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }
        CFG["output_snapshot"].write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        print("Dashboard build completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
