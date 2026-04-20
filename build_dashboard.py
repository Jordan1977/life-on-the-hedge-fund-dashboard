#!/usr/bin/env python3
"""
Life on the Hedge Fund — build_dashboard.py
Institutional Portfolio Analytics Terminal
Trinity College Dublin · Investment Analysis

Architecture: Python is the single source of truth.
This script downloads prices → computes all metrics →
writes docs/index.html (fully static, no browser-side data fetching).

Fixed issues vs Copilot-modified version:
  1. download_prices: rewritten to handle all yfinance MultiIndex layouts
  2. pd.Timestamp.utcnow() replaced with datetime.now(timezone.utc)
  3. All other logic preserved exactly
"""
from __future__ import annotations

import json
import math
import os
import textwrap
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

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
    "subtitle": "Institutional Portfolio Analytics Terminal",
    "school": "Trinity College Dublin",
    "school_tag": "#1 in Ireland",
    "course_context": "Academic portfolio analytics project",
    "benchmark": "QQQ",
    "benchmark_secondary": "SPY",
    "inception": "2025-03-06",
    "aum_initial": None,
    "risk_free_rate": 0.0450,
    "annualisation_factor": 252,
    "rolling_window": 30,
    "lookback_buffer_days": 420,
    "mc_seed": 42,
    "mc_paths": 600,
    "news_items": 10,
    "output_html": DOCS / "index.html",
    "output_snapshot": DATA_DIR / "dashboard_snapshot.json",
}

PALETTE = {
    "bg": "#06080d",
    "panel": "#0b1018",
    "panel_2": "#111826",
    "card": "#0d1420",
    "card_2": "#111b2a",
    "border": "#1d2a3f",
    "border_2": "#2a3d5a",
    "text": "#dde7f3",
    "muted": "#91a4bf",
    "dim": "#5d708d",
    "grid": "#172232",
    "green": "#21d07a",
    "red": "#f45b69",
    "amber": "#ffbe55",
    "blue": "#4d8dff",
    "cyan": "#45d7ff",
    "purple": "#b085ff",
}

RISK_BUCKET_COLORS = {
    "CORE": PALETTE["blue"],
    "GROWTH": PALETTE["green"],
    "SPECULATIVE": PALETTE["amber"],
}

SECTOR_COLORS = {
    "AI / Semiconductors": "#4d8dff",
    "AI / Tech Platform": "#6ea0ff",
    "AI / Defence Tech": "#7d7cf7",
    "AI / AdTech": "#915df8",
    "AI / Voice": "#bf8dff",
    "Defense / Aerospace": "#6d78d6",
    "Space Economy": "#8b59d3",
    "Energy Transition": "#ffbe55",
    "Crypto Infrastructure": "#ff7d72",
    "Bitcoin Mining": "#db4b42",
    "Fintech / Retail": "#21d07a",
    "Mobility / Platform": "#45d7ff",
    "Social / AI Data": "#4fd3c4",
}

SCENARIOS = [
    {"name": "Broad market correction", "type": "benchmark", "shock": -0.10, "description": "10% QQQ drawdown mapped through portfolio beta."},
    {"name": "Growth selloff", "type": "bucket", "bucket": "GROWTH", "shock": -0.18, "description": "De-rating of high-duration growth exposures."},
    {"name": "Speculative risk-off", "type": "bucket", "bucket": "SPECULATIVE", "shock": -0.25, "description": "High-beta / retail / crypto / space unwind."},
    {"name": "AI valuation compression", "type": "theme", "theme_keywords": ["AI"], "shock": -0.17, "description": "Multiple compression across AI-linked holdings."},
    {"name": "Rates shock", "type": "custom", "mapping": {"GROWTH": -0.12, "SPECULATIVE": -0.16, "CORE": -0.06}, "description": "Higher real yields hit long-duration equities."},
    {"name": "Crypto crash", "type": "tickers", "tickers": ["COIN", "MARA", "HOOD"], "shock": -0.28, "description": "Crypto-beta sleeve reprices sharply lower."},
]


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


def fmt_x(x: float, decimals: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}x"


def pnl_color(x: float) -> str:
    return PALETTE["green"] if x >= 0 else PALETTE["red"]


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        return float(str(x).replace(",", ""))
    except Exception:
        return default


def annualized_return(total_return: float, n_periods: int, af: int = 252) -> float:
    if n_periods <= 0:
        return np.nan
    return (1 + total_return) ** (af / n_periods) - 1


def max_drawdown_from_returns(returns: pd.Series) -> float:
    wealth = (1 + returns.fillna(0)).cumprod()
    dd = wealth / wealth.cummax() - 1
    return float(dd.min())


def downside_deviation(returns: pd.Series, mar_daily: float = 0.0, af: int = 252) -> float:
    downside = np.minimum(returns - mar_daily, 0)
    return float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(af))


def omega_ratio(returns: pd.Series, mar_daily: float = 0.0) -> float:
    diff = returns - mar_daily
    gains = diff[diff > 0].sum()
    losses = -diff[diff < 0].sum()
    if losses == 0:
        return np.nan
    return float(gains / losses)


def treynor_ratio(ann_return: float, rf: float, beta: float) -> float:
    if pd.isna(beta) or abs(beta) < 1e-12:
        return np.nan
    return float((ann_return - rf) / beta)


def capture_ratio(port: pd.Series, bench: pd.Series, upside: bool = True) -> float:
    mask = bench > 0 if upside else bench < 0
    if mask.sum() == 0:
        return np.nan
    b = bench[mask].mean()
    if abs(b) < 1e-12:
        return np.nan
    return float(port[mask].mean() / b)


def html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ============================================================
# DATA INGESTION
# ============================================================
def load_holdings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {
        "ticker", "name", "quantity", "buy_price", "sector", "theme",
        "risk_bucket", "inception_date"
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"holdings.csv missing columns: {sorted(missing)}")
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df["quantity"] = df["quantity"].astype(float)
    df["buy_price"] = df["buy_price"].astype(float)
    df["cost_basis"] = df["quantity"] * df["buy_price"]
    return df


def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Robust yfinance downloader that handles all MultiIndex column layouts
    produced by different yfinance versions.

    yfinance >=0.2.x with multiple tickers returns a MultiIndex DataFrame.
    The column order can be either (field, ticker) or (ticker, field) depending
    on the version and the group_by parameter.  We normalise to a plain
    ticker → Close price DataFrame regardless.
    """
    end_buffer = (pd.Timestamp(end) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    # ── Download without group_by so yfinance uses its default layout ──────────
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end_buffer,
        auto_adjust=True,
        progress=False,
        threads=True,
        # Do NOT pass group_by="ticker" — it produces inconsistent MultiIndex
        # layouts across yfinance versions and is the root cause of the crash.
    )

    if raw.empty:
        raise RuntimeError("No market data returned by yfinance. Check your internet connection or try again.")

    # ── Normalise columns ───────────────────────────────────────────────────────
    if isinstance(raw.columns, pd.MultiIndex):
        # Determine the MultiIndex level order.
        # yfinance default (no group_by) gives (field, ticker).
        # We want the "Close" field for each ticker.
        level_values_0 = [str(v) for v in raw.columns.get_level_values(0)]
        level_values_1 = [str(v) for v in raw.columns.get_level_values(1)]

        frames = []

        if "Close" in level_values_0:
            # Layout: (field, ticker) — yfinance default for multiple tickers
            for t in tickers:
                if ("Close", t) in raw.columns:
                    frames.append(raw[("Close", t)].rename(t))
                else:
                    # Ticker might have slightly different case — try case-insensitive
                    matched = [c for c in raw.columns if c[0] == "Close" and str(c[1]).upper() == t.upper()]
                    if matched:
                        frames.append(raw[matched[0]].rename(t))

        elif "Close" in level_values_1:
            # Layout: (ticker, field) — produced by group_by="ticker"
            for t in tickers:
                col = (t, "Close")
                if col in raw.columns:
                    frames.append(raw[col].rename(t))
                else:
                    matched = [c for c in raw.columns if str(c[0]).upper() == t.upper() and c[1] == "Close"]
                    if matched:
                        frames.append(raw[matched[0]].rename(t))

        else:
            # Fallback: try to find any column containing "Close" for each ticker
            for t in tickers:
                for col in raw.columns:
                    if "close" in str(col).lower() and t.upper() in str(col).upper():
                        frames.append(raw[col].rename(t))
                        break

        if not frames:
            # Last resort: if only one ticker was downloaded, raw may be flat
            if "Close" in raw.columns:
                frames = [raw["Close"].rename(tickers[0])]
            else:
                raise RuntimeError(
                    f"Could not extract Close prices from yfinance response.\n"
                    f"Columns returned: {list(raw.columns)[:20]}"
                )

        close = pd.concat(frames, axis=1)

    else:
        # Single ticker or already-flat response
        if "Close" in raw.columns:
            close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            close = raw.copy()

    # ── Clean index ─────────────────────────────────────────────────────────────
    close.index = pd.to_datetime(close.index)
    if hasattr(close.index, "tz") and close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    close = close.sort_index().ffill(limit=5).dropna(how="all")

    # ── Validate: warn about any ticker with all-NaN data ──────────────────────
    for t in tickers:
        if t in close.columns and close[t].isna().all():
            print(f"  WARNING: all prices for {t} are NaN — check ticker symbol.")

    missing_tickers = [t for t in tickers if t not in close.columns]
    if missing_tickers:
        print(f"  WARNING: these tickers were not found in the response: {missing_tickers}")

    print(f"  Downloaded: {len(close)} sessions, {len(close.columns)} tickers")
    return close


# ============================================================
# ANALYTICS ENGINE
# ============================================================
def build_market_frame(prices: pd.DataFrame, holdings: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    px = prices.copy()
    positions_mv = pd.DataFrame(index=px.index)
    for _, row in holdings.iterrows():
        t = row["ticker"]
        if t not in px.columns:
            print(f"  WARNING: {t} not in price data — skipping in portfolio NAV")
            continue
        positions_mv[t] = px[t] * row["quantity"]

    portfolio_nav = positions_mv.sum(axis=1)
    benchmark = CFG["benchmark"]
    benchmark_secondary = CFG["benchmark_secondary"]

    init_nav = holdings["cost_basis"].sum()
    qqq_units = init_nav / px[benchmark].dropna().iloc[0]
    qqq_nav = px[benchmark] * qqq_units

    if benchmark_secondary in px.columns:
        spy_units = init_nav / px[benchmark_secondary].dropna().iloc[0]
        spy_nav = px[benchmark_secondary] * spy_units
    else:
        spy_nav = pd.Series(index=px.index, dtype=float)

    returns = portfolio_nav.pct_change().fillna(0)
    qqq_returns = qqq_nav.pct_change().fillna(0)
    spy_returns = spy_nav.pct_change().fillna(0) if not spy_nav.empty else pd.Series(index=px.index, dtype=float)

    cum = (1 + returns).cumprod()
    qqq_cum = (1 + qqq_returns).cumprod()
    spy_cum = (1 + spy_returns).cumprod() if not spy_returns.empty else pd.Series(index=px.index, dtype=float)

    frame = {
        "prices": px,
        "positions_mv": positions_mv,
        "portfolio_nav": portfolio_nav,
        "benchmark_nav": qqq_nav,
        "secondary_nav": spy_nav,
        "returns": returns,
        "benchmark_returns": qqq_returns,
        "secondary_returns": spy_returns,
        "base100_portfolio": cum * 100,
        "base100_benchmark": qqq_cum * 100,
        "base100_secondary": spy_cum * 100 if not spy_nav.empty else pd.Series(index=px.index, dtype=float),
        "drawdown": (portfolio_nav / portfolio_nav.cummax() - 1) * 100,
        "benchmark_drawdown": (qqq_nav / qqq_nav.cummax() - 1) * 100,
        "secondary_drawdown": (spy_nav / spy_nav.cummax() - 1) * 100 if not spy_nav.empty else pd.Series(index=px.index, dtype=float),
    }
    return frame


def compute_core_metrics(frame: Dict[str, pd.Series]) -> Dict[str, float]:
    af = CFG["annualisation_factor"]
    rf = CFG["risk_free_rate"]
    rfd = rf / af

    r = frame["returns"]
    b = frame["benchmark_returns"]
    n = len(r)

    total_return = float(frame["portfolio_nav"].iloc[-1] / frame["portfolio_nav"].iloc[0] - 1)
    benchmark_total_return = float(frame["benchmark_nav"].iloc[-1] / frame["benchmark_nav"].iloc[0] - 1)
    ann_return = annualized_return(total_return, n, af)
    benchmark_ann_return = annualized_return(benchmark_total_return, n, af)

    vol = float(r.std() * math.sqrt(af))
    bench_vol = float(b.std() * math.sqrt(af))
    down_dev = downside_deviation(r, mar_daily=rfd, af=af)
    bench_down_dev = downside_deviation(b, mar_daily=rfd, af=af)

    sharpe = float((ann_return - rf) / vol) if vol else np.nan
    sortino = float((ann_return - rf) / down_dev) if down_dev else np.nan
    benchmark_sharpe = float((benchmark_ann_return - rf) / bench_vol) if bench_vol else np.nan
    benchmark_sortino = float((benchmark_ann_return - rf) / bench_down_dev) if bench_down_dev else np.nan

    beta = float(r.cov(b) / b.var()) if b.var() else np.nan
    corr = float(r.corr(b))
    jensen_alpha = float(ann_return - (rf + beta * (benchmark_ann_return - rf))) if not pd.isna(beta) else np.nan

    active = r - b
    tracking_error = float(active.std() * math.sqrt(af))
    information_ratio = float((active.mean() * af) / tracking_error) if tracking_error else np.nan

    max_dd = max_drawdown_from_returns(r)
    bench_max_dd = max_drawdown_from_returns(b)
    calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else np.nan

    var_95 = float(np.percentile(r, 5))
    cvar_95 = float(r[r <= var_95].mean())
    skew = float(r.skew())
    kurt = float(r.kurtosis())
    upside_capture = capture_ratio(r, b, True)
    downside_capture = capture_ratio(r, b, False)
    treynor = treynor_ratio(ann_return, rf, beta)
    omega = omega_ratio(r, mar_daily=rfd)

    daily_pnl = float(frame["portfolio_nav"].iloc[-1] - frame["portfolio_nav"].iloc[-2])
    daily_return = float(r.iloc[-1])
    total_pnl = float(frame["portfolio_nav"].iloc[-1] - frame["portfolio_nav"].iloc[0])
    hit_ratio = float((r > 0).mean())
    benchmark_hit_ratio = float((b > 0).mean())
    simple_alpha = total_return - benchmark_total_return

    rolling_window = CFG["rolling_window"]
    rolling_vol = r.rolling(rolling_window).std() * math.sqrt(af) * 100
    rolling_beta = r.rolling(rolling_window).cov(b) / b.rolling(rolling_window).var()
    rolling_sharpe = ((r.rolling(rolling_window).mean() - rfd) / r.rolling(rolling_window).std()) * math.sqrt(af)
    rolling_excess = ((1 + active).rolling(rolling_window).apply(np.prod, raw=True) - 1) * 100

    monthly_port = (1 + r).resample("ME").prod() - 1
    monthly_bench = (1 + b).resample("ME").prod() - 1
    annual_port = (1 + r).resample("YE").prod() - 1
    annual_bench = (1 + b).resample("YE").prod() - 1

    return {
        "current_nav": float(frame["portfolio_nav"].iloc[-1]),
        "daily_pnl": daily_pnl,
        "daily_return": daily_return,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "benchmark_total_return": benchmark_total_return,
        "alpha_vs_benchmark": simple_alpha,
        "annualized_return": ann_return,
        "benchmark_annualized_return": benchmark_ann_return,
        "annualized_volatility": vol,
        "benchmark_annualized_volatility": bench_vol,
        "downside_deviation": down_dev,
        "benchmark_downside_deviation": bench_down_dev,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "benchmark_sharpe_ratio": benchmark_sharpe,
        "benchmark_sortino_ratio": benchmark_sortino,
        "beta": beta,
        "correlation": corr,
        "jensen_alpha": jensen_alpha,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "max_drawdown": max_dd,
        "benchmark_max_drawdown": bench_max_dd,
        "calmar_ratio": calmar,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "skewness": skew,
        "kurtosis": kurt,
        "upside_capture": upside_capture,
        "downside_capture": downside_capture,
        "treynor_ratio": treynor,
        "omega_ratio": omega,
        "hit_ratio": hit_ratio,
        "benchmark_hit_ratio": benchmark_hit_ratio,
        "sessions": n,
        "rolling_volatility": rolling_vol,
        "rolling_beta": rolling_beta,
        "rolling_sharpe": rolling_sharpe,
        "rolling_excess_return": rolling_excess,
        "monthly_portfolio": monthly_port,
        "monthly_benchmark": monthly_bench,
        "annual_portfolio": annual_port,
        "annual_benchmark": annual_bench,
    }


def compute_position_analytics(frame: Dict, holdings: pd.DataFrame) -> pd.DataFrame:
    px = frame["prices"]
    benchmark_returns = frame["benchmark_returns"]
    nav = frame["portfolio_nav"].iloc[-1]
    rows = []
    for _, row in holdings.iterrows():
        ticker = row["ticker"]
        if ticker not in px.columns:
            continue
        ser = px[ticker].dropna()
        returns = ser.pct_change().dropna()
        latest_price = float(ser.iloc[-1])
        prev_price = float(ser.iloc[-2]) if len(ser) >= 2 else latest_price
        quantity = float(row["quantity"])
        market_value = latest_price * quantity
        cost_basis = float(row["cost_basis"])
        pnl = market_value - cost_basis
        ret = pnl / cost_basis
        weight = market_value / nav
        contribution = pnl / holdings["cost_basis"].sum()

        def trailing(days: int):
            if len(ser) <= days:
                return np.nan
            return float(ser.iloc[-1] / ser.iloc[-days - 1] - 1)

        aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
        aligned.columns = ["r", "b"]
        beta = float(aligned["r"].cov(aligned["b"]) / aligned["b"].var()) if len(aligned) > 10 and aligned["b"].var() else np.nan

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
            "contribution": contribution,
            "perf_1d": latest_price / prev_price - 1,
            "perf_5d": trailing(5),
            "perf_1m": trailing(21),
            "perf_inception": ret,
            "beta_vs_benchmark": beta,
            "sector": row["sector"],
            "theme": row["theme"],
            "risk_bucket": row["risk_bucket"],
        })
    pos = pd.DataFrame(rows).sort_values(["weight", "market_value"], ascending=[False, False]).reset_index(drop=True)
    return pos


def compute_structure_analytics(pos: pd.DataFrame, frame: Dict, metrics: Dict) -> Dict[str, pd.DataFrame]:
    sector = pos.groupby("sector", as_index=False).agg(
        weight=("weight", "sum"),
        market_value=("market_value", "sum"),
        pnl=("pnl", "sum"),
        positions=("ticker", "count"),
    ).sort_values("weight", ascending=False)
    theme = pos.groupby("theme", as_index=False).agg(
        weight=("weight", "sum"),
        market_value=("market_value", "sum"),
        pnl=("pnl", "sum"),
        positions=("ticker", "count"),
    ).sort_values("weight", ascending=False)
    bucket = pos.groupby("risk_bucket", as_index=False).agg(
        weight=("weight", "sum"),
        market_value=("market_value", "sum"),
        pnl=("pnl", "sum"),
        positions=("ticker", "count"),
    ).sort_values("weight", ascending=False)

    hhi = float((pos["weight"] ** 2).sum())
    eff_n = float(1 / hhi) if hhi else np.nan
    top5 = float(pos["weight"].head(5).sum())

    daily_corr = frame["positions_mv"].pct_change().corr()
    corr_lens = daily_corr.where(np.triu(np.ones(daily_corr.shape), k=1).astype(bool)).stack().sort_values(ascending=False)

    return {
        "sector": sector,
        "theme": theme,
        "bucket": bucket,
        "hhi": hhi,
        "effective_n": eff_n,
        "top5_weight": top5,
        "position_corr": daily_corr,
        "most_correlated_pairs": corr_lens.head(5).reset_index(name="corr").rename(columns={"level_0": "ticker_1", "level_1": "ticker_2"}),
    }


def build_daily_ledger(frame: Dict) -> pd.DataFrame:
    nav = frame["portfolio_nav"]
    ledger = pd.DataFrame({
        "date": nav.index,
        "nav": nav.values,
        "daily_pnl": nav.diff().fillna(0).values,
        "daily_return": frame["returns"].values,
        "benchmark_return": frame["benchmark_returns"].values,
        "active_return": (frame["returns"] - frame["benchmark_returns"]).values,
        "drawdown": frame["drawdown"].values / 100,
    })
    return ledger.tail(40).iloc[::-1].reset_index(drop=True)


def build_return_heatmap(monthly: pd.Series) -> pd.DataFrame:
    heat = monthly.copy().to_frame("return")
    heat["year"] = heat.index.year
    heat["month"] = heat.index.strftime("%b")
    pivot = heat.pivot(index="year", columns="month", values="return")
    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=[m for m in months_order if m in pivot.columns])
    return pivot


def build_stress_tests(pos: pd.DataFrame, metrics: Dict) -> pd.DataFrame:
    rows = []
    portfolio_nav = pos["market_value"].sum()
    beta = metrics["beta"]
    for scenario in SCENARIOS:
        shocked = pos[["ticker", "market_value", "risk_bucket", "theme"]].copy()
        shocked["shock"] = 0.0
        if scenario["type"] == "benchmark":
            shocked["shock"] = beta * scenario["shock"]
        elif scenario["type"] == "bucket":
            shocked.loc[shocked["risk_bucket"] == scenario["bucket"], "shock"] = scenario["shock"]
        elif scenario["type"] == "theme":
            mask = shocked["theme"].str.contains("|".join(scenario["theme_keywords"]), case=False, na=False)
            shocked.loc[mask, "shock"] = scenario["shock"]
        elif scenario["type"] == "tickers":
            shocked.loc[shocked["ticker"].isin(scenario["tickers"]), "shock"] = scenario["shock"]
        elif scenario["type"] == "custom":
            shocked["shock"] = shocked["risk_bucket"].map(scenario["mapping"]).fillna(0.0)
        shocked["pnl_impact"] = shocked["market_value"] * shocked["shock"]
        total_pnl = shocked["pnl_impact"].sum()
        rows.append({
            "scenario": scenario["name"],
            "description": scenario["description"],
            "estimated_pnl_impact": total_pnl,
            "estimated_nav_after": portfolio_nav + total_pnl,
            "estimated_return_impact": total_pnl / portfolio_nav,
        })
    return pd.DataFrame(rows).sort_values("estimated_pnl_impact")


def build_forecast(frame: Dict, metrics: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(CFG["mc_seed"])
    r = frame["returns"].dropna()
    b = frame["benchmark_returns"].dropna()
    mu = float(r.mean())
    sigma = float(r.std())
    bmu = float(b.mean())
    bsigma = float(b.std())
    start_nav = float(frame["portfolio_nav"].iloc[-1])
    horizons = {"3M": 63, "6M": 126, "12M": 252, "15Y": 252 * 15}

    summary_rows = []
    path_rows = []
    for label, h in horizons.items():
        shocks = np.random.normal(mu, sigma, size=(CFG["mc_paths"], h))
        wealth = start_nav * np.cumprod(1 + shocks, axis=1)
        ending = wealth[:, -1]
        summary_rows.append({
            "horizon": label,
            "start_nav": start_nav,
            "p05": np.percentile(ending, 5),
            "p25": np.percentile(ending, 25),
            "median": np.percentile(ending, 50),
            "p75": np.percentile(ending, 75),
            "p95": np.percentile(ending, 95),
        })

        horizon_steps = list(range(h + 1))
        bull = start_nav * np.cumprod(np.r_[1, np.repeat(bmu + 0.75 * bsigma, h)])
        base = start_nav * np.cumprod(np.r_[1, np.repeat(mu, h)])
        bear = start_nav * np.cumprod(np.r_[1, np.repeat(mu - 0.75 * sigma, h)])
        p05_path = np.percentile(np.c_[np.repeat(start_nav, CFG["mc_paths"]).reshape(-1, 1), wealth], 5, axis=0)
        p95_path = np.percentile(np.c_[np.repeat(start_nav, CFG["mc_paths"]).reshape(-1, 1), wealth], 95, axis=0)
        for step, bull_v, base_v, bear_v, low_v, high_v in zip(horizon_steps, bull, base, bear, p05_path, p95_path):
            path_rows.append({
                "horizon": label,
                "step": step,
                "bull": bull_v,
                "base": base_v,
                "bear": bear_v,
                "mc_low": low_v,
                "mc_high": high_v,
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(path_rows)


def build_news(holdings: pd.DataFrame) -> pd.DataFrame:
    items = []
    for ticker in holdings["ticker"].tolist():
        try:
            tk = yf.Ticker(ticker)
            news = getattr(tk, "news", None) or []
            for article in news[:6]:
                content = article.get("content", {}) if isinstance(article, dict) else {}
                title = content.get("title") or article.get("title")
                url = content.get("canonicalUrl", {}).get("url") or article.get("link") or article.get("url")
                provider = content.get("provider", {}).get("displayName") or article.get("publisher") or "Yahoo Finance"
                pub = content.get("pubDate") or article.get("providerPublishTime")
                if isinstance(pub, str):
                    try:
                        pub = pd.to_datetime(pub, utc=True)
                    except Exception:
                        pub = pd.NaT
                elif pub is not None:
                    pub = pd.to_datetime(pub, unit="s", utc=True, errors="coerce")
                else:
                    pub = pd.NaT
                if title and url:
                    items.append({
                        "ticker": ticker,
                        "title": title,
                        "source": provider,
                        "published_at": pub,
                        "url": url,
                    })
        except Exception:
            continue
    if not items:
        return pd.DataFrame(columns=["ticker", "title", "source", "published_at", "url"])

    news_df = pd.DataFrame(items)
    news_df["published_at"] = pd.to_datetime(news_df["published_at"], utc=True, errors="coerce")
    news_df = news_df.sort_values("published_at", ascending=False)
    news_df["norm_title"] = news_df["title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    news_df = news_df.drop_duplicates(subset=["norm_title"]).drop(columns=["norm_title"])
    return news_df.head(CFG["news_items"]).reset_index(drop=True)


def build_intelligence(metrics: Dict, pos: pd.DataFrame, structure: Dict, stress: pd.DataFrame) -> List[Tuple[str, str]]:
    top = pos.nlargest(3, "contribution")
    bottom = pos.nsmallest(3, "contribution")
    dominant_sector = structure["sector"].iloc[0]
    dominant_bucket = structure["bucket"].sort_values("weight", ascending=False).iloc[0]
    worst_stress = stress.iloc[0]

    dna = (
        f"The portfolio is unmistakably a concentrated US equity book with a high-conviction, thematic growth bias. "
        f"Top five positions represent {structure['top5_weight']*100:.1f}% of NAV, while HHI sits at {structure['hhi']:.3f} "
        f"for an effective position count of {structure['effective_n']:.1f}."
    )
    performance = (
        f"Performance has been driven mainly by {', '.join(top['ticker'].tolist())}, with cumulative contribution led by "
        f"{top.iloc[0]['ticker']} at {top.iloc[0]['contribution']*100:+.1f}% of initial capital. "
        f"Main detractors have been {', '.join(bottom['ticker'].tolist())}."
    )
    risk = (
        f"Portfolio beta versus {CFG['benchmark']} is {metrics['beta']:.2f}x and annualized volatility is {metrics['annualized_volatility']*100:.1f}%. "
        f"This is deliberate high-beta positioning rather than closet indexing. The harshest modelled shock is '{worst_stress['scenario']}', "
        f"with an estimated one-step NAV hit of {worst_stress['estimated_return_impact']*100:.1f}%."
    )
    benchmark = (
        f"Alpha versus {CFG['benchmark']} stands at {metrics['alpha_vs_benchmark']*100:+.1f}% since inception, with an information ratio of "
        f"{metrics['information_ratio']:.2f}. Benchmark comparison is intentionally growth-centric because the opportunity cost is Nasdaq-heavy, not broad-market defensive equity exposure."
    )
    structure_comment = (
        f"The largest exposure today is {dominant_sector['sector']} at {dominant_sector['weight']*100:.1f}% of NAV. "
        f"Risk bucket mix is led by {dominant_bucket['risk_bucket']} at {dominant_bucket['weight']*100:.1f}% of NAV, consistent with the project brief: long-duration themes, winner concentration, and no rebalancing."
    )

    return [
        ("Portfolio DNA", dna),
        ("What worked", performance),
        ("Risk lens", risk),
        ("Benchmark lens", benchmark),
        ("Concentration lens", structure_comment),
    ]


# ============================================================
# CHARTS
# ============================================================
def plot_layout(title: str, height: int = 360) -> dict:
    return dict(
        title={"text": title, "x": 0.01, "font": {"size": 14, "color": PALETTE["text"]}},
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["panel"],
        margin={"l": 55, "r": 20, "t": 55, "b": 45},
        font={"family": "JetBrains Mono, monospace", "size": 11, "color": PALETTE["text"]},
        xaxis={"gridcolor": PALETTE["grid"], "zeroline": False, "showline": False, "tickfont": {"color": PALETTE["muted"]}},
        yaxis={"gridcolor": PALETTE["grid"], "zeroline": False, "showline": False, "tickfont": {"color": PALETTE["muted"]}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.01, "bgcolor": PALETTE["card"]},
        hoverlabel={"bgcolor": PALETTE["card"], "bordercolor": PALETTE["border"], "font": {"color": PALETTE["text"]}},
        hovermode="x unified",
        height=height,
    )


def make_charts(frame: Dict, metrics: Dict, pos: pd.DataFrame, structure: Dict, heatmap: pd.DataFrame, stress: pd.DataFrame, forecast_paths: pd.DataFrame) -> Dict[str, dict]:
    charts = {}

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["base100_portfolio"].index, y=frame["base100_portfolio"], name=CFG["portfolio_name"], mode="lines", line={"color": PALETTE["green"], "width": 2.5}))
    fig.add_trace(go.Scatter(x=frame["base100_benchmark"].index, y=frame["base100_benchmark"], name=CFG["benchmark"], mode="lines", line={"color": PALETTE["blue"], "width": 2}))
    if not frame["secondary_nav"].empty:
        fig.add_trace(go.Scatter(x=frame["base100_secondary"].index, y=frame["base100_secondary"], name=CFG["benchmark_secondary"], mode="lines", line={"color": PALETTE["purple"], "width": 1.6, "dash": "dot"}))
    fig.update_layout(**plot_layout("Portfolio vs Benchmark — Base 100", 400))
    charts["perf"] = fig.to_plotly_json()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["drawdown"].index, y=frame["drawdown"], name=CFG["portfolio_name"], fill="tozeroy", line={"color": PALETTE["red"], "width": 2}))
    fig.add_trace(go.Scatter(x=frame["benchmark_drawdown"].index, y=frame["benchmark_drawdown"], name=CFG["benchmark"], line={"color": PALETTE["blue"], "width": 1.8}))
    fig.update_layout(**plot_layout("Drawdown from Peak (%)"))
    charts["drawdown"] = fig.to_plotly_json()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics["rolling_volatility"].index, y=metrics["rolling_volatility"], name="Portfolio", line={"color": PALETTE["amber"], "width": 2}))
    fig.update_layout(**plot_layout(f"Rolling {CFG['rolling_window']}-Day Volatility (Ann. %)"))
    charts["rolling_vol"] = fig.to_plotly_json()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics["rolling_beta"].index, y=metrics["rolling_beta"], name="Rolling beta", line={"color": PALETTE["cyan"], "width": 2}))
    fig.add_hline(y=1.0, line={"color": PALETTE["dim"], "dash": "dot"})
    fig.update_layout(**plot_layout(f"Rolling {CFG['rolling_window']}-Day Beta vs {CFG['benchmark']}"))
    charts["rolling_beta"] = fig.to_plotly_json()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics["rolling_sharpe"].index, y=metrics["rolling_sharpe"], name="Rolling Sharpe", line={"color": PALETTE["green"], "width": 2}))
    fig.update_layout(**plot_layout(f"Rolling {CFG['rolling_window']}-Day Sharpe"))
    charts["rolling_sharpe"] = fig.to_plotly_json()

    month_df = pd.DataFrame({
        "date": metrics["monthly_portfolio"].index,
        "portfolio": metrics["monthly_portfolio"].values * 100,
        "benchmark": metrics["monthly_benchmark"].values * 100,
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(x=month_df["date"], y=month_df["portfolio"], name=CFG["portfolio_name"], marker_color=np.where(month_df["portfolio"] >= 0, PALETTE["green"], PALETTE["red"])))
    fig.add_trace(go.Scatter(x=month_df["date"], y=month_df["benchmark"], name=CFG["benchmark"], line={"color": PALETTE["blue"], "width": 2}))
    fig.update_layout(**plot_layout("Monthly Returns (%)"))
    charts["monthly_returns"] = fig.to_plotly_json()

    fig = go.Figure(data=go.Heatmap(
        z=(heatmap * 100).values,
        x=list(heatmap.columns),
        y=list(map(str, heatmap.index)),
        colorscale=[[0, PALETTE["red"]], [0.5, PALETTE["panel_2"]], [1, PALETTE["green"]]],
        text=np.vectorize(lambda x: "" if pd.isna(x) else f"{x*100:+.1f}%")(heatmap.values),
        texttemplate="%{text}",
        hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(**plot_layout("Monthly Return Heatmap", 300))
    charts["heatmap"] = fig.to_plotly_json()

    fig = go.Figure(data=[go.Bar(
        x=pos["weight"].iloc[:10] * 100,
        y=pos["ticker"].iloc[:10],
        orientation="h",
        marker_color=[RISK_BUCKET_COLORS.get(x, PALETTE["blue"]) for x in pos["risk_bucket"].iloc[:10]],
        text=[f"{w*100:.1f}%" for w in pos["weight"].iloc[:10]],
        textposition="auto",
    )])
    fig.update_layout(**plot_layout("Top Weights (%)"))
    fig.update_yaxes(autorange="reversed")
    charts["top_weights"] = fig.to_plotly_json()

    fig = go.Figure(data=[go.Pie(
        labels=structure["sector"]["sector"],
        values=structure["sector"]["weight"] * 100,
        hole=0.55,
        marker={"colors": [SECTOR_COLORS.get(s, PALETTE["blue"]) for s in structure["sector"]["sector"]]},
        sort=False,
    )])
    fig.update_layout(**plot_layout("Sector Allocation (%)", 360))
    charts["sector_alloc"] = fig.to_plotly_json()

    fig = go.Figure(data=[go.Pie(
        labels=structure["theme"]["theme"],
        values=structure["theme"]["weight"] * 100,
        hole=0.55,
        marker={"colors": [PALETTE["blue"], PALETTE["purple"], PALETTE["green"], PALETTE["amber"], PALETTE["cyan"], "#8f9fb5", "#6f80a0", "#b0b9c8", "#d0d6df", "#5c8bd8", "#7db0ff", "#2ecf91", "#ff9e66"]},
        sort=False,
    )])
    fig.update_layout(**plot_layout("Thematic Allocation (%)", 360))
    charts["theme_alloc"] = fig.to_plotly_json()

    fig = go.Figure(data=[go.Bar(
        x=pos["ticker"],
        y=pos["pnl"],
        marker_color=np.where(pos["pnl"] >= 0, PALETTE["green"], PALETTE["red"]),
        text=[fmt_currency(x, 0) for x in pos["pnl"]],
        textposition="outside",
    )])
    fig.update_layout(**plot_layout("Position P&L Attribution ($)"))
    charts["pnl_attr"] = fig.to_plotly_json()

    fig = go.Figure(data=[go.Bar(
        x=stress["estimated_pnl_impact"],
        y=stress["scenario"],
        orientation="h",
        marker_color=np.where(stress["estimated_pnl_impact"] >= 0, PALETTE["green"], PALETTE["red"]),
        text=[fmt_currency(x, 0) for x in stress["estimated_pnl_impact"]],
        textposition="auto",
    )])
    fig.update_layout(**plot_layout("Stress Testing — Estimated P&L Impact", 360))
    fig.update_yaxes(autorange="reversed")
    charts["stress"] = fig.to_plotly_json()

    fp = forecast_paths[forecast_paths["horizon"] == "12M"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fp["step"], y=fp["mc_high"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fp["step"], y=fp["mc_low"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(77,141,255,0.15)", name="Monte Carlo 5-95%"))
    fig.add_trace(go.Scatter(x=fp["step"], y=fp["bull"], name="Bull", line={"color": PALETTE["green"], "width": 2}))
    fig.add_trace(go.Scatter(x=fp["step"], y=fp["base"], name="Base", line={"color": PALETTE["blue"], "width": 2}))
    fig.add_trace(go.Scatter(x=fp["step"], y=fp["bear"], name="Bear", line={"color": PALETTE["red"], "width": 2}))
    fig.update_layout(**plot_layout("12M Scenario Envelope — Model-Based Paths", 360))
    charts["forecast"] = fig.to_plotly_json()

    return charts


# ============================================================
# HTML BUILDERS
# ============================================================
def kpi_card(label: str, value: str, sub: str = "", tone: str = "blue") -> str:
    return f"""
    <div class="kpi-card {tone}">
      <div class="kpi-label">{html_escape(label)}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """


def build_positions_table(pos: pd.DataFrame) -> str:
    rows = []
    for _, r in pos.iterrows():
        rows.append(f"""
        <tr>
          <td class="mono strong">{r['ticker']}</td>
          <td>{html_escape(r['name'])}</td>
          <td>{html_escape(r['sector'])}</td>
          <td>{html_escape(r['theme'])}</td>
          <td><span class="bucket {r['risk_bucket'].lower()}">{html_escape(r['risk_bucket'])}</span></td>
          <td class="num">{r['quantity']:,.0f}</td>
          <td class="num">{fmt_currency(r['buy_price'],2)}</td>
          <td class="num">{fmt_currency(r['latest_price'],2)}</td>
          <td class="num">{fmt_currency(r['market_value'],0)}</td>
          <td class="num" style="color:{pnl_color(r['pnl'])}">{fmt_currency(r['pnl'],0)}</td>
          <td class="num" style="color:{pnl_color(r['return'])}">{fmt_pct(r['return']*100,1)}</td>
          <td class="num">{r['weight']*100:.1f}%</td>
          <td class="num">{r['contribution']*100:+.1f}%</td>
          <td class="num" style="color:{pnl_color(r['perf_1d'])}">{fmt_pct(r['perf_1d']*100,1)}</td>
          <td class="num" style="color:{pnl_color(r['perf_5d'])}">{fmt_pct(r['perf_5d']*100,1)}</td>
          <td class="num" style="color:{pnl_color(r['perf_1m'])}">{fmt_pct(r['perf_1m']*100,1)}</td>
          <td class="num">{r['beta_vs_benchmark']:.2f}</td>
        </tr>
        """)
    return "\n".join(rows)


def build_metrics_table(metrics: Dict) -> str:
    rows = [
        ("Current NAV", fmt_currency(metrics["current_nav"], 0), "Current marked-to-market net asset value."),
        ("Total P&L", fmt_currency(metrics["total_pnl"], 0), "Absolute profit and loss since inception."),
        ("Total return", fmt_pct(metrics["total_return"]*100), "Portfolio return since inception."),
        ("Daily P&L", fmt_currency(metrics["daily_pnl"], 0), "Latest session marked-to-market change."),
        ("Daily return", fmt_pct(metrics["daily_return"]*100), "Latest session return."),
        ("Alpha vs benchmark", fmt_pct(metrics["alpha_vs_benchmark"]*100), "Simple excess return versus QQQ."),
        ("Beta vs benchmark", f"{metrics['beta']:.2f}", "CAPM sensitivity versus QQQ."),
        ("Sharpe ratio", f"{metrics['sharpe_ratio']:.2f}", "Annualized excess return per unit of volatility."),
        ("Sortino ratio", f"{metrics['sortino_ratio']:.2f}", "Annualized excess return per unit of downside deviation."),
        ("Max drawdown", fmt_pct(metrics['max_drawdown']*100), "Largest peak-to-trough portfolio drawdown."),
        ("Annualized volatility", fmt_pct(metrics['annualized_volatility']*100), "Realized annualized volatility."),
        ("Information ratio", f"{metrics['information_ratio']:.2f}", "Annualized active return divided by tracking error."),
        ("Tracking error", fmt_pct(metrics['tracking_error']*100), "Annualized standard deviation of active returns."),
        ("Hit ratio", fmt_pct(metrics['hit_ratio']*100), "Share of positive return days."),
        ("Annualized return", fmt_pct(metrics['annualized_return']*100), "Compounded annualized return."),
        ("Downside deviation", fmt_pct(metrics['downside_deviation']*100), "Annualized downside deviation with rf as MAR."),
        ("Calmar ratio", f"{metrics['calmar_ratio']:.2f}", "Annualized return divided by absolute max drawdown."),
        ("VaR 95% (1D)", fmt_pct(metrics['var_95']*100), "Historical daily 5th percentile return."),
        ("CVaR 95% (1D)", fmt_pct(metrics['cvar_95']*100), "Expected shortfall conditional on the left tail."),
        ("Skewness", f"{metrics['skewness']:.2f}", "Return distribution asymmetry."),
        ("Kurtosis", f"{metrics['kurtosis']:.2f}", "Excess kurtosis of daily returns."),
        ("Upside capture", fmt_x(metrics['upside_capture']), "Mean capture versus benchmark up days."),
        ("Downside capture", fmt_x(metrics['downside_capture']), "Mean capture versus benchmark down days."),
        ("Jensen alpha", fmt_pct(metrics['jensen_alpha']*100), "CAPM alpha using displayed risk-free rate."),
        ("Treynor ratio", f"{metrics['treynor_ratio']:.2f}", "Annualized excess return per unit of beta."),
        ("Omega ratio", f"{metrics['omega_ratio']:.2f}", "Gain/loss ratio above the daily rf threshold."),
    ]
    html_rows = []
    for label, value, note in rows:
        html_rows.append(f"<tr><td>{label}</td><td class='num strong'>{value}</td><td>{note}</td></tr>")
    return "\n".join(html_rows)


def build_monthly_annual_table(metrics: Dict) -> str:
    month = pd.DataFrame({
        "Month": metrics["monthly_portfolio"].index.strftime("%Y-%m"),
        "Portfolio": metrics["monthly_portfolio"].values,
        "QQQ": metrics["monthly_benchmark"].values,
        "Active": (metrics["monthly_portfolio"] - metrics["monthly_benchmark"]).values,
    }).tail(12).iloc[::-1]
    year = pd.DataFrame({
        "Year": metrics["annual_portfolio"].index.year.astype(str),
        "Portfolio": metrics["annual_portfolio"].values,
        "QQQ": metrics["annual_benchmark"].values,
        "Active": (metrics["annual_portfolio"] - metrics["annual_benchmark"]).values,
    }).iloc[::-1]
    parts = ["<div class='split-table'>"]
    parts.append("<div><div class='mini-title'>Monthly returns</div><table class='data-table'><thead><tr><th>Month</th><th>Portfolio</th><th>QQQ</th><th>Active</th></tr></thead><tbody>")
    for _, r in month.iterrows():
        parts.append(f"<tr><td>{r['Month']}</td><td class='num' style='color:{pnl_color(r['Portfolio'])}'>{fmt_pct(r['Portfolio']*100,1)}</td><td class='num'>{fmt_pct(r['QQQ']*100,1)}</td><td class='num' style='color:{pnl_color(r['Active'])}'>{fmt_pct(r['Active']*100,1)}</td></tr>")
    parts.append("</tbody></table></div>")
    parts.append("<div><div class='mini-title'>Annual returns</div><table class='data-table'><thead><tr><th>Year</th><th>Portfolio</th><th>QQQ</th><th>Active</th></tr></thead><tbody>")
    for _, r in year.iterrows():
        parts.append(f"<tr><td>{r['Year']}</td><td class='num' style='color:{pnl_color(r['Portfolio'])}'>{fmt_pct(r['Portfolio']*100,1)}</td><td class='num'>{fmt_pct(r['QQQ']*100,1)}</td><td class='num' style='color:{pnl_color(r['Active'])}'>{fmt_pct(r['Active']*100,1)}</td></tr>")
    parts.append("</tbody></table></div></div>")
    return "".join(parts)


def build_structure_table(structure: Dict) -> str:
    rows = []
    for _, r in structure["sector"].iterrows():
        rows.append(f"<tr><td>{html_escape(r['sector'])}</td><td class='num'>{r['weight']*100:.1f}%</td><td class='num'>{fmt_currency(r['market_value'],0)}</td><td class='num' style='color:{pnl_color(r['pnl'])}'>{fmt_currency(r['pnl'],0)}</td><td class='num'>{int(r['positions'])}</td></tr>")
    return "\n".join(rows)


def build_daily_ledger_table(ledger: pd.DataFrame) -> str:
    rows = []
    for _, r in ledger.iterrows():
        rows.append(
            f"<tr><td>{pd.to_datetime(r['date']).strftime('%Y-%m-%d')}</td>"
            f"<td class='num'>{fmt_currency(r['nav'],0)}</td>"
            f"<td class='num' style='color:{pnl_color(r['daily_pnl'])}'>{fmt_currency(r['daily_pnl'],0)}</td>"
            f"<td class='num' style='color:{pnl_color(r['daily_return'])}'>{fmt_pct(r['daily_return']*100,2)}</td>"
            f"<td class='num'>{fmt_pct(r['benchmark_return']*100,2)}</td>"
            f"<td class='num' style='color:{pnl_color(r['active_return'])}'>{fmt_pct(r['active_return']*100,2)}</td>"
            f"<td class='num' style='color:{pnl_color(r['drawdown'])}'>{fmt_pct(r['drawdown']*100,2)}</td></tr>"
        )
    return "\n".join(rows)


def build_news_table(news: pd.DataFrame) -> str:
    if news.empty:
        return "<div class='empty-state'>News retrieval failed or returned no items. Core analytics remain fully available because news is non-blocking by design.</div>"
    rows = []
    for _, r in news.iterrows():
        published = pd.to_datetime(r["published_at"], utc=True, errors="coerce")
        published_text = published.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(published) else "Unknown"
        rows.append(
            f"<tr><td class='mono strong'>{r['ticker']}</td>"
            f"<td><a href='{html_escape(str(r['url']))}' target='_blank' rel='noopener noreferrer'>{html_escape(str(r['title']))}</a></td>"
            f"<td>{html_escape(str(r['source']))}</td>"
            f"<td>{published_text}</td></tr>"
        )
    return "\n".join(rows)


def build_forecast_table(summary: pd.DataFrame) -> str:
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            f"<tr><td class='strong'>{r['horizon']}</td>"
            f"<td class='num'>{fmt_currency(r['start_nav'],0)}</td>"
            f"<td class='num red'>{fmt_currency(r['p05'],0)}</td>"
            f"<td class='num'>{fmt_currency(r['p25'],0)}</td>"
            f"<td class='num green'>{fmt_currency(r['median'],0)}</td>"
            f"<td class='num'>{fmt_currency(r['p75'],0)}</td>"
            f"<td class='num amber'>{fmt_currency(r['p95'],0)}</td></tr>"
        )
    return "\n".join(rows)


def build_stress_table(stress: pd.DataFrame) -> str:
    rows = []
    for _, r in stress.iterrows():
        color = pnl_color(r["estimated_pnl_impact"])
        rows.append(
            f"<tr><td class='strong'>{html_escape(r['scenario'])}</td>"
            f"<td>{html_escape(r['description'])}</td>"
            f"<td class='num' style='color:{color}'>{fmt_currency(r['estimated_pnl_impact'],0)}</td>"
            f"<td class='num' style='color:{color}'>{fmt_pct(r['estimated_return_impact']*100,1)}</td>"
            f"<td class='num'>{fmt_currency(r['estimated_nav_after'],0)}</td></tr>"
        )
    return "\n".join(rows)


def generate_html(
    holdings, frame, metrics, positions, structure, ledger,
    heatmap, stress, forecast_summary, forecast_paths,
    news, intelligence, charts
) -> str:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    inception = CFG["inception"]
    init_aum = holdings["cost_basis"].sum()
    rf_pct = CFG["risk_free_rate"] * 100
    rf_display = f"{rf_pct:.2f}%"

    chart_json = json.dumps({k: v for k, v in charts.items()})

    kpis_row1 = "".join([
        kpi_card("Current NAV", fmt_currency(metrics["current_nav"], 0), f"Inception {inception}", "green"),
        kpi_card("Total P&L", fmt_currency(metrics["total_pnl"], 0), f"Initial AUM {fmt_currency(init_aum,0)}", "green" if metrics["total_pnl"] >= 0 else "red"),
        kpi_card("Total Return", fmt_pct(metrics["total_return"]*100), f"vs QQQ {fmt_pct(metrics['benchmark_total_return']*100)}", "green" if metrics["total_return"] >= 0 else "red"),
        kpi_card("Daily P&L", fmt_currency(metrics["daily_pnl"], 0), fmt_pct(metrics["daily_return"]*100), "green" if metrics["daily_pnl"] >= 0 else "red"),
        kpi_card("Alpha vs QQQ", fmt_pct(metrics["alpha_vs_benchmark"]*100), "Simple excess return", "green" if metrics["alpha_vs_benchmark"] >= 0 else "red"),
    ])
    kpis_row2 = "".join([
        kpi_card("Beta", f"{metrics['beta']:.2f}x", f"Correlation {metrics['correlation']:.2f}", "amber"),
        kpi_card("Sharpe", f"{metrics['sharpe_ratio']:.2f}", f"rf = {rf_display}", "blue"),
        kpi_card("Sortino", f"{metrics['sortino_ratio']:.2f}", "Downside-adjusted", "blue"),
        kpi_card("Max Drawdown", fmt_pct(metrics["max_drawdown"]*100), f"vs QQQ {fmt_pct(metrics['benchmark_max_drawdown']*100)}", "red"),
        kpi_card("Ann. Volatility", fmt_pct(metrics["annualized_volatility"]*100), f"vs QQQ {fmt_pct(metrics['benchmark_annualized_volatility']*100)}", "amber"),
        kpi_card("Info Ratio", f"{metrics['information_ratio']:.2f}", f"TE {fmt_pct(metrics['tracking_error']*100)}", "blue"),
        kpi_card("Hit Ratio", fmt_pct(metrics["hit_ratio"]*100), f"{metrics['sessions']} sessions", "purple"),
    ])

    intel_html = "".join(
        f"<div class='intel-block'><div class='intel-title'>{html_escape(title)}</div><p>{html_escape(body)}</p></div>"
        for title, body in intelligence
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{html_escape(CFG['portfolio_name'])} — Analytics Terminal</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {{
  --bg: {PALETTE['bg']}; --panel: {PALETTE['panel']}; --panel2: {PALETTE['panel_2']};
  --card: {PALETTE['card']}; --card2: {PALETTE['card_2']};
  --border: {PALETTE['border']}; --border2: {PALETTE['border_2']};
  --text: {PALETTE['text']}; --muted: {PALETTE['muted']}; --dim: {PALETTE['dim']};
  --grid: {PALETTE['grid']};
  --green: {PALETTE['green']}; --red: {PALETTE['red']}; --amber: {PALETTE['amber']};
  --blue: {PALETTE['blue']}; --cyan: {PALETTE['cyan']}; --purple: {PALETTE['purple']};
  --mono: 'JetBrains Mono', monospace; --sans: 'Inter', system-ui, sans-serif;
}}
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html {{ scroll-behavior: smooth; }}
body {{ background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 13px; line-height: 1.55; min-height: 100vh; overflow-x: hidden; }}
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: var(--panel); }}
::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 2px; }}

/* ── LAYOUT ── */
.shell {{ display: flex; min-height: 100vh; }}
.sidebar {{ width: 210px; min-width: 210px; background: var(--panel); border-right: 1px solid var(--border); display: flex; flex-direction: column; position: sticky; top: 0; height: 100vh; overflow-y: auto; z-index: 100; flex-shrink: 0; }}
.content {{ flex: 1; min-width: 0; display: flex; flex-direction: column; }}
main {{ padding: 20px 24px 60px; }}

/* ── SIDEBAR ── */
.sb-header {{ padding: 18px 16px 14px; border-bottom: 1px solid var(--border); }}
.sb-name {{ font-family: var(--mono); font-size: 11.5px; font-weight: 700; color: var(--green); line-height: 1.3; }}
.sb-tagline {{ font-family: var(--mono); font-size: 8px; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; }}
.sb-pill {{ display: inline-flex; align-items: center; gap: 5px; background: rgba(33,208,122,0.08); border: 1px solid rgba(33,208,122,0.18); border-radius: 3px; padding: 3px 8px; margin-top: 8px; font-family: var(--mono); font-size: 7.5px; color: var(--green); letter-spacing: 0.5px; }}
.pulse {{ width: 5px; height: 5px; border-radius: 50%; background: var(--green); box-shadow: 0 0 4px var(--green); animation: blink 2s infinite; }}
@keyframes blink {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.2; }} }}
.sb-meta {{ padding: 12px 16px; border-bottom: 1px solid var(--border); display: flex; flex-direction: column; gap: 6px; }}
.sb-row {{ display: flex; justify-content: space-between; align-items: center; }}
.sb-label {{ font-family: var(--mono); font-size: 7.5px; color: var(--dim); letter-spacing: 1.5px; text-transform: uppercase; }}
.sb-value {{ font-family: var(--mono); font-size: 10px; font-weight: 600; color: var(--text); }}
.sb-value.green {{ color: var(--green); }} .sb-value.red {{ color: var(--red); }} .sb-value.amber {{ color: var(--amber); }}
.sb-nav {{ padding: 10px 0; flex: 1; }}
.nav-group {{ margin-bottom: 4px; }}
.nav-group-label {{ font-family: var(--mono); font-size: 7px; color: var(--dim); letter-spacing: 2px; text-transform: uppercase; padding: 7px 16px 3px; }}
.nav-link {{ display: flex; align-items: center; gap: 8px; padding: 7px 16px; font-family: var(--mono); font-size: 9.5px; color: var(--muted); text-decoration: none; letter-spacing: 0.4px; border-left: 2px solid transparent; transition: all 0.12s; }}
.nav-link:hover, .nav-link.active {{ color: var(--text); background: rgba(77,141,255,0.06); border-left-color: var(--blue); }}
.nav-link.active {{ color: var(--green); border-left-color: var(--green); }}
.sb-footer {{ padding: 12px 16px; border-top: 1px solid var(--border); font-family: var(--mono); font-size: 7.5px; color: var(--dim); line-height: 1.8; }}

/* ── TOPBAR ── */
.topbar {{ background: rgba(6,8,13,0.97); backdrop-filter: blur(20px); border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; padding: 0 24px; height: 48px; position: sticky; top: 0; z-index: 90; }}
.tb-left {{ display: flex; align-items: center; gap: 16px; }}
.tb-title {{ font-family: var(--mono); font-size: 10.5px; font-weight: 700; color: var(--text); letter-spacing: 0.3px; }}
.tb-divider {{ width: 1px; height: 14px; background: var(--border); }}
.tb-stat {{ font-family: var(--mono); font-size: 9.5px; display: flex; align-items: center; gap: 5px; }}
.tb-stat .lbl {{ color: var(--dim); }}
.tb-stat .val {{ font-weight: 600; }}
.tb-right {{ display: flex; align-items: center; gap: 14px; font-family: var(--mono); font-size: 8.5px; color: var(--dim); }}

/* ── SECTION HEADERS ── */
.section-header {{ font-family: var(--mono); font-size: 8px; font-weight: 700; color: var(--muted); letter-spacing: 3px; text-transform: uppercase; margin: 28px 0 14px; display: flex; align-items: center; gap: 10px; }}
.section-header::after {{ content: ''; flex: 1; height: 1px; background: var(--border); }}
.section-num {{ color: var(--dim); font-size: 7px; font-weight: 400; }}

/* ── KPI CARDS ── */
.kpi-row {{ display: grid; gap: 10px; margin-bottom: 10px; }}
.kpi-row-5 {{ grid-template-columns: repeat(5, 1fr); }}
.kpi-row-7 {{ grid-template-columns: repeat(7, 1fr); }}
.kpi-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; position: relative; overflow: hidden; }}
.kpi-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }}
.kpi-card.green::before {{ background: var(--green); }} .kpi-card.red::before {{ background: var(--red); }}
.kpi-card.blue::before {{ background: var(--blue); }} .kpi-card.amber::before {{ background: var(--amber); }}
.kpi-card.purple::before {{ background: var(--purple); }} .kpi-card.cyan::before {{ background: var(--cyan); }}
.kpi-label {{ font-family: var(--mono); font-size: 8px; color: var(--muted); letter-spacing: 2px; text-transform: uppercase; margin-bottom: 7px; }}
.kpi-value {{ font-family: var(--mono); font-size: 19px; font-weight: 700; line-height: 1; letter-spacing: -0.5px; }}
.kpi-sub {{ font-family: var(--mono); font-size: 8.5px; color: var(--muted); margin-top: 5px; }}

/* ── GRID ── */
.grid {{ display: grid; gap: 14px; margin-bottom: 14px; }}
.g2 {{ grid-template-columns: 1fr 1fr; }}
.g3 {{ grid-template-columns: 1fr 1fr 1fr; }}
.g4 {{ grid-template-columns: 1fr 1fr 1fr 1fr; }}
.g65 {{ grid-template-columns: 3fr 2fr; }}
.g35 {{ grid-template-columns: 2fr 3fr; }}

/* ── CARDS ── */
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: 9px; overflow: hidden; margin-bottom: 14px; }}
.card-header {{ display: flex; align-items: center; justify-content: space-between; padding: 10px 16px; border-bottom: 1px solid var(--border); }}
.card-title {{ font-family: var(--mono); font-size: 8px; font-weight: 700; color: var(--muted); letter-spacing: 2px; text-transform: uppercase; }}
.card-badge {{ font-family: var(--mono); font-size: 7.5px; color: var(--dim); padding: 2px 8px; background: var(--card2); border-radius: 3px; border: 1px solid var(--border); }}
.card-body {{ padding: 14px 16px; }}
.chart-wrap {{ padding: 4px 4px 6px; }}

/* ── TABLES ── */
.table-wrap {{ overflow-x: auto; }}
.data-table {{ width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 10.5px; }}
.data-table th {{ font-size: 8px; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase; padding: 9px 12px; border-bottom: 1px solid var(--border); text-align: left; background: var(--card2); font-weight: 600; white-space: nowrap; }}
.data-table td {{ padding: 7.5px 12px; border-bottom: 1px solid rgba(29,42,63,0.5); white-space: nowrap; }}
.data-table tr:last-child td {{ border-bottom: none; }}
.data-table tr:hover td {{ background: rgba(77,141,255,0.04); }}
.num {{ text-align: right; }}
.mono {{ font-family: var(--mono); }}
.strong {{ font-weight: 700; }}
.green {{ color: var(--green); }} .red {{ color: var(--red); }} .amber {{ color: var(--amber); }}

/* ── RISK BUCKETS ── */
.bucket {{ font-family: var(--mono); font-size: 7.5px; padding: 2px 7px; border-radius: 3px; text-transform: uppercase; letter-spacing: 0.5px; }}
.bucket.core {{ background: rgba(77,141,255,0.1); color: var(--blue); border: 1px solid rgba(77,141,255,0.2); }}
.bucket.growth {{ background: rgba(33,208,122,0.1); color: var(--green); border: 1px solid rgba(33,208,122,0.2); }}
.bucket.speculative {{ background: rgba(255,190,85,0.1); color: var(--amber); border: 1px solid rgba(255,190,85,0.2); }}

/* ── INTELLIGENCE ── */
.intel-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }}
.intel-block {{ background: var(--card2); border: 1px solid var(--border); border-radius: 7px; padding: 14px; }}
.intel-title {{ font-family: var(--mono); font-size: 8px; font-weight: 700; color: var(--blue); letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; padding-bottom: 7px; border-bottom: 1px solid var(--border); }}
.intel-block p {{ font-size: 12px; line-height: 1.85; color: var(--text); }}

/* ── SPLIT TABLE ── */
.split-table {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
.mini-title {{ font-family: var(--mono); font-size: 8px; font-weight: 700; color: var(--muted); letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; }}

/* ── MISC ── */
.empty-state {{ font-family: var(--mono); font-size: 10px; color: var(--dim); padding: 20px; text-align: center; }}
.disclaimer {{ font-family: var(--mono); font-size: 8.5px; color: var(--dim); border: 1px solid var(--border); background: var(--card2); border-radius: 6px; padding: 10px 14px; margin-bottom: 14px; line-height: 1.8; }}
.disclaimer strong {{ color: var(--amber); }}

/* ── FOOTER ── */
footer {{ border-top: 1px solid var(--border); padding: 18px 24px; display: flex; align-items: center; justify-content: space-between; font-family: var(--mono); font-size: 8.5px; color: var(--dim); line-height: 1.8; }}

/* ── RESPONSIVE ── */
@media (max-width: 1200px) {{ .sidebar {{ display: none; }} .kpi-row-7 {{ grid-template-columns: repeat(4,1fr); }} }}
@media (max-width: 768px) {{ main {{ padding: 12px 14px 40px; }} .kpi-row-5, .kpi-row-7 {{ grid-template-columns: 1fr 1fr; }} .g2,.g3,.g4,.g65,.g35 {{ grid-template-columns: 1fr; }} .intel-grid {{ grid-template-columns: 1fr; }} .split-table {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="shell">

<!-- SIDEBAR -->
<aside class="sidebar">
  <div class="sb-header">
    <div class="sb-name">Life on the<br>Hedge Fund</div>
    <div class="sb-tagline">Trinity College Dublin &bull; #1 Ireland</div>
    <div class="sb-pill"><div class="pulse"></div>AUTO-REBUILT</div>
  </div>
  <div class="sb-meta">
    <div class="sb-row"><span class="sb-label">NAV</span><span class="sb-value green">{fmt_currency(metrics['current_nav'],0)}</span></div>
    <div class="sb-row"><span class="sb-label">P&L</span><span class="sb-value {'green' if metrics['total_pnl']>=0 else 'red'}">{fmt_currency(metrics['total_pnl'],0)}</span></div>
    <div class="sb-row"><span class="sb-label">Return</span><span class="sb-value {'green' if metrics['total_return']>=0 else 'red'}">{fmt_pct(metrics['total_return']*100)}</span></div>
    <div class="sb-row"><span class="sb-label">Alpha</span><span class="sb-value {'green' if metrics['alpha_vs_benchmark']>=0 else 'red'}">{fmt_pct(metrics['alpha_vs_benchmark']*100)}</span></div>
    <div class="sb-row"><span class="sb-label">Beta</span><span class="sb-value amber">{metrics['beta']:.2f}x</span></div>
    <div class="sb-row"><span class="sb-label">Sharpe</span><span class="sb-value">{metrics['sharpe_ratio']:.2f}</span></div>
    <div class="sb-row"><span class="sb-label">Max DD</span><span class="sb-value red">{fmt_pct(metrics['max_drawdown']*100)}</span></div>
  </div>
  <nav class="sb-nav">
    <div class="nav-group">
      <div class="nav-group-label">Analytics</div>
      <a href="#overview" class="nav-link active">&#9632; Overview</a>
      <a href="#performance" class="nav-link">&#9632; Performance</a>
      <a href="#risk" class="nav-link">&#9632; Risk</a>
    </div>
    <div class="nav-group">
      <div class="nav-group-label">Portfolio</div>
      <a href="#positions" class="nav-link">&#9632; Positions</a>
      <a href="#structure" class="nav-link">&#9632; Structure</a>
      <a href="#stress" class="nav-link">&#9632; Stress Test</a>
    </div>
    <div class="nav-group">
      <div class="nav-group-label">Outlook</div>
      <a href="#scenarios" class="nav-link">&#9632; Scenarios</a>
      <a href="#intelligence" class="nav-link">&#9632; Intelligence</a>
      <a href="#news" class="nav-link">&#9632; News</a>
    </div>
  </nav>
  <div class="sb-footer">
    Updated: {now_utc}<br>
    Inception: {inception}<br>
    AUM: {fmt_currency(init_aum,0)}<br>
    Bench: {CFG['benchmark']} &bull; rf={rf_display}
  </div>
</aside>

<!-- CONTENT -->
<div class="content">
<div class="topbar">
  <div class="tb-left">
    <div class="tb-title">LHFUND &middot; ANALYTICS TERMINAL</div>
    <div class="tb-divider"></div>
    <div class="tb-stat"><span class="lbl">NAV</span><span class="val" style="color:var(--green)">{fmt_currency(metrics['current_nav'],0)}</span></div>
    <div class="tb-divider"></div>
    <div class="tb-stat"><span class="lbl">Return</span><span class="val" style="color:{'var(--green)' if metrics['total_return']>=0 else 'var(--red)'}">{fmt_pct(metrics['total_return']*100)}</span></div>
    <div class="tb-divider"></div>
    <div class="tb-stat"><span class="lbl">Alpha</span><span class="val" style="color:{'var(--green)' if metrics['alpha_vs_benchmark']>=0 else 'var(--red)'}">{fmt_pct(metrics['alpha_vs_benchmark']*100)}</span></div>
    <div class="tb-divider"></div>
    <div class="tb-stat"><span class="lbl">Beta</span><span class="val" style="color:var(--amber)">{metrics['beta']:.2f}x</span></div>
    <div class="tb-divider"></div>
    <div class="tb-stat"><span class="lbl">Sharpe</span><span class="val">{metrics['sharpe_ratio']:.2f}</span></div>
  </div>
  <div class="tb-right">
    <div style="display:flex;align-items:center;gap:5px;color:var(--green)"><div class="pulse"></div>AUTO-REBUILT HOURLY</div>
    <span>Build: {now_utc}</span>
  </div>
</div>

<main>

<!-- OVERVIEW -->
<div id="overview"></div>
<div class="kpi-row kpi-row-5">{kpis_row1}</div>
<div class="kpi-row kpi-row-7">{kpis_row2}</div>

<!-- PERFORMANCE -->
<div id="performance" class="section-header"><span class="section-num">02</span> Performance</div>
<div class="card">
  <div class="card-header"><span class="card-title">Portfolio vs Benchmark — Base 100 from {inception}</span><span class="card-badge">NAV-weighted &bull; Zero rebalancing &bull; 15-year horizon</span></div>
  <div class="chart-wrap" id="perf"></div>
</div>
<div class="grid g2">
  <div class="card">
    <div class="card-header"><span class="card-title">Drawdown from Peak</span><span class="card-badge">vs {CFG['benchmark']}</span></div>
    <div class="chart-wrap" id="drawdown"></div>
  </div>
  <div class="card">
    <div class="card-header"><span class="card-title">Monthly Returns</span></div>
    <div class="chart-wrap" id="monthly_returns"></div>
  </div>
</div>
<div class="card">
  <div class="card-header"><span class="card-title">Monthly Return Heatmap</span></div>
  <div class="chart-wrap" id="heatmap"></div>
</div>
<div class="card">
  <div class="card-header"><span class="card-title">Monthly &amp; Annual Return Tables</span></div>
  <div class="card-body">{build_monthly_annual_table(metrics)}</div>
</div>

<!-- RISK -->
<div id="risk" class="section-header"><span class="section-num">03</span> Risk Analytics</div>
<div class="grid g3">
  <div class="card">
    <div class="card-header"><span class="card-title">Rolling {CFG['rolling_window']}-Day Volatility</span></div>
    <div class="chart-wrap" id="rolling_vol"></div>
  </div>
  <div class="card">
    <div class="card-header"><span class="card-title">Rolling {CFG['rolling_window']}-Day Beta</span></div>
    <div class="chart-wrap" id="rolling_beta"></div>
  </div>
  <div class="card">
    <div class="card-header"><span class="card-title">Rolling {CFG['rolling_window']}-Day Sharpe</span></div>
    <div class="chart-wrap" id="rolling_sharpe"></div>
  </div>
</div>
<div class="card">
  <div class="card-header"><span class="card-title">Complete Risk / Return Metrics</span><span class="card-badge">AF=252 &bull; rf={rf_display}</span></div>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr><th>Metric</th><th class="num">Value</th><th>Description</th></tr></thead>
      <tbody>{build_metrics_table(metrics)}</tbody>
    </table>
  </div>
</div>

<!-- POSITIONS -->
<div id="positions" class="section-header"><span class="section-num">04</span> Positions</div>
<div class="grid g2">
  <div class="card">
    <div class="card-header"><span class="card-title">Top Weights (%)</span></div>
    <div class="chart-wrap" id="top_weights"></div>
  </div>
  <div class="card">
    <div class="card-header"><span class="card-title">P&amp;L Attribution by Position</span></div>
    <div class="chart-wrap" id="pnl_attr"></div>
  </div>
</div>
<div class="card">
  <div class="card-header"><span class="card-title">Holdings — {len(positions)} Positions &bull; NAV {fmt_currency(metrics['current_nav'],0)} &bull; {now_utc}</span><span class="card-badge">Buy &amp; Hold &bull; No rebalancing since {inception}</span></div>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr>
        <th>Ticker</th><th>Name</th><th>Sector</th><th>Theme</th><th>Risk</th>
        <th class="num">Qty</th><th class="num">Buy</th><th class="num">Price</th>
        <th class="num">Value</th><th class="num">P&amp;L</th><th class="num">Return</th>
        <th class="num">Weight</th><th class="num">Contrib</th>
        <th class="num">1D</th><th class="num">5D</th><th class="num">1M</th><th class="num">Beta</th>
      </tr></thead>
      <tbody>{build_positions_table(positions)}</tbody>
    </table>
  </div>
</div>
<div class="card">
  <div class="card-header"><span class="card-title">Daily Ledger</span><span class="card-badge">Last 40 sessions</span></div>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr><th>Date</th><th class="num">NAV</th><th class="num">P&amp;L</th><th class="num">Day %</th><th class="num">Bench %</th><th class="num">Active %</th><th class="num">DD %</th></tr></thead>
      <tbody>{build_daily_ledger_table(ledger)}</tbody>
    </table>
  </div>
</div>

<!-- STRUCTURE -->
<div id="structure" class="section-header"><span class="section-num">05</span> Portfolio Structure</div>
<div class="grid g2">
  <div class="card">
    <div class="card-header"><span class="card-title">Sector Allocation</span></div>
    <div class="chart-wrap" id="sector_alloc"></div>
  </div>
  <div class="card">
    <div class="card-header"><span class="card-title">Thematic Allocation</span></div>
    <div class="chart-wrap" id="theme_alloc"></div>
  </div>
</div>
<div class="card">
  <div class="card-header"><span class="card-title">Sector Detail</span><span class="card-badge">HHI={structure['hhi']:.3f} &bull; Eff. N={structure['effective_n']:.1f} &bull; Top5={structure['top5_weight']*100:.1f}%</span></div>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr><th>Sector</th><th class="num">Weight</th><th class="num">Market Value</th><th class="num">P&amp;L</th><th class="num">Positions</th></tr></thead>
      <tbody>{build_structure_table(structure)}</tbody>
    </table>
  </div>
</div>

<!-- STRESS TEST -->
<div id="stress" class="section-header"><span class="section-num">06</span> Stress Testing</div>
<div class="card">
  <div class="card-header"><span class="card-title">Scenario P&amp;L Impact</span><span class="card-badge">Beta-adjusted &bull; position-level shocks</span></div>
  <div class="chart-wrap" id="stress"></div>
</div>
<div class="card">
  <div class="card-header"><span class="card-title">Stress Scenario Detail</span></div>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr><th>Scenario</th><th>Description</th><th class="num">P&amp;L Impact</th><th class="num">Return Impact</th><th class="num">NAV After</th></tr></thead>
      <tbody>{build_stress_table(stress)}</tbody>
    </table>
  </div>
</div>

<!-- SCENARIOS / FORECAST -->
<div id="scenarios" class="section-header"><span class="section-num">07</span> Scenario Projections</div>
<div class="disclaimer"><strong>Model disclaimer:</strong> The projections below are generated via a {CFG['mc_paths']}-path Monte Carlo simulation using the portfolio's own historical return distribution. They are <strong>model-based scenario envelopes, not forecasts or guarantees</strong>. Past distribution of returns does not predict future performance. Markets are non-stationary. This section is provided for analytical framing consistent with a {CFG['inception']}→15-year investment horizon thesis.</div>
<div class="card">
  <div class="card-header"><span class="card-title">12M Scenario Envelope — Bull / Base / Bear + Monte Carlo</span><span class="card-badge">bootstrap &bull; historical distribution</span></div>
  <div class="chart-wrap" id="forecast"></div>
</div>
<div class="card">
  <div class="card-header"><span class="card-title">Projection Summary by Horizon</span></div>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr><th>Horizon</th><th class="num">Start NAV</th><th class="num red">P5 (Bear)</th><th class="num">P25</th><th class="num green">Median</th><th class="num">P75</th><th class="num amber">P95 (Bull)</th></tr></thead>
      <tbody>{build_forecast_table(forecast_summary)}</tbody>
    </table>
  </div>
</div>

<!-- INTELLIGENCE -->
<div id="intelligence" class="section-header"><span class="section-num">08</span> Portfolio Intelligence</div>
<div class="card">
  <div class="card-header"><span class="card-title">Auto-generated Intelligence &bull; {now_utc}</span></div>
  <div class="card-body"><div class="intel-grid">{intel_html}</div></div>
</div>

<!-- NEWS -->
<div id="news" class="section-header"><span class="section-num">09</span> Portfolio News</div>
<div class="card">
  <div class="card-header"><span class="card-title">Latest News &bull; Non-blocking &bull; Build time only</span></div>
  <div class="table-wrap">
    <table class="data-table">
      <thead><tr><th>Ticker</th><th>Headline</th><th>Source</th><th>Published</th></tr></thead>
      <tbody>{build_news_table(news)}</tbody>
    </table>
  </div>
</div>

</main>

<footer>
  <div>{html_escape(CFG['portfolio_name'])} &bull; {html_escape(CFG['school'])} &bull; {html_escape(CFG['course_context'])}<br>Inception: {inception} &bull; AUM: {fmt_currency(init_aum,0)} &bull; Benchmark: {CFG['benchmark']} &bull; rf={rf_display}</div>
  <div>Generated: {now_utc}<br>Data: Yahoo Finance via yfinance &bull; Not investment advice</div>
</footer>
</div><!-- .content -->
</div><!-- .shell -->

<script>
const CHARTS = {chart_json};
const PLOT_CFG = {{responsive: true, displayModeBar: false, scrollZoom: false}};
const observer = new IntersectionObserver((entries) => {{
  entries.forEach(entry => {{
    if (entry.isIntersecting) {{
      const id = entry.target.id;
      if (CHARTS[id]) {{
        try {{
          Plotly.newPlot(id, CHARTS[id].data, CHARTS[id].layout, PLOT_CFG);
        }} catch(e) {{ console.warn('Chart error:', id, e); }}
      }}
      observer.unobserve(entry.target);
    }}
  }});
}}, {{ rootMargin: '300px' }});
Object.keys(CHARTS).forEach(id => {{
  const el = document.getElementById(id);
  if (el) observer.observe(el);
}});
const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('[id]');
window.addEventListener('scroll', () => {{
  let current = '';
  sections.forEach(s => {{ if (window.scrollY >= s.offsetTop - 120) current = s.id; }});
  navLinks.forEach(l => {{ l.classList.toggle('active', l.getAttribute('href') === '#' + current); }});
}}, {{ passive: true }});
</script>
</body>
</html>"""


# ============================================================
# ORCHESTRATION
# ============================================================
def main():
    print("Loading holdings...")
    holdings = load_holdings(ROOT / "holdings.csv")
    CFG["aum_initial"] = float(holdings["cost_basis"].sum())
    print(f"  Loaded {len(holdings)} holdings  |  Initial AUM: ${CFG['aum_initial']:,.2f}")

    tickers = holdings["ticker"].tolist() + [CFG["benchmark"], CFG["benchmark_secondary"]]

    # FIX: use datetime.now(timezone.utc) instead of deprecated pd.Timestamp.utcnow()
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"\nDownloading prices: {CFG['inception']} → {today_str}")
    prices = download_prices(tickers, CFG["inception"], today_str)

    print("\nBuilding analytics...")
    frame = build_market_frame(prices, holdings)
    metrics = compute_core_metrics(frame)
    positions = compute_position_analytics(frame, holdings)
    structure = compute_structure_analytics(positions, frame, metrics)
    ledger = build_daily_ledger(frame)
    heatmap = build_return_heatmap(metrics["monthly_portfolio"])
    stress = build_stress_tests(positions, metrics)
    forecast_summary, forecast_paths = build_forecast(frame, metrics)

    try:
        news = build_news(holdings)
        print(f"  News: {len(news)} articles fetched")
    except Exception:
        news = pd.DataFrame(columns=["ticker", "title", "source", "published_at", "url"])
        print("  News: skipped (non-blocking)")

    print("\nBuilding intelligence and charts...")
    intelligence = build_intelligence(metrics, positions, structure, stress)
    charts = make_charts(frame, metrics, positions, structure, heatmap, stress, forecast_paths)

    print("\nGenerating HTML...")
    html = generate_html(
        holdings, frame, metrics, positions, structure, ledger,
        heatmap, stress, forecast_summary, forecast_paths,
        news, intelligence, charts
    )

    CFG["output_html"].write_text(html, encoding="utf-8")

    snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "portfolio_name": CFG["portfolio_name"],
            "benchmark": CFG["benchmark"],
            "risk_free_rate": CFG["risk_free_rate"],
            "inception": CFG["inception"],
        },
        "overview": {
            "initial_aum": float(holdings["cost_basis"].sum()),
            "current_nav": metrics["current_nav"],
            "total_pnl": metrics["total_pnl"],
            "total_return": metrics["total_return"],
            "daily_pnl": metrics["daily_pnl"],
            "daily_return": metrics["daily_return"],
        },
        "risk": {
            "alpha_vs_benchmark": metrics["alpha_vs_benchmark"],
            "beta": metrics["beta"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "annualized_volatility": metrics["annualized_volatility"],
            "information_ratio": metrics["information_ratio"],
            "tracking_error": metrics["tracking_error"],
            "var_95": metrics["var_95"],
            "cvar_95": metrics["cvar_95"],
            "jensen_alpha": metrics["jensen_alpha"],
        },
        "positions": positions.to_dict(orient="records"),
    }
    CFG["output_snapshot"].write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")

    print(f"\n{'='*64}")
    print(f"  Dashboard → {CFG['output_html']}")
    print(f"  Snapshot  → {CFG['output_snapshot']}")
    print(f"  NAV: ${metrics['current_nav']:,.2f}  |  Return: {metrics['total_return']*100:+.2f}%")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}  |  Beta: {metrics['beta']:.3f}")
    print(f"{'='*64}")
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nBuild failed: {exc}")
        traceback.print_exc()
        raise
