#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DATA = ROOT / "data"
DOCS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

CFG = {
    "portfolio_name": "Life on the Hedge Fund",
    "subtitle": "Institutional Portfolio Analytics Terminal",
    "school": "Trinity College Dublin",
    "school_tag": "#1 in Ireland",
    "benchmark": "QQQ",
    "benchmark_2": "SPY",
    "risk_free_rate": 0.045,
    "annualisation_factor": 252,
    "rolling_window": 30,
    "news_items": 10,
    "mc_paths": 400,
    "seed": 42,
}

P = {
    "bg": "#06080d",
    "panel": "#0c111a",
    "card": "#101827",
    "card2": "#131d2f",
    "border": "#23324d",
    "grid": "#1a263a",
    "text": "#dde7f3",
    "muted": "#8ea4c0",
    "dim": "#5b6f8d",
    "green": "#21d07a",
    "red": "#f45b69",
    "amber": "#ffbe55",
    "blue": "#4d8dff",
    "cyan": "#45d7ff",
    "purple": "#b085ff",
}

BUCKET_COLORS = {
    "CORE": P["blue"],
    "GROWTH": P["green"],
    "SPECULATIVE": P["amber"],
}

SCENARIOS = [
    {"name": "Broad market correction", "type": "beta", "shock": -0.10, "description": "10% QQQ correction mapped through portfolio beta."},
    {"name": "Growth selloff", "type": "bucket", "bucket": "GROWTH", "shock": -0.18, "description": "Long-duration growth de-rating."},
    {"name": "Speculative unwind", "type": "bucket", "bucket": "SPECULATIVE", "shock": -0.25, "description": "High-beta speculative sleeve drawdown."},
    {"name": "AI valuation compression", "type": "theme", "keyword": "AI", "shock": -0.17, "description": "AI-linked positions re-rate lower."},
    {"name": "Rates shock", "type": "map", "mapping": {"CORE": -0.06, "GROWTH": -0.12, "SPECULATIVE": -0.16}, "description": "Higher yields pressure equity duration."},
    {"name": "Crypto crash", "type": "tickers", "tickers": ["COIN", "MARA", "HOOD"], "shock": -0.28, "description": "Crypto beta sleeve reprices sharply."},
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)


def fmt_currency(x: float, d: int = 0) -> str:
    return f"{'-' if x < 0 else ''}${abs(x):,.{d}f}"


def fmt_pct(x: float, d: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.{d}f}%"


def safe_div(a: float, b: float) -> float:
    return np.nan if abs(b) < 1e-12 else a / b


def esc(s: Any) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def pnl_color(x: float) -> str:
    return P["green"] if x >= 0 else P["red"]


def load_holdings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ticker", "name", "quantity", "buy_price", "sector", "theme", "risk_bucket", "inception_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"holdings.csv missing columns: {sorted(missing)}")
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df["quantity"] = df["quantity"].astype(float)
    df["buy_price"] = df["buy_price"].astype(float)
    df["cost_basis"] = df["quantity"] * df["buy_price"]
    return df


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    end_buffer = (pd.Timestamp(end) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    raw = yf.download(tickers=tickers, start=start, end=end_buffer, auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        raise RuntimeError("No market data returned by yfinance")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        elif "Close" in raw.columns.get_level_values(1):
            frames = []
            for t in tickers:
                if (t, "Close") in raw.columns:
                    frames.append(raw[(t, "Close")].rename(t))
            close = pd.concat(frames, axis=1)
        else:
            raise RuntimeError("Unable to locate Close prices in yfinance response")
    else:
        if "Close" in raw.columns:
            close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            close = raw.copy()

    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    close = close.sort_index().ffill(limit=5).dropna(how="all")
    return close


def annualized_return(total_return: float, periods: int, af: int) -> float:
    return np.nan if periods <= 0 else (1 + total_return) ** (af / periods) - 1


def downside_deviation(r: pd.Series, mar_daily: float, af: int) -> float:
    downside = np.minimum(r - mar_daily, 0)
    return float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(af))


def omega_ratio(r: pd.Series, mar_daily: float) -> float:
    diff = r - mar_daily
    gains = diff[diff > 0].sum()
    losses = -diff[diff < 0].sum()
    return np.nan if losses == 0 else float(gains / losses)


def capture_ratio(port: pd.Series, bench: pd.Series, upside: bool) -> float:
    mask = bench > 0 if upside else bench < 0
    if mask.sum() == 0:
        return np.nan
    denom = bench[mask].mean()
    return np.nan if abs(denom) < 1e-12 else float(port[mask].mean() / denom)


def build_frame(prices: pd.DataFrame, holdings: pd.DataFrame) -> dict[str, Any]:
    positions_mv = pd.DataFrame(index=prices.index)
    for _, row in holdings.iterrows():
        positions_mv[row["ticker"]] = prices[row["ticker"]] * row["quantity"]
    nav = positions_mv.sum(axis=1)

    init_aum = float(holdings["cost_basis"].sum())
    qqq_units = init_aum / prices[CFG["benchmark"]].dropna().iloc[0]
    qqq_nav = prices[CFG["benchmark"]] * qqq_units
    spy_nav = pd.Series(dtype=float)
    if CFG["benchmark_2"] in prices.columns:
        spy_units = init_aum / prices[CFG["benchmark_2"]].dropna().iloc[0]
        spy_nav = prices[CFG["benchmark_2"]] * spy_units

    r = nav.pct_change().fillna(0)
    b = qqq_nav.pct_change().fillna(0)
    s = spy_nav.pct_change().fillna(0) if not spy_nav.empty else pd.Series(index=prices.index, dtype=float)

    return {
        "prices": prices,
        "positions_mv": positions_mv,
        "nav": nav,
        "qqq_nav": qqq_nav,
        "spy_nav": spy_nav,
        "r": r,
        "b": b,
        "s": s,
        "base100_nav": (1 + r).cumprod() * 100,
        "base100_qqq": (1 + b).cumprod() * 100,
        "base100_spy": (1 + s).cumprod() * 100 if not spy_nav.empty else pd.Series(dtype=float),
        "dd": (nav / nav.cummax() - 1),
        "dd_b": (qqq_nav / qqq_nav.cummax() - 1),
    }


def compute_metrics(frame: dict[str, Any]) -> dict[str, Any]:
    af = CFG["annualisation_factor"]
    rf = CFG["risk_free_rate"]
    rfd = rf / af
    r, b = frame["r"], frame["b"]
    n = len(r)

    total_return = float(frame["nav"].iloc[-1] / frame["nav"].iloc[0] - 1)
    benchmark_total_return = float(frame["qqq_nav"].iloc[-1] / frame["qqq_nav"].iloc[0] - 1)
    ann_return = annualized_return(total_return, n, af)
    benchmark_ann_return = annualized_return(benchmark_total_return, n, af)
    vol = float(r.std() * math.sqrt(af))
    vol_b = float(b.std() * math.sqrt(af))
    down = downside_deviation(r, rfd, af)
    down_b = downside_deviation(b, rfd, af)
    sharpe = safe_div(ann_return - rf, vol)
    sortino = safe_div(ann_return - rf, down)
    sharpe_b = safe_div(benchmark_ann_return - rf, vol_b)
    sortino_b = safe_div(benchmark_ann_return - rf, down_b)
    beta = safe_div(float(r.cov(b)), float(b.var()))
    corr = float(r.corr(b))
    jensen_alpha = ann_return - (rf + beta * (benchmark_ann_return - rf)) if not pd.isna(beta) else np.nan
    active = r - b
    tracking_error = float(active.std() * math.sqrt(af))
    information_ratio = safe_div(float(active.mean() * af), tracking_error)
    max_dd = float(frame["dd"].min())
    max_dd_b = float(frame["dd_b"].min())
    calmar = safe_div(ann_return, abs(max_dd))
    var95 = float(np.percentile(r, 5))
    cvar95 = float(r[r <= var95].mean())
    skew = float(r.skew())
    kurt = float(r.kurtosis())
    up_cap = capture_ratio(r, b, True)
    down_cap = capture_ratio(r, b, False)
    treynor = safe_div(ann_return - rf, beta)
    omega = omega_ratio(r, rfd)
    hit_ratio = float((r > 0).mean())
    daily_pnl = float(frame["nav"].iloc[-1] - frame["nav"].iloc[-2])
    daily_ret = float(r.iloc[-1])
    total_pnl = float(frame["nav"].iloc[-1] - frame["nav"].iloc[0])
    w = CFG["rolling_window"]
    monthly_port = (1 + r).resample("ME").prod() - 1
    monthly_bench = (1 + b).resample("ME").prod() - 1
    annual_port = (1 + r).resample("YE").prod() - 1
    annual_bench = (1 + b).resample("YE").prod() - 1

    return {
        "current_nav": float(frame["nav"].iloc[-1]),
        "daily_pnl": daily_pnl,
        "daily_return": daily_ret,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "benchmark_total_return": benchmark_total_return,
        "alpha_vs_benchmark": total_return - benchmark_total_return,
        "annualized_return": ann_return,
        "benchmark_annualized_return": benchmark_ann_return,
        "annualized_volatility": vol,
        "benchmark_annualized_volatility": vol_b,
        "downside_deviation": down,
        "benchmark_downside_deviation": down_b,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "benchmark_sharpe_ratio": sharpe_b,
        "benchmark_sortino_ratio": sortino_b,
        "beta": beta,
        "correlation": corr,
        "jensen_alpha": jensen_alpha,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "max_drawdown": max_dd,
        "benchmark_max_drawdown": max_dd_b,
        "calmar_ratio": calmar,
        "var_95": var95,
        "cvar_95": cvar95,
        "skewness": skew,
        "kurtosis": kurt,
        "upside_capture": up_cap,
        "downside_capture": down_cap,
        "treynor_ratio": treynor,
        "omega_ratio": omega,
        "hit_ratio": hit_ratio,
        "sessions": n,
        "rolling_volatility": r.rolling(w).std() * math.sqrt(af) * 100,
        "rolling_beta": r.rolling(w).cov(b) / b.rolling(w).var(),
        "rolling_sharpe": ((r.rolling(w).mean() - rfd) / r.rolling(w).std()) * math.sqrt(af),
        "monthly_portfolio": monthly_port,
        "monthly_benchmark": monthly_bench,
        "annual_portfolio": annual_port,
        "annual_benchmark": annual_bench,
    }


def compute_positions(frame: dict[str, Any], holdings: pd.DataFrame) -> pd.DataFrame:
    px = frame["prices"]
    bench = frame["b"]
    nav = float(frame["nav"].iloc[-1])
    rows = []
    for _, row in holdings.iterrows():
        t = row["ticker"]
        ser = px[t].dropna()
        latest = float(ser.iloc[-1])
        prev = float(ser.iloc[-2]) if len(ser) >= 2 else latest
        mv = latest * row["quantity"]
        pnl = mv - row["cost_basis"]
        ret = safe_div(pnl, row["cost_basis"])
        weight = safe_div(mv, nav)
        contribution = safe_div(pnl, holdings["cost_basis"].sum())

        def trailing(days: int):
            return np.nan if len(ser) <= days else float(ser.iloc[-1] / ser.iloc[-days - 1] - 1)

        aligned = pd.concat([ser.pct_change(), bench], axis=1, join="inner").dropna()
        aligned.columns = ["r", "b"]
        beta = safe_div(float(aligned["r"].cov(aligned["b"])), float(aligned["b"].var())) if len(aligned) > 10 else np.nan

        rows.append({
            "ticker": t,
            "name": row["name"],
            "quantity": float(row["quantity"]),
            "buy_price": float(row["buy_price"]),
            "latest_price": latest,
            "market_value": mv,
            "pnl": pnl,
            "return": ret,
            "weight": weight,
            "contribution": contribution,
            "perf_1d": latest / prev - 1,
            "perf_5d": trailing(5),
            "perf_1m": trailing(21),
            "perf_inception": ret,
            "beta_vs_benchmark": beta,
            "sector": row["sector"],
            "theme": row["theme"],
            "risk_bucket": row["risk_bucket"],
        })
    return pd.DataFrame(rows).sort_values(["weight", "market_value"], ascending=[False, False]).reset_index(drop=True)


def compute_structure(pos: pd.DataFrame) -> dict[str, Any]:
    sector = pos.groupby("sector", as_index=False).agg(weight=("weight", "sum"), market_value=("market_value", "sum"), pnl=("pnl", "sum"), positions=("ticker", "count")).sort_values("weight", ascending=False)
    theme = pos.groupby("theme", as_index=False).agg(weight=("weight", "sum"), market_value=("market_value", "sum"), pnl=("pnl", "sum")).sort_values("weight", ascending=False)
    bucket = pos.groupby("risk_bucket", as_index=False).agg(weight=("weight", "sum"), market_value=("market_value", "sum"), pnl=("pnl", "sum")).sort_values("weight", ascending=False)
    hhi = float((pos["weight"] ** 2).sum())
    eff_n = safe_div(1.0, hhi)
    top5 = float(pos["weight"].head(5).sum())
    return {"sector": sector, "theme": theme, "bucket": bucket, "hhi": hhi, "effective_n": eff_n, "top5_weight": top5}


def build_ledger(frame: dict[str, Any]) -> pd.DataFrame:
    nav = frame["nav"]
    ledger = pd.DataFrame({
        "date": nav.index,
        "nav": nav.values,
        "daily_pnl": nav.diff().fillna(0).values,
        "daily_return": frame["r"].values,
        "benchmark_return": frame["b"].values,
        "active_return": (frame["r"] - frame["b"]).values,
        "drawdown": frame["dd"].values,
    })
    return ledger.tail(40).iloc[::-1].reset_index(drop=True)


def build_heatmap(monthly: pd.Series) -> pd.DataFrame:
    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.strftime("%b")
    heat = df.pivot(index="year", columns="month", values="ret")
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return heat.reindex(columns=[m for m in order if m in heat.columns])


def build_stress(pos: pd.DataFrame, metrics: dict[str, Any]) -> pd.DataFrame:
    nav = float(pos["market_value"].sum())
    rows = []
    for s in SCENARIOS:
        shocked = pos[["ticker", "market_value", "risk_bucket", "theme"]].copy()
        shocked["shock"] = 0.0
        if s["type"] == "beta":
            shocked["shock"] = metrics["beta"] * s["shock"]
        elif s["type"] == "bucket":
            shocked.loc[shocked["risk_bucket"] == s["bucket"], "shock"] = s["shock"]
        elif s["type"] == "theme":
            shocked.loc[shocked["theme"].str.contains(s["keyword"], case=False, na=False), "shock"] = s["shock"]
        elif s["type"] == "map":
            shocked["shock"] = shocked["risk_bucket"].map(s["mapping"]).fillna(0.0)
        elif s["type"] == "tickers":
            shocked.loc[shocked["ticker"].isin(s["tickers"]), "shock"] = s["shock"]
        shocked["pnl_impact"] = shocked["market_value"] * shocked["shock"]
        pnl = float(shocked["pnl_impact"].sum())
        rows.append({
            "scenario": s["name"],
            "description": s["description"],
            "estimated_pnl_impact": pnl,
            "estimated_return_impact": safe_div(pnl, nav),
            "estimated_nav_after": nav + pnl,
        })
    return pd.DataFrame(rows).sort_values("estimated_pnl_impact")


def build_forecast(frame: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(CFG["seed"])
    r = frame["r"].dropna()
    b = frame["b"].dropna()
    mu, sigma = float(r.mean()), float(r.std())
    bmu, bsig = float(b.mean()), float(b.std())
    start_nav = float(frame["nav"].iloc[-1])
    horizons = {"3M": 63, "6M": 126, "12M": 252, "15Y": 252 * 15}
    summary_rows, path_rows = [], []

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

        bull = start_nav * np.cumprod(np.r_[1, np.repeat(bmu + 0.75 * bsig, h)])
        base = start_nav * np.cumprod(np.r_[1, np.repeat(mu, h)])
        bear = start_nav * np.cumprod(np.r_[1, np.repeat(mu - 0.75 * sigma, h)])
        wealth_full = np.c_[np.repeat(start_nav, CFG["mc_paths"]).reshape(-1, 1), wealth]
        low = np.percentile(wealth_full, 5, axis=0)
        high = np.percentile(wealth_full, 95, axis=0)
        for step in range(h + 1):
            path_rows.append({"horizon": label, "step": step, "bull": bull[step], "base": base[step], "bear": bear[step], "mc_low": low[step], "mc_high": high[step]})

    return pd.DataFrame(summary_rows), pd.DataFrame(path_rows)


def build_news(holdings: pd.DataFrame) -> pd.DataFrame:
    items = []
    for t in holdings["ticker"].tolist():
        try:
            tk = yf.Ticker(t)
            news = getattr(tk, "news", None) or []
            for article in news[:6]:
                content = article.get("content", {}) if isinstance(article, dict) else {}
                title = content.get("title") or article.get("title")
                url = content.get("canonicalUrl", {}).get("url") or article.get("link") or article.get("url")
                source = content.get("provider", {}).get("displayName") or article.get("publisher") or "Yahoo Finance"
                pub = content.get("pubDate") or article.get("providerPublishTime")
                if isinstance(pub, str):
                    pub = pd.to_datetime(pub, utc=True, errors="coerce")
                elif pub is not None:
                    pub = pd.to_datetime(pub, unit="s", utc=True, errors="coerce")
                else:
                    pub = pd.NaT
                if title and url:
                    items.append({"ticker": t, "title": title, "source": source, "published_at": pub, "url": url})
        except Exception:
            continue
    if not items:
        return pd.DataFrame(columns=["ticker", "title", "source", "published_at", "url"])
    news = pd.DataFrame(items)
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True, errors="coerce")
    news["norm"] = news["title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    news = news.sort_values("published_at", ascending=False).drop_duplicates(subset=["norm"]).drop(columns=["norm"])
    return news.head(CFG["news_items"]).reset_index(drop=True)


def build_intelligence(metrics: dict[str, Any], pos: pd.DataFrame, structure: dict[str, Any], stress: pd.DataFrame) -> list[tuple[str, str]]:
    top = pos.nlargest(3, "contribution")
    bottom = pos.nsmallest(3, "contribution")
    dom_sector = structure["sector"].iloc[0]
    dom_bucket = structure["bucket"].iloc[0]
    worst = stress.iloc[0]
    return [
        ("Portfolio DNA", f"The book is a concentrated US equity portfolio with a high-conviction, high-beta, thematic growth profile. Top 5 positions represent {structure['top5_weight']*100:.1f}% of NAV, HHI is {structure['hhi']:.3f}, and the effective position count is {structure['effective_n']:.1f}."),
        ("What worked", f"Performance has been led by {', '.join(top['ticker'].tolist())}. The top contributor is {top.iloc[0]['ticker']} at {top.iloc[0]['contribution']*100:+.1f}% of initial capital."),
        ("What hurt", f"The main detractors have been {', '.join(bottom['ticker'].tolist())}, showing the cost of keeping speculative sleeves through drawdowns rather than rebalancing."),
        ("Risk lens", f"Beta versus {CFG['benchmark']} is {metrics['beta']:.2f}x, realized annualized volatility is {metrics['annualized_volatility']*100:.1f}%, and the harshest modelled scenario is {worst['scenario']} with an estimated NAV hit of {worst['estimated_return_impact']*100:.1f}%."),
        ("Benchmark lens", f"Alpha versus {CFG['benchmark']} stands at {metrics['alpha_vs_benchmark']*100:+.1f}% since inception. The benchmark choice is deliberate because the opportunity set is Nasdaq-heavy growth rather than defensive broad market equity."),
        ("Structure lens", f"The largest sector is {dom_sector['sector']} at {dom_sector['weight']*100:.1f}% of NAV, while the dominant risk bucket is {dom_bucket['risk_bucket']} at {dom_bucket['weight']*100:.1f}% of NAV. That is fully consistent with the Life on the Hedge Fund mandate."),
    ]


def plot_layout(title: str, height: int = 350) -> dict[str, Any]:
    return {
        "title": {"text": title, "x": 0.01, "font": {"size": 14, "color": P["text"]}},
        "paper_bgcolor": P["bg"],
        "plot_bgcolor": P["panel"],
        "margin": {"l": 50, "r": 20, "t": 50, "b": 40},
        "font": {"family": "JetBrains Mono, monospace", "size": 11, "color": P["text"]},
        "xaxis": {"gridcolor": P["grid"], "zeroline": False, "tickfont": {"color": P["muted"]}},
        "yaxis": {"gridcolor": P["grid"], "zeroline": False, "tickfont": {"color": P["muted"]}},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.01},
        "hoverlabel": {"bgcolor": P["card"], "bordercolor": P["border"], "font": {"color": P["text"]}},
        "hovermode": "x unified",
        "height": height,
    }


def make_charts(frame: dict[str, Any], metrics: dict[str, Any], pos: pd.DataFrame, structure: dict[str, Any], heatmap: pd.DataFrame, stress: pd.DataFrame, forecast_paths: pd.DataFrame) -> dict[str, Any]:
    charts = {}

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["base100_nav"].index, y=frame["base100_nav"], name=CFG["portfolio_name"], mode="lines", line={"color": P["green"], "width": 2.5}))
    fig.add_trace(go.Scatter(x=frame["base100_qqq"].index, y=frame["base100_qqq"], name=CFG["benchmark"], mode="lines", line={"color": P["blue"], "width": 2}))
    if not frame["spy_nav"].empty:
        fig.add_trace(go.Scatter(x=frame["base100_spy"].index, y=frame["base100_spy"], name=CFG["benchmark_2"], mode="lines", line={"color": P["purple"], "width": 1.5, "dash": "dot"}))
    fig.update_layout(**plot_layout("Portfolio vs Benchmark — Base 100", 380))
    charts["perf"] = fig.to_plotly_json()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["dd"].index, y=frame["dd"] * 100, name=CFG["portfolio_name"], fill="tozeroy", line={"color": P["red"], "width": 2}))
    fig.add_trace(go.Scatter(x=frame["dd_b"].index, y=frame["dd_b"] * 100, name=CFG["benchmark"], line={"color": P["blue"], "width": 1.6}))
    fig.update_layout(**plot_layout("Drawdown from Peak (%)"))
    charts["drawdown"] = fig.to_plotly_json()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics["rolling_volatility"].index, y=metrics["rolling_volatility"], name="Rolling vol", line={"color": P["amber"], "width": 2}))
    fig.update_layout(**plot_layout(f"Rolling {CFG['rolling_window']}-Day Volatility (Ann. %)"))
    charts["rolling_vol"] = fig.to_plotly_json()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics["rolling_beta"].index, y=metrics["rolling_beta"], name="Rolling beta", line={"color": P["cyan"], "width": 2}))
    fig.add_hline(y=1.0, line={"color": P["dim"], "dash": "dot"})
    fig.update_layout(**plot_layout(f"Rolling {CFG['rolling_window']}-Day Beta vs {CFG['benchmark']}"))
    charts["rolling_beta"] = fig.to_plotly_json()

    month_df = pd.DataFrame({"date": metrics["monthly_portfolio"].index, "portfolio": metrics["monthly_portfolio"].values * 100, "benchmark": metrics["monthly_benchmark"].values * 100})
    fig = go.Figure()
    fig.add_trace(go.Bar(x=month_df["date"], y=month_df["portfolio"], name=CFG["portfolio_name"], marker_color=np.where(month_df["portfolio"] >= 0, P["green"], P["red"]).tolist()))
    fig.add_trace(go.Scatter(x=month_df["date"], y=month_df["benchmark"], name=CFG["benchmark"], line={"color": P["blue"], "width": 2}))
    fig.update_layout(**plot_layout("Monthly Returns (%)"))
    charts["monthly"] = fig.to_plotly_json()

    z = (heatmap * 100).values
    text = np.vectorize(lambda x: "" if pd.isna(x) else f"{x:+.1f}%")(z)
    fig = go.Figure(data=go.Heatmap(z=z, x=list(heatmap.columns), y=list(map(str, heatmap.index)), colorscale=[[0, P["red"]], [0.5, P["panel"]], [1, P["green"]]], text=text, texttemplate="%{text}", hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>"))
    fig.update_layout(**plot_layout("Monthly Return Heatmap", 300))
    charts["heatmap"] = fig.to_plotly_json()

    fig = go.Figure(data=[go.Bar(x=pos["weight"].head(10) * 100, y=pos["ticker"].head(10), orientation="h", marker_color=[BUCKET_COLORS.get(x, P["blue"]) for x in pos["risk_bucket"].head(10)], text=[f"{x*100:.1f}%" for x in pos["weight"].head(10)], textposition="auto")])
    fig.update_layout(**plot_layout("Top Weights (%)"))
    fig.update_yaxes(autorange="reversed")
    charts["weights"] = fig.to_plotly_json()

    fig = go.Figure(data=[go.Pie(labels=structure["sector"]["sector"], values=structure["sector"]["weight"] * 100, hole=0.55, sort=False)])
    fig.update_layout(**plot_layout("Sector Allocation (%)", 350))
    charts["sector"] = fig.to_plotly_json()

    fig = go.Figure(data=[go.Bar(x=pos["ticker"], y=pos["pnl"], marker_color=np.where(pos["pnl"] >= 0, P["green"], P["red"]).tolist(), text=[fmt_currency(x, 0) for x in pos["pnl"]], textposition="outside")])
    fig.update_layout(**plot_layout("Position P&L Attribution ($)"))
    charts["pnl"] = fig.to_plotly_json()

    fig = go.Figure(data=[go.Bar(x=stress["estimated_pnl_impact"], y=stress["scenario"], orientation="h", marker_color=np.where(stress["estimated_pnl_impact"] >= 0, P["green"], P["red"]).tolist(), text=[fmt_currency(x, 0) for x in stress["estimated_pnl_impact"]], textposition="auto")])
    fig.update_layout(**plot_layout("Stress Testing — Estimated P&L Impact"))
    fig.update_yaxes(autorange="reversed")
    charts["stress"] = fig.to_plotly_json()

    fp = forecast_paths[forecast_paths["horizon"] == "12M"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fp["step"], y=fp["mc_high"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fp["step"], y=fp["mc_low"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(77,141,255,0.15)", name="Monte Carlo 5-95%"))
    fig.add_trace(go.Scatter(x=fp["step"], y=fp["bull"], name="Bull", line={"color": P["green"], "width": 2}))
    fig.add_trace(go.Scatter(x=fp["step"], y=fp["base"], name="Base", line={"color": P["blue"], "width": 2}))
    fig.add_trace(go.Scatter(x=fp["step"], y=fp["bear"], name="Bear", line={"color": P["red"], "width": 2}))
    fig.update_layout(**plot_layout("12M Scenario Envelope — Model-Based Paths"))
    charts["forecast"] = fig.to_plotly_json()

    return charts


def positions_table(pos: pd.DataFrame) -> str:
    rows = []
    for _, r in pos.iterrows():
        beta_display = "—" if pd.isna(r["beta_vs_benchmark"]) else f"{r['beta_vs_benchmark']:.2f}"
        rows.append(
            f"<tr>"
            f"<td class='mono strong'>{r['ticker']}</td>"
            f"<td>{esc(r['name'])}</td>"
            f"<td>{esc(r['sector'])}</td>"
            f"<td>{esc(r['theme'])}</td>"
            f"<td><span class='bucket {str(r['risk_bucket']).lower()}'>{esc(r['risk_bucket'])}</span></td>"
            f"<td class='num'>{r['quantity']:,.0f}</td>"
            f"<td class='num'>{fmt_currency(r['buy_price'],2)}</td>"
            f"<td class='num'>{fmt_currency(r['latest_price'],2)}</td>"
            f"<td class='num'>{fmt_currency(r['market_value'],0)}</td>"
            f"<td class='num' style='color:{pnl_color(r['pnl'])}'>{fmt_currency(r['pnl'],0)}</td>"
            f"<td class='num' style='color:{pnl_color(r['return'])}'>{fmt_pct(r['return']*100,1)}</td>"
            f"<td class='num'>{r['weight']*100:.1f}%</td>"
            f"<td class='num'>{r['contribution']*100:+.1f}%</td>"
            f"<td class='num' style='color:{pnl_color(r['perf_1d'])}'>{fmt_pct(r['perf_1d']*100,1)}</td>"
            f"<td class='num' style='color:{pnl_color(r['perf_5d'])}'>{fmt_pct(r['perf_5d']*100,1)}</td>"
            f"<td class='num' style='color:{pnl_color(r['perf_1m'])}'>{fmt_pct(r['perf_1m']*100,1)}</td>"
            f"<td class='num'>{beta_display}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def metrics_table(m: dict[str, Any]) -> str:
    rows = [
        ("Current NAV", fmt_currency(m["current_nav"], 0), "Current marked-to-market net asset value."),
        ("Total P&L", fmt_currency(m["total_pnl"], 0), "Absolute P&L since inception."),
        ("Total return", fmt_pct(m["total_return"] * 100), "Portfolio return since inception."),
        ("Daily P&L", fmt_currency(m["daily_pnl"], 0), "Latest session P&L."),
        ("Daily return", fmt_pct(m["daily_return"] * 100), "Latest session return."),
        ("Alpha vs benchmark", fmt_pct(m["alpha_vs_benchmark"] * 100), "Simple excess return versus QQQ."),
        ("Beta vs benchmark", f"{m['beta']:.2f}x", "CAPM sensitivity versus QQQ."),
        ("Sharpe ratio", f"{m['sharpe_ratio']:.2f}", "Annualized excess return per unit of volatility."),
        ("Sortino ratio", f"{m['sortino_ratio']:.2f}", "Annualized excess return per unit of downside deviation."),
        ("Max drawdown", fmt_pct(m["max_drawdown"] * 100), "Largest peak-to-trough drawdown."),
        ("Annualized volatility", fmt_pct(m["annualized_volatility"] * 100), "Realized annualized volatility."),
        ("Information ratio", f"{m['information_ratio']:.2f}", "Active return divided by tracking error."),
        ("Tracking error", fmt_pct(m["tracking_error"] * 100), "Annualized active risk."),
        ("Hit ratio", fmt_pct(m["hit_ratio"] * 100), "Share of positive days."),
        ("Annualized return", fmt_pct(m["annualized_return"] * 100), "Compounded annualized return."),
        ("Downside deviation", fmt_pct(m["downside_deviation"] * 100), "Annualized downside deviation."),
        ("Calmar ratio", f"{m['calmar_ratio']:.2f}", "Annualized return divided by max drawdown."),
        ("VaR 95% (1D)", fmt_pct(m["var_95"] * 100), "Historical daily 5th percentile."),
        ("CVaR 95% (1D)", fmt_pct(m["cvar_95"] * 100), "Expected shortfall."),
        ("Skewness", f"{m['skewness']:.2f}", "Return asymmetry."),
        ("Kurtosis", f"{m['kurtosis']:.2f}", "Excess kurtosis."),
        ("Upside capture", f"{m['upside_capture']:.2f}x", "Capture on benchmark up days."),
        ("Downside capture", f"{m['downside_capture']:.2f}x", "Capture on benchmark down days."),
        ("Jensen alpha", fmt_pct(m["jensen_alpha"] * 100), "CAPM alpha using displayed risk-free rate."),
        ("Treynor ratio", f"{m['treynor_ratio']:.2f}", "Excess return per unit of beta."),
        ("Omega ratio", f"{m['omega_ratio']:.2f}", "Gain/loss ratio above the daily rf threshold."),
    ]
    return "\n".join([f"<tr><td>{a}</td><td class='num strong'>{b}</td><td>{c}</td></tr>" for a, b, c in rows])


def monthly_annual_table(m: dict[str, Any]) -> str:
    month = pd.DataFrame({"Month": m["monthly_portfolio"].index.strftime("%Y-%m"), "Portfolio": m["monthly_portfolio"].values, "QQQ": m["monthly_benchmark"].values, "Active": (m["monthly_portfolio"] - m["monthly_benchmark"]).values}).tail(12).iloc[::-1]
    year = pd.DataFrame({"Year": m["annual_portfolio"].index.year.astype(str), "Portfolio": m["annual_portfolio"].values, "QQQ": m["annual_benchmark"].values, "Active": (m["annual_portfolio"] - m["annual_benchmark"]).values}).iloc[::-1]
    parts = ["<div class='split'>", "<div><div class='mini'>Monthly returns</div><table class='data-table'><thead><tr><th>Month</th><th>Portfolio</th><th>QQQ</th><th>Active</th></tr></thead><tbody>"]
    for _, r in month.iterrows():
        parts.append(f"<tr><td>{r['Month']}</td><td class='num' style='color:{pnl_color(r['Portfolio'])}'>{fmt_pct(r['Portfolio']*100,1)}</td><td class='num'>{fmt_pct(r['QQQ']*100,1)}</td><td class='num' style='color:{pnl_color(r['Active'])}'>{fmt_pct(r['Active']*100,1)}</td></tr>")
    parts.append("</tbody></table></div><div><div class='mini'>Annual returns</div><table class='data-table'><thead><tr><th>Year</th><th>Portfolio</th><th>QQQ</th><th>Active</th></tr></thead><tbody>")
    for _, r in year.iterrows():
        parts.append(f"<tr><td>{r['Year']}</td><td class='num' style='color:{pnl_color(r['Portfolio'])}'>{fmt_pct(r['Portfolio']*100,1)}</td><td class='num'>{fmt_pct(r['QQQ']*100,1)}</td><td class='num' style='color:{pnl_color(r['Active'])}'>{fmt_pct(r['Active']*100,1)}</td></tr>")
    parts.append("</tbody></table></div></div>")
    return "".join(parts)


def sector_table(s: dict[str, Any]) -> str:
    return "\n".join([f"<tr><td>{esc(r['sector'])}</td><td class='num'>{r['weight']*100:.1f}%</td><td class='num'>{fmt_currency(r['market_value'],0)}</td><td class='num' style='color:{pnl_color(r['pnl'])}'>{fmt_currency(r['pnl'],0)}</td><td class='num'>{int(r['positions'])}</td></tr>" for _, r in s["sector"].iterrows()])


def ledger_table(ledger: pd.DataFrame) -> str:
    out = []
    for _, r in ledger.iterrows():
        out.append(f"<tr><td>{pd.to_datetime(r['date']).strftime('%Y-%m-%d')}</td><td class='num'>{fmt_currency(r['nav'],0)}</td><td class='num' style='color:{pnl_color(r['daily_pnl'])}'>{fmt_currency(r['daily_pnl'],0)}</td><td class='num' style='color:{pnl_color(r['daily_return'])}'>{fmt_pct(r['daily_return']*100,2)}</td><td class='num'>{fmt_pct(r['benchmark_return']*100,2)}</td><td class='num' style='color:{pnl_color(r['active_return'])}'>{fmt_pct(r['active_return']*100,2)}</td><td class='num' style='color:{pnl_color(r['drawdown'])}'>{fmt_pct(r['drawdown']*100,2)}</td></tr>")
    return "\n".join(out)


def news_table(news: pd.DataFrame) -> str:
    if news.empty:
        return "<div class='empty'>News retrieval failed or returned no items. Core analytics still build because news is non-blocking by design.</div>"
    out = []
    for _, r in news.iterrows():
        ts = pd.to_datetime(r["published_at"], utc=True, errors="coerce")
        stamp = ts.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(ts) else "Unknown"
        out.append(f"<tr><td class='mono strong'>{r['ticker']}</td><td><a href='{esc(r['url'])}' target='_blank' rel='noopener noreferrer'>{esc(r['title'])}</a></td><td>{esc(r['source'])}</td><td>{stamp}</td></tr>")
    return "\n".join(out)


def forecast_table(summary: pd.DataFrame) -> str:
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            f"<tr>"
            f"<td class='strong'>{r['horizon']}</td>"
            f"<td class='num'>{fmt_currency(r['start_nav'],0)}</td>"
            f"<td class='num' style='color:{P['red']}'>{fmt_currency(r['p05'],0)}</td>"
            f"<td class='num'>{fmt_currency(r['p25'],0)}</td>"
            f"<td class='num' style='color:{P['green']}'>{fmt_currency(r['median'],0)}</td>"
            f"<td class='num'>{fmt_currency(r['p75'],0)}</td>"
            f"<td class='num' style='color:{P['amber']}'>{fmt_currency(r['p95'],0)}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def stress_table(stress: pd.DataFrame) -> str:
    return "\n".join([f"<tr><td class='strong'>{esc(r['scenario'])}</td><td>{esc(r['description'])}</td><td class='num' style='color:{pnl_color(r['estimated_pnl_impact'])}'>{fmt_currency(r['estimated_pnl_impact'],0)}</td><td class='num' style='color:{pnl_color(r['estimated_return_impact'])}'>{fmt_pct(r['estimated_return_impact']*100,1)}</td><td class='num'>{fmt_currency(r['estimated_nav_after'],0)}</td></tr>" for _, r in stress.iterrows()])


def kpi(label: str, value: str, sub: str, tone: str) -> str:
    return f"<div class='kpi {tone}'><div class='kpi-label'>{esc(label)}</div><div class='kpi-value'>{value}</div><div class='kpi-sub'>{esc(sub)}</div></div>"


def generate_html(holdings: pd.DataFrame, metrics: dict[str, Any], pos: pd.DataFrame, structure: dict[str, Any], ledger: pd.DataFrame, heatmap: pd.DataFrame, stress: pd.DataFrame, forecast_summary: pd.DataFrame, forecast_paths: pd.DataFrame, news: pd.DataFrame, intelligence: list[tuple[str, str]], charts: dict[str, Any]) -> str:
    chart_json = json.dumps(charts, cls=NumpyEncoder)
    init_aum = float(holdings["cost_basis"].sum())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    inception = str(holdings["inception_date"].iloc[0])

    kpis_1 = "".join([
        kpi("Current NAV", fmt_currency(metrics["current_nav"], 0), f"Inception {inception}", "green"),
        kpi("Total P&L", fmt_currency(metrics["total_pnl"], 0), f"Initial AUM {fmt_currency(init_aum,0)}", "green" if metrics["total_pnl"] >= 0 else "red"),
        kpi("Total Return", fmt_pct(metrics["total_return"] * 100), f"vs QQQ {fmt_pct(metrics['benchmark_total_return'] * 100)}", "green" if metrics["total_return"] >= 0 else "red"),
        kpi("Daily P&L", fmt_currency(metrics["daily_pnl"], 0), fmt_pct(metrics["daily_return"] * 100), "green" if metrics["daily_pnl"] >= 0 else "red"),
        kpi("Alpha vs QQQ", fmt_pct(metrics["alpha_vs_benchmark"] * 100), "Simple excess return", "green" if metrics["alpha_vs_benchmark"] >= 0 else "red"),
    ])
    kpis_2 = "".join([
        kpi("Beta", f"{metrics['beta']:.2f}x", f"Correlation {metrics['correlation']:.2f}", "amber"),
        kpi("Sharpe", f"{metrics['sharpe_ratio']:.2f}", f"rf {CFG['risk_free_rate']*100:.2f}%", "blue"),
        kpi("Sortino", f"{metrics['sortino_ratio']:.2f}", "Downside-adjusted", "blue"),
        kpi("Max Drawdown", fmt_pct(metrics["max_drawdown"] * 100), f"vs QQQ {fmt_pct(metrics['benchmark_max_drawdown'] * 100)}", "red"),
        kpi("Ann. Volatility", fmt_pct(metrics["annualized_volatility"] * 100), f"vs QQQ {fmt_pct(metrics['benchmark_annualized_volatility'] * 100)}", "amber"),
        kpi("Info Ratio", f"{metrics['information_ratio']:.2f}", f"TE {fmt_pct(metrics['tracking_error'] * 100)}", "blue"),
        kpi("Hit Ratio", fmt_pct(metrics["hit_ratio"] * 100), f"{metrics['sessions']} sessions", "purple"),
    ])
    intel_html = "".join([f"<div class='intel'><div class='intel-title'>{esc(t)}</div><p>{esc(b)}</p></div>" for t, b in intelligence])

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{esc(CFG['portfolio_name'])} — Analytics Terminal</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root{{--bg:{P['bg']};--panel:{P['panel']};--card:{P['card']};--card2:{P['card2']};--border:{P['border']};--grid:{P['grid']};--text:{P['text']};--muted:{P['muted']};--dim:{P['dim']};--green:{P['green']};--red:{P['red']};--amber:{P['amber']};--blue:{P['blue']};--cyan:{P['cyan']};--purple:{P['purple']};--mono:'JetBrains Mono',monospace;--sans:'Inter',system-ui,sans-serif;}}
*{{box-sizing:border-box;margin:0;padding:0}} html{{scroll-behavior:smooth}} body{{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:13px;line-height:1.55;overflow-x:hidden}} a{{color:var(--blue)}}
.shell{{display:flex;min-height:100vh}} .sidebar{{width:220px;min-width:220px;background:var(--panel);border-right:1px solid var(--border);padding:16px;position:sticky;top:0;height:100vh;overflow-y:auto}} .content{{flex:1;min-width:0}}
.brand{{font-family:var(--mono);font-size:12px;font-weight:700;color:var(--green)}} .tag{{font-family:var(--mono);font-size:8px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;margin-top:4px}} .pill{{display:inline-flex;gap:6px;align-items:center;margin-top:10px;padding:4px 8px;border:1px solid rgba(33,208,122,.2);border-radius:4px;background:rgba(33,208,122,.08);font-family:var(--mono);font-size:8px;color:var(--green)}} .dot{{width:5px;height:5px;border-radius:50%;background:var(--green)}}
.sidebox{{margin-top:16px;padding:12px;background:var(--card);border:1px solid var(--border);border-radius:10px}} .srow{{display:flex;justify-content:space-between;gap:8px;margin:7px 0}} .slabel{{font-family:var(--mono);font-size:8px;color:var(--dim);text-transform:uppercase}} .sval{{font-family:var(--mono);font-size:10px;font-weight:600}}
.nav{{margin-top:18px;display:flex;flex-direction:column;gap:6px}} .nav a{{text-decoration:none;padding:8px 10px;border-radius:8px;background:transparent;color:var(--muted);font-family:var(--mono);font-size:10px}} .nav a:hover{{background:rgba(77,141,255,.08);color:var(--text)}}
.topbar{{position:sticky;top:0;z-index:50;height:50px;padding:0 22px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;background:rgba(6,8,13,.94);backdrop-filter:blur(12px)}} .tb-left{{display:flex;align-items:center;gap:14px;font-family:var(--mono);font-size:10px}} .tb-muted{{color:var(--dim)}} .tb-right{{font-family:var(--mono);font-size:9px;color:var(--dim)}}
main{{padding:20px 22px 48px}} .hero{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}} .hero2{{display:grid;grid-template-columns:repeat(7,1fr);gap:12px;margin-top:12px}} .kpi{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px;min-height:90px}} .kpi.green{{border-top:2px solid var(--green)}} .kpi.red{{border-top:2px solid var(--red)}} .kpi.blue{{border-top:2px solid var(--blue)}} .kpi.amber{{border-top:2px solid var(--amber)}} .kpi.purple{{border-top:2px solid var(--purple)}} .kpi-label{{font-family:var(--mono);font-size:8px;color:var(--muted);letter-spacing:1.6px;text-transform:uppercase}} .kpi-value{{font-family:var(--mono);font-size:24px;font-weight:700;margin-top:8px}} .kpi-sub{{font-family:var(--mono);font-size:9px;color:var(--dim);margin-top:6px}}
.section{{margin-top:26px}} .section-title{{font-family:var(--mono);font-size:9px;font-weight:700;color:var(--muted);letter-spacing:3px;text-transform:uppercase;margin-bottom:12px}} .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} .card{{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden}} .card-head{{display:flex;justify-content:space-between;align-items:center;padding:12px 14px;border-bottom:1px solid var(--border)}} .card-title{{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:1.8px;text-transform:uppercase}} .card-body{{padding:14px}}
.data-table{{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:10.5px}} .data-table th{{text-align:left;padding:10px;border-bottom:1px solid var(--border);color:var(--muted);font-size:8px;letter-spacing:1.5px;text-transform:uppercase;background:var(--card2)}} .data-table td{{padding:9px 10px;border-bottom:1px solid rgba(35,50,77,.55)}} .num{{text-align:right}} .mono{{font-family:var(--mono)}} .strong{{font-weight:700}}
.bucket{{display:inline-block;padding:3px 7px;border-radius:999px;font-family:var(--mono);font-size:8px;border:1px solid var(--border)}} .bucket.core{{color:var(--blue)}} .bucket.growth{{color:var(--green)}} .bucket.speculative{{color:var(--amber)}}
.split{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} .mini{{font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px}} .intel-wrap{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} .intel{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px}} .intel-title{{font-family:var(--mono);font-size:9px;color:var(--cyan);letter-spacing:1.8px;text-transform:uppercase;margin-bottom:10px}} .intel p{{color:var(--text);line-height:1.8}}
.empty{{padding:16px;color:var(--muted)}}
@media (max-width:1200px){{.hero,.hero2{{grid-template-columns:repeat(3,1fr)}} .grid-2,.split,.intel-wrap{{grid-template-columns:1fr}} .sidebar{{display:none}} }}
@media (max-width:768px){{main{{padding:16px}} .hero,.hero2{{grid-template-columns:1fr 1fr}} }}
</style>
</head>
<body>
<div class="shell">
  <aside class="sidebar">
    <div class="brand">{esc(CFG['portfolio_name'])}</div>
    <div class="tag">{esc(CFG['subtitle'])}</div>
    <div class="pill"><span class="dot"></span> Python source of truth</div>
    <div class="sidebox">
      <div class="srow"><span class="slabel">School</span><span class="sval">{esc(CFG['school'])}</span></div>
      <div class="srow"><span class="slabel">Tag</span><span class="sval">{esc(CFG['school_tag'])}</span></div>
      <div class="srow"><span class="slabel">Benchmark</span><span class="sval">QQQ</span></div>
      <div class="srow"><span class="slabel">AUM</span><span class="sval">{fmt_currency(init_aum,0)}</span></div>
      <div class="srow"><span class="slabel">Rebalancing</span><span class="sval">None</span></div>
      <div class="srow"><span class="slabel">Horizon</span><span class="sval">15 years</span></div>
      <div class="srow"><span class="slabel">Updated</span><span class="sval">{esc(now)}</span></div>
    </div>
    <nav class="nav">
      <a href="#overview">Overview</a>
      <a href="#analytics">Analytics</a>
      <a href="#positions">Positions</a>
      <a href="#structure">Structure</a>
      <a href="#risk">Risk</a>
      <a href="#forecast">Forecast</a>
      <a href="#news">News</a>
      <a href="#intelligence">Intelligence</a>
    </nav>
  </aside>
  <div class="content">
    <div class="topbar"><div class="tb-left"><span>{esc(CFG['portfolio_name'])}</span><span class="tb-muted">concentrated US equity portfolio · QQQ benchmark · no rebalancing</span></div><div class="tb-right">Last updated {esc(now)}</div></div>
    <main>
      <section id="overview" class="section">
        <div class="section-title">Overview</div>
        <div class="hero">{kpis_1}</div>
        <div class="hero2">{kpis_2}</div>
      </section>

      <section id="analytics" class="section">
        <div class="section-title">Analytics</div>
        <div class="card"><div class="card-head"><div class="card-title">Portfolio vs benchmark</div></div><div class="card-body"><div id="perf"></div></div></div>
        <div class="grid-2" style="margin-top:14px">
          <div class="card"><div class="card-head"><div class="card-title">Drawdown</div></div><div class="card-body"><div id="drawdown"></div></div></div>
          <div class="card"><div class="card-head"><div class="card-title">Monthly returns</div></div><div class="card-body"><div id="monthly"></div></div></div>
        </div>
        <div class="grid-2" style="margin-top:14px">
          <div class="card"><div class="card-head"><div class="card-title">Rolling volatility</div></div><div class="card-body"><div id="rolling_vol"></div></div></div>
          <div class="card"><div class="card-head"><div class="card-title">Rolling beta</div></div><div class="card-body"><div id="rolling_beta"></div></div></div>
        </div>
        <div class="card" style="margin-top:14px"><div class="card-head"><div class="card-title">Monthly return heatmap</div></div><div class="card-body"><div id="heatmap"></div></div></div>
      </section>

      <section id="positions" class="section">
        <div class="section-title">Positions</div>
        <div class="grid-2">
          <div class="card"><div class="card-head"><div class="card-title">Top weights</div></div><div class="card-body"><div id="weights"></div></div></div>
          <div class="card"><div class="card-head"><div class="card-title">P&amp;L attribution</div></div><div class="card-body"><div id="pnl"></div></div></div>
        </div>
        <div class="card" style="margin-top:14px"><div class="card-head"><div class="card-title">Positions monitor</div></div><div class="card-body" style="padding:0;overflow:auto"><table class="data-table"><thead><tr><th>Ticker</th><th>Name</th><th>Sector</th><th>Theme</th><th>Risk</th><th class="num">Qty</th><th class="num">Buy</th><th class="num">Last</th><th class="num">Value</th><th class="num">P&amp;L</th><th class="num">Return</th><th class="num">Weight</th><th class="num">Contrib</th><th class="num">1D</th><th class="num">5D</th><th class="num">1M</th><th class="num">Beta</th></tr></thead><tbody>{positions_table(pos)}</tbody></table></div></div>
      </section>

      <section id="structure" class="section">
        <div class="section-title">Structure</div>
        <div class="grid-2">
          <div class="card"><div class="card-head"><div class="card-title">Sector allocation</div></div><div class="card-body"><div id="sector"></div></div></div>
          <div class="card"><div class="card-head"><div class="card-title">Concentration metrics</div></div><div class="card-body"><div class="srow"><span class="slabel">HHI</span><span class="sval">{structure['hhi']:.3f}</span></div><div class="srow"><span class="slabel">Effective N</span><span class="sval">{structure['effective_n']:.2f}</span></div><div class="srow"><span class="slabel">Top 5 weight</span><span class="sval">{structure['top5_weight']*100:.1f}%</span></div><div class="srow"><span class="slabel">Benchmark correlation</span><span class="sval">{metrics['correlation']:.2f}</span></div></div></div>
        </div>
        <div class="card" style="margin-top:14px"><div class="card-head"><div class="card-title">Sector table</div></div><div class="card-body" style="padding:0;overflow:auto"><table class="data-table"><thead><tr><th>Sector</th><th class="num">Weight</th><th class="num">Value</th><th class="num">P&amp;L</th><th class="num">Positions</th></tr></thead><tbody>{sector_table(structure)}</tbody></table></div></div>
      </section>

      <section id="risk" class="section">
        <div class="section-title">Risk</div>
        <div class="card"><div class="card-head"><div class="card-title">Risk and performance metrics</div></div><div class="card-body" style="padding:0;overflow:auto"><table class="data-table"><thead><tr><th>Metric</th><th class="num">Value</th><th>Comment</th></tr></thead><tbody>{metrics_table(metrics)}</tbody></table></div></div>
        <div class="card" style="margin-top:14px"><div class="card-head"><div class="card-title">Monthly and annual returns</div></div><div class="card-body">{monthly_annual_table(metrics)}</div></div>
        <div class="card" style="margin-top:14px"><div class="card-head"><div class="card-title">Daily ledger</div></div><div class="card-body" style="padding:0;overflow:auto"><table class="data-table"><thead><tr><th>Date</th><th class="num">NAV</th><th class="num">Daily P&amp;L</th><th class="num">Daily return</th><th class="num">QQQ</th><th class="num">Active</th><th class="num">Drawdown</th></tr></thead><tbody>{ledger_table(ledger)}</tbody></table></div></div>
        <div class="card" style="margin-top:14px"><div class="card-head"><div class="card-title">Stress testing</div></div><div class="card-body"><div id="stress"></div><div style="margin-top:14px;overflow:auto"><table class="data-table"><thead><tr><th>Scenario</th><th>Description</th><th class="num">P&amp;L impact</th><th class="num">Return impact</th><th class="num">NAV after</th></tr></thead><tbody>{stress_table(stress)}</tbody></table></div></div></div>
      </section>

      <section id="forecast" class="section">
        <div class="section-title">Forecast</div>
        <div class="card"><div class="card-head"><div class="card-title">Scenario-based forward envelope</div></div><div class="card-body"><div id="forecast"></div><p style="margin-top:12px;color:var(--muted)">These are model-based scenario envelopes using historical drift and volatility assumptions. They are not predictions of certainty.</p><div style="margin-top:14px;overflow:auto"><table class="data-table"><thead><tr><th>Horizon</th><th class="num">Start NAV</th><th class="num">P05</th><th class="num">P25</th><th class="num">Median</th><th class="num">P75</th><th class="num">P95</th></tr></thead><tbody>{forecast_table(forecast_summary)}</tbody></table></div></div></div>
      </section>

      <section id="news" class="section">
        <div class="section-title">Latest portfolio news</div>
        <div class="card"><div class="card-head"><div class="card-title">Python-built news layer</div></div><div class="card-body" style="padding:0;overflow:auto"><table class="data-table"><thead><tr><th>Ticker</th><th>Title</th><th>Source</th><th>Published</th></tr></thead><tbody>{news_table(news)}</tbody></table></div></div>
      </section>

      <section id="intelligence" class="section">
        <div class="section-title">Portfolio intelligence</div>
        <div class="intel-wrap">{intel_html}</div>
      </section>
    </main>
  </div>
</div>
<script>
const charts = {chart_json};
Plotly.newPlot('perf', charts.perf.data, charts.perf.layout, {{displayModeBar:false,responsive:true}});
Plotly.newPlot('drawdown', charts.drawdown.data, charts.drawdown.layout, {{displayModeBar:false,responsive:true}});
Plotly.newPlot('monthly', charts.monthly.data, charts.monthly.layout, {{displayModeBar:false,responsive:true}});
Plotly.newPlot('rolling_vol', charts.rolling_vol.data, charts.rolling_vol.layout, {{displayModeBar:false,responsive:true}});
Plotly.newPlot('rolling_beta', charts.rolling_beta.data, charts.rolling_beta.layout, {{displayModeBar:false,responsive:true}});
Plotly.newPlot('heatmap', charts.heatmap.data, charts.heatmap.layout, {{displayModeBar:false,responsive:true}});
Plotly.newPlot('weights', charts.weights.data, charts.weights.layout, {{displayModeBar:false,responsive:true}});
Plotly.newPlot('sector', charts.sector.data, charts.sector.layout, {{displayModeBar:false,responsive:true}});
Plotly.newPlot('pnl', charts.pnl.data, charts.pnl.layout, {{displayModeBar:false,responsive:true}});
Plotly.newPlot('stress', charts.stress.data, charts.stress.layout, {{displayModeBar:false,responsive:true}});
Plotly.newPlot('forecast', charts.forecast.data, charts.forecast.layout, {{displayModeBar:false,responsive:true}});
</script>
</body>
</html>'''


def main() -> None:
    print("Loading holdings...")
    holdings = load_holdings(ROOT / "holdings.csv")
    init_aum = float(holdings["cost_basis"].sum())
    print(f"  Loaded {len(holdings)} holdings  |  Initial AUM: {fmt_currency(init_aum,2)}")

    inception = str(holdings["inception_date"].min())
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tickers = holdings["ticker"].tolist() + [CFG["benchmark"], CFG["benchmark_2"]]

    print(f"Downloading prices: {inception} → {end}")
    prices = download_prices(tickers, inception, end)
    print(f"  Downloaded: {len(prices)} sessions, {len(prices.columns)} tickers")

    print("Building analytics...")
    frame = build_frame(prices, holdings)
    metrics = compute_metrics(frame)
    positions = compute_positions(frame, holdings)
    structure = compute_structure(positions)
    ledger = build_ledger(frame)
    heatmap = build_heatmap(metrics["monthly_portfolio"])
    stress = build_stress(positions, metrics)
    forecast_summary, forecast_paths = build_forecast(frame)
    news = build_news(holdings)
    print(f"  News: {len(news)} articles fetched")

    print("Building intelligence and charts...")
    intelligence = build_intelligence(metrics, positions, structure, stress)
    charts = make_charts(frame, metrics, positions, structure, heatmap, stress, forecast_paths)

    print("Generating HTML...")
    html = generate_html(holdings, metrics, positions, structure, ledger, heatmap, stress, forecast_summary, forecast_paths, news, intelligence, charts)
    (DOCS / "index.html").write_text(html, encoding="utf-8")

    snapshot = {
        "updated_at": datetime.now(timezone.utc),
        "portfolio_name": CFG["portfolio_name"],
        "benchmark": CFG["benchmark"],
        "initial_aum": init_aum,
        "positions": positions.to_dict(orient="records"),
        "metrics": metrics,
        "stress": stress.to_dict(orient="records"),
        "forecast_summary": forecast_summary.to_dict(orient="records"),
        "news": news.to_dict(orient="records"),
    }
    with open(DATA / "dashboard_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, cls=NumpyEncoder)

    print("Build complete:")
    print(f"  HTML     -> {DOCS / 'index.html'}")
    print(f"  Snapshot -> {DATA / 'dashboard_snapshot.json'}")


if __name__ == "__main__":
    main()
