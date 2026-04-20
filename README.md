# Life on the Hedge Fund
- Drawdown chart
- Rolling volatility
- Rolling beta
- Monthly return heatmap
- Monthly and annual return tables
- Daily ledger table

### Position monitor

- Ticker, company, quantity, buy price, latest price
- Market value, P&L, return, weight, contribution
- 1D / 5D / 1M / inception performance
- Individual beta vs benchmark
- Sector, theme and risk bucket

### Structure and risk

- Sector allocation
- Thematic allocation
- Concentration metrics
- HHI
- Effective number of positions
- Correlation to benchmark
- VaR and CVaR
- Jensen alpha
- Treynor ratio
- Omega ratio
- Stress testing section

### Forward-looking section

- Scenario-based model envelopes over 3M / 6M / 12M / 15Y
- Bull / base / bear path overlays
- Monte Carlo percentile table
- Explicitly framed as estimation, not certainty

## Repository structure

```text
.
├── build_dashboard.py
├── holdings.csv
├── requirements.txt
├── data/
│   └── dashboard_snapshot.json
├── docs/
│   └── index.html
└── .github/
    └── workflows/
        └── update-dashboard.yml
