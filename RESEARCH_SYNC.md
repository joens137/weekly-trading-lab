# Automated public research snapshots

The GitHub Actions workflow `Publish weekly research results` runs every Monday at 23:15 UTC
(after the regular US equity close) and can also be started with **Run workflow**.

It installs requirements, downloads Yahoo Finance ETF history, executes the existing
`engine.py` with fixed research settings, and commits the following files under
`public_results/` on `main`:

- `sync_status.json`: last successful generation time, market date and experiment settings.
- `strategy_leaderboard.csv`: nine-strategy backtest comparison.
- `historical_backtest.csv`: full 104-week trade-by-trade historical simulation.
- `dual_momentum_trades.csv`: subset for Dual Momentum.
- `weekly_forecasts.csv`: **immutable per-date prediction entries**, scored once their next-open-to-five-sessions-later returns become available.

The `📡 Synced research` tab in Streamlit reads these files from the repository checkout.

## Initial setup

1. Merge the feature pull request into `main`.
2. Go to **GitHub > weekly-trading-lab > Actions > Publish weekly research results**.
3. Click **Run workflow** (choose `main`) to publish the first snapshot.
4. Check the workflow logs for successful data generation and commit. Refresh Streamlit.
5. In the repository Settings > Actions > General, enable workflow read/write permissions
   if GitHub rejects the commit with a permissions error.

The workflow is scheduled once a week, but GitHub schedules can be delayed or skipped.
The `sync_status.json` timestamp shows whether it actually ran. Market holidays may cause
two scheduled runs to reference the same last market close; duplicates are not added.

## Important boundaries

- **Public repository:** every synced file is publicly readable. No brokerage credentials,
  personally identifiable data, or private app-session records belong here.
- This is an *independent standardized research run*. It does **not** scrape or mirror
  a user's Streamlit session or local `saved_forecasts.csv`.
- These virtual forecasts are generated using market data available when the workflow
  runs. A workflow can run after the market close; daily history may be delayed.
- There is no automatic trading or investment recommendation.
- Weekly score calculations use gross simulated price returns, and do **not** deduct
  turnover fees. Historical portfolio backtests include the engine's approximate fees.
- Backtests roll forward with the historical window; old leaderboard values are not
  independently versioned beyond Git commit history.
- The journal keys forecasts by the observed market date and strategy. Existing
  predictions are not rewritten, even if model settings later change. Avoid modifying
  fixed settings in `sync_research_results.py` without versioning the strategy.
- No dedicated database is installed. The Git-committed CSV journal is sufficient for
  an initial public experiment but not a private or transactional production store.
