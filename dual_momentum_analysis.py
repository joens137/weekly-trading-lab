"""Dual momentum investigation: audit holdings, risk, attribution and parameter stability."""
import numpy as np
import pandas as pd
import streamlit as st
from engine import TICKERS, turnover_cost, realized_week


def _equity_and_drawdown(frame, stake):
    x = frame.sort_values("week").copy()
    x["equity"] = stake * (1 + x["actual"]).cumprod()
    peak = pd.concat([pd.Series([stake]), x["equity"].reset_index(drop=True)], ignore_index=True).cummax().iloc[1:].to_numpy()
    x["drawdown_pct"] = 100 * (x["equity"].to_numpy() / peak - 1)
    return x


def _parameter_replay(opens, closes, dates, lookback, bps, stake):
    """One-ETF strategy using only information at each signal date; same execution dates."""
    previous = {}
    records = []
    for day in dates:
        i = closes.index.get_indexer([pd.Timestamp(day)])[0]
        if i < 253 or i + 6 >= len(opens):
            continue
        s = closes.iloc[:i + 1]
        last = s.iloc[-1]
        # Change the relative-momentum horizon while retaining the same safety gate.
        signal = last / s.iloc[-1 - lookback] - 1
        selected = str(signal.idxmax())
        ma200 = s.rolling(200).mean().iloc[-1]
        allocation = {selected: 1.0} if signal[selected] > 0 and last[selected] > ma200[selected] else {}
        gross = realized_week(opens, i, allocation)
        fee = turnover_cost(allocation, previous, bps)
        records.append({"signal_date":str(day),"holding":selected if allocation else "CASH",
                        "actual":gross-fee,"gross":gross,"fees":fee})
        previous = allocation
    if not records:
        return None
    d = pd.DataFrame(records)
    curve = stake * (1 + d["actual"]).cumprod()
    running_peak = pd.concat([pd.Series([stake]), curve],ignore_index=True).cummax().iloc[1:]
    return {"Lookback":f"{lookback} sessions","Ending $":float(curve.iloc[-1]),
            "Total return %":float(100 * (curve.iloc[-1]/stake-1)),
            "Max drawdown %":float(100 * (curve.to_numpy()/running_peak.to_numpy()-1).min()),
            "Weeks invested %":float(100 * (d["holding"]!="CASH").mean()),
            "Weeks":len(d)}


def render_dual_momentum(bt, opens, closes, stake, bps):
    st.subheader("Dual Momentum Investigation")
    st.caption("Audit of the existing backtest, using the same signal dates and next-opening-price execution. No live trades.")
    target = bt.loc[bt["strategy"] == "Dual momentum"].sort_values("week").copy()
    spy = bt.loc[bt["strategy"] == "SPY hold"].sort_values("week").copy()
    qqq = bt.loc[bt["strategy"] == "QQQ hold"].sort_values("week").copy()
    if target.empty:
        st.info("Run the backtest to display dual momentum analysis.")
        return

    target["asset"] = target["holding"].str.extract(r"^([A-Z]+)")[0].fillna("CASH")
    target.loc[target["holding"] == "CASH","asset"] = "CASH"
    target = _equity_and_drawdown(target, stake)
    spy = _equity_and_drawdown(spy, stake)
    qqq = _equity_and_drawdown(qqq, stake)
    final = target["equity"].iloc[-1]
    invested = (target["asset"] != "CASH")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Dual momentum ending balance",f"${final:,.2f}",f"{100*(final/stake-1):+.1f}%")
    c2.metric("SPY ending balance",f"${spy['equity'].iloc[-1]:,.2f}")
    c3.metric("Worst drawdown",f"{target['drawdown_pct'].min():.1f}%")
    c4.metric("Time invested",f"{100*invested.mean():.1f}%")

    st.markdown("#### Weekly portfolio values: identical test dates")
    comp = pd.DataFrame({"Dual momentum":target.set_index("exit_date")["equity"],
                         "SPY":spy.set_index("exit_date")["equity"],
                         "QQQ":qqq.set_index("exit_date")["equity"]})
    st.line_chart(comp)
    st.caption("Charts reflect historical simulations, not guaranteed future gains.")

    left,right = st.columns(2)
    with left:
        st.markdown("#### What did the strategy hold?")
        count = target.groupby("asset").agg(Weeks=("week","size"),
                                            Sum_of_weekly_net_returns_pct=("actual",lambda x:100*x.sum()),
                                            Avg_weekly_return_pct=("actual",lambda x:100*x.mean()))
        count = count.sort_values("Weeks",ascending=False)
        st.bar_chart(count["Weeks"])
        st.dataframe(count,use_container_width=True)
        st.caption("Summed weekly returns describe attribution but do not equal compounded contribution. Market exposure is determined from recorded holdings.")
    with right:
        st.markdown("#### Risk and trading behavior")
        st.line_chart(target.set_index("exit_date")["drawdown_pct"])
        changes = (target["asset"] != target["asset"].shift(1)).sum() - 1
        st.metric("Holding changes between weeks",int(max(0,changes)))
        st.metric("Total simulated trading costs",f"${(stake * target['fees']).sum():.2f}",
                  help="Indicative non-compounded cost on constant $ stake, not actual compounded-dollar fees.")
        st.caption("Drawdown measures decline from a previous portfolio peak.")

    st.markdown("#### Period-by-period results")
    phases = []
    for phase, part in target.groupby("phase",sort=False):
        end = stake * (1 + part["actual"]).prod()
        peak = pd.concat([pd.Series([stake]),stake*(1+part["actual"]).cumprod().reset_index(drop=True)],ignore_index=True)
        dd = (peak/peak.cummax()-1).min()
        phases.append({"Period":phase,"Weeks":len(part),"Return %":100*(end/stake-1),
                       "Max drawdown %":100*dd,
                       "Cash weeks %":100*(part["asset"]=="CASH").mean()})
    st.dataframe(pd.DataFrame(phases),hide_index=True,use_container_width=True)
    st.warning("The development / validation / holdout labels describe chronological segments in this backtest. If the rule was chosen after inspecting these results, the holdout is no longer truly untouched.")

    st.markdown("#### Stress test: momentum lookback")
    st.write("This is a diagnostic, not a search for the highest-return setting. Each variation uses the same test dates, 200-day trend filter, next-open entry, and trading-cost assumption.")
    scenarios=[]
    dates = target["signal_date"].tolist()
    for n in [63,126,189,252]:
        row = _parameter_replay(opens,closes,dates,n,bps,stake)
        if row:scenarios.append(row)
    if scenarios:
        st.dataframe(pd.DataFrame(scenarios),hide_index=True,use_container_width=True)
    st.caption("These single-lookback variants are not equivalent to the original weighted 3/6/12-month rule; they probe whether the concept is sensitive to the signal horizon.")

    st.markdown("#### Trade-by-trade journal")
    st.dataframe(target[["week","phase","signal_date","entry_date","exit_date","asset",
                         "predicted","gross","fees","actual","equity","drawdown_pct"]],
                 hide_index=True,use_container_width=True)
    st.download_button("Download dual momentum trade audit",
                       target.to_csv(index=False),"dual_momentum_trade_audit.csv","text/csv")
    st.markdown("**How it works:** rank SPY, QQQ, IWM, GLD and TLT by 40% of their 63-session return plus 30% of their 126-session return plus 30% of their 252-session return. Invest in the highest score only if that score is positive and its close exceeds its 200-session moving average; otherwise hold cash. Re-evaluate every five sessions. Transaction fees use an approximate change-in-target-weights calculation.")
