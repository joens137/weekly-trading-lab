from pathlib import Path
import json
import pandas as pd
import streamlit as st
import yfinance as yf
from engine import (TICKERS,STRATEGIES,normalize_data,features_for,get_predictions,
                    weights_for,run_backtest,performance,score_log)
from dual_momentum_analysis import render_dual_momentum

st.set_page_config(page_title='Weekly Trading Lab | Research Edition',page_icon='📈',layout='wide')
st.markdown('''<style>.block-container{max-width:1500px;padding-top:1.7rem}
[data-testid="stMetric"]{padding:1rem;border:1px solid #8193a535;border-radius:14px;background:#7b94ae0d}
h1,h2,h3{letter-spacing:-.02em}</style>''',unsafe_allow_html=True)
HERE=Path(__file__).resolve().parent
LOG=HERE/'saved_forecasts.csv'

@st.cache_data(ttl=3600,show_spinner=False)
def prices():
    raw=yf.download(list(TICKERS),period='10y',interval='1d',auto_adjust=True,progress=False,
                    group_by='ticker',threads=True)
    return normalize_data(raw)

@st.cache_data(show_spinner=False)
def history(opens, closes, n, bps, fast_mode):
    return run_backtest(opens,closes,n,bps,model_enabled=not fast_mode)

@st.cache_data(show_spinner=False)
def current_choices(opens, closes):
    f=features_for(opens,closes)
    ridge,forest=get_predictions(f,len(closes)-1)
    return weights_for(closes,len(closes)-1,ridge,forest)

st.title('📈 Weekly Trading Lab')
st.caption('Research Edition · nine trading strategies · walk-forward forecasting · virtual money only')
st.info('Backtests are simulations, not predictions of guaranteed gains. This tool never places real trades.')
with st.sidebar:
    st.header('Research settings')
    stake=st.number_input('Virtual dollars per strategy',min_value=10.0,max_value=100000.,value=100.,step=50.)
    weeks=st.slider('Historical weeks to evaluate',26,156,104,13)
    bps=st.slider('Cost per traded notional (basis points)',0,30,5)
    fast=st.checkbox('Fast mode (no trained ML)',value=False,
                     help='Explore rule-based strategies quickly. ML strategies become inactive/neutral until full mode runs.')
    if st.button('🔄 Refresh Yahoo Finance data'):
        prices.clear();history.clear();current_choices.clear();st.rerun()
    st.divider()
    st.caption('Decisions after a daily close. Simulated execution at the next open. Exit five sessions later. All strategies use the same signal dates.')

try:
    with st.spinner('Downloading ten years of adjusted ETF prices…'):
        opens,closes=prices()
except Exception as e:
    st.error(f'Market prices could not be downloaded: {e}')
    st.stop()
st.caption(f'Latest price observation: {closes.index[-1].date()} · {len(closes):,} daily bars across five ETFs')
try:
    with st.spinner('Walk-forward simulation is running. Full ML mode retrains two models every week and may take several minutes on first use…'):
        bt=history(opens,closes,weeks,bps,fast)
        board,curves=performance(bt,stake)
except Exception as e:
    st.error(f'Historical simulation failed: {e}')
    st.stop()
if board.empty:
    st.error('No historical weeks could be tested.');st.stop()

val=bt[bt.phase=='Validation']
val_board,_=performance(val,stake)
selected=str(val_board.iloc[0]['Strategy'])
test=bt[bt.phase=='Holdout']
test_board,_=performance(test,stake)
test_pick=test_board[test_board.Strategy==selected].iloc[0]
spy_test=test_board[test_board.Strategy=='SPY hold'].iloc[0]

m1,m2,m3,m4=st.columns(4)
m1.metric('Selected on validation',selected)
m2.metric('Selected strategy: holdout',f"{test_pick['Total return %']:+.1f}%")
m3.metric('SPY: same holdout',f"{spy_test['Total return %']:+.1f}%")
m4.metric('Holdout prediction accuracy',f"{test_pick['Weekly accuracy %']:.0f}%")
st.caption('The selected strategy is ranked by validation-period return, then evaluated on the later holdout. Repeatedly changing settings after seeing holdout performance makes it less independent.')

tabs=st.tabs(['🏆 Strategy competition','🔎 Dual Momentum','📡 Synced research','🎯 Prediction accuracy','🧪 Validation & holdout','🔮 Next-week forecasts','💵 Paper portfolios','📖 Methods'])
with tabs[0]:
    st.subheader('Historical portfolio growth')
    pivot=curves.pivot(index='exit_date',columns='strategy',values='equity')
    st.line_chart(pivot)
    st.subheader('Performance leaderboard: full replay')
    st.dataframe(board,hide_index=True,use_container_width=True)
    st.download_button('Download complete backtest',bt.to_csv(index=False),'research_backtest.csv','text/csv')
with tabs[1]:
    render_dual_momentum(bt,opens,closes,stake,bps)
with tabs[2]:
    st.subheader('Automated GitHub research snapshots')
    st.write('These results are generated independently by a scheduled GitHub Actions job, not taken from this browser session. The public files contain simulated research results only.')
    public=HERE/'public_results'
    statusfile=public/'sync_status.json'
    if not statusfile.exists():
        st.info('No published snapshot yet. After this feature is merged, run the "Publish weekly research results" workflow once from GitHub Actions. Weekly updates are scheduled thereafter.')
    else:
        try:
            status=json.loads(statusfile.read_text(encoding='utf-8'))
            a,b,c=st.columns(3)
            a.metric('Last synced (UTC)',str(status.get('generated_at_utc','Unknown')))
            b.metric('Latest market date',str(status.get('latest_market_date','Unknown')))
            c.metric('Scored forward forecasts',str(status.get('forecasts_scored','Unknown')))
            st.caption('Standardized run: 104 weeks, $100 per strategy, estimated five-basis-point transaction costs and trained ML models.')
            for filename,heading in [('strategy_leaderboard.csv','Published strategy leaderboard'),
                                      ('dual_momentum_trades.csv','Published dual momentum trades'),
                                      ('weekly_forecasts.csv','Saved forward forecasts and scored outcomes')]:
                path=public/filename
                if path.exists():
                    data=pd.read_csv(path)
                    st.markdown(f'#### {heading}')
                    st.dataframe(data.tail(200) if filename!='strategy_leaderboard.csv' else data,hide_index=True,use_container_width=True)
                    st.download_button(f'Download {filename}',path.read_bytes(),file_name=filename,mime='text/csv',key=f'sync_{filename}')
        except Exception as e:
            st.warning(f'Could not display synced research data: {e}')
    st.caption('Scheduled workflows can run late or fail. Review the GitHub Actions history if the timestamp is old. Forecast snapshots are immutable for a given market date and strategy; historical backtests recalculate on a moving data window.')
with tabs[3]:
    st.subheader('Forecasts versus reality')
    strategy=st.selectbox('View strategy',list(STRATEGIES))
    s=bt[bt.strategy==strategy].copy()
    x=s.set_index('exit_date')[['predicted','gross','baseline_up','baseline_zero']]*100
    x.columns=['Model forecast (%)','Actual price return (%)','Always-up heuristic (%)','Predict zero (%)']
    st.line_chart(x)
    left,right=st.columns(2)
    with left:st.metric('Correct up/down calls',f"{100*s.direction_correct.mean():.1f}%")
    with right:st.metric('Mean absolute forecast error',f"{s.error_pp.mean():.2f} percentage points")
    st.write('The always-up and predict-zero lines are simple forecasting baselines. They are not guaranteed investment strategies.')
    st.dataframe(s[['signal_date','entry_date','exit_date','holding','predicted','gross','actual','error_pp','direction_correct']],
                 hide_index=True,use_container_width=True)
with tabs[4]:
    st.subheader('Chronological evaluation')
    st.write('The trading model is re-trained using only labels observable at each historical signal date. The performance display is divided into development, validation, and later holdout segments.')
    for phase in ['Development','Validation','Holdout']:
        p=bt[bt.phase==phase]
        frame,_=performance(p,stake)
        st.markdown(f'**{phase} · {p.week.nunique()} evaluated weeks · {p.signal_date.min()} to {p.exit_date.max()}**')
        st.dataframe(frame[['Strategy','Total return %','Weekly accuracy %','Prediction MAE (pp)','Max drawdown %','Sharpe (approx)']],
                     hide_index=True,use_container_width=True)
    st.warning('Only select settings and strategies using development and validation. Changing the model after seeing the holdout can overfit the holdout.')
with tabs[5]:
    st.subheader('Lock in forecasts before future outcomes are known')
    try:
        choices=current_choices(opens,closes)
    except Exception as e:
        st.error(f'Latest model forecast could not be created: {e}')
        choices={}
    today=str(closes.index[-1].date())
    forecast_rows=[]
    for strategy,(weights,predicted) in choices.items():
        forecast_rows.append(dict(as_of=today,strategy=strategy,weights=repr(weights),
            holding=', '.join(f'{k}: {v:.0%}' for k,v in weights.items()) or 'CASH',
            predicted=float(predicted),starting_value=float(stake),
            forecast_value=float(stake*(1+predicted)),model='Ridge + random forest; strategy-specific'))
    forecast_df=pd.DataFrame(forecast_rows)
    if not forecast_df.empty:
        v=forecast_df[['strategy','holding','predicted','forecast_value']].copy()
        v['predicted']=v['predicted'].map(lambda x:f'{x:+.2%}')
        v['forecast_value']=v['forecast_value'].map(lambda x:f'${x:,.2f}')
        st.dataframe(v,hide_index=True,use_container_width=True)
        st.caption('Forecasts are speculative point estimates. A prediction of +1% does not mean there is a reliable 1% expected gain.')
        if st.button('💾 Save all nine forecasts',type='primary'):
            try:existing=pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
            except Exception:existing=pd.DataFrame()
            if not existing.empty and today in existing.get('as_of',pd.Series(dtype=str)).astype(str).values:
                st.info('A snapshot from this market date is already saved. Original forecasts were not overwritten.')
            else:
                merged=pd.concat([existing,forecast_df],ignore_index=True)
                merged.to_csv(LOG,index=False)
                st.success('Saved. Reopen this dashboard after six more daily price observations (entry plus five holding sessions) to score the forecast.')
    if LOG.exists():
        try:
            saved=pd.read_csv(LOG)
            scored=score_log(opens,closes,saved)
            st.subheader('Recorded predictions and their outcomes')
            st.dataframe(scored,hide_index=True,use_container_width=True)
            st.download_button('Download saved forecast journal',scored.to_csv(index=False),'forecast_journal.csv','text/csv')
        except Exception as e:st.warning(f'Could not read saved forecast file: {e}')
with tabs[6]:
    st.subheader('Virtual portfolio competition')
    st.write(f'Every strategy begins with **${stake:,.2f}**. Results compound non-overlapping five-session trades independently.')
    st.bar_chart(board.set_index('Strategy')['Final $'])
    st.dataframe(board[['Strategy','Final $','Total return %','Max drawdown %','Sharpe (approx)','Profit factor','Win rate %']],
                 hide_index=True,use_container_width=True)
    st.caption('Trading costs are simulated from changes in target weights. Fractional shares and cash at zero yield are assumed.')
with tabs[7]:
    st.markdown('''
### Strategy definitions
- **SPY / QQQ hold:** Long-only market baselines; buy and remain invested.
- **Diversified mix:** SPY 40%, QQQ 20%, IWM 10%, GLD 15%, TLT 15%.
- **Momentum rotation:** Buy the highest 20-session momentum ETF above its 200-day moving average; otherwise cash.
- **ML ETF picker:** Choose an ETF with the highest positive random-forest forecast; otherwise cash.
- **Dual momentum:** Compare 3/6/12-month momentum, require positive absolute trend and a price above the 200-day average.
- **Mean reversion:** Buy the most oversold of SPY/QQQ/IWM if its 20-day decline exceeds 3% and its long-term trend remains positive; otherwise cash.
- **Volatility adjusted:** Inverse-volatility weights for ETFs above their 200-day moving averages, capped at 40% per ETF; unallocated cash stays idle.
- **ML ensemble:** Average a regularized linear model and random forest, select the strongest positive forecast; otherwise cash.

### Evaluation integrity
At decision-day close **t**, models may train only on labels whose outcome was already observed by **t**.
Entry is at adjusted opening price **t+1**, with liquidation at opening **t+6**. Historical decisions are five sessions apart, preventing overlapping trades.
The research dashboard splits evaluated weeks into 60% development, 20% validation, 20% later holdout. These are *evaluation segments*, not a random shuffle.
Forecast errors compare estimated versus gross underlying portfolio returns, while net strategy performance deducts turnover costs.

### Limitations
Adjusted historical OHLC prices may not match broker fills; using next-open prices does **not** simulate opening auction orders, slippage, bid–ask spreads, dividends timing, taxes, liquidity limits, financing, or portfolio drift precisely. Transaction-cost estimates are simplified.
The same five ETFs and strategy definitions were chosen by humans, which can introduce selection bias. The holdout is weakened each time someone uses it to choose a new model. Short holding horizons are particularly noisy. Sharpe ratios are approximations using 52 five-session periods/year.
Cash has no interest return. This app never connects to a brokerage or places real orders.

**To keep prior saved forecasts:** Copy `saved_forecasts.csv` from your old lab into this folder. Back it up first. Older single-ticker or diversified-MIX records can be scored; unknown legacy formats remain visible but may require manual migration.
''')
