"""Pure strategy engine, separated from Streamlit for reproducible testing."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TICKERS = ('SPY', 'QQQ', 'IWM', 'GLD', 'TLT')
MIX = {'SPY': .4, 'QQQ': .2, 'IWM': .1, 'GLD': .15, 'TLT': .15}
STRATEGIES = ('SPY hold', 'QQQ hold', 'Diversified mix', 'Momentum rotation',
              'ML ETF picker', 'Dual momentum', 'Mean reversion',
              'Volatility adjusted', 'ML ensemble')
FEATURES = ('r5', 'r10', 'r20', 'r60', 'r120', 'vol20', 'vol60', 'ma50', 'ma200', 'rsi14')


def normalize_data(raw):
    """Accept {ticker: {'Open': series, 'Close': series}} or standard yfinance dataframe."""
    if isinstance(raw, dict):
        opens = pd.DataFrame({t:raw[t]['Open'] for t in TICKERS})
        closes = pd.DataFrame({t:raw[t]['Close'] for t in TICKERS})
    else:
        try:
            opens = pd.DataFrame({t:raw[t]['Open'] for t in TICKERS})
            closes = pd.DataFrame({t:raw[t]['Close'] for t in TICKERS})
        except (KeyError, TypeError):
            opens = raw['Open'][list(TICKERS)]
            closes = raw['Close'][list(TICKERS)]
    opens, closes = opens.astype(float), closes.astype(float)
    idx = opens.index.intersection(closes.index)
    opens, closes = opens.loc[idx].sort_index(), closes.loc[idx].sort_index()
    common = opens.notna().all(axis=1) & closes.notna().all(axis=1)
    opens, closes = opens.loc[common], closes.loc[common]
    if len(closes) < 700:
        raise ValueError('Need at least 700 complete daily bars across all five ETFs.')
    if (opens <= 0).any().any() or (closes <= 0).any().any():
        raise ValueError('Price data contains nonpositive values.')
    return opens, closes


def features_for(opens, closes):
    """Features available as of each close; labels use next open -> open five sessions later.
    Label at j becomes knowable only at index j+6. Never train j > i-6 at decision i.
    """
    d = {}
    for t in TICKERS:
        s, o = closes[t], opens[t]
        returns = s.pct_change()
        z = pd.DataFrame(index=closes.index)
        for n in (5, 10, 20, 60, 120):
            z[f'r{n}'] = s.pct_change(n)
        z['vol20'] = returns.rolling(20).std()
        z['vol60'] = returns.rolling(60).std()
        z['ma50'] = s / s.rolling(50).mean() - 1
        z['ma200'] = s / s.rolling(200).mean() - 1
        delta = s.diff()
        ups = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        downs = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        z['rsi14'] = 100 * ups / (ups + downs).replace(0, np.nan)
        z['target'] = o.shift(-6) / o.shift(-1) - 1
        d[t] = z.replace([np.inf,-np.inf],np.nan)
    return d


def get_predictions(feat, i, use_ml=True):
    """Train only on fully-known observations. Return per-ticker forecasts for 2 estimators.
    For index i, j+6 <= i means training examples j <= i-6.
    iloc[:i-5] includes exactly j = 0 ... i-6.
    """
    if not use_ml:
        return {t:0.0 for t in TICKERS}, {t:0.0 for t in TICKERS}
    blocks=[feat[t].iloc[:i-5].dropna(subset=list(FEATURES)+['target']) for t in TICKERS]
    train = pd.concat(blocks, ignore_index=True)
    if len(train) < 500:
        raise ValueError('Not enough known historical labels for the model.')
    X, y = train[list(FEATURES)], train['target'].clip(-.3,.3)
    models = [
        make_pipeline(StandardScaler(),Ridge(alpha=120)),
        RandomForestRegressor(n_estimators=32, max_depth=5, min_samples_leaf=25,
                              max_features=.8, random_state=37, n_jobs=-1),
    ]
    current = pd.concat([feat[t].iloc[[i]][list(FEATURES)] for t in TICKERS],ignore_index=True)
    if current.isna().any().any():
        raise ValueError('Current feature observations are incomplete.')
    out=[]
    for m in models:
        m.fit(X, y)
        predicted=np.clip(m.predict(current),-.12,.12)
        out.append(dict(zip(TICKERS,map(float,predicted))))
    return out[0],out[1]


def weights_for(closes, i, ridge, forest):
    """Returns strategy => (weights, predicted weekly return). cash implied by sum(weights)<1."""
    series=closes.iloc[:i+1]
    last=series.iloc[-1]
    ma200=series.rolling(200).mean().iloc[-1]
    mom20=last / series.iloc[-21]-1
    mom63=last / series.iloc[-64]-1
    mom126=last / series.iloc[-127]-1
    mom252=last / series.iloc[-253]-1
    r=series.pct_change().tail(61)
    vol20=r.tail(20).std().clip(lower=.002)
    forest_pred=forest
    avg={t:(ridge[t]+forest[t])/2 for t in TICKERS}
    def pos(t):return ({t:1.0} if t is not None else {})
    def forecast(w,pred=avg):return float(sum(pred[t]*v for t,v in w.items()))
    trend=[t for t in TICKERS if last[t]>ma200[t]]
    momentum=max(trend,key=lambda t:float(mom20[t])) if trend else None
    ml=max(TICKERS,key=lambda t:forest_pred[t]); ml=ml if forest_pred[ml]>0 else None
    # Dual momentum: relative 3/6/12m momentum + absolute positive long-run momentum.
    dscore={t:float(.4*mom63[t]+.3*mom126[t]+.3*mom252[t]) for t in TICKERS}
    dual=max(dscore,key=dscore.get)
    dual=dual if dscore[dual]>0 and last[dual]>ma200[dual] else None
    # Mean reversion: buy most oversold liquid equity ETF if above 200 MA;
    # otherwise sit in cash. This is deliberately an easy-to-audit baseline.
    candidates=[t for t in ('SPY','QQQ','IWM') if last[t]>ma200[t] and mom20[t]<-.03]
    revert=min(candidates,key=lambda t:float(mom20[t])) if candidates else None
    # Volatility inverse weighted stocks/bonds/gold, with 200 MA safety gate.
    eligible=[t for t in TICKERS if last[t]>ma200[t]]
    if eligible:
        inv={t:1/float(vol20[t]) for t in eligible}
        total=sum(inv.values()); riskw={t:w/total for t,w in inv.items()}
        # Cap each ETF at 40% and leave remainder cash rather than re-leverage.
        riskw={t:min(w,.4) for t,w in riskw.items()}
    else:riskw={}
    ensemble=max(TICKERS,key=lambda t:avg[t]); ensemble=ensemble if avg[ensemble]>0 else None
    result={
        'SPY hold':(pos('SPY'),ridge['SPY']),
        'QQQ hold':(pos('QQQ'),ridge['QQQ']),
        'Diversified mix':(MIX.copy(), forecast(MIX)),
        'Momentum rotation':(pos(momentum),float(mom20[momentum]/4) if momentum else 0.0),
        'ML ETF picker':(pos(ml),float(forest_pred[ml]) if ml else 0.0),
        'Dual momentum':(pos(dual),float(dscore[dual]/13) if dual else 0.0),
        'Mean reversion':(pos(revert),float(-mom20[revert]/4) if revert else 0.0),
        'Volatility adjusted':(riskw,forecast(riskw)),
        'ML ensemble':(pos(ensemble),float(avg[ensemble]) if ensemble else 0.0),
    }
    return result


def realized_week(opens, i, weights):
    """Enter at next session's open (i+1), exit five sessions later (i+6)."""
    if i+6 >= len(opens):raise IndexError('Future opening prices not yet observed.')
    entry,exit=opens.iloc[i+1], opens.iloc[i+6]
    return float(sum(weight*(exit[t]/entry[t]-1) for t,weight in weights.items()))


def turnover_cost(new, old, bps):
    """Simple absolute target-weight turnover cost; includes sells and buys.
    Positions drift during holding period, so this is still an approximation.
    """
    return bps/10000 * sum(abs(new.get(t,0)-old.get(t,0)) for t in TICKERS)


def run_backtest(opens, closes, weeks=104, bps=5, model_enabled=True):
    feat=features_for(opens,closes)
    # Signals every five sessions, with non-overlapping open-to-open holding windows.
    # Keep a minimum 520-session training warm-up.
    i0=max(520,len(closes)-6-5*weeks)
    indexes=list(range(i0,len(closes)-6,5))
    results=[]
    previous={s:{} for s in STRATEGIES}
    for wi,i in enumerate(indexes):
        ridge,forest=get_predictions(feat,i,model_enabled)
        portfolios=weights_for(closes,i,ridge,forest)
        phase=('Development' if wi < .6*len(indexes) else
               'Validation' if wi < .8*len(indexes) else 'Holdout')
        for strategy,(weights,forecast) in portfolios.items():
            # Charges turnover on entering/rebalancing; no arbitrary per-week fee for hold portfolios.
            fees=turnover_cost(weights,previous[strategy],bps)
            actual_gross=realized_week(opens,i,weights)
            actual=actual_gross-fees
            base='SPY'
            # Simple always-up baseline based on positive recent drift, not ML.
            baseline_up=max(0.,float(closes['SPY'].iloc[i]/closes['SPY'].iloc[i-21]-1)/4)
            results.append(dict(week=wi+1,phase=phase,signal_date=str(closes.index[i].date()),
                 entry_date=str(opens.index[i+1].date()),exit_date=str(opens.index[i+6].date()),
                 strategy=strategy,holding='/'.join(f'{t}:{w:.2f}' for t,w in weights.items()) or 'CASH',
                 weights=repr(weights),predicted=float(forecast),actual=float(actual),
                 gross=float(actual_gross),fees=float(fees),
                 baseline_up=baseline_up,baseline_zero=0.0,
                 error_pp=abs(forecast-actual_gross)*100,
                 direction_correct=int((forecast>0)==(actual_gross>0)),
                 up_baseline_error_pp=abs(baseline_up-actual_gross)*100,
                 zero_baseline_error_pp=abs(actual_gross)*100))
            previous[strategy]=weights
    return pd.DataFrame(results)


def performance(data, capital=100):
    if data.empty:return pd.DataFrame(), pd.DataFrame()
    curves=[]; summary=[]
    for strategy, g in data.groupby('strategy',sort=False):
        g=g.sort_values('week').copy()
        g['equity']=capital*(1+g['actual']).cumprod()
        line=pd.concat([pd.Series([capital]),g['equity'].reset_index(drop=True)],ignore_index=True)
        drawdown=line/line.cummax()-1
        vol=g['actual'].std()
        # Approximate annualization with 52 equal-length five-session weeks.
        sharpe=(g['actual'].mean()/vol*np.sqrt(52)) if vol and np.isfinite(vol) else np.nan
        profits=g.loc[g['actual']>0,'actual'].sum()
        losses=-g.loc[g['actual']<0,'actual'].sum()
        summary.append({'Strategy':strategy,'Weeks':len(g),'Final $':float(g['equity'].iloc[-1]),
                        'Total return %':100*(float(g['equity'].iloc[-1])/capital-1),
                        'Weekly accuracy %':100*g['direction_correct'].mean(),
                        'Prediction MAE (pp)':g['error_pp'].mean(),
                        'Always-up MAE (pp)':g['up_baseline_error_pp'].mean(),
                        'Zero MAE (pp)':g['zero_baseline_error_pp'].mean(),
                        'Win rate %':100*(g['actual']>0).mean(),
                        'Max drawdown %':100*drawdown.min(),
                        'Sharpe (approx)':float(sharpe),
                        'Profit factor':float(profits/losses) if losses else np.nan,
                        'Avg weekly return %':100*g['actual'].mean()})
        curves.append(g)
    return pd.DataFrame(summary).sort_values('Final $',ascending=False).reset_index(drop=True),pd.concat(curves)


def score_log(opens, closes, saved):
    if saved.empty:return saved
    out=[]
    for _,raw in saved.iterrows():
        row=raw.to_dict()
        date=pd.Timestamp(row.get('as_of',row.get('signal_date')))
        if date not in closes.index:
            row.update(status='Date not found',actual=np.nan);out.append(row);continue
        i=closes.index.get_loc(date)
        if i+6>=len(opens):
            row.update(status='Pending',actual=np.nan);out.append(row);continue
        import ast
        try:weights=ast.literal_eval(row['weights'])
        except (ValueError,TypeError,SyntaxError,KeyError):
            holding=str(row.get('holding','CASH'))
            if holding=='CASH':weights={}
            elif holding=='MIX':weights=MIX.copy()
            elif holding in TICKERS:weights={holding:1.0}
            else:row.update(status='Unsupported legacy holding',actual=np.nan);out.append(row);continue
        actual=realized_week(opens,i,weights)
        predicted=float(row.get('predicted',row.get('forecast',0)))
        capital=float(row.get('starting_value',100))
        row.update(status='Scored',actual=actual,actual_value=capital*(1+actual),
                   error_pp=100*abs(predicted-actual),
                   direction_correct=int((predicted>0)==(actual>0)))
        out.append(row)
    return pd.DataFrame(out)
