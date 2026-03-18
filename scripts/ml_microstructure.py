"""ML Strategy using microstructure features: cross-pair cascade, volume, funding, price action.
Walk-forward validation on 1h data."""
import ccxt, pandas as pd, numpy as np, time, lightgbm as lgb
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

data_dir = Path('data')
binance = ccxt.binance({'options':{'defaultType':'future'},'enableRateLimit':True})

pairs = ['BTC/USDT:USDT','ETH/USDT:USDT','SOL/USDT:USDT','XRP/USDT:USDT','DOGE/USDT:USDT',
         'BNB/USDT:USDT','ADA/USDT:USDT','AVAX/USDT:USDT','LINK/USDT:USDT','SUI/USDT:USDT']

def calc_atr(h,l,c,p=14):
    return pd.concat([h-l,abs(h-c.shift()),abs(l-c.shift())],axis=1).max(axis=1).rolling(p,min_periods=1).mean()

# Download 1h klines
klines_1h = {}
for pair in pairs:
    base = pair.split('/')[0]
    print(f'{base}...', end=' ', flush=True)
    all_k = []
    since = int((pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=90)).timestamp()*1000)
    while True:
        ohlcv = binance.fetch_ohlcv(pair, '1h', since=since, limit=1500)
        if not ohlcv: break
        all_k.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1500: break
        time.sleep(0.1)
    if all_k:
        df = pd.DataFrame(all_k, columns=['ts','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        klines_1h[base] = df
        print(f'{len(df)} bars')

# Load funding data
funding = {}
fund_dir = data_dir/'funding'
if fund_dir.exists():
    for f in fund_dir.glob('*_funding.csv'):
        base = f.name.replace('_USDT_USDT_funding.csv','')
        df = pd.read_csv(f)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        df['fundingRate'] = pd.to_numeric(df['fundingRate'])
        funding[base] = df.set_index('timestamp')['fundingRate']

btc = klines_1h.get('BTC')
eth = klines_1h.get('ETH')
btc_atr = calc_atr(btc['high'],btc['low'],btc['close'])
coins = [c for c in klines_1h if c not in ['BTC','ETH']]
print(f'\nBuilding features for {len(coins)} coins...')

rows = []
for coin in coins:
    h1 = klines_1h[coin]
    atr = calc_atr(h1['high'],h1['low'],h1['close'])
    vol_ma = h1['volume'].rolling(24).mean()
    fr = funding.get(coin)
    common = h1.index.intersection(btc.index).intersection(eth.index)
    
    for ts in common:
        i = h1.index.get_loc(ts)
        ib = btc.index.get_loc(ts)
        ie = eth.index.get_loc(ts)
        if i<24 or ib<24 or ie<24 or i>=len(h1)-5: continue
        a = atr.iloc[i]
        if pd.isna(a) or a==0: continue
        ba = btc_atr.iloc[ib]; ba = ba if not pd.isna(ba) and ba>0 else 1
        vm = vol_ma.iloc[i]; vm = vm if not pd.isna(vm) and vm>0 else 1
        
        btc_ret = [(btc['close'].iloc[ib]-btc['close'].iloc[ib-n])/btc['close'].iloc[ib-n] for n in [1,2,4,8]]
        eth_ret = [(eth['close'].iloc[ie]-eth['close'].iloc[ie-n])/eth['close'].iloc[ie-n] for n in [1,2,4,8]]
        coin_ret = [(h1['close'].iloc[i]-h1['close'].iloc[i-n])/h1['close'].iloc[i-n] for n in [1,2,4,8]]
        
        fr_val = 0
        if fr is not None:
            fr_before = fr[fr.index <= ts]
            if len(fr_before) > 0: fr_val = fr_before.iloc[-1] * 100
        
        future = h1.iloc[i+1:min(i+5,len(h1))]
        if len(future) < 3: continue
        max_up = (future['high'].max() - h1['close'].iloc[i]) / a
        max_dn = (h1['close'].iloc[i] - future['low'].min()) / a
        
        rows.append({
            'ts':ts,'coin':coin,'atr':a,'entry_price':h1['open'].iloc[i+1],
            'btc_ret1':btc_ret[0],'btc_ret4':btc_ret[2],'btc_ret8':btc_ret[3],
            'eth_ret1':eth_ret[0],'eth_ret4':eth_ret[2],
            'coin_ret1':coin_ret[0],'coin_ret4':coin_ret[2],'coin_ret8':coin_ret[3],
            'btc_lead_1h':btc_ret[0]-coin_ret[0],'btc_lead_4h':btc_ret[2]-coin_ret[2],
            'eth_lead_1h':eth_ret[0]-coin_ret[0],
            'vol_ratio':h1['volume'].iloc[i]/vm,
            'vol_trend':h1['volume'].iloc[i-2:i+1].mean()/(h1['volume'].iloc[i-8:i-2].mean()+1e-10),
            'vol_spike':h1['volume'].iloc[i]/(h1['volume'].iloc[i-4:i].max()+1e-10),
            'body':(h1['close'].iloc[i]-h1['open'].iloc[i])/a,
            'upper_wick':(h1['high'].iloc[i]-max(h1['close'].iloc[i],h1['open'].iloc[i]))/a,
            'lower_wick':(min(h1['close'].iloc[i],h1['open'].iloc[i])-h1['low'].iloc[i])/a,
            'range_4h':(h1['high'].iloc[i-4:i+1].max()-h1['low'].iloc[i-4:i+1].min())/a,
            'mom_acc':coin_ret[0]-coin_ret[1],'funding':fr_val,
            'hour':ts.hour,'dow':ts.dayofweek,
            'big_up':int(max_up>=1.5),'big_dn':int(max_dn>=1.5),
        })

df = pd.DataFrame(rows).sort_values('ts').reset_index(drop=True)
print(f'\nDataset: {len(df)} samples')

feats = ['btc_ret1','btc_ret4','btc_ret8','eth_ret1','eth_ret4',
         'coin_ret1','coin_ret4','coin_ret8','btc_lead_1h','btc_lead_4h','eth_lead_1h',
         'vol_ratio','vol_trend','vol_spike','body','upper_wick','lower_wick','range_4h',
         'mom_acc','funding','hour','dow']

train_size = int(len(df)*0.5)
test_block = int(len(df)*0.1)

for target, direction, sl_m, tp_m in [
    ('big_dn','SHORT',0.5,1.5),('big_dn','SHORT',0.3,1.5),('big_dn','SHORT',0.5,2.5),
    ('big_up','LONG',0.5,1.5),('big_up','LONG',0.3,1.5),('big_up','LONG',0.5,2.5),
]:
    results = {th:[] for th in [0.25,0.30,0.40,0.50]}
    for start in range(0, len(df)-train_size-test_block, test_block):
        train = df.iloc[start:start+train_size]
        test = df.iloc[start+train_size:start+train_size+test_block]
        model = lgb.LGBMClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,
                                    min_child_samples=30,subsample=0.7,colsample_bytree=0.8,
                                    reg_alpha=0.1,reg_lambda=0.1,verbose=-1)
        model.fit(train[feats], train[target])
        probs = model.predict_proba(test[feats])[:,1]
        for th in results:
            for ii in test[probs>=th].index:
                row=df.iloc[ii]; ep=row['entry_price'];a=row['atr']
                if ep==0 or a==0: continue
                h1=klines_1h.get(row['coin'])
                if h1 is None: continue
                ti=h1.index.searchsorted(row['ts'])+1
                if ti>=len(h1)-1: continue
                sl=ep-a*sl_m if direction=='LONG' else ep+a*sl_m
                tp=ep+a*tp_m if direction=='LONG' else ep-a*tp_m
                xp=ep
                for j in range(ti,min(ti+5,len(h1))):
                    b=h1.iloc[j]
                    if direction=='LONG':
                        if b['low']<=sl:xp=sl;break
                        if b['high']>=tp:xp=tp;break
                    else:
                        if b['high']>=sl:xp=sl;break
                        if b['low']<=tp:xp=tp;break
                else: xp=h1['close'].iloc[min(ti+4,len(h1)-1)]
                results[th].append(((xp-ep)/ep if direction=='LONG' else (ep-xp)/ep)-0.0004)
    
    print(f'\n{direction} {target} SL{sl_m}/TP{tp_m}:')
    for th in sorted(results):
        pnls=results[th]
        if not pnls: continue
        w=sum(1 for p in pnls if p>0);n=len(pnls)
        sw=sum(p for p in pnls if p>0);sl2=abs(sum(p for p in pnls if p<=0)) or 0.001
        print(f'  P>={th:.2f} | {n:>4} | WR {w/n*100:.1f}% | PF {sw/sl2:.2f} | Avg {np.mean(pnls)*100:+.3f}%')

# Feature importance
print(f'\nTop features:')
imp=dict(zip(feats,model.feature_importances_))
for k,v in sorted(imp.items(),key=lambda x:-x[1])[:10]:
    print(f'  {k:15s}: {v}')

# COMPOUND BACKTEST of best config
print(f'\n{"="*70}')
print(f'COMPOUND BACKTEST — Best config with $50')
print(f'{"="*70}')
for leverage in [3,5]:
    for n_slots in [3,5]:
        balance=50.0;trade_list=[];active=[];pk=50;dd=0;monthly={}
        for start in range(0,len(df)-train_size-test_block,test_block):
            train=df.iloc[start:start+train_size]
            test=df.iloc[start+train_size:start+train_size+test_block]
            m_up=lgb.LGBMClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,min_child_samples=30,subsample=0.7,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=0.1,verbose=-1)
            m_dn=lgb.LGBMClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,min_child_samples=30,subsample=0.7,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=0.1,verbose=-1)
            m_up.fit(train[feats],train['big_up']); m_dn.fit(train[feats],train['big_dn'])
            p_up=m_up.predict_proba(test[feats])[:,1]; p_dn=m_dn.predict_proba(test[feats])[:,1]
            
            for ii in range(len(test)):
                row=test.iloc[ii]; ts=row['ts']
                active=[e for e in active if e>ts]
                if len(active)>=n_slots: continue
                pu,pd2 = p_up[ii],p_dn[ii]
                if max(pu,pd2)<0.35: continue
                d='LONG' if pu>pd2 else 'SHORT'
                ep=row['entry_price'];a=row['atr']
                if ep==0 or a==0: continue
                h1=klines_1h.get(row['coin'])
                if h1 is None: continue
                ti=h1.index.searchsorted(ts)+1
                if ti>=len(h1)-1: continue
                sl_m,tp_m=0.5,1.5
                sl=ep-a*sl_m if d=='LONG' else ep+a*sl_m
                tp=ep+a*tp_m if d=='LONG' else ep-a*tp_m
                pos=balance*(1.0/n_slots)*leverage
                xp=ep; exit_ts=h1.index[min(ti+4,len(h1)-1)]
                for j in range(ti,min(ti+5,len(h1))):
                    b=h1.iloc[j]
                    if d=='LONG':
                        if b['low']<=sl:xp=sl;exit_ts=b.name;break
                        if b['high']>=tp:xp=tp;exit_ts=b.name;break
                    else:
                        if b['high']>=sl:xp=sl;exit_ts=b.name;break
                        if b['low']<=tp:xp=tp;exit_ts=b.name;break
                else: xp=h1['close'].iloc[min(ti+4,len(h1)-1)]
                active.append(exit_ts)
                pnl_pct=(xp-ep)/ep if d=='LONG' else (ep-xp)/ep
                net=pos*pnl_pct-pos*0.0004
                balance+=net; balance=max(balance,0.5)
                pk=max(pk,balance); dd=max(dd,(pk-balance)/pk*100)
                trade_list.append({'net':net,'bal':balance,'ts':ts})
                monthly[ts.strftime('%Y-%m')]=balance
        
        if not trade_list: continue
        w=sum(1 for t in trade_list if t['net']>0)
        days=(trade_list[-1]['ts']-trade_list[0]['ts']).total_seconds()/86400
        print(f'\n{n_slots}slot {leverage}x | {len(trade_list)} trades ({len(trade_list)/days*7:.0f}/wk) | WR {w/len(trade_list)*100:.1f}% | $50->${balance:,.0f} | DD {dd:.0f}%')
        for m,b in sorted(monthly.items()): print(f'  {m}: ${b:,.0f}')
