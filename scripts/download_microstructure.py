"""Download microstructure data from Binance Futures API:
- 1h klines WITH taker buy/sell volume
- Open Interest history
- Long/Short account ratio
- Taker buy/sell volume ratio
"""
import requests, pandas as pd, numpy as np, time, json
from pathlib import Path

data_dir = Path('data')

with open('config/pairs_20.json') as f:
    pairs = [p['symbol'] for p in json.load(f)['pairs']]

def to_binance(pair):
    return pair.split('/')[0] + 'USDT'

end_ms = int(pd.Timestamp.now(tz='UTC').timestamp()*1000)
start_ms = end_ms - 90*24*3600*1000

# 1. KLINES WITH TAKER BUY VOLUME (1h)
print('=== Downloading 1h klines with taker buy volume ===')
klines_dir = data_dir / 'klines_1h'
klines_dir.mkdir(exist_ok=True)

for pair in pairs:
    sym = to_binance(pair)
    base = pair.split('/')[0]
    if (klines_dir / f'{base}_1h.csv').exists():
        print(f'{sym}: already exists, skipping')
        continue
    all_k = []
    s = start_ms
    while s < end_ms:
        url = f'https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1h&startTime={s}&limit=1500'
        r = requests.get(url)
        if r.status_code != 200:
            print(f'Error {sym}: {r.text}')
            break
        data = r.json()
        if not data:
            break
        all_k.extend(data)
        s = data[-1][0] + 1
        time.sleep(0.1)
    
    if all_k:
        df = pd.DataFrame(all_k, columns=['ts','open','high','low','close','volume','close_time',
                                          'quote_vol','n_trades','taker_buy_vol','taker_buy_quote','_'])
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        for c in ['open','high','low','close','volume','taker_buy_vol','quote_vol']:
            df[c] = pd.to_numeric(df[c])
        df['n_trades'] = pd.to_numeric(df['n_trades'])
        df['taker_sell_vol'] = df['volume'] - df['taker_buy_vol']
        df['buy_ratio'] = df['taker_buy_vol'] / df['volume'].replace(0, np.nan)
        df.to_csv(klines_dir / f'{base}_1h.csv', index=False)
        print(f'{sym}: {len(df)} bars')

# 2. OPEN INTEREST KLINES (1h) - limited to 500 records
print('\n=== Downloading OI klines ===')
oi_dir = data_dir / 'oi_1h'
oi_dir.mkdir(exist_ok=True)

for pair in pairs:
    sym = to_binance(pair)
    base = pair.split('/')[0]
    if (oi_dir / f'{base}_oi.csv').exists():
        print(f'{sym}: already exists')
        continue
    url = f'https://fapi.binance.com/futures/data/openInterestHist?symbol={sym}&period=1h&limit=500'
    r = requests.get(url)
    if r.status_code != 200:
        print(f'Error {sym}: {r.text}')
        continue
    data = r.json()
    if data:
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.to_csv(oi_dir / f'{base}_oi.csv', index=False)
        print(f'{sym}: {len(df)} OI records')
    time.sleep(0.3)

# 3. LONG/SHORT RATIO (1h)
print('\n=== Downloading Long/Short ratio ===')
ls_dir = data_dir / 'longshort_1h'
ls_dir.mkdir(exist_ok=True)

for pair in pairs:
    sym = to_binance(pair)
    base = pair.split('/')[0]
    if (ls_dir / f'{base}_ls.csv').exists():
        print(f'{sym}: already exists')
        continue
    url = f'https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={sym}&period=1h&limit=500'
    r = requests.get(url)
    if r.status_code != 200:
        print(f'Error {sym}: {r.text}')
        continue
    data = r.json()
    if data:
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.to_csv(ls_dir / f'{base}_ls.csv', index=False)
        print(f'{sym}: {len(df)} L/S records')
    time.sleep(0.3)

# 4. TAKER BUY/SELL VOLUME RATIO (1h)
print('\n=== Downloading Taker Buy/Sell ratio ===')
taker_dir = data_dir / 'taker_1h'
taker_dir.mkdir(exist_ok=True)

for pair in pairs:
    sym = to_binance(pair)
    base = pair.split('/')[0]
    if (taker_dir / f'{base}_taker.csv').exists():
        print(f'{sym}: already exists')
        continue
    url = f'https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={sym}&period=1h&limit=500'
    r = requests.get(url)
    if r.status_code != 200:
        print(f'Error {sym}: {r.text}')
        continue
    data = r.json()
    if data:
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.to_csv(taker_dir / f'{base}_taker.csv', index=False)
        print(f'{sym}: {len(df)} taker records')
    time.sleep(0.3)

print('\nDone!')
