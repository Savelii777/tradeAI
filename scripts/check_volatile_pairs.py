#!/usr/bin/env python3
"""Check model predictions on volatile pairs"""

import sys
import joblib
import ccxt
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_mtf import MTFFeatureEngine

MODEL_DIR = Path("models/v8_improved")

def add_volume_features(df):
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df

def calculate_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def main():
    models = {
        'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
        'features': joblib.load(MODEL_DIR / 'feature_names.joblib')
    }
    
    binance = ccxt.binance({'options': {'defaultType': 'future'}})
    mtf_fe = MTFFeatureEngine()
    
    cumsum_patterns = ['bars_since_swing', 'consecutive_up', 'consecutive_down', 'obv', 'volume_delta_cumsum', 'swing_high_price', 'swing_low_price']
    features = [f for f in models['features'] if not any(p in f.lower() for p in cumsum_patterns)]
    
    pairs = ['PIPPIN/USDT:USDT', 'HYPE/USDT:USDT', 'TAO/USDT:USDT', 'ASTER/USDT:USDT', 'SUI/USDT:USDT']
    
    for pair in pairs:
        print(f'\n{"="*60}')
        print(f'ANALYZING: {pair}')
        print(f'{"="*60}')
        
        try:
            data = {}
            for tf in ['1m', '5m', '15m']:
                candles = binance.fetch_ohlcv(pair, tf, limit=1000)
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)
                data[tf] = df
            
            ft = mtf_fe.align_timeframes(data['1m'], data['5m'], data['15m'])
            ft = ft.join(data['5m'][['open', 'high', 'low', 'close', 'volume']])
            ft = add_volume_features(ft)
            ft['atr'] = calculate_atr(ft)
            ft = ft.dropna()
            
            for f in features:
                if f not in ft.columns:
                    ft[f] = 0.0
            
            X = ft[features].values.astype(np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            proba = models['direction'].predict_proba(X)
            preds = np.argmax(proba, axis=1)
            confs = np.max(proba, axis=1)
            
            print(f'Total candles: {len(ft)}')
            print(f'Direction: DOWN={np.mean(preds==0)*100:.1f}%, SIDEWAYS={np.mean(preds==1)*100:.1f}%, UP={np.mean(preds==2)*100:.1f}%')
            print(f'Avg confidence: {confs.mean():.3f}')
            
            long_signals = (preds == 2) & (confs > 0.5)
            short_signals = (preds == 0) & (confs > 0.5)
            print(f'LONG signals (conf>0.5): {long_signals.sum()}')
            print(f'SHORT signals (conf>0.5): {short_signals.sum()}')
            
            if long_signals.sum() > 0:
                print(f'   LONG avg conf: {confs[long_signals].mean():.3f}')
            if short_signals.sum() > 0:
                print(f'   SHORT avg conf: {confs[short_signals].mean():.3f}')
            
            print(f'\nLast 10 predictions:')
            for i in range(-10, 0):
                ts = ft.index[i]
                p = preds[i]
                c = confs[i]
                dir_str = ['SHORT', 'SIDE', 'LONG'][p]
                print(f'   {ts}: {dir_str} conf={c:.2f}')
                
        except Exception as e:
            print(f'Error: {e}')
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
