#!/usr/bin/env python3
"""
LIVE TRADING SCANNER v4 - Стабильная версия
- Загружает 1500 свечей (достаточно для стабильных фичей)
- Работает в цикле каждые N минут
- Логирует все сигналы
"""

import sys
import time
import json
import joblib
import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_mtf import MTFFeatureEngine

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path(__file__).parent.parent / "models" / "v8_improved"
PAIRS_FILE = Path(__file__).parent.parent / "config" / "pairs_list.json"
LOG_FILE = Path(__file__).parent.parent / "logs" / "live_scanner.log"

# Thresholds
MIN_CONF = 0.50
MIN_TIMING = 0.8
MIN_STRENGTH = 1.4

# Сколько свечей загружать (1500 достаточно для стабильных фичей)
CANDLES_M5 = 1500
CANDLES_M1 = 1500
CANDLES_M15 = 500

# Интервал сканирования (минуты)
SCAN_INTERVAL_MINUTES = 5


# ============================================================
# FEATURES
# ============================================================
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ============================================================
# DATA LOADING
# ============================================================
def fetch_candles(exchange, pair: str, timeframe: str, total_needed: int, verbose: bool = False) -> Optional[pd.DataFrame]:
    """Загружает много свечей несколькими запросами"""
    try:
        all_candles = []
        limit = 1000
        
        candles = exchange.fetch_ohlcv(pair, timeframe, limit=limit)
        all_candles = candles
        
        while len(all_candles) < total_needed:
            oldest = all_candles[0][0]
            tf_ms = {'1m': 60000, '5m': 300000, '15m': 900000}[timeframe]
            since = oldest - limit * tf_ms
            
            candles = exchange.fetch_ohlcv(pair, timeframe, since=since, limit=limit)
            if not candles:
                break
            
            new = [c for c in candles if c[0] < oldest]
            if not new:
                break
            
            all_candles = new + all_candles
            time.sleep(0.05)
        
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        if verbose:
            print(f"      {timeframe}: {len(df)} candles ({df.index[0].strftime('%m-%d %H:%M')} → {df.index[-1].strftime('%m-%d %H:%M')})")
        
        return df
    except Exception as e:
        print(f"    Error fetching {pair} {timeframe}: {e}")
        return None


def log_signal(signal: Dict):
    """Логирует сигнал в файл"""
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(signal, default=str) + '\n')


# ============================================================
# MAIN SCANNER
# ============================================================
class LiveScanner:
    def __init__(self):
        # Load models
        self.models = {
            'direction': joblib.load(MODEL_DIR / 'direction_model.joblib'),
            'timing': joblib.load(MODEL_DIR / 'timing_model.joblib'),
            'strength': joblib.load(MODEL_DIR / 'strength_model.joblib'),
        }
        self.features = joblib.load(MODEL_DIR / 'feature_names.joblib')
        
        # Load pairs
        with open(PAIRS_FILE) as f:
            self.pairs = [p['symbol'] for p in json.load(f)['pairs'][:20]]
        
        # Init
        self.binance = ccxt.binance({'options': {'defaultType': 'future'}})
        self.mtf_fe = MTFFeatureEngine()
        
        print(f"Loaded model with {len(self.features)} features")
        print(f"Trading {len(self.pairs)} pairs")
        print(f"Thresholds: CONF>={MIN_CONF}, TIMING>={MIN_TIMING}, STRENGTH>={MIN_STRENGTH}")
    
    def scan_pair(self, pair: str, verbose: bool = False) -> Optional[Dict]:
        """Сканирует одну пару"""
        try:
            # Fetch data
            if verbose:
                print(f"    Loading data...")
            
            data = {}
            for tf, n in [('1m', CANDLES_M1), ('5m', CANDLES_M5), ('15m', CANDLES_M15)]:
                data[tf] = fetch_candles(self.binance, pair, tf, n, verbose=verbose)
                if data[tf] is None or len(data[tf]) < 100:
                    return None
            
            # Build features
            ft = self.mtf_fe.align_timeframes(data['1m'], data['5m'], data['15m'])
            ft = ft.join(data['5m'][['open', 'high', 'low', 'close', 'volume']])
            ft = add_volume_features(ft)
            ft['atr'] = calculate_atr(ft)
            ft = ft.dropna()
            
            if verbose:
                print(f"    Features: {len(ft.columns)} cols, {len(ft)} rows after dropna")
            
            if len(ft) < 10:
                return None
            
            # Fill missing features
            for f in self.features:
                if f not in ft.columns:
                    ft[f] = 0.0
            
            # Get CLOSED candle (index -2)
            row = ft.iloc[[-2]]
            X = row[self.features].values.astype(np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Predict
            dir_proba = self.models['direction'].predict_proba(X)
            dir_pred = int(np.argmax(dir_proba))
            dir_conf = float(np.max(dir_proba))
            timing = float(self.models['timing'].predict(X)[0])
            strength = float(self.models['strength'].predict(X)[0])
            
            direction = ['SHORT', 'SIDEWAYS', 'LONG'][dir_pred]
            ts = row.index[0]
            close_price = row['close'].iloc[0]
            atr = row['atr'].iloc[0]
            
            return {
                'pair': pair,
                'direction': direction,
                'dir_pred': dir_pred,
                'timestamp': ts,
                'price': close_price,
                'conf': dir_conf,
                'timing': timing,
                'strength': strength,
                'atr': atr,
                'proba': dir_proba[0].tolist()
            }
            
        except Exception as e:
            print(f"    Error scanning {pair}: {e}")
            return None
    
    def run_scan(self, verbose: bool = False) -> List[Dict]:
        """Полный скан всех пар"""
        now = datetime.now(timezone.utc)
        print(f"\n{'='*70}")
        print(f"SCAN at {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Loading: M1={CANDLES_M1}, M5={CANDLES_M5}, M15={CANDLES_M15} candles per pair")
        print(f"{'='*70}")
        
        signals = []
        
        for i, pair in enumerate(self.pairs):
            print(f"[{i+1:2d}/{len(self.pairs)}] {pair:20s}", end=" " if not verbose else "\n", flush=True)
            
            result = self.scan_pair(pair, verbose=verbose)
            
            if result is None:
                print("ERROR")
                continue
            
            if result['dir_pred'] == 1:  # SIDEWAYS
                print(f"SIDEWAYS conf={result['conf']:.2f}")
                continue
            
            # Check filters
            passes = (
                result['conf'] >= MIN_CONF and 
                result['timing'] >= MIN_TIMING and 
                result['strength'] >= MIN_STRENGTH
            )
            
            if passes:
                print(f"✅ {result['direction']} @ {result['price']:.6f} | "
                      f"C={result['conf']:.3f} T={result['timing']:.2f} S={result['strength']:.2f}")
                signals.append(result)
                log_signal(result)
            else:
                reject = []
                if result['conf'] < MIN_CONF: 
                    reject.append(f"C={result['conf']:.2f}<{MIN_CONF}")
                if result['timing'] < MIN_TIMING: 
                    reject.append(f"T={result['timing']:.2f}<{MIN_TIMING}")
                if result['strength'] < MIN_STRENGTH: 
                    reject.append(f"S={result['strength']:.2f}<{MIN_STRENGTH}")
                print(f"❌ {result['direction']} | {', '.join(reject)}")
        
        # Summary
        print(f"\n{'='*70}")
        print(f"FOUND {len(signals)} VALID SIGNALS")
        
        if signals:
            for sig in signals:
                print(f"  📌 {sig['pair']} {sig['direction']} @ {sig['price']:.6f}")
                print(f"     Conf={sig['conf']:.3f} Timing={sig['timing']:.2f} Strength={sig['strength']:.2f}")
        
        return signals
    
    def run_loop(self, verbose: bool = False):
        """Запускает бесконечный цикл сканирования"""
        self.verbose = verbose
        print(f"\n🚀 Starting scanner loop (interval: {SCAN_INTERVAL_MINUTES} min)")
        print(f"   Verbose mode: {'ON' if verbose else 'OFF'}")
        print(f"   Press Ctrl+C to stop\n")
        
        while True:
            try:
                self.run_scan(verbose=self.verbose)
                
                # Wait until next scan
                now = datetime.now(timezone.utc)
                # Выравниваем по минутам (сканируем в начале каждой 5-минутки)
                next_min = (now.minute // SCAN_INTERVAL_MINUTES + 1) * SCAN_INTERVAL_MINUTES
                if next_min >= 60:
                    next_min = 0
                    wait_seconds = (60 - now.minute) * 60 + (SCAN_INTERVAL_MINUTES * 60) - now.second
                else:
                    wait_seconds = (next_min - now.minute) * 60 - now.second
                
                # Добавляем 30 сек чтобы свеча точно закрылась
                wait_seconds += 30
                
                print(f"\n⏳ Next scan in {wait_seconds // 60}m {wait_seconds % 60}s...")
                time.sleep(wait_seconds)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Scanner stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Error in scan loop: {e}")
                print("   Retrying in 60 seconds...")
                time.sleep(60)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true', help='Run single scan and exit')
    parser.add_argument('--interval', type=int, default=5, help='Scan interval in minutes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed loading info')
    args = parser.parse_args()
    
    if args.interval:
        SCAN_INTERVAL_MINUTES = args.interval
    
    scanner = LiveScanner()
    
    if args.once:
        scanner.run_scan(verbose=args.verbose)
    else:
        scanner.run_loop(verbose=args.verbose)
