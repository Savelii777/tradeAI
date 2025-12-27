#!/usr/bin/env python3
"""
Paper Trading Script (V7 Sniper)
- Loads trained models from models/v7_sniper_final/
- Connects to Binance via CCXT
- Fetches live data every 5 minutes
- Generates signals
- Sends alerts to Telegram (or prints to console)
"""

import sys
import time
import json
import joblib
import ccxt
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.feature_engine import FeatureEngine
from train_mtf import MTFFeatureEngine

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = Path("models/v7_sniper_final")
PAIRS_FILE = Path("config/pairs_list.json")
TIMEFRAMES = ['1m', '5m', '15m']
LOOKBACK = 1000 # Candles to fetch

# Thresholds (Must match training)
MIN_CONF = 0.55
MIN_TIMING = 0.60
MIN_STRENGTH = 2.0

# Telegram Config (Replace with your own or load from env)
TELEGRAM_TOKEN = "8270168075:AAHkJ_bbJGgk4fV3r0_Gc8NQb07O_zUMBJc"
TELEGRAM_CHAT_ID = "677822370"

# ============================================================
# UTILS
# ============================================================
def load_models():
    logger.info(f"Loading models from {MODEL_DIR}...")
    try:
        models = {
            'direction': joblib.load(MODEL_DIR / 'direction_model.pkl'),
            'timing': joblib.load(MODEL_DIR / 'timing_model.pkl'),
            'strength': joblib.load(MODEL_DIR / 'strength_model.pkl'),
            'features': joblib.load(MODEL_DIR / 'features.pkl')
        }
        return models
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        sys.exit(1)

def get_pairs():
    with open(PAIRS_FILE, "r") as f:
        data = json.load(f)
        # Use top 20 pairs as in training
        return [p['symbol'] for p in data['pairs']][:20]

def fetch_live_data(exchange, symbol):
    """Fetch latest candles for all required timeframes."""
    data = {}
    clean_symbol = symbol.replace('_', '/')
    if '/' not in clean_symbol:
        clean_symbol = f"{clean_symbol[:-4]}/{clean_symbol[-4:]}"
        
    try:
        for tf in TIMEFRAMES:
            candles = exchange.fetch_ohlcv(clean_symbol, tf, limit=LOOKBACK)
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            data[tf] = df
        return data
    except Exception as e:
        logger.warning(f"Error fetching {symbol}: {e}")
        return None

# ============================================================
# FEATURE ENGINEERING (Must match training exactly)
# ============================================================
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_zscore'] = (df['volume'] - df['vol_sma_20']) / df['volume'].rolling(20).std()
    df['price_change'] = df['close'].diff()
    df['obv'] = np.where(df['price_change'] > 0, df['volume'], -df['volume']).cumsum()
    df['obv_sma'] = pd.Series(df['obv']).rolling(20).mean()
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
    df['vol_momentum'] = df['volume'].pct_change(5)
    return df

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def prepare_features(data, mtf_fe):
    m1 = data['1m']
    m5 = data['5m']
    m15 = data['15m']
    
    ft = mtf_fe.align_timeframes(m1, m5, m15)
    ft = ft.join(m5[['open', 'high', 'low', 'close', 'volume']])
    ft = add_volume_features(ft)
    ft['atr'] = calculate_atr(ft)
    
    return ft.dropna()

# ============================================================
# SIGNAL GENERATION
# ============================================================
def check_signal(df, models, pair):
    """Check the LATEST candle for a signal."""
    if len(df) < 2: return None, None
    
    # Use iloc[-2] to get the last COMPLETED candle
    # iloc[-1] is the current forming candle (unstable features)
    latest = df.iloc[[-2]] 
    
    feature_cols = models['features']
    
    # Ensure all features exist
    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        logger.warning(f"Missing features for {pair}: {missing}")
        return None, None
        
    X = latest[feature_cols].values
    
    # Predict
    dir_proba = models['direction'].predict_proba(X)[0]
    dir_pred = np.argmax(dir_proba)
    dir_conf = np.max(dir_proba)
    
    timing_prob = models['timing'].predict_proba(X)[0][1]
    strength_pred = models['strength'].predict(X)[0]
    
    stats = {
        'pair': pair,
        'dir': dir_pred,
        'conf': dir_conf,
        'timing': timing_prob,
        'strength': strength_pred
    }
    
    # Filter
    if dir_pred == 1: return None, stats # Sideways
    if dir_conf < MIN_CONF: return None, stats
    if timing_prob < MIN_TIMING: return None, stats
    if strength_pred < MIN_STRENGTH: return None, stats
    
    direction = 'LONG' if dir_pred == 2 else 'SHORT'
    
    return {
        'timestamp': latest.index[0],
        'pair': pair,
        'direction': direction,
        'price': latest['close'].iloc[0],
        'atr': latest['atr'].iloc[0],
        'conf': dir_conf,
        'timing': timing_prob,
        'strength': strength_pred
    }, stats

def send_telegram_alert(signal):
    """Send alert to Telegram."""
    msg = (
        f"🚨 <b>SNIPER SIGNAL</b> 🚨\n\n"
        f"Symbol: <b>{signal['pair']}</b>\n"
        f"Direction: <b>{signal['direction']}</b>\n"
        f"Price: {signal['price']:.5f}\n"
        f"ATR: {signal['atr']:.5f}\n\n"
        f"Confidence: {signal['conf']:.1%}\n"
        f"Timing Score: {signal['timing']:.1%}\n"
        f"Exp. Strength: {signal['strength']:.2f} ATR\n"
        f"Time: {signal['timestamp']}"
    )
    print("\n" + "="*50)
    print(msg.replace("<b>", "").replace("</b>", ""))
    print("="*50 + "\n")
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': msg,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=data)
        if response.status_code != 200:
            logger.error(f"Telegram send failed: {response.text}")
    except Exception as e:
        logger.error(f"Telegram error: {e}")

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    logger.info("Starting Paper Trading Bot (V7 Sniper)...")
    
    models = load_models()
    pairs = get_pairs()
    exchange = ccxt.binance()
    mtf_fe = MTFFeatureEngine()
    
    logger.info(f"Monitoring {len(pairs)} pairs.")
    
    # Track last processed candle time to avoid duplicates
    last_processed = {}
    
    while True:
        logger.info(f"Scanning... {datetime.now().strftime('%H:%M:%S')}")
        
        best_candidate = None
        
        for pair in pairs:
            try:
                data = fetch_live_data(exchange, pair)
                if not data: continue
                
                df = prepare_features(data, mtf_fe)
                signal, stats = check_signal(df, models, pair)
                
                # Track best candidate for logging
                if stats:
                    # Priority: Directional > Sideways
                    is_directional = stats['dir'] != 1
                    
                    if best_candidate is None:
                        best_candidate = stats
                    else:
                        best_is_directional = best_candidate['dir'] != 1
                        
                        if is_directional and not best_is_directional:
                            best_candidate = stats
                        elif is_directional == best_is_directional:
                            # If both directional or both sideways, pick highest confidence
                            if stats['conf'] > best_candidate['conf']:
                                best_candidate = stats
                
                if signal:
                    # Check if we already alerted for this candle
                    last_time = last_processed.get(pair)
                    current_time = signal['timestamp']
                    
                    if last_time != current_time:
                        send_telegram_alert(signal)
                        last_processed[pair] = current_time
                    
            except Exception as e:
                logger.error(f"Error processing {pair}: {e}")
        
        if best_candidate:
            dir_map = {0: 'SHORT', 1: 'SIDEWAYS', 2: 'LONG'}
            d_str = dir_map.get(best_candidate['dir'], 'UNKNOWN')
            logger.info(f"Best: {best_candidate['pair']} [{d_str}] (Conf: {best_candidate['conf']:.1%}, Timing: {best_candidate['timing']:.1%}, Str: {best_candidate['strength']:.2f})")
        else:
            logger.info("No data fetched.")
                
        # Wait for next 5m candle close
        # Sleep 60s to avoid spamming, but in production align with clock
        time.sleep(60)

if __name__ == '__main__':
    main()
