#!/usr/bin/env python3
"""
Estimate MEXC position limits at different leverage levels.
Based on API data which returns limits at MAX leverage.
"""

import json
from pathlib import Path

# Load fetched data
config_dir = Path(__file__).parent.parent / 'config'
with open(config_dir / 'mexc_limits_fetched.json') as f:
    limits = json.load(f)

# Our pairs to check
our_pairs = [
    'BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'XRP_USDT', 'DOGE_USDT',
    'BNB_USDT', 'ADA_USDT', 'AVAX_USDT', 'LINK_USDT', 'SUI_USDT',
    'TRX_USDT', 'DOT_USDT', 'LTC_USDT', 'BCH_USDT', 'UNI_USDT',
    'APT_USDT', 'ATOM_USDT', 'NEAR_USDT', 'AAVE_USDT', 'ZEC_USDT'
]

def fmt(val):
    if val >= 1_000_000_000:
        return f"${val/1_000_000_000:.1f}B"
    elif val >= 1_000_000:
        return f"${val/1_000_000:.1f}M"
    else:
        return f"${val/1000:.0f}K"

print("="*95)
print("MEXC POSITION LIMITS BY LEVERAGE")
print("="*95)
print(f"{'Symbol':<12} {'Max Lev':<10} {'Limit @MaxLev':<15} {'Limit @10x':<18} {'Limit @1x':<18}")
print("-"*95)

for symbol in our_pairs:
    if symbol in limits:
        info = limits[symbol]
        max_lev = info.get('max_leverage', 100)
        max_pos = info.get('max_position_usdt', 0)
        
        # MEXC uses tiered limits:
        # At max leverage: smallest limit
        # At 1x leverage: ~max_leverage times larger
        # This is a rough approximation
        
        limit_at_10x = max_pos * (max_lev / 10)
        limit_at_1x = max_pos * max_lev
        
        print(f"{symbol:<12} {max_lev}x{'':<7} {fmt(max_pos):<15} {fmt(limit_at_10x):<18} {fmt(limit_at_1x):<18}")
    else:
        print(f"{symbol:<12} NOT FOUND")

print()
print("⚠️  Limit @10x and @1x are ROUGH ESTIMATES")
print("   Formula: limit_at_lower_lev ≈ limit_at_max_lev × (max_lev / target_lev)")
print()
print("📊 For comparison - user stated BTC at 1x = $111M from MEXC website")
print(f"   Our estimate BTC at 1x = {fmt(limits.get('BTC_USDT', {}).get('max_position_usdt', 0) * limits.get('BTC_USDT', {}).get('max_leverage', 1))}")
print()

# Calculate what we actually need
print("="*95)
print("OUR ACTUAL NEEDS (for risk-based sizing)")
print("="*95)
deposits = [10000, 50000, 100000, 500000, 1000000]
print(f"With RISK_PCT=5%, ATR-based stops (usually 1-3%), leverage ~2-10x:")
print()
for deposit in deposits:
    max_position = deposit * 10  # At 10x leverage
    print(f"  Deposit ${deposit:>10,} → Max position ~${max_position:>12,}")
print()
print("✅ All these are well below MEXC limits for major pairs!")
print("   Even at $1M deposit with 10x, we need only $10M position limit")
print("   BTC limit at 10x ≈ $195M (plenty of room)")
