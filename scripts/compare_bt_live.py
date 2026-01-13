#!/usr/bin/env python3
"""Compare backtest vs live trading parameters."""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import re

def extract_params(filepath: str) -> dict:
    """Extract key trading parameters from a script."""
    params = {}
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract numeric constants
    patterns = {
        'MIN_CONFIDENCE': r'MIN_CONFIDENCE\s*=\s*([\d.]+)',
        'STOP_LOSS_PCT': r'STOP_LOSS_PCT\s*=\s*([\d.]+)',
        'TAKE_PROFIT_R': r'TAKE_PROFIT_R\s*=\s*([\d.]+)',
        'MAX_LEVERAGE': r'MAX_LEVERAGE\s*=\s*([\d.]+)',
        'BREAKEVEN_TRIGGER_R': r'BREAKEVEN_TRIGGER_R\s*=\s*([\d.]+)',
        'USE_DYNAMIC_LEVERAGE': r'USE_DYNAMIC_LEVERAGE\s*=\s*(True|False)',
        'USE_AGGRESSIVE_TRAIL': r'USE_AGGRESSIVE_TRAIL\s*=\s*(True|False)',
        'MAX_POSITION_SIZE': r'MAX_POSITION_SIZE\s*=\s*([\d.]+)',
        'MARGIN_LIMIT_PCT': r'MARGIN_LIMIT_PCT\s*=\s*([\d.]+)',
        'MIN_TIMING_SCORE': r'MIN_TIMING_SCORE\s*=\s*([\d.]+)',
        'MIN_STRENGTH_SCORE': r'MIN_STRENGTH_SCORE\s*=\s*([\d.]+)',
    }
    
    for name, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            val = match.group(1)
            if val in ('True', 'False'):
                params[name] = val == 'True'
            else:
                params[name] = float(val)
    
    return params

def main():
    print('=' * 80)
    print('BACKTEST vs LIVE PARAMETER COMPARISON')
    print('=' * 80)
    
    backtest_params = extract_params('scripts/train_v3_dynamic.py')
    live_params = extract_params('scripts/live_trading_v10_csv.py')
    
    # All parameters to check
    all_params = sorted(set(backtest_params.keys()) | set(live_params.keys()))
    
    print(f'\n{"Parameter":<25} | {"Backtest":>15} | {"Live":>15} | Match?')
    print('-' * 70)
    
    issues = []
    for param in all_params:
        bt_val = backtest_params.get(param, 'N/A')
        live_val = live_params.get(param, 'N/A')
        
        if bt_val == live_val:
            match = '✅'
        elif bt_val == 'N/A' or live_val == 'N/A':
            match = '⚠️'
            issues.append((param, bt_val, live_val, 'Missing in one script'))
        else:
            match = '❌'
            issues.append((param, bt_val, live_val, 'Different values'))
        
        print(f'{param:<25} | {str(bt_val):>15} | {str(live_val):>15} | {match}')
    
    # Check trailing stop logic
    print('\n' + '=' * 80)
    print('TRAILING STOP LOGIC')
    print('=' * 80)
    
    # Check if trailing stop logic exists in both
    with open('scripts/train_v3_dynamic.py', 'r') as f:
        bt_content = f.read()
    with open('scripts/live_trading_v10_csv.py', 'r') as f:
        live_content = f.read()
    
    checks = [
        ('Breakeven trigger', 'breakeven_active', 'breakeven'),
        ('Progressive trailing', 'Progressive Trailing', 'trail'),
        ('Dynamic leverage', 'USE_DYNAMIC_LEVERAGE', 'dynamic.*leverage'),
        ('Stop loss handling', 'stop_loss', 'stop_loss'),
        ('Take profit logic', 'take_profit', 'take_profit'),
    ]
    
    for name, bt_pattern, live_pattern in checks:
        bt_has = bool(re.search(bt_pattern, bt_content, re.IGNORECASE))
        live_has = bool(re.search(live_pattern, live_content, re.IGNORECASE))
        
        if bt_has and live_has:
            status = '✅ Both have'
        elif bt_has and not live_has:
            status = '❌ MISSING in live!'
            issues.append((name, 'Yes', 'No', 'Missing in live'))
        elif not bt_has and live_has:
            status = '⚠️ Only in live'
        else:
            status = '⚠️ Neither has'
        
        print(f'{name:<25}: {status}')
    
    # Summary
    print('\n' + '=' * 80)
    if issues:
        print('⚠️ DIFFERENCES FOUND:')
        for param, bt, live, reason in issues:
            print(f'  - {param}: Backtest={bt}, Live={live} ({reason})')
    else:
        print('✅ ALL PARAMETERS MATCH!')
    print('=' * 80)

if __name__ == '__main__':
    main()
