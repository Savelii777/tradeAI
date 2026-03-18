# Smart Money Concepts Features for ULTRA mode
# These features capture institutional trading patterns

SMC_FEATURES = [
    # Market Structure
    'smc_structure_uptrend',
    'smc_structure_downtrend',
    'smc_structure_bos_bull',
    'smc_structure_bos_bear',
    'smc_structure_choch_bull',
    'smc_structure_choch_bear',
    'smc_dist_to_swing_high',
    'smc_dist_to_swing_low',
    
    # Fair Value Gaps
    'smc_fvg_bullish',
    'smc_fvg_bearish',
    'smc_fvg_bullish_size',
    'smc_fvg_bearish_size',
    'smc_fvg_bullish_recent',
    'smc_fvg_bearish_recent',
    
    # Order Blocks  
    'smc_ob_bullish',
    'smc_ob_bearish',
    'smc_ob_bullish_strength',
    'smc_ob_bearish_strength',
    'smc_dist_to_ob_bull',
    'smc_dist_to_ob_bear',
    
    # Liquidity
    'smc_dist_to_liquidity_high',
    'smc_dist_to_liquidity_low',
    'smc_liquidity_equal_highs',
    'smc_liquidity_equal_lows',
    'smc_liquidity_sweep_high',
    'smc_liquidity_sweep_low',
    
    # Premium/Discount
    'smc_premium_discount_position',
    'smc_in_premium',
    'smc_in_discount',
    'smc_in_extreme_premium',
    'smc_in_extreme_discount',
]
