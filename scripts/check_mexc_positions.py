#!/usr/bin/env python3
"""Check MEXC positions and orders"""

import ccxt
from loguru import logger

MEXC_API_KEY = "mx0vglp7RP0pQYiNA2"
MEXC_API_SECRET = "25817ec107364a55976a23ca6f19d470"

mexc = ccxt.mexc({
    'apiKey': MEXC_API_KEY,
    'secret': MEXC_API_SECRET,
    'timeout': 30000,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
        'adjustForTimeDifference': True,
        'recvWindow': 60000
    }
})

logger.info("Checking MEXC account...")

# Balance
try:
    balance = mexc.fetch_balance({'type': 'swap'})
    logger.info(f"💰 Balance: ${balance['USDT']['free']:.2f}")
except Exception as e:
    logger.error(f"Balance error: {e}")

# Positions
try:
    positions = mexc.fetch_positions()
    logger.info(f"\n📊 Positions: {len(positions)}")
    for pos in positions:
        contracts = float(pos.get('contracts', 0))
        if contracts > 0:
            logger.info(f"  {pos['symbol']}: {contracts} contracts")
            logger.info(f"    Entry: {pos.get('entryPrice')}")
            logger.info(f"    PnL: {pos.get('unrealizedPnl')}")
except Exception as e:
    logger.error(f"Positions error: {e}")

# Open orders
try:
    orders = mexc.fetch_open_orders()
    logger.info(f"\n📋 Open orders: {len(orders)}")
    for order in orders:
        logger.info(f"  {order['symbol']}: {order['side']} {order['amount']}")
        logger.info(f"    Type: {order['type']}, Status: {order['status']}")
except Exception as e:
    logger.error(f"Orders error: {e}")

logger.info("\n✅ Check complete")

