"""
AI Trading Bot - Main Entry Point
Orchestrates all components and runs the trading bot.
Version 2.1 - Multi-pair scanner + M1 sniper + Aggressive trading
"""

import asyncio
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_yaml_config, setup_logger
from src.utils.constants import PositionSide
from src.data import DataCollector, DataStorage, DataPreprocessor
from src.features import FeatureEngine
from src.models import EnsembleModel, ModelTrainer
from src.strategy import DecisionEngine
from src.execution import ExchangeAPI, OrderManager, PositionManager
from src.risk import RiskLimits, DrawdownController
from src.risk.risk_manager import RiskManager, load_risk_config
from src.monitoring import AlertManager, Dashboard, trading_logger
from src.scanner import PairScanner, M1Sniper, AggressiveSizer


class TradingBot:
    """
    Main trading bot orchestrator v2.1.
    
    New features:
    - Multi-pair scanning
    - M1 precise entry
    - Aggressive 100% deposit usage with leverage
    - Single position mode
    
    Coordinates all components:
    - Data collection
    - Multi-pair scanning
    - Feature generation
    - Model prediction
    - M1 sniper entry
    - Decision making
    - Order execution
    - Risk management
    - Monitoring
    """
    
    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        trading_params_path: str = "config/trading_params.yaml"
    ):
        """
        Initialize the trading bot.
        
        Args:
            config_path: Path to main settings file.
            trading_params_path: Path to trading parameters file.
        """
        # Load configuration
        self.config = load_yaml_config(config_path)
        self.trading_params = load_yaml_config(trading_params_path)
        
        # Extract settings
        self.symbol = self.config['trading']['symbol']
        self.timeframe = self.config['data']['primary_timeframe']
        
        # v2.1: Aggressive mode settings
        self.aggressive_mode = self.config['trading'].get('aggressive_mode', False)
        self.single_position = self.config['trading'].get('single_position', True)
        self.use_scanner = self.config['trading'].get('use_scanner', False)
        
        # Component initialization flags
        self._initialized = False
        self._running = False
        
        # Components will be initialized in setup()
        self.data_collector: Optional[DataCollector] = None
        self.data_storage: Optional[DataStorage] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.feature_engine: Optional[FeatureEngine] = None
        self.model: Optional[EnsembleModel] = None
        self.decision_engine: Optional[DecisionEngine] = None
        self.exchange: Optional[ExchangeAPI] = None
        self.order_manager: Optional[OrderManager] = None
        self.position_manager: Optional[PositionManager] = None
        self.risk_limits: Optional[RiskLimits] = None
        self.drawdown_controller: Optional[DrawdownController] = None
        self.risk_manager: Optional[RiskManager] = None  # V1 + Risk Management
        self.alert_manager: Optional[AlertManager] = None
        self.dashboard: Optional[Dashboard] = None
        
        # v2.1: New components
        self.scanner: Optional[PairScanner] = None
        self.sniper: Optional[M1Sniper] = None
        self.aggressive_sizer: Optional[AggressiveSizer] = None
        
    async def setup(self) -> None:
        """Initialize all components."""
        logger.info("Setting up trading bot v2.1...")
        
        # Setup logging
        setup_logger(
            log_file=self.config.get('app', {}).get('log_file', 'logs/trading_bot.log'),
            log_level=self.config.get('app', {}).get('log_level', 'INFO')
        )
        
        # Initialize data components
        self.preprocessor = DataPreprocessor()
        self.feature_engine = FeatureEngine(self.trading_params.get('features', {}))
        
        # Initialize exchange API
        exchange_config = self.config.get('exchange', {})
        self.exchange = ExchangeAPI(
            exchange_id=exchange_config.get('name', 'binance'),
            testnet=exchange_config.get('testnet', True),
            config={'market_type': 'future'}  # v2.1: Use futures for leverage
        )
        await self.exchange.initialize()
        
        # Initialize data collector
        self.data_collector = DataCollector(
            exchange_id=exchange_config.get('name', 'binance'),
            symbol=self.symbol.replace('USDT', '/USDT'),
            testnet=exchange_config.get('testnet', True)
        )
        await self.data_collector.start()
        
        # Initialize execution components
        self.order_manager = OrderManager(
            self.exchange,
            self.trading_params.get('execution', {})
        )
        
        self.position_manager = PositionManager(
            self.order_manager,
            self.trading_params.get('exit', {})
        )
        
        # Initialize risk components
        self.risk_limits = RiskLimits(self.trading_params.get('risk', {}))
        self.drawdown_controller = DrawdownController(self.trading_params.get('risk', {}))
        
        # V1 + Risk Management: Load RiskManager
        try:
            risk_config = load_risk_config('config/risk_management.yaml')
            self.risk_manager = RiskManager(risk_config)
            logger.info(f"RiskManager initialized: max_risk={self.risk_manager.max_risk_per_trade:.1%}, "
                       f"max_dd={self.risk_manager.max_drawdown:.1%}")
        except Exception as e:
            logger.warning(f"RiskManager not initialized: {e}")
            self.risk_manager = None
        
        # Initialize monitoring
        self.alert_manager = AlertManager(
            self.config.get('notifications', {})
        )
        
        if self.config.get('monitoring', {}).get('dashboard', {}).get('enabled', False):
            self.dashboard = Dashboard(
                self.config.get('monitoring', {}).get('dashboard', {})
            )
            
        # Load or create model
        model_path = self.config.get('models', {}).get('save_path', './models/saved/')
        if os.path.exists(f"{model_path}/ensemble_meta.joblib"):
            self.model = EnsembleModel()
            self.model.load(model_path)
            logger.info("Loaded existing model")
        else:
            self.model = EnsembleModel(
                model_config=self.trading_params.get('model_params', {})
            )
            logger.info("Created new model (needs training)")
            
        # Initialize decision engine
        self.decision_engine = DecisionEngine(
            model=self.model,
            feature_engine=self.feature_engine,
            config={
                'symbol': self.symbol,
                'signals': self.trading_params.get('entry', {}),
                'filters': self.trading_params.get('filters', {}),
                'position_sizing': self.trading_params.get('position_sizing', {})
            }
        )
        
        # v2.1: Initialize scanner components
        if self.use_scanner:
            scanner_config = self.trading_params.get('scanner', {})
            self.scanner = PairScanner(
                model=self.model,
                feature_engine=self.feature_engine,
                config=scanner_config
            )
            logger.info(f"Scanner initialized with {len(self.scanner.pairs)} pairs")
            
        # v2.1: Initialize M1 sniper
        sniper_config = self.trading_params.get('sniper', {})
        self.sniper = M1Sniper(config=sniper_config)
        logger.info("M1 Sniper initialized")
        
        # v2.1: Initialize aggressive sizer
        if self.aggressive_mode:
            aggressive_config = self.trading_params.get('aggressive_sizing', {})
            self.aggressive_sizer = AggressiveSizer(config=aggressive_config)
            logger.info("Aggressive sizer initialized (100% deposit with leverage)")
        
        self._initialized = True
        logger.info("Trading bot v2.1 setup complete")
        
    async def start(self) -> None:
        """Start the trading bot."""
        if not self._initialized:
            await self.setup()
            
        logger.info(f"Starting trading bot v2.1 for {self.symbol}")
        if self.use_scanner:
            logger.info(f"Scanner mode: scanning {len(self.scanner.pairs)} pairs")
        if self.aggressive_mode:
            logger.info("Aggressive mode: 100% deposit with leverage")
            
        self._running = True
        
        # Send startup alert
        await self.alert_manager.info(
            "Bot Started v2.1",
            f"Trading bot started - Scanner: {self.use_scanner}, Aggressive: {self.aggressive_mode}"
        )
        
        # Start main loop
        try:
            if self.use_scanner:
                await self._scanner_loop()
            else:
                await self._main_loop()
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
            await self.alert_manager.critical("Bot Error", str(e))
        finally:
            await self.stop()
            
    async def _scanner_loop(self) -> None:
        """
        Main trading loop with multi-pair scanner (v2.1).
        
        Flow:
        1. Check if position is open -> manage it
        2. If no position -> scan all pairs for opportunity
        3. Find best opportunity -> use M1 sniper for entry
        4. Execute trade with aggressive sizing
        """
        scan_interval = self.trading_params.get('scanner', {}).get('scan_interval', 60)
        
        while self._running:
            try:
                # Get account state
                balance_info = await self.exchange.get_balance('USDT')
                account_balance = balance_info.get('total', 10000)
                
                # Update drawdown tracking
                self.drawdown_controller.update(account_balance)
                self.risk_limits.update_balance(account_balance)
                
                # Check if trading is allowed
                position_multiplier = self.drawdown_controller.get_position_multiplier()
                if position_multiplier == 0:
                    logger.warning("Trading blocked by drawdown controller")
                    await asyncio.sleep(60)
                    continue
                    
                # Check for open positions
                all_positions = self.position_manager.get_open_positions()
                
                if all_positions:
                    # v2.1: Single position mode - manage existing position
                    position = all_positions[0]
                    
                    # Fetch current price for position symbol
                    symbol_ccxt = position.symbol.replace('USDT', '/USDT')
                    df = await self.data_collector.fetch_ohlcv(
                        symbol=symbol_ccxt,
                        timeframe='1m',
                        limit=50
                    )
                    
                    if not df.empty:
                        current_price = df['close'].iloc[-1]
                        atr = self._calculate_atr(df)
                        
                        await self.position_manager.update_position(
                            position.id,
                            current_price,
                            atr
                        )
                        
                    logger.debug(f"Managing position: {position.symbol} @ {position.entry_price:.4f}")
                    await asyncio.sleep(10)
                    continue
                    
                # No position - scan for opportunities
                logger.info("Scanning for opportunities...")
                
                scan_results = await self.scanner.scan_all_pairs(
                    self.data_collector,
                    limit=500
                )
                
                # Get best opportunity
                best = self.scanner.get_best_opportunity()
                
                if best is None:
                    logger.info("No valid opportunities found")
                    await asyncio.sleep(scan_interval)
                    continue
                    
                logger.info(f"Best opportunity: {best.symbol} score={best.score:.1f}, "
                           f"direction={'LONG' if best.direction == 1 else 'SHORT'}, "
                           f"prob={best.direction_probability:.1%}")
                
                # Use M1 sniper for precise entry
                entry = await self.sniper.snipe_entry(
                    symbol=best.symbol,
                    direction=best.direction,
                    data_collector=self.data_collector,
                    scan_result=best,
                    account_balance=account_balance,
                    min_leverage=self.trading_params.get('aggressive_sizing', {}).get('min_leverage', 5),
                    max_leverage=self.trading_params.get('aggressive_sizing', {}).get('max_leverage', 20)
                )
                
                if entry is None:
                    logger.info(f"Sniper timeout for {best.symbol} - no entry found")
                    await asyncio.sleep(30)
                    continue
                    
                # Execute trade
                await self._execute_aggressive_trade(entry, account_balance)
                
                # Wait before next scan
                await asyncio.sleep(scan_interval)
                
            except Exception as e:
                logger.error(f"Error in scanner loop: {e}")
                await asyncio.sleep(30)
                
    async def _execute_aggressive_trade(
        self,
        entry,  # SniperEntry
        account_balance: float
    ) -> None:
        """Execute an aggressive trade from sniper entry."""
        # Check risk limits
        risk_check = self.risk_limits.check_trade_allowed(
            account_balance=account_balance,
            trade_risk=entry.risk_amount,
            position_value=entry.position_value
        )
        
        if not risk_check['allowed']:
            logger.warning(f"Trade blocked by risk limits: {risk_check['reasons']}")
            return
            
        # Open position
        side = PositionSide.LONG if entry.direction == 1 else PositionSide.SHORT
        
        try:
            # Log position details
            if self.aggressive_sizer:
                calc = self.aggressive_sizer.calculate(
                    deposit=account_balance,
                    entry_price=entry.entry_price,
                    stop_loss=entry.stop_loss,
                    direction=entry.direction
                )
                self.aggressive_sizer.log_position_summary(calc, entry.direction)
            
            position = await self.position_manager.open_position(
                symbol=entry.symbol,
                side=side,
                quantity=entry.position_size,
                entry_price=entry.entry_price,
                stop_loss=entry.stop_loss,
                take_profit=entry.take_profit,
                atr=entry.metadata.get('atr', 0),
                metadata={
                    'leverage': entry.leverage,
                    'entry_type': entry.entry_type,
                    'trigger_reason': entry.trigger_reason,
                    'risk_percent': entry.risk_percent,
                    'sniper_entry': True
                }
            )
            
            await self.alert_manager.alert_position_opened(
                symbol=entry.symbol,
                side=side.value,
                size=entry.position_size,
                entry_price=entry.entry_price
            )
            
            logger.info(f"Position opened: {entry.symbol} {side.value} @ {entry.entry_price:.4f}, "
                       f"leverage={entry.leverage}x, size={entry.position_size:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            await self.alert_manager.critical("Trade Execution Failed", str(e))
            
    async def _main_loop(self) -> None:
        """Main trading loop (original v1.0 mode)."""
        while self._running:
            try:
                # Fetch latest market data
                df = await self.data_collector.fetch_ohlcv(
                    timeframe=self.timeframe,
                    limit=500
                )
                
                if df.empty:
                    logger.warning("No market data received")
                    await asyncio.sleep(5)
                    continue
                    
                # Get account state
                balance_info = await self.exchange.get_balance('USDT')
                account_state = {
                    'balance': balance_info.get('total', 10000),
                    'equity': balance_info.get('total', 10000)
                }
                
                # Update drawdown tracking
                self.drawdown_controller.update(account_state['equity'])
                self.risk_limits.update_balance(account_state['equity'])
                
                # Check if trading is allowed
                position_multiplier = self.drawdown_controller.get_position_multiplier()
                if position_multiplier == 0:
                    logger.warning("Trading blocked by drawdown controller")
                    await asyncio.sleep(60)
                    continue
                    
                # Get current positions
                open_positions = self.position_manager.get_open_positions(self.symbol)
                current_position = open_positions[0] if open_positions else None
                
                # Make trading decision
                decision = self.decision_engine.make_decision(
                    market_data=df,
                    account_state=account_state,
                    current_position=current_position.__dict__ if current_position else None
                )
                
                # Execute decision
                if decision.action in ['buy', 'sell'] and decision.filters_passed:
                    await self._execute_trade(decision, account_state)
                elif decision.action == 'close' and current_position:
                    await self._close_position(current_position)
                    
                # Update open positions
                current_price = df['close'].iloc[-1]
                atr = self._calculate_atr(df)
                
                for position in open_positions:
                    await self.position_manager.update_position(
                        position.id,
                        current_price,
                        atr
                    )
                    
                # Log status periodically
                logger.debug(f"Price: {current_price:.2f}, "
                           f"Positions: {len(open_positions)}, "
                           f"Balance: {account_state['balance']:.2f}")
                           
                # Wait for next iteration
                await asyncio.sleep(10)  # 10 second loop
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)
                
    async def _execute_trade(
        self,
        decision,
        account_state: Dict
    ) -> None:
        """Execute a trading decision."""
        # V1 + Risk Management: Check RiskManager first
        if self.risk_manager:
            symbol = decision.symbol if hasattr(decision, 'symbol') else self.symbol
            can_trade, reason = self.risk_manager.can_trade(symbol, account_state['balance'])
            
            if not can_trade:
                logger.warning(f"Trade blocked by RiskManager: {reason}")
                return
                
            # Use RiskManager for position sizing
            stop_loss_pct = abs(decision.stop_loss - decision.entry_price) / decision.entry_price
            position_size_value = self.risk_manager.calculate_position_size(
                capital=account_state['balance'],
                stop_loss_pct=stop_loss_pct,
                signal_confidence=decision.confidence if hasattr(decision, 'confidence') else 1.0
            )
            # Adjust decision position size based on RiskManager
            decision.position_size = position_size_value / decision.entry_price
        
        # Check legacy risk limits
        risk_check = self.risk_limits.check_trade_allowed(
            account_balance=account_state['balance'],
            trade_risk=decision.risk_assessment.get('risk_amount', 0),
            position_value=decision.position_size * decision.entry_price
        )
        
        if not risk_check['allowed']:
            logger.warning(f"Trade blocked by risk limits: {risk_check['reasons']}")
            return
            
        # Open position
        side = PositionSide.LONG if decision.action == 'buy' else PositionSide.SHORT
        
        try:
            position = await self.position_manager.open_position(
                symbol=decision.symbol,
                side=side,
                quantity=decision.position_size,
                entry_price=decision.entry_price,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                atr=decision.signal.volatility if decision.signal else 0,
                metadata={'decision_id': id(decision)}
            )
            
            await self.alert_manager.alert_position_opened(
                symbol=decision.symbol,
                side=side.value,
                size=decision.position_size,
                entry_price=decision.entry_price
            )
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            await self.alert_manager.critical("Trade Execution Failed", str(e))
            
    async def _close_position(self, position) -> None:
        """Close a position."""
        try:
            closed = await self.position_manager.close_position(
                position.id,
                reason="signal"
            )
            
            if closed:
                # Record in legacy risk_limits
                self.risk_limits.record_trade_result(
                    pnl=closed.realized_pnl,
                    is_win=closed.realized_pnl > 0
                )
                
                # V1 + Risk Management: Record in RiskManager
                if self.risk_manager:
                    self.risk_manager.record_trade_result(
                        pnl=closed.realized_pnl,
                        is_win=closed.realized_pnl > 0
                    )
                
                self.decision_engine.update_after_trade(
                    is_win=closed.realized_pnl > 0,
                    pnl=closed.realized_pnl
                )
                
                await self.alert_manager.alert_position_closed(
                    symbol=position.symbol,
                    pnl=closed.realized_pnl,
                    pnl_percent=closed.pnl_percent
                )
                
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            
    def _calculate_atr(self, df, period: int = 14) -> float:
        """Calculate ATR from dataframe."""
        import pandas as pd
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        return tr.ewm(span=period, adjust=False).mean().iloc[-1]
        
    async def stop(self) -> None:
        """Stop the trading bot."""
        logger.info("Stopping trading bot...")
        self._running = False
        
        # Close all open positions
        if self.position_manager:
            open_positions = self.position_manager.get_open_positions()
            for position in open_positions:
                await self.position_manager.close_position(
                    position.id,
                    reason="shutdown"
                )
                
        # Close connections
        if self.data_collector:
            await self.data_collector.stop()
            
        if self.exchange:
            await self.exchange.close()
            
        await self.alert_manager.info("Bot Stopped", "Trading bot shutdown complete")
        logger.info("Trading bot stopped")
        
    def handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self._running = False


async def main():
    """Main entry point."""
    # Create bot instance
    bot = TradingBot()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, bot.handle_shutdown)
    signal.signal(signal.SIGTERM, bot.handle_shutdown)
    
    # Run bot
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
