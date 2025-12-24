"""
AI Trading Bot - Main Entry Point
Orchestrates all components and runs the trading bot.
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
from src.data import DataCollector, DataStorage, DataPreprocessor
from src.features import FeatureEngine
from src.models import EnsembleModel, ModelTrainer
from src.strategy import DecisionEngine
from src.execution import ExchangeAPI, OrderManager, PositionManager
from src.risk import RiskLimits, DrawdownController
from src.monitoring import AlertManager, Dashboard, trading_logger


class TradingBot:
    """
    Main trading bot orchestrator.
    
    Coordinates all components:
    - Data collection
    - Feature generation
    - Model prediction
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
        self.alert_manager: Optional[AlertManager] = None
        self.dashboard: Optional[Dashboard] = None
        
    async def setup(self) -> None:
        """Initialize all components."""
        logger.info("Setting up trading bot...")
        
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
            config={'market_type': 'spot'}
        )
        await self.exchange.initialize()
        
        # Initialize data collector
        self.data_collector = DataCollector(
            exchange_id=exchange_config.get('name', 'binance'),
            symbol=self.symbol.replace('USDT', '/USDT'),  # Convert to ccxt format
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
        
        self._initialized = True
        logger.info("Trading bot setup complete")
        
    async def start(self) -> None:
        """Start the trading bot."""
        if not self._initialized:
            await self.setup()
            
        logger.info(f"Starting trading bot for {self.symbol}")
        self._running = True
        
        # Send startup alert
        await self.alert_manager.info(
            "Bot Started",
            f"Trading bot started for {self.symbol}"
        )
        
        # Start main loop
        try:
            await self._main_loop()
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
            await self.alert_manager.critical("Bot Error", str(e))
        finally:
            await self.stop()
            
    async def _main_loop(self) -> None:
        """Main trading loop."""
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
        # Check risk limits
        risk_check = self.risk_limits.check_trade_allowed(
            account_balance=account_state['balance'],
            trade_risk=decision.risk_assessment.get('risk_amount', 0),
            position_value=decision.position_size * decision.entry_price
        )
        
        if not risk_check['allowed']:
            logger.warning(f"Trade blocked by risk limits: {risk_check['reasons']}")
            return
            
        # Open position
        from src.utils.constants import PositionSide
        
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
                self.risk_limits.record_trade_result(
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
