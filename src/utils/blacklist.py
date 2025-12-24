"""
AI Trading Bot - Pair Blacklist System

Dynamic blacklist for trading pairs based on performance metrics.
Automatically excludes underperforming pairs.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path

from loguru import logger


@dataclass
class PairMetrics:
    """Performance metrics for a trading pair."""
    pair: str
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    @property
    def profit_factor(self) -> float:
        """Calculate profit factor."""
        if self.gross_loss == 0:
            return float('inf') if self.gross_profit > 0 else 0.0
        return self.gross_profit / abs(self.gross_loss)
    
    @property
    def avg_pnl(self) -> float:
        """Calculate average PnL per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'pair': self.pair,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'total_pnl': self.total_pnl,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_pnl': self.avg_pnl,
            'last_updated': self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PairMetrics':
        """Create from dictionary."""
        return cls(
            pair=data['pair'],
            total_trades=data.get('total_trades', 0),
            winning_trades=data.get('winning_trades', 0),
            total_pnl=data.get('total_pnl', 0.0),
            gross_profit=data.get('gross_profit', 0.0),
            gross_loss=data.get('gross_loss', 0.0),
            last_updated=datetime.fromisoformat(data.get('last_updated', datetime.now().isoformat()))
        )


@dataclass
class BlacklistReason:
    """Reason for blacklisting a pair."""
    reason: str
    metric_name: str
    metric_value: float
    threshold: float
    added_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'reason': self.reason,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'added_at': self.added_at.isoformat()
        }


class PairBlacklist:
    """
    Dynamic blacklist for trading pairs.
    
    Features:
    - Static blacklist for permanently excluded pairs
    - Dynamic blacklist based on performance metrics
    - Automatic blacklisting based on thresholds
    - Cooldown period for temporary blacklists
    - Persistence to file
    """
    
    # Default thresholds
    DEFAULT_MIN_WIN_RATE = 0.45
    DEFAULT_MIN_PROFIT_FACTOR = 1.0
    DEFAULT_MIN_TRADES = 10  # Minimum trades before blacklisting
    DEFAULT_MAX_DRAWDOWN = 0.20  # 20% max drawdown per pair
    
    def __init__(
        self,
        static_blacklist: Optional[List[str]] = None,
        min_win_rate: float = DEFAULT_MIN_WIN_RATE,
        min_profit_factor: float = DEFAULT_MIN_PROFIT_FACTOR,
        min_trades_for_evaluation: int = DEFAULT_MIN_TRADES,
        max_drawdown: float = DEFAULT_MAX_DRAWDOWN,
        cooldown_hours: int = 24,  # How long pair stays blacklisted
        persistence_path: Optional[str] = None,
    ):
        """
        Initialize the blacklist.
        
        Args:
            static_blacklist: List of permanently blacklisted pairs.
            min_win_rate: Minimum win rate threshold.
            min_profit_factor: Minimum profit factor threshold.
            min_trades_for_evaluation: Minimum trades before evaluating.
            max_drawdown: Maximum drawdown before blacklisting.
            cooldown_hours: Hours a pair stays blacklisted.
            persistence_path: Path to save/load blacklist state.
        """
        self.static_blacklist: Set[str] = set(static_blacklist or [])
        self.dynamic_blacklist: Dict[str, BlacklistReason] = {}
        self.pair_metrics: Dict[str, PairMetrics] = {}
        
        # Thresholds
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor
        self.min_trades_for_evaluation = min_trades_for_evaluation
        self.max_drawdown = max_drawdown
        self.cooldown_hours = cooldown_hours
        
        # Persistence
        self.persistence_path = Path(persistence_path) if persistence_path else None
        
        # Load from file if exists
        if self.persistence_path and self.persistence_path.exists():
            self.load()
        
        logger.info(f"Blacklist initialized: {len(self.static_blacklist)} static, "
                   f"{len(self.dynamic_blacklist)} dynamic")
    
    def add_to_static_blacklist(self, pair: str, reason: str = "Manual") -> None:
        """Add pair to static (permanent) blacklist."""
        self.static_blacklist.add(pair)
        logger.warning(f"Added {pair} to static blacklist: {reason}")
        self._save_if_configured()
    
    def remove_from_static_blacklist(self, pair: str) -> bool:
        """Remove pair from static blacklist."""
        if pair in self.static_blacklist:
            self.static_blacklist.remove(pair)
            logger.info(f"Removed {pair} from static blacklist")
            self._save_if_configured()
            return True
        return False
    
    def update_pair_metrics(
        self,
        pair: str,
        trade_result: Dict
    ) -> None:
        """
        Update metrics for a pair after a trade.
        
        Args:
            pair: Trading pair symbol
            trade_result: Dict with 'pnl', 'is_win' keys
        """
        if pair not in self.pair_metrics:
            self.pair_metrics[pair] = PairMetrics(pair=pair)
        
        metrics = self.pair_metrics[pair]
        metrics.total_trades += 1
        metrics.total_pnl += trade_result['pnl']
        metrics.last_updated = datetime.now()
        
        if trade_result.get('is_win', trade_result['pnl'] > 0):
            metrics.winning_trades += 1
            metrics.gross_profit += trade_result['pnl']
        else:
            metrics.gross_loss += abs(trade_result['pnl'])
        
        # Check if pair should be blacklisted
        self._evaluate_pair(pair)
    
    def update_from_backtest(
        self,
        pair: str,
        trades: List[Dict]
    ) -> None:
        """
        Update metrics from backtest results.
        
        Args:
            pair: Trading pair symbol
            trades: List of trade dictionaries with 'pnl' key
        """
        metrics = PairMetrics(pair=pair)
        
        for trade in trades:
            pnl = trade['pnl']
            metrics.total_trades += 1
            metrics.total_pnl += pnl
            
            if pnl > 0:
                metrics.winning_trades += 1
                metrics.gross_profit += pnl
            else:
                metrics.gross_loss += abs(pnl)
        
        metrics.last_updated = datetime.now()
        self.pair_metrics[pair] = metrics
        
        # Evaluate
        self._evaluate_pair(pair)
    
    def update_from_performance(
        self,
        pair_metrics: Dict[str, Dict]
    ) -> None:
        """
        Bulk update from performance dictionary.
        
        Args:
            pair_metrics: Dict mapping pair to metrics dict
        """
        for pair, metrics in pair_metrics.items():
            self.pair_metrics[pair] = PairMetrics(
                pair=pair,
                total_trades=metrics.get('total_trades', 0),
                winning_trades=int(metrics.get('win_rate', 0) * metrics.get('total_trades', 0)),
                total_pnl=metrics.get('total_pnl', 0),
                gross_profit=metrics.get('gross_profit', 0),
                gross_loss=metrics.get('gross_loss', 0),
            )
            self._evaluate_pair(pair)
    
    def _evaluate_pair(self, pair: str) -> None:
        """Evaluate if pair should be blacklisted."""
        if pair in self.static_blacklist:
            return
        
        if pair not in self.pair_metrics:
            return
        
        metrics = self.pair_metrics[pair]
        
        # Need minimum trades
        if metrics.total_trades < self.min_trades_for_evaluation:
            return
        
        # Check win rate
        if metrics.win_rate < self.min_win_rate:
            self._add_to_dynamic_blacklist(
                pair,
                reason=f"Win rate {metrics.win_rate:.1%} < {self.min_win_rate:.1%}",
                metric_name="win_rate",
                metric_value=metrics.win_rate,
                threshold=self.min_win_rate
            )
            return
        
        # Check profit factor
        if metrics.profit_factor < self.min_profit_factor:
            self._add_to_dynamic_blacklist(
                pair,
                reason=f"Profit factor {metrics.profit_factor:.2f} < {self.min_profit_factor:.1f}",
                metric_name="profit_factor",
                metric_value=metrics.profit_factor,
                threshold=self.min_profit_factor
            )
            return
        
        # If pair passes, remove from dynamic blacklist
        if pair in self.dynamic_blacklist:
            del self.dynamic_blacklist[pair]
            logger.info(f"Removed {pair} from dynamic blacklist (metrics improved)")
    
    def _add_to_dynamic_blacklist(
        self,
        pair: str,
        reason: str,
        metric_name: str,
        metric_value: float,
        threshold: float
    ) -> None:
        """Add pair to dynamic blacklist."""
        self.dynamic_blacklist[pair] = BlacklistReason(
            reason=reason,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold
        )
        logger.warning(f"Blacklisted {pair}: {reason}")
        self._save_if_configured()
    
    def is_allowed(self, pair: str) -> bool:
        """
        Check if pair is allowed for trading.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            True if pair is allowed, False if blacklisted
        """
        # Check static blacklist
        if pair in self.static_blacklist:
            return False
        
        # Check dynamic blacklist with cooldown
        if pair in self.dynamic_blacklist:
            reason = self.dynamic_blacklist[pair]
            cooldown_expired = datetime.now() - reason.added_at > timedelta(hours=self.cooldown_hours)
            
            if cooldown_expired:
                # Cooldown expired, remove from blacklist
                del self.dynamic_blacklist[pair]
                logger.info(f"Cooldown expired for {pair}, removed from blacklist")
                return True
            
            return False
        
        return True
    
    def get_blacklist_reason(self, pair: str) -> Optional[str]:
        """Get reason why pair is blacklisted."""
        if pair in self.static_blacklist:
            return "Permanently blacklisted"
        
        if pair in self.dynamic_blacklist:
            return self.dynamic_blacklist[pair].reason
        
        return None
    
    def is_blacklisted(self, pair: str) -> bool:
        """
        Check if pair is blacklisted.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            True if pair is blacklisted, False if allowed
        """
        return not self.is_allowed(pair)
    
    def get_allowed_pairs(self, pairs: List[str]) -> List[str]:
        """Filter list of pairs to only allowed ones."""
        return [p for p in pairs if self.is_allowed(p)]
    
    def get_blacklisted_pairs(self) -> Dict[str, str]:
        """Get all blacklisted pairs with reasons."""
        result = {}
        
        for pair in self.static_blacklist:
            result[pair] = "Permanently blacklisted"
        
        for pair, reason in self.dynamic_blacklist.items():
            result[pair] = reason.reason
        
        return result
    
    def get_pair_stats(self, pair: str) -> Optional[Dict]:
        """Get performance stats for a pair."""
        if pair in self.pair_metrics:
            return self.pair_metrics[pair].to_dict()
        return None
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get performance stats for all pairs."""
        return {pair: m.to_dict() for pair, m in self.pair_metrics.items()}
    
    def _save_if_configured(self) -> None:
        """Save to file if persistence path is configured."""
        if self.persistence_path:
            self.save()
    
    def save(self, path: Optional[str] = None) -> None:
        """Save blacklist state to file."""
        save_path = Path(path) if path else self.persistence_path
        if not save_path:
            return
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'static_blacklist': list(self.static_blacklist),
            'dynamic_blacklist': {
                pair: reason.to_dict() 
                for pair, reason in self.dynamic_blacklist.items()
            },
            'pair_metrics': {
                pair: metrics.to_dict() 
                for pair, metrics in self.pair_metrics.items()
            },
            'config': {
                'min_win_rate': self.min_win_rate,
                'min_profit_factor': self.min_profit_factor,
                'min_trades_for_evaluation': self.min_trades_for_evaluation,
                'max_drawdown': self.max_drawdown,
                'cooldown_hours': self.cooldown_hours
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Blacklist saved to {save_path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """Load blacklist state from file."""
        load_path = Path(path) if path else self.persistence_path
        if not load_path or not load_path.exists():
            return
        
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        self.static_blacklist = set(data.get('static_blacklist', []))
        
        self.dynamic_blacklist = {}
        for pair, reason_data in data.get('dynamic_blacklist', {}).items():
            self.dynamic_blacklist[pair] = BlacklistReason(
                reason=reason_data['reason'],
                metric_name=reason_data['metric_name'],
                metric_value=reason_data['metric_value'],
                threshold=reason_data['threshold'],
                added_at=datetime.fromisoformat(reason_data['added_at'])
            )
        
        self.pair_metrics = {}
        for pair, metrics_data in data.get('pair_metrics', {}).items():
            self.pair_metrics[pair] = PairMetrics.from_dict(metrics_data)
        
        config = data.get('config', {})
        self.min_win_rate = config.get('min_win_rate', self.DEFAULT_MIN_WIN_RATE)
        self.min_profit_factor = config.get('min_profit_factor', self.DEFAULT_MIN_PROFIT_FACTOR)
        self.min_trades_for_evaluation = config.get('min_trades_for_evaluation', self.DEFAULT_MIN_TRADES)
        self.max_drawdown = config.get('max_drawdown', self.DEFAULT_MAX_DRAWDOWN)
        self.cooldown_hours = config.get('cooldown_hours', 24)
        
        logger.info(f"Blacklist loaded from {load_path}")
    
    def reset_dynamic_blacklist(self) -> None:
        """Clear all dynamic blacklist entries."""
        count = len(self.dynamic_blacklist)
        self.dynamic_blacklist.clear()
        logger.info(f"Cleared {count} entries from dynamic blacklist")
        self._save_if_configured()
    
    def reset_pair_metrics(self, pair: Optional[str] = None) -> None:
        """Reset metrics for a pair or all pairs."""
        if pair:
            if pair in self.pair_metrics:
                del self.pair_metrics[pair]
                logger.info(f"Reset metrics for {pair}")
        else:
            count = len(self.pair_metrics)
            self.pair_metrics.clear()
            logger.info(f"Reset metrics for {count} pairs")
        self._save_if_configured()
    
    def __repr__(self) -> str:
        return (f"PairBlacklist(static={len(self.static_blacklist)}, "
                f"dynamic={len(self.dynamic_blacklist)}, "
                f"tracked_pairs={len(self.pair_metrics)})")


# Default instance with ADA blacklisted
DEFAULT_BLACKLIST = PairBlacklist(
    static_blacklist=['ADA/USDT:USDT'],
    min_win_rate=0.45,
    min_profit_factor=1.0
)
