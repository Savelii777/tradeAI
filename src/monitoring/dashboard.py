"""
AI Trading Bot - Dashboard
Web-based monitoring dashboard.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

# Dash imports will be used when dashboard is run
# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import plotly.graph_objs as go


class Dashboard:
    """
    Web-based monitoring dashboard.
    
    Displays:
    - Account status
    - Open positions
    - Recent trades
    - Performance metrics
    - Model status
    - Alerts
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """
        Initialize dashboard.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.port = self.config.get('port', 8050)
        self.host = self.config.get('host', '0.0.0.0')
        
        self._data_sources: Dict[str, Any] = {}
        self._app = None
        
    def register_data_source(
        self,
        name: str,
        source: Any
    ) -> None:
        """
        Register a data source for the dashboard.
        
        Args:
            name: Data source name.
            source: Data source object.
        """
        self._data_sources[name] = source
        logger.debug(f"Registered dashboard data source: {name}")
        
    def create_app(self):
        """Create the Dash application."""
        try:
            import dash
            from dash import dcc, html
        except ImportError:
            logger.warning("Dash not installed. Dashboard unavailable.")
            return None
            
        self._app = dash.Dash(__name__)
        
        self._app.layout = html.Div([
            # Header
            html.Div([
                html.H1("AI Trading Bot Dashboard"),
                html.Div(id='last-update', style={'float': 'right'})
            ], className='header'),
            
            # Main content
            html.Div([
                # Account section
                html.Div([
                    html.H2("Account Status"),
                    html.Div(id='account-info')
                ], className='section'),
                
                # Positions section
                html.Div([
                    html.H2("Open Positions"),
                    html.Div(id='positions-table')
                ], className='section'),
                
                # Performance section
                html.Div([
                    html.H2("Performance"),
                    dcc.Graph(id='equity-chart'),
                    html.Div(id='performance-metrics')
                ], className='section'),
                
                # Recent trades section
                html.Div([
                    html.H2("Recent Trades"),
                    html.Div(id='trades-table')
                ], className='section'),
                
                # Model status section
                html.Div([
                    html.H2("Model Status"),
                    html.Div(id='model-status')
                ], className='section'),
                
                # Alerts section
                html.Div([
                    html.H2("Recent Alerts"),
                    html.Div(id='alerts-list')
                ], className='section'),
            ], className='main-content'),
            
            # Auto-refresh interval
            dcc.Interval(
                id='refresh-interval',
                interval=5000,  # 5 seconds
                n_intervals=0
            )
        ])
        
        self._setup_callbacks()
        
        return self._app
        
    def _setup_callbacks(self) -> None:
        """Setup Dash callbacks for live updates."""
        from dash.dependencies import Input, Output
        
        @self._app.callback(
            Output('last-update', 'children'),
            Input('refresh-interval', 'n_intervals')
        )
        def update_timestamp(n):
            return f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
        @self._app.callback(
            Output('account-info', 'children'),
            Input('refresh-interval', 'n_intervals')
        )
        def update_account_info(n):
            from dash import html
            
            account = self._get_account_data()
            if not account:
                return "No account data available"
                
            return html.Div([
                html.P(f"Balance: ${account.get('balance', 0):.2f}"),
                html.P(f"Equity: ${account.get('equity', 0):.2f}"),
                html.P(f"Daily P&L: ${account.get('daily_pnl', 0):.2f}"),
                html.P(f"Open Positions: {account.get('open_positions', 0)}")
            ])
            
    def _get_account_data(self) -> Dict[str, Any]:
        """Get account data from data sources."""
        if 'account' in self._data_sources:
            return self._data_sources['account'].get_state()
        return {}
        
    def _get_positions_data(self) -> List[Dict]:
        """Get positions data from data sources."""
        if 'positions' in self._data_sources:
            return self._data_sources['positions'].get_open_positions()
        return []
        
    def _get_trades_data(self) -> List[Dict]:
        """Get trades data from data sources."""
        if 'trades' in self._data_sources:
            return self._data_sources['trades'].get_recent_trades()
        return []
        
    def _get_model_data(self) -> Dict[str, Any]:
        """Get model data from data sources."""
        if 'model' in self._data_sources:
            return self._data_sources['model'].get_status()
        return {}
        
    def _get_alerts_data(self) -> List[Dict]:
        """Get alerts data from data sources."""
        if 'alerts' in self._data_sources:
            return self._data_sources['alerts'].get_recent_alerts()
        return []
        
    def run(self, debug: bool = False) -> None:
        """
        Run the dashboard server.
        
        Args:
            debug: Enable debug mode.
        """
        if self._app is None:
            self.create_app()
            
        if self._app:
            logger.info(f"Starting dashboard on {self.host}:{self.port}")
            self._app.run_server(
                host=self.host,
                port=self.port,
                debug=debug
            )
        else:
            logger.warning("Cannot start dashboard - app not created")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        return {
            'data_sources': list(self._data_sources.keys()),
            'port': self.port,
            'running': self._app is not None
        }
