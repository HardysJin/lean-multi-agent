"""
策略模块
包含各种交易策略实现

所有策略都继承自BaseStrategy基类，只负责生成信号，不管理仓位。
仓位管理由llm_backtest或其他执行层负责。
"""

from .base_strategy import BaseStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .double_ema_channel import DoubleEmaChannelStrategy
from .grid_trading import GridTradingStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .simple_actions import BuyStrategy, SellStrategy, HoldStrategy
from .strategy_factory import StrategyFactory

__all__ = [
    'BaseStrategy',
    'BuyAndHoldStrategy',
    'DoubleEmaChannelStrategy',
    'GridTradingStrategy',
    'MeanReversionStrategy',
    'MomentumStrategy',
    'BuyStrategy',
    'SellStrategy',
    'HoldStrategy',
    'StrategyFactory',
]
