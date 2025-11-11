"""
回测引擎模块
提供简单高效的策略回测功能
"""

from .simple_backtest import BacktestEngine
from .strategy_comparison import StrategyComparison

__all__ = ['BacktestEngine', 'StrategyComparison']
