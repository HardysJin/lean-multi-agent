"""
策略工厂
统一管理和创建所有交易策略
"""

from typing import Dict, Any, Optional
from .double_ema_channel import DoubleEmaChannelStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .grid_trading import GridTradingStrategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .simple_actions import BuyStrategy, SellStrategy, HoldStrategy


class StrategyFactory:
    """
    策略工厂类
    负责创建和管理所有可用的交易策略
    """
    
    # 支持的策略映射
    STRATEGY_MAP = {
        'double_ema_channel': DoubleEmaChannelStrategy,
        'buy_and_hold': BuyAndHoldStrategy,
        'grid_trading': GridTradingStrategy,
        'momentum': MomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'buy': BuyStrategy,
        'sell': SellStrategy,
        'hold': HoldStrategy,
    }
    
    # 策略别名
    STRATEGY_ALIASES = {
        'ema_channel': 'double_ema_channel',
        'ema': 'double_ema_channel',
        'bah': 'buy_and_hold',
        'hold': 'hold',
        'grid': 'grid_trading',
        'mom': 'momentum',
        'mr': 'mean_reversion',
        'reversion': 'mean_reversion',
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str, **kwargs) -> Any:
        """
        创建策略实例
        
        Args:
            strategy_name: 策略名称
            **kwargs: 策略参数
        
        Returns:
            策略实例
        
        Raises:
            ValueError: 如果策略名称不支持
        """
        # 处理别名
        if strategy_name in cls.STRATEGY_ALIASES:
            strategy_name = cls.STRATEGY_ALIASES[strategy_name]
        
        # 获取策略类
        strategy_class = cls.STRATEGY_MAP.get(strategy_name)
        
        if strategy_class is None:
            available = list(cls.STRATEGY_MAP.keys())
            raise ValueError(
                f"不支持的策略: {strategy_name}. "
                f"可用策略: {', '.join(available)}"
            )
        
        # 创建策略实例
        return strategy_class(**kwargs)
    
    @classmethod
    def list_strategies(cls) -> Dict[str, Dict[str, Any]]:
        """
        列出所有可用策略及其信息
        
        Returns:
            策略信息字典
        """
        strategies_info = {}
        
        for name, strategy_class in cls.STRATEGY_MAP.items():
            # 创建临时实例获取策略信息
            temp_instance = strategy_class()
            strategies_info[name] = temp_instance.get_strategy_info()
        
        return strategies_info
    
    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        获取特定策略的信息
        
        Args:
            strategy_name: 策略名称
        
        Returns:
            策略信息字典，如果策略不存在返回None
        """
        # 处理别名
        if strategy_name in cls.STRATEGY_ALIASES:
            strategy_name = cls.STRATEGY_ALIASES[strategy_name]
        
        strategy_class = cls.STRATEGY_MAP.get(strategy_name)
        
        if strategy_class is None:
            return None
        
        temp_instance = strategy_class()
        return temp_instance.get_strategy_info()
