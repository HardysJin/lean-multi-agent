"""
交易策略基类
定义所有策略的统一接口，策略只负责生成信号，不管理仓位
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class BaseStrategy(ABC):
    """
    交易策略基类
    
    所有具体策略都应该继承这个基类并实现抽象方法
    
    核心原则：
    - 策略只生成信号（buy/sell/hold），不管理仓位
    - 仓位管理由llm_backtest或其他执行层负责
    - 策略保持无状态或仅维护必要的技术指标状态
    """
    
    def __init__(self, **kwargs):
        """
        初始化策略
        
        Args:
            **kwargs: 策略特定参数
        """
        self.name = self.__class__.__name__
        self.parameters = kwargs
    
    @abstractmethod
    def get_required_data_points(self) -> int:
        """
        返回策略需要的最小数据点数
        
        Returns:
            最小数据点数（用于指标计算）
        """
        pass
    
    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame, **context) -> Dict[str, Any]:
        """
        基于市场数据生成交易信号
        
        这是策略的核心方法，必须由子类实现
        
        Args:
            market_data: DataFrame包含OHLCV数据，列名：Open, High, Low, Close, Volume
            **context: 额外的上下文信息（如当前持仓状态、账户信息等）
                      策略可以使用这些信息来生成更好的信号，但不应该修改它们
        
        Returns:
            交易信号字典，必须包含以下字段：
            {
                'action': str,  # 'buy', 'sell', 或 'hold'
                'reason': str,  # 信号原因说明
                'confidence': float,  # 信号置信度 (0.0-1.0)
                'price': float,  # 当前价格
                'indicators': dict,  # 可选：相关技术指标值
                ... 其他策略特定字段
            }
        """
        pass
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        返回策略的基本信息
        
        Returns:
            策略信息字典
        """
        return {
            'name': self.name,
            'description': self.__doc__.strip() if self.__doc__ else '',
            'parameters': self.parameters,
            'required_data_points': self.get_required_data_points()
        }
    
    def reset(self):
        """
        重置策略状态
        
        某些策略可能需要在回测开始时重置内部状态
        子类可以重写此方法
        """
        pass
    
    def validate_market_data(self, market_data: pd.DataFrame) -> bool:
        """
        验证市场数据是否有效
        
        Args:
            market_data: 市场数据DataFrame
        
        Returns:
            数据是否有效
        """
        if market_data is None or len(market_data) == 0:
            return False
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in market_data.columns:
                return False
        
        return True
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.parameters.items())})"
    
    def __repr__(self) -> str:
        """对象表示"""
        return self.__str__()
