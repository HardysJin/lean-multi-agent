"""
买入持有策略 (Buy and Hold)
最基础的投资策略，在期初买入并持有到期末
"""

from typing import Dict, Any
import pandas as pd
from backend.strategies.base_strategy import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """
    买入持有策略
    
    策略逻辑：
    - 在第一个交易日买入
    - 持有到最后
    - 不做任何调整
    
    这是最简单的基准策略，常用于对比其他策略的表现
    """
    
    def __init__(self):
        """初始化策略"""
        super().__init__()
        self.has_generated_buy_signal = False
    
    def get_required_data_points(self) -> int:
        """返回策略需要的最小数据点数"""
        return 1  # 买入持有只需要1个数据点
    
    def generate_signals(self, market_data: pd.DataFrame, **context) -> Dict[str, Any]:
        """
        生成交易信号
        
        Args:
            market_data: DataFrame包含OHLCV数据
            **context: 上下文信息（如持仓状态、入场价格等）
        
        Returns:
            交易信号字典
        """
        if not self.validate_market_data(market_data):
            return {
                'action': 'hold',
                'reason': '无市场数据',
                'confidence': 0.0
            }
        
        current_price = market_data['Close'].iloc[-1]
        
        # 从上下文获取当前持仓状态
        current_position = context.get('position', 0)
        entry_price = context.get('entry_price', 0)
        
        # 如果还没持仓，生成买入信号
        # 注意：只要当前无持仓，就应该买入（不依赖内部状态标记）
        if current_position == 0:
            return {
                'action': 'buy',
                'reason': '买入持有策略：初始买入',
                'confidence': 1.0,
                'price': current_price,
                'indicators': {}
            }
        
        # 已经持仓，持续持有
        else:
            profit_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
            return {
                'action': 'hold',
                'reason': f'买入持有中 (盈利: {profit_pct:.2f}%)' if entry_price > 0 else '买入持有中',
                'confidence': 1.0,
                'price': current_price,
                'profit_pct': profit_pct
            }
    
    def reset(self):
        """重置策略状态"""
        self.has_generated_buy_signal = False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """返回策略信息"""
        info = super().get_strategy_info()
        info.update({
            'name': 'buy_and_hold',
            'description': '买入持有策略（基准策略）',
            'parameters': {},
            'risk_level': 'low',
            'suitable_market': ['bullish', 'any']
        })
        return info
