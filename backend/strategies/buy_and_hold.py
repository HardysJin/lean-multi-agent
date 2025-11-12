"""
买入持有策略 (Buy and Hold)
最基础的投资策略，在期初买入并持有到期末
"""

from typing import Dict, Any
import pandas as pd


class BuyAndHoldStrategy:
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
        self.position = 0
        self.entry_price = 0
        self.has_bought = False
    
    def get_required_data_points(self) -> int:
        """返回策略需要的最小数据点数"""
        return 1  # 买入持有只需要1个数据点
    
    def generate_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成交易信号
        
        Args:
            market_data: DataFrame包含OHLCV数据
        
        Returns:
            交易信号字典
        """
        if len(market_data) == 0:
            return {
                'action': 'hold',
                'reason': '无市场数据',
                'confidence': 0.0
            }
        
        current_price = market_data['Close'].iloc[-1]
        
        # 如果还没买入，生成买入信号
        if not self.has_bought and self.position == 0:
            return {
                'action': 'buy',
                'reason': '买入持有策略：初始买入',
                'confidence': 1.0,
                'price': current_price,
                'indicators': {}
            }
        
        # 已经买入，持续持有
        else:
            profit_pct = ((current_price / self.entry_price) - 1) * 100 if self.entry_price > 0 else 0
            return {
                'action': 'hold',
                'reason': f'买入持有中 (盈利: {profit_pct:.2f}%)',
                'confidence': 1.0,
                'price': current_price,
                'profit_pct': profit_pct
            }
    
    def execute_trade(self, action: str, price: float):
        """
        执行交易并更新状态
        
        Args:
            action: 交易动作
            price: 交易价格
        """
        if action == 'buy':
            self.position = 1
            self.entry_price = price
            self.has_bought = True
        elif action == 'sell':
            self.position = 0
            # 注意：买入持有策略不应该卖出，除非强制清仓
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """返回策略信息"""
        return {
            'name': 'buy_and_hold',
            'description': '买入持有策略（基准策略）',
            'parameters': {},
            'risk_level': 'low',
            'suitable_market': ['bullish', 'any']
        }
