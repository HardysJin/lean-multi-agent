"""
简单执行策略 (Simple Action Strategies)
用于执行LLM的买入/卖出/持有决策信号
"""

from typing import Dict, Any
import pandas as pd
from backend.strategies.base_strategy import BaseStrategy


class BuyStrategy(BaseStrategy):
    """
    买入策略
    
    策略逻辑：
    - 接收LLM的买入信号
    - 根据建议的仓位比例买入
    - 支持增仓或建仓
    
    这是一个执行策略，用于将LLM的买入决策转化为实际交易信号
    """
    
    def __init__(self, target_exposure: float = 0.7):
        """
        初始化策略
        
        Args:
            target_exposure: 目标仓位比例 (0.0-1.0)，默认70%
        """
        super().__init__(target_exposure=target_exposure)
        self.target_exposure = max(0.0, min(1.0, target_exposure))
    
    def get_required_data_points(self) -> int:
        """返回策略需要的最小数据点数"""
        return 1  # 买入信号只需要当前价格
    
    def generate_signals(self, market_data: pd.DataFrame, **context) -> Dict[str, Any]:
        """
        生成交易信号
        
        Args:
            market_data: DataFrame包含OHLCV数据
            **context: 上下文信息（如当前持仓、账户信息等）
                - position: 当前持仓数量
                - portfolio_value: 组合总价值
                - cash: 可用现金
                - entry_price: 入场价格
        
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
        
        # 从上下文获取账户信息
        current_position = context.get('position', 0)
        portfolio_value = context.get('portfolio_value', 0)
        cash = context.get('cash', 0)
        entry_price = context.get('entry_price', 0)
        
        # 计算当前仓位比例
        if portfolio_value > 0:
            position_value = current_position * current_price
            current_exposure = position_value / portfolio_value
        else:
            current_exposure = 0.0
        
        # 如果当前仓位低于目标仓位，执行买入
        if current_exposure < self.target_exposure - 0.01:  # 留1%容差
            # 计算需要达到的目标仓位价值
            target_value = portfolio_value * self.target_exposure
            current_value = current_position * current_price
            need_to_buy_value = target_value - current_value
            
            # 检查是否有足够现金
            if cash < need_to_buy_value:
                # 现金不足，用全部现金买入
                actual_buy_value = cash
                actual_exposure = (current_value + actual_buy_value) / portfolio_value
                
                return {
                    'action': 'buy',
                    'reason': f'LLM买入信号：目标{self.target_exposure*100:.0f}%仓位，实际{actual_exposure*100:.1f}%（现金限制）',
                    'confidence': 0.9,
                    'price': current_price,
                    'target_exposure': self.target_exposure,
                    'actual_exposure': actual_exposure,
                    'indicators': {
                        'current_exposure': f'{current_exposure*100:.1f}%',
                        'target_exposure': f'{self.target_exposure*100:.0f}%',
                        'cash_available': cash,
                        'buy_value': actual_buy_value
                    }
                }
            else:
                # 现金充足，按目标仓位买入
                return {
                    'action': 'buy',
                    'reason': f'LLM买入信号：当前{current_exposure*100:.1f}% → 目标{self.target_exposure*100:.0f}%仓位',
                    'confidence': 1.0,
                    'price': current_price,
                    'target_exposure': self.target_exposure,
                    'indicators': {
                        'current_exposure': f'{current_exposure*100:.1f}%',
                        'target_exposure': f'{self.target_exposure*100:.0f}%',
                        'buy_value': need_to_buy_value
                    }
                }
        
        # 已经达到或超过目标仓位，持有
        else:
            profit_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
            return {
                'action': 'hold',
                'reason': f'已达目标仓位{self.target_exposure*100:.0f}% (当前{current_exposure*100:.1f}%)',
                'confidence': 1.0,
                'price': current_price,
                'profit_pct': profit_pct,
                'indicators': {
                    'current_exposure': f'{current_exposure*100:.1f}%',
                    'target_exposure': f'{self.target_exposure*100:.0f}%'
                }
            }
    
    def reset(self):
        """重置策略状态"""
        pass


class SellStrategy(BaseStrategy):
    """
    卖出策略
    
    策略逻辑：
    - 接收LLM的卖出信号
    - 根据建议的仓位比例卖出或清仓
    - 支持减仓或全部退出
    
    这是一个执行策略，用于将LLM的卖出决策转化为实际交易信号
    """
    
    def __init__(self, target_exposure: float = 0.0):
        """
        初始化策略
        
        Args:
            target_exposure: 目标仓位比例 (0.0-1.0)，默认0%（全部卖出）
        """
        super().__init__(target_exposure=target_exposure)
        self.target_exposure = max(0.0, min(1.0, target_exposure))
    
    def get_required_data_points(self) -> int:
        """返回策略需要的最小数据点数"""
        return 1  # 卖出信号只需要当前价格
    
    def generate_signals(self, market_data: pd.DataFrame, **context) -> Dict[str, Any]:
        """
        生成交易信号
        
        Args:
            market_data: DataFrame包含OHLCV数据
            **context: 上下文信息（如当前持仓、账户信息等）
                - position: 当前持仓数量
                - portfolio_value: 组合总价值
                - entry_price: 入场价格
        
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
        
        # 从上下文获取账户信息
        current_position = context.get('position', 0)
        portfolio_value = context.get('portfolio_value', 0)
        entry_price = context.get('entry_price', 0)
        
        # 计算盈亏
        profit_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        
        # 如果没有持仓，无需卖出
        if current_position == 0:
            return {
                'action': 'hold',
                'reason': 'LLM卖出信号：但当前无持仓',
                'confidence': 1.0,
                'price': current_price,
                'indicators': {
                    'current_exposure': '0.0%',
                    'target_exposure': f'{self.target_exposure*100:.0f}%'
                }
            }
        
        # 计算当前仓位比例
        if portfolio_value > 0:
            position_value = current_position * current_price
            current_exposure = position_value / portfolio_value
        else:
            current_exposure = 0.0
        
        # 如果当前仓位高于目标仓位，执行卖出
        if current_exposure > self.target_exposure + 0.01:  # 留1%容差
            if self.target_exposure == 0.0:
                # 全部清仓
                return {
                    'action': 'sell',
                    'reason': f'LLM卖出信号：清仓退出 (盈利: {profit_pct:+.2f}%)',
                    'confidence': 1.0,
                    'price': current_price,
                    'profit_pct': profit_pct,
                    'target_exposure': 0.0,
                    'indicators': {
                        'current_exposure': f'{current_exposure*100:.1f}%',
                        'target_exposure': '0.0%',
                        'profit': f'{profit_pct:+.2f}%'
                    }
                }
            else:
                # 减仓到目标比例
                return {
                    'action': 'sell',
                    'reason': f'LLM卖出信号：减仓 {current_exposure*100:.1f}% → {self.target_exposure*100:.0f}% (盈利: {profit_pct:+.2f}%)',
                    'confidence': 1.0,
                    'price': current_price,
                    'profit_pct': profit_pct,
                    'target_exposure': self.target_exposure,
                    'indicators': {
                        'current_exposure': f'{current_exposure*100:.1f}%',
                        'target_exposure': f'{self.target_exposure*100:.0f}%',
                        'profit': f'{profit_pct:+.2f}%'
                    }
                }
        
        # 已经低于或等于目标仓位，持有
        else:
            if current_position > 0:
                return {
                    'action': 'hold',
                    'reason': f'已达目标仓位{self.target_exposure*100:.0f}% (当前{current_exposure*100:.1f}%, 盈利: {profit_pct:+.2f}%)',
                    'confidence': 1.0,
                    'price': current_price,
                    'profit_pct': profit_pct,
                    'indicators': {
                        'current_exposure': f'{current_exposure*100:.1f}%',
                        'target_exposure': f'{self.target_exposure*100:.0f}%',
                        'profit': f'{profit_pct:+.2f}%'
                    }
                }
            else:
                return {
                    'action': 'hold',
                    'reason': '已清仓，保持观望',
                    'confidence': 1.0,
                    'price': current_price,
                    'indicators': {
                        'current_exposure': '0.0%',
                        'target_exposure': f'{self.target_exposure*100:.0f}%'
                    }
                }
    
    def reset(self):
        """重置策略状态"""
        pass


class HoldStrategy(BaseStrategy):
    """
    持有策略
    
    策略逻辑：
    - 接收LLM的持有信号
    - 维持当前仓位不变
    - 观察市场等待更好的时机
    
    这是一个执行策略，用于将LLM的持有决策转化为实际交易信号
    """
    
    def __init__(self):
        """初始化策略"""
        super().__init__()
    
    def get_required_data_points(self) -> int:
        """返回策略需要的最小数据点数"""
        return 1  # 持有信号只需要当前价格
    
    def generate_signals(self, market_data: pd.DataFrame, **context) -> Dict[str, Any]:
        """
        生成交易信号
        
        Args:
            market_data: DataFrame包含OHLCV数据
            **context: 上下文信息（如当前持仓、账户信息等）
                - position: 当前持仓数量
                - portfolio_value: 组合总价值
                - entry_price: 入场价格
                - cash: 可用现金
        
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
        
        # 从上下文获取账户信息
        current_position = context.get('position', 0)
        portfolio_value = context.get('portfolio_value', 0)
        entry_price = context.get('entry_price', 0)
        cash = context.get('cash', 0)
        
        # 计算当前仓位比例
        if portfolio_value > 0:
            position_value = current_position * current_price
            current_exposure = position_value / portfolio_value
            cash_ratio = cash / portfolio_value
        else:
            current_exposure = 0.0
            cash_ratio = 1.0
        
        # 如果有持仓，显示盈亏
        if current_position > 0 and entry_price > 0:
            profit_pct = ((current_price / entry_price) - 1) * 100
            
            return {
                'action': 'hold',
                'reason': f'LLM持有信号：维持当前仓位 (盈利: {profit_pct:+.2f}%)',
                'confidence': 1.0,
                'price': current_price,
                'profit_pct': profit_pct,
                'indicators': {
                    'current_exposure': f'{current_exposure*100:.1f}%',
                    'cash_ratio': f'{cash_ratio*100:.1f}%',
                    'profit': f'{profit_pct:+.2f}%',
                    'position_value': f'${position_value:,.2f}'
                }
            }
        
        # 如果无持仓（100%现金）
        elif current_position == 0:
            return {
                'action': 'hold',
                'reason': 'LLM持有信号：保持观望，等待入场时机',
                'confidence': 1.0,
                'price': current_price,
                'indicators': {
                    'current_exposure': '0.0%',
                    'cash_ratio': '100.0%',
                    'status': '空仓观望'
                }
            }
        
        # 有持仓但无入场价格信息
        else:
            return {
                'action': 'hold',
                'reason': f'LLM持有信号：维持当前仓位 ({current_exposure*100:.1f}%)',
                'confidence': 1.0,
                'price': current_price,
                'indicators': {
                    'current_exposure': f'{current_exposure*100:.1f}%',
                    'cash_ratio': f'{cash_ratio*100:.1f}%'
                }
            }
    
    def reset(self):
        """重置策略状态"""
        pass
