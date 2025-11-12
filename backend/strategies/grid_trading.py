"""
网格交易策略 (Grid Trading)
在预设的价格区间内建立多个买卖价格档位（网格），在网格点位自动买入卖出
适合震荡行情，通过频繁买卖获取价差收益
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


class GridTradingStrategy:
    """
    网格交易策略
    
    策略逻辑：
    - 设定价格上限、下限，均分为N个网格
    - 价格下跌到网格支撑位时买入
    - 价格上涨到网格阻力位时卖出
    - 适合横盘震荡行情
    
    参数：
    - price_lower: 价格下限
    - price_upper: 价格上限
    - grid_num: 网格数量
    - initial_grids: 初始建仓网格数（买入几格）
    """
    
    def __init__(
        self,
        price_lower: float = None,
        price_upper: float = None,
        grid_num: int = 10,
        initial_grids: int = 5,
        auto_adjust: bool = True,  # 自动根据历史数据调整网格范围
        lookback_days: int = 30    # 回看天数用于自动调整
    ):
        """
        初始化网格交易策略
        
        Args:
            price_lower: 价格下限（None则自动计算）
            price_upper: 价格上限（None则自动计算）
            grid_num: 网格数量
            initial_grids: 初始建仓网格数
            auto_adjust: 是否自动调整网格范围
            lookback_days: 用于自动调整的回看天数
        """
        self.price_lower = price_lower
        self.price_upper = price_upper
        self.grid_num = grid_num
        self.initial_grids = initial_grids
        self.auto_adjust = auto_adjust
        self.lookback_days = lookback_days
        
        # 网格配置
        self.grids = []  # 网格价格点位
        self.grid_size = 0  # 每格大小
        self.initialized = False
        
        # 持仓状态
        self.positions = {}  # {grid_index: quantity} 每个网格的持仓
        self.total_position = 0
        self.last_action_price = 0
    
    def get_required_data_points(self) -> int:
        """返回策略需要的最小数据点数"""
        return self.lookback_days if self.auto_adjust else 1
        
    def _initialize_grids(self, market_data: pd.DataFrame):
        """
        初始化网格
        根据历史数据自动计算网格范围
        """
        if self.auto_adjust or self.price_lower is None or self.price_upper is None:
            # 使用最近N天的数据计算价格区间
            recent_data = market_data.tail(self.lookback_days)
            low = recent_data['Low'].min()
            high = recent_data['High'].max()
            
            # 扩展10%作为缓冲
            price_range = high - low
            self.price_lower = low - price_range * 0.1
            self.price_upper = high + price_range * 0.1
        
        # 生成网格点位
        self.grid_size = (self.price_upper - self.price_lower) / self.grid_num
        self.grids = [self.price_lower + i * self.grid_size for i in range(self.grid_num + 1)]
        
        self.initialized = True
        
    def _find_grid_level(self, price: float) -> int:
        """
        找到价格对应的网格层级
        
        Returns:
            网格索引 (0到grid_num)
        """
        if price <= self.price_lower:
            return 0
        if price >= self.price_upper:
            return self.grid_num
        
        return int((price - self.price_lower) / self.grid_size)
    
    def generate_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成交易信号
        
        Args:
            market_data: DataFrame包含OHLCV数据
        
        Returns:
            交易信号字典
        """
        if len(market_data) < self.lookback_days:
            return {
                'action': 'hold',
                'reason': f'数据不足，需要至少{self.lookback_days}个数据点',
                'confidence': 0.0
            }
        
        # 初始化网格
        if not self.initialized:
            self._initialize_grids(market_data)
        
        latest = market_data.iloc[-1]
        current_price = latest['Close']
        current_grid = self._find_grid_level(current_price)
        
        # 价格超出网格范围
        if current_price < self.price_lower * 0.95:
            return {
                'action': 'hold',
                'reason': f'价格${current_price:.2f}低于网格下限${self.price_lower:.2f}，等待回归',
                'confidence': 0.0,
                'grid_info': {
                    'current_grid': current_grid,
                    'total_grids': self.grid_num,
                    'grid_range': f'${self.price_lower:.2f} - ${self.price_upper:.2f}'
                }
            }
        
        if current_price > self.price_upper * 1.05:
            return {
                'action': 'hold',
                'reason': f'价格${current_price:.2f}高于网格上限${self.price_upper:.2f}，等待回落',
                'confidence': 0.0,
                'grid_info': {
                    'current_grid': current_grid,
                    'total_grids': self.grid_num,
                    'grid_range': f'${self.price_lower:.2f} - ${self.price_upper:.2f}'
                }
            }
        
        # 初始建仓：空仓时在中间档位开始建仓
        if self.total_position == 0 and current_grid <= self.grid_num // 2:
            return {
                'action': 'buy',
                'reason': f'初始建仓：价格在网格{current_grid}/{self.grid_num}，位于下半区',
                'confidence': 0.7,
                'price': current_price,
                'grid_info': {
                    'current_grid': current_grid,
                    'grid_price': self.grids[current_grid],
                    'total_grids': self.grid_num
                }
            }
        
        # 有持仓情况下的交易逻辑
        if self.total_position > 0:
            # 计算平均成本
            avg_cost = sum(self.grids[g] * q for g, q in self.positions.items()) / self.total_position if self.total_position > 0 else 0
            profit_pct = ((current_price / avg_cost) - 1) * 100 if avg_cost > 0 else 0
            
            # 卖出信号：价格上涨到上方网格
            if current_grid > self.grid_num // 2 and profit_pct > 2:  # 至少2%利润
                return {
                    'action': 'sell',
                    'reason': f'网格卖出：价格在网格{current_grid}/{self.grid_num}，盈利{profit_pct:.2f}%',
                    'confidence': 0.8,
                    'price': current_price,
                    'profit_pct': profit_pct,
                    'grid_info': {
                        'current_grid': current_grid,
                        'avg_cost': avg_cost,
                        'positions': len(self.positions)
                    }
                }
            
            # 加仓信号：价格下跌到下方网格且未在此网格建仓
            elif current_grid < self.grid_num // 2 and current_grid not in self.positions:
                return {
                    'action': 'buy',
                    'reason': f'网格加仓：价格回落到网格{current_grid}/{self.grid_num}',
                    'confidence': 0.75,
                    'price': current_price,
                    'grid_info': {
                        'current_grid': current_grid,
                        'avg_cost': avg_cost,
                        'positions': len(self.positions)
                    }
                }
        
        # 持有
        grid_position = "上半区" if current_grid > self.grid_num // 2 else "下半区"
        return {
            'action': 'hold',
            'reason': f'在网格{current_grid}/{self.grid_num}({grid_position})，等待交易机会',
            'confidence': 0.5,
            'grid_info': {
                'current_grid': current_grid,
                'total_grids': self.grid_num,
                'positions': len(self.positions),
                'total_position': self.total_position
            }
        }
    
    def execute_trade(self, action: str, price: float):
        """
        执行交易并更新持仓状态
        
        Args:
            action: 交易动作
            price: 交易价格
        """
        grid_level = self._find_grid_level(price)
        
        if action == 'buy':
            # 记录在该网格的持仓
            if grid_level not in self.positions:
                self.positions[grid_level] = 0
            self.positions[grid_level] += 1
            self.total_position += 1
            self.last_action_price = price
            
        elif action == 'sell':
            # 卖出最高网格的持仓（先进先出）
            if self.positions:
                highest_grid = max(self.positions.keys())
                self.positions[highest_grid] -= 1
                if self.positions[highest_grid] <= 0:
                    del self.positions[highest_grid]
                self.total_position -= 1
                self.last_action_price = price
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """返回策略信息"""
        return {
            'name': 'grid_trading',
            'description': '网格交易策略（适合震荡行情）',
            'parameters': {
                'grid_num': self.grid_num,
                'price_range': f'${self.price_lower:.2f} - ${self.price_upper:.2f}' if self.initialized else 'auto',
                'grid_size': f'${self.grid_size:.2f}' if self.initialized else 'auto'
            },
            'risk_level': 'medium',
            'suitable_market': ['sideways', 'range-bound']
        }
