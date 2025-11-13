"""
网格交易策略 (Grid Trading)
在预设的价格区间内建立多个买卖价格档位（网格），在网格点位自动买入卖出
适合震荡行情，通过频繁买卖获取价差收益
"""

from typing import Dict, Any, List
import pandas as pd
from backend.config.config_loader import get_config
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
        self.config = get_config()
        self.price_lower = price_lower
        self.price_upper = price_upper
        self.grid_num = grid_num
        self.initial_grids = initial_grids
        self.auto_adjust = auto_adjust
        self.lookback_days = 30 # self.config.system.lookback_days
        
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
        
        # 价格超出网格范围的处理
        # 价格大幅低于网格下限 - 分情况处理，避免盲目抄底
        if current_price < self.price_lower * 0.95:
            drop_ratio = (self.price_lower - current_price) / self.price_lower
            
            # 策略1：如果已有持仓，谨慎加仓（防止继续下跌越套越深）
            if self.total_position > 0:
                # 已有持仓时，降低买入信心
                # 跌破5%: conf=0.4, 跌破10%: conf=0.5, 跌破20%: conf=0.6
                confidence = min(0.4 + drop_ratio * 0.5, 0.6)
                
                return {
                    'action': 'buy',
                    'reason': f'持仓中谨慎加仓：价格${current_price:.2f}低于网格下限${self.price_lower:.2f}（-{drop_ratio:.1%}），小幅分批建仓',
                    'confidence': confidence,
                    'price': current_price,
                    'grid_info': {
                        'current_grid': current_grid,
                        'total_grids': self.grid_num,
                        'grid_range': f'${self.price_lower:.2f} - ${self.price_upper:.2f}',
                        'drop_from_lower': f'{drop_ratio:.1%}',
                        'existing_position': self.total_position
                    }
                }
            
            # 策略2：空仓时，等待进一步确认或极端超跌
            else:
                # 跌破幅度不大（5-15%）：观望等待
                if drop_ratio < 0.15:
                    return {
                        'action': 'hold',
                        'reason': f'价格${current_price:.2f}低于网格下限${self.price_lower:.2f}（-{drop_ratio:.1%}），但跌幅未达极端水平，观望等待',
                        'confidence': 0.0,
                        'grid_info': {
                            'current_grid': current_grid,
                            'total_grids': self.grid_num,
                            'grid_range': f'${self.price_lower:.2f} - ${self.price_upper:.2f}',
                            'drop_from_lower': f'{drop_ratio:.1%}'
                        }
                    }
                
                # 跌破幅度较大（15-30%）：适度抄底
                elif drop_ratio < 0.30:
                    confidence = 0.5 + (drop_ratio - 0.15) * 0.67  # 0.5 to 0.6
                    return {
                        'action': 'buy',
                        'reason': f'适度抄底：价格${current_price:.2f}低于网格下限${self.price_lower:.2f}（-{drop_ratio:.1%}），分批建仓',
                        'confidence': confidence,
                        'price': current_price,
                        'grid_info': {
                            'current_grid': current_grid,
                            'total_grids': self.grid_num,
                            'grid_range': f'${self.price_lower:.2f} - ${self.price_upper:.2f}',
                            'drop_from_lower': f'{drop_ratio:.1%}'
                        }
                    }
                
                # 跌破幅度极大（>30%）：标记为极端机会，但仍保持谨慎
                else:
                    confidence = min(0.65 + (drop_ratio - 0.30) * 0.5, 0.8)  # 0.65 to 0.8
                    return {
                        'action': 'buy',
                        'reason': f'极端超跌抄底：价格${current_price:.2f}低于网格下限${self.price_lower:.2f}（-{drop_ratio:.1%}），可能是底部',
                        'confidence': confidence,
                        'extreme_opportunity': True,  # 仅在极端情况下标记
                        'price': current_price,
                        'grid_info': {
                            'current_grid': current_grid,
                            'total_grids': self.grid_num,
                            'grid_range': f'${self.price_lower:.2f} - ${self.price_upper:.2f}',
                            'drop_from_lower': f'{drop_ratio:.1%}'
                        }
                    }
        
        # 价格大幅高于网格上限 - 卖出获利
        if current_price > self.price_upper * 1.05:
            # 如果有持仓，卖出获利；如果没持仓，等待回落
            if self.total_position > 0:
                # 计算超出幅度
                exceed_ratio = (current_price - self.price_upper) / self.price_upper
                confidence = min(0.8 + exceed_ratio, 0.95)
                
                return {
                    'action': 'sell',
                    'reason': f'价格${current_price:.2f}高于网格上限${self.price_upper:.2f}（+{exceed_ratio:.1%}），获利了结',
                    'confidence': confidence,
                    'price': current_price,
                    'grid_info': {
                        'current_grid': current_grid,
                        'total_grids': self.grid_num,
                        'grid_range': f'${self.price_lower:.2f} - ${self.price_upper:.2f}',
                        'exceed_upper': f'{exceed_ratio:.1%}'
                    }
                }
            else:
                return {
                    'action': 'hold',
                    'reason': f'价格${current_price:.2f}高于网格上限${self.price_upper:.2f}，空仓等待回落',
                    'confidence': 0.0,
                    'grid_info': {
                        'current_grid': current_grid,
                        'total_grids': self.grid_num,
                        'grid_range': f'${self.price_lower:.2f} - ${self.price_upper:.2f}'
                    }
                }
        
        # 初始建仓：空仓时在中间档位开始建仓
        if self.total_position == 0 and current_grid <= self.grid_num // 2:
            # 初始建仓信心：位置越低，信心越高
            # 网格0: 0.9, 网格1: 0.82, 网格2: 0.74, 网格5: 0.5
            # 使用反向计算：距离底部越近，信心越高
            distance_to_bottom = current_grid / (self.grid_num // 2)  # 0 at bottom, 1 at mid
            confidence = 0.9 - distance_to_bottom * 0.4  # 0.9 at bottom, 0.5 at mid
            confidence = max(0.5, min(0.9, confidence))  # 限制在[0.5, 0.9]
            
            return {
                'action': 'buy',
                'reason': f'初始建仓：价格在网格{current_grid}/{self.grid_num}，位于下半区',
                'confidence': confidence,
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
                # 卖出信心：
                # 1. 基于网格位置：越接近上限，信心越高（0.3 -> 0.7）
                # 2. 基于盈利幅度：盈利越多，信心越高
                grid_position_factor = (current_grid - self.grid_num // 2) / (self.grid_num // 2)  # 0 to 1
                profit_factor = min(profit_pct / 30, 1.0)  # 盈利30%时达到1.0
                
                # 综合信心：位置权重60%，盈利权重40%
                confidence = 0.3 + grid_position_factor * 0.4 + profit_factor * 0.2
                confidence = max(0.3, min(0.9, confidence))  # 限制在[0.3, 0.9]
                
                return {
                    'action': 'sell',
                    'reason': f'网格卖出：价格在网格{current_grid}/{self.grid_num}，盈利{profit_pct:.2f}%',
                    'confidence': confidence,
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
                # 加仓信心：越接近下限，信心越高（0.2 -> 0.9）
                # 距离下限越近，买入信心越强
                distance_to_bottom = current_grid / (self.grid_num // 2)  # 1 at mid, 0 at bottom
                confidence = 0.9 - distance_to_bottom * 0.4  # 0.2 at mid, 0.9 at bottom
                confidence = max(0.2, min(0.9, confidence))
                
                return {
                    'action': 'buy',
                    'reason': f'网格加仓：价格回落到网格{current_grid}/{self.grid_num}',
                    'confidence': confidence,
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
