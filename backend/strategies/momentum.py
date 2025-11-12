"""
动量策略 (Momentum Strategy)
基于价格动量（价格变化的速度）进行交易
适合趋势明显的行情，追涨杀跌
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


class MomentumStrategy:
    """
    动量策略
    
    策略逻辑：
    - 计算价格动量（N日收益率）
    - 动量为正且强势时买入
    - 动量转负或减弱时卖出
    - 结合成交量和移动平均线过滤信号
    
    参数：
    - momentum_period: 动量计算周期（天）
    - entry_threshold: 买入动量阈值（百分比）
    - exit_threshold: 卖出动量阈值（百分比）
    - ma_period: 移动平均线周期（用于趋势确认）
    - volume_multiplier: 成交量放大倍数（用于确认突破）
    """
    
    def __init__(
        self,
        momentum_period: int = 20,
        entry_threshold: float = 1.5,   # 1.5%上涨动量（进一步降低买入门槛）
        exit_threshold: float = -1.0,   # -1%下跌动量（更及时止损）
        ma_period: int = 50,
        volume_multiplier: float = 1.0,  # 1.0倍（不要求成交量放大）
        use_rsi_filter: bool = False,   # 关闭RSI过滤器（避免错过买入机会）
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30
    ):
        """
        初始化动量策略
        
        Args:
            momentum_period: 动量计算周期
            entry_threshold: 买入动量阈值（%）
            exit_threshold: 卖出动量阈值（%）
            ma_period: 移动平均线周期
            volume_multiplier: 成交量倍数
            use_rsi_filter: 是否使用RSI过滤
            rsi_period: RSI周期
            rsi_overbought: RSI超买线
            rsi_oversold: RSI超卖线
        """
        self.momentum_period = momentum_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.ma_period = ma_period
        self.volume_multiplier = volume_multiplier
        self.use_rsi_filter = use_rsi_filter
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        
        # 持仓状态
        self.position = 0
        self.entry_price = 0
        self.entry_momentum = 0
    
    def get_required_data_points(self) -> int:
        """返回策略需要的最小数据点数"""
        return max(self.momentum_period, self.ma_period, self.rsi_period) + 1
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """
        计算RSI指标
        
        Args:
            prices: 价格序列
            period: RSI周期
        
        Returns:
            RSI值 (0-100)
        """
        if len(prices) < period + 1:
            return 50.0  # 默认中性值
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def generate_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成交易信号
        
        Args:
            market_data: DataFrame包含OHLCV数据
        
        Returns:
            交易信号字典
        """
        required_length = max(self.momentum_period, self.ma_period, self.rsi_period) + 1
        if len(market_data) < required_length:
            return {
                'action': 'hold',
                'reason': f'数据不足，需要至少{required_length}个数据点',
                'confidence': 0.0
            }
        
        df = market_data.copy()
        
        # 计算动量指标
        df['momentum'] = df['Close'].pct_change(self.momentum_period) * 100  # 百分比形式
        df['ma'] = df['Close'].rolling(window=self.ma_period).mean()
        df['volume_ma'] = df['Volume'].rolling(window=5).mean()
        
        # 获取最新数据
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        current_price = latest['Close']
        momentum = latest['momentum']
        ma = latest['ma']
        volume_ratio = latest['Volume'] / latest['volume_ma'] if latest['volume_ma'] > 0 else 1.0
        
        # 计算RSI
        rsi = self._calculate_rsi(df['Close'], self.rsi_period) if self.use_rsi_filter else 50.0
        
        # 无持仓时的买入信号
        if self.position == 0:
            # 动量买入条件
            momentum_strong = momentum > self.entry_threshold
            price_above_ma = current_price > ma
            volume_confirmed = volume_ratio >= self.volume_multiplier
            rsi_ok = not self.use_rsi_filter or (rsi > self.rsi_oversold and rsi < self.rsi_overbought)
            
            # 放宽买入条件：动量强或价格在均线上方+成交量确认
            if (momentum_strong or price_above_ma) and volume_confirmed and rsi_ok:
                confidence = 0.7
                if momentum_strong and price_above_ma and volume_confirmed:
                    confidence = 0.85
                
                reasons = []
                if momentum_strong:
                    reasons.append(f'{self.momentum_period}日动量{momentum:.2f}%')
                if price_above_ma:
                    reasons.append(f'价格在MA{self.ma_period}上方')
                if volume_confirmed:
                    reasons.append(f'成交量{volume_ratio:.2f}倍')
                
                return {
                    'action': 'buy',
                    'reason': f'动量买入：{", ".join(reasons)}',
                    'confidence': confidence,
                    'price': current_price,
                    'indicators': {
                        'momentum': momentum,
                        'ma': ma,
                        'volume_ratio': volume_ratio,
                        'rsi': rsi
                    }
                }
            
            else:
                # 说明为什么不买
                reasons = []
                if not momentum_strong and not price_above_ma:
                    reasons.append(f'动量{momentum:.2f}% < {self.entry_threshold}%')
                    reasons.append(f'价格${current_price:.2f} < MA${ma:.2f}')
                if not volume_confirmed:
                    reasons.append(f'成交量{volume_ratio:.2f}倍 < {self.volume_multiplier}倍')
                if not rsi_ok:
                    reasons.append(f'RSI{rsi:.1f}超买')
                
                return {
                    'action': 'hold',
                    'reason': f'等待机会：{"; ".join(reasons)}',
                    'confidence': 0.0,
                    'indicators': {
                        'momentum': momentum,
                        'ma': ma,
                        'volume_ratio': volume_ratio,
                        'rsi': rsi
                    }
                }
        
        # 持仓时的卖出信号
        else:
            profit_pct = ((current_price / self.entry_price) - 1) * 100 if self.entry_price > 0 else 0
            
            # 动量转弱卖出
            momentum_weak = momentum < self.exit_threshold
            momentum_declining = momentum < previous['momentum']  # 动量递减
            price_below_ma = current_price < ma
            take_profit = profit_pct > 6.0  # 止盈：盈利超过6%（降低止盈门槛）
            
            # 卖出条件：动量转负 或 价格跌破均线+动量下降 或 止盈
            if momentum_weak or (price_below_ma and momentum_declining) or take_profit:
                reasons = []
                if take_profit:
                    reasons.append(f'止盈{profit_pct:.2f}%')
                if momentum_weak:
                    reasons.append(f'动量转弱{momentum:.2f}%')
                if price_below_ma:
                    reasons.append(f'跌破MA{self.ma_period}')
                if momentum_declining:
                    reasons.append('动量递减')
                
                return {
                    'action': 'sell',
                    'reason': f'动量卖出：{", ".join(reasons)}，盈利{profit_pct:.2f}%',
                    'confidence': 0.8,
                    'price': current_price,
                    'profit_pct': profit_pct,
                    'indicators': {
                        'momentum': momentum,
                        'momentum_change': momentum - previous['momentum'],
                        'ma': ma,
                        'rsi': rsi
                    }
                }
            
            # 持有
            else:
                return {
                    'action': 'hold',
                    'reason': f'动量持有：动量{momentum:.2f}%，盈利{profit_pct:.2f}%',
                    'confidence': 0.6,
                    'price': current_price,
                    'profit_pct': profit_pct,
                    'indicators': {
                        'momentum': momentum,
                        'ma': ma,
                        'rsi': rsi
                    }
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
        elif action == 'sell':
            self.position = 0
            self.entry_price = 0
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """返回策略信息"""
        return {
            'name': 'momentum',
            'description': '动量策略（追涨杀跌，适合趋势行情）',
            'parameters': {
                'momentum_period': self.momentum_period,
                'entry_threshold': f'{self.entry_threshold}%',
                'exit_threshold': f'{self.exit_threshold}%',
                'ma_period': self.ma_period,
                'use_rsi_filter': self.use_rsi_filter
            },
            'risk_level': 'medium-high',
            'suitable_market': ['trending', 'bullish', 'volatile']
        }
