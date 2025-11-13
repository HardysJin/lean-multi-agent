"""
均值回归策略 (Mean Reversion Strategy)
基于价格回归均值的理论：价格偏离均值过多时会回归
适合震荡行情，在超买时卖出、超卖时买入
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from backend.strategies.base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略
    
    策略逻辑：
    - 计算价格的移动平均线作为均值
    - 使用布林带或RSI识别超买/超卖
    - 价格偏离均值过多（超卖）时买入
    - 价格回归均值或超买时卖出
    
    参数：
    - ma_period: 均值周期（移动平均线）
    - std_multiplier: 布林带标准差倍数
    - rsi_period: RSI周期
    - rsi_oversold: RSI超卖线
    - rsi_overbought: RSI超买线
    - use_bollinger: 是否使用布林带
    - use_rsi: 是否使用RSI
    """
    
    def __init__(
        self,
        ma_period: int = 20,
        std_multiplier: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        use_bollinger: bool = True,
        use_rsi: bool = True,
        profit_target: float = 3.0,    # 止盈目标（%）
        stop_loss: float = 5.0          # 止损线（%）
    ):
        """
        初始化均值回归策略
        
        Args:
            ma_period: 移动平均线周期
            std_multiplier: 布林带标准差倍数
            rsi_period: RSI周期
            rsi_oversold: RSI超卖阈值
            rsi_overbought: RSI超买阈值
            use_bollinger: 是否使用布林带
            use_rsi: 是否使用RSI
            profit_target: 止盈目标（%）
            stop_loss: 止损线（%）
        """
        super().__init__(
            ma_period=ma_period,
            std_multiplier=std_multiplier,
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            use_bollinger=use_bollinger,
            use_rsi=use_rsi,
            profit_target=profit_target,
            stop_loss=stop_loss
        )
        self.ma_period = ma_period
        self.std_multiplier = std_multiplier
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.use_bollinger = use_bollinger
        self.use_rsi = use_rsi
        self.profit_target = profit_target
        self.stop_loss = stop_loss
    
    def get_required_data_points(self) -> int:
        """返回策略需要的最小数据点数"""
        return max(self.ma_period, self.rsi_period) + 1
    
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
        
        # 避免除零
        loss = loss.replace(0, 0.0001)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def generate_signals(self, market_data: pd.DataFrame, **context) -> Dict[str, Any]:
        """
        生成交易信号
        
        Args:
            market_data: DataFrame包含OHLCV数据
            **context: 上下文信息（如持仓状态、入场价格等）
        
        Returns:
            交易信号字典
        """
        required_length = max(self.ma_period, self.rsi_period) + 1
        if len(market_data) < required_length:
            return {
                'action': 'hold',
                'reason': f'数据不足，需要至少{required_length}个数据点',
                'confidence': 0.0
            }
        
        df = market_data.copy()
        
        # 计算均值回归指标
        df['ma'] = df['Close'].rolling(window=self.ma_period).mean()
        df['std'] = df['Close'].rolling(window=self.ma_period).std()
        
        # 布林带
        df['bb_upper'] = df['ma'] + (df['std'] * self.std_multiplier)
        df['bb_lower'] = df['ma'] - (df['std'] * self.std_multiplier)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['ma'] * 100  # 布林带宽度（%）
        
        # 价格相对位置（0-1之间，0.5为中位）
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 获取最新数据
        latest = df.iloc[-1]
        current_price = latest['Close']
        ma = latest['ma']
        bb_upper = latest['bb_upper']
        bb_lower = latest['bb_lower']
        bb_position = latest['bb_position']
        bb_width = latest['bb_width']
        
        # 计算RSI
        rsi = self._calculate_rsi(df['Close'], self.rsi_period) if self.use_rsi else 50.0
        
        # 价格偏离度
        deviation_pct = ((current_price / ma) - 1) * 100 if ma > 0 else 0
        
        # 从上下文获取当前持仓状态
        current_position = context.get('position', 0)
        entry_price = context.get('entry_price', 0)
        
        # 无持仓时的买入信号
        if current_position == 0:
            # 超卖信号检测
            bollinger_oversold = self.use_bollinger and (current_price < bb_lower or bb_position < 0.2)
            rsi_oversold = self.use_rsi and (rsi < self.rsi_oversold)
            
            # 买入条件：布林带下轨或RSI超卖
            if bollinger_oversold or rsi_oversold:
                confidence = 0.6
                reasons = []
                
                if bollinger_oversold:
                    reasons.append(f'价格触及布林下轨(${bb_lower:.2f})')
                    confidence += 0.1
                if rsi_oversold:
                    reasons.append(f'RSI超卖({rsi:.1f})')
                    confidence += 0.1
                if deviation_pct < -3:
                    reasons.append(f'偏离均值{deviation_pct:.1f}%')
                    confidence += 0.05
                
                return {
                    'action': 'buy',
                    'reason': f'均值回归买入：{", ".join(reasons)}',
                    'confidence': min(confidence, 0.9),
                    'price': current_price,
                    'indicators': {
                        'ma': ma,
                        'bb_upper': bb_upper,
                        'bb_lower': bb_lower,
                        'bb_position': bb_position,
                        'rsi': rsi,
                        'deviation_pct': deviation_pct
                    }
                }
            
            else:
                # 说明为什么不买
                return {
                    'action': 'hold',
                    'reason': f'等待超卖：RSI{rsi:.1f}，布林位置{bb_position:.2f}，偏离{deviation_pct:.1f}%',
                    'confidence': 0.0,
                    'indicators': {
                        'ma': ma,
                        'bb_position': bb_position,
                        'rsi': rsi,
                        'deviation_pct': deviation_pct
                    }
                }
        
        # 持仓时的卖出信号
        else:
            profit_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
            
            # 卖出条件
            profit_target_reached = profit_pct >= self.profit_target
            stop_loss_triggered = profit_pct <= -self.stop_loss
            
            # 回归均值卖出
            mean_reversion_sell = current_price > ma and deviation_pct > 1  # 回归到均值上方
            
            # 超买卖出
            bollinger_overbought = self.use_bollinger and (current_price > bb_upper or bb_position > 0.8)
            rsi_overbought = self.use_rsi and (rsi > self.rsi_overbought)
            
            if stop_loss_triggered:
                return {
                    'action': 'sell',
                    'reason': f'止损：亏损{profit_pct:.2f}%',
                    'confidence': 0.95,
                    'price': current_price,
                    'profit_pct': profit_pct,
                    'indicators': {
                        'rsi': rsi,
                        'bb_position': bb_position
                    }
                }
            
            elif profit_target_reached:
                return {
                    'action': 'sell',
                    'reason': f'止盈：盈利{profit_pct:.2f}%达到目标{self.profit_target}%',
                    'confidence': 0.9,
                    'price': current_price,
                    'profit_pct': profit_pct,
                    'indicators': {
                        'rsi': rsi,
                        'bb_position': bb_position
                    }
                }
            
            elif mean_reversion_sell or bollinger_overbought or rsi_overbought:
                reasons = []
                confidence = 0.7
                
                if mean_reversion_sell:
                    reasons.append(f'回归均值(${ma:.2f})')
                    confidence += 0.05
                if bollinger_overbought:
                    reasons.append(f'触及布林上轨(${bb_upper:.2f})')
                    confidence += 0.05
                if rsi_overbought:
                    reasons.append(f'RSI超买({rsi:.1f})')
                    confidence += 0.05
                
                return {
                    'action': 'sell',
                    'reason': f'均值回归卖出：{", ".join(reasons)}，盈利{profit_pct:.2f}%',
                    'confidence': min(confidence, 0.9),
                    'price': current_price,
                    'profit_pct': profit_pct,
                    'indicators': {
                        'ma': ma,
                        'bb_position': bb_position,
                        'rsi': rsi,
                        'deviation_pct': deviation_pct
                    }
                }
            
            # 持有
            else:
                return {
                    'action': 'hold',
                    'reason': f'持有中：盈利{profit_pct:.2f}%，等待卖出信号（RSI{rsi:.1f}，布林{bb_position:.2f}）',
                    'confidence': 0.5,
                    'price': current_price,
                    'profit_pct': profit_pct,
                    'indicators': {
                        'ma': ma,
                        'bb_position': bb_position,
                        'rsi': rsi,
                        'deviation_pct': deviation_pct
                    }
                }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """返回策略信息"""
        indicators_used = []
        if self.use_bollinger:
            indicators_used.append(f'布林带({self.ma_period}期,{self.std_multiplier}σ)')
        if self.use_rsi:
            indicators_used.append(f'RSI({self.rsi_period}期)')
        
        info = super().get_strategy_info()
        info.update({
            'name': 'mean_reversion',
            'description': '均值回归策略（超卖买入，回归卖出）',
            'parameters': {
                'ma_period': self.ma_period,
                'indicators': ', '.join(indicators_used),
                'profit_target': f'{self.profit_target}%',
                'stop_loss': f'{self.stop_loss}%'
            },
            'risk_level': 'medium',
            'suitable_market': ['sideways', 'range-bound', 'volatile']
        })
        return info
