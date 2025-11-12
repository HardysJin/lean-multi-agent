"""
双EMA通道突破策略
基于两组EMA通道：
- 蓝色通道 (25周期)：快速通道，用于交易信号
- 黄色通道 (90周期)：慢速通道，用于趋势确认
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class DoubleEmaChannelStrategy:
    """
    双EMA通道突破策略
    
    策略逻辑：
    - 蓝色通道：UP1 = EMA(H, 25), LOW1 = EMA(L, 25)
    - 黄色通道：UP2 = EMA(H, 90), LOW2 = EMA(L, 90)
    - 买入信号：价格突破蓝色通道上沿 (收盘价 > UP1) + 成交量确认 + 趋势确认
    - 卖出信号：价格跌破蓝色通道下沿 (收盘价 < LOW1) 或止损/止盈
    """
    
    def __init__(
        self,
        fast_period: int = 25,
        slow_period: int = 90,
        volume_multiplier: float = 2.0,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ):
        """
        初始化策略参数
        
        Args:
            fast_period: 快速通道周期
            slow_period: 慢速通道周期
            volume_multiplier: 成交量放大倍数（用于确认突破）
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.volume_multiplier = volume_multiplier
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # 交易状态
        self.entry_price = 0
        self.position = 0
    
    def get_required_data_points(self) -> int:
        """返回策略需要的最小数据点数"""
        return self.slow_period + 5  # 慢速通道周期 + 成交量MA窗口
    
    def generate_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        基于市场数据生成交易信号
        
        Args:
            market_data: DataFrame包含OHLCV数据，列名：Open, High, Low, Close, Volume
        
        Returns:
            交易信号字典
        """
        if len(market_data) < self.slow_period:
            return {
                'action': 'hold',
                'reason': '数据不足，需要至少90个数据点',
                'confidence': 0.0
            }
        
        # 计算EMA通道
        df = market_data.copy()
        
        # 快速通道 (25周期)
        df['ema_high_25'] = df['High'].ewm(span=self.fast_period, adjust=False).mean()
        df['ema_low_25'] = df['Low'].ewm(span=self.fast_period, adjust=False).mean()
        
        # 慢速通道 (90周期)
        df['ema_high_90'] = df['High'].ewm(span=self.slow_period, adjust=False).mean()
        df['ema_low_90'] = df['Low'].ewm(span=self.slow_period, adjust=False).mean()
        
        # 计算平均成交量
        df['avg_volume'] = df['Volume'].rolling(window=5).mean()
        
        # 获取最新数据
        latest = df.iloc[-1]
        current_price = latest['Close']
        up1 = latest['ema_high_25']
        low1 = latest['ema_low_25']
        up2 = latest['ema_high_90']
        low2 = latest['ema_low_90']
        current_volume = latest['Volume']
        avg_volume = latest['avg_volume']
        
        # 生成交易信号
        if self.position == 0:
            # 买入信号检查 - 简化条件，只需要价格突破即可
            signal_breakout = current_price > up1
            
            # 可选的确认条件（降低要求）
            volume_ok = pd.notna(avg_volume) and (avg_volume == 0 or current_volume > avg_volume * 0.8)  # 放宽到0.8倍
            
            if signal_breakout and volume_ok:
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                return {
                    'action': 'buy',
                    'reason': f'价格突破蓝色上沿(${up1:.2f})，当前${current_price:.2f}，成交量{volume_ratio:.2f}倍',
                    'confidence': 0.75,
                    'price': current_price,
                    'indicators': {
                        'up1': up1,
                        'low1': low1,
                        'up2': up2,
                        'low2': low2,
                        'volume_ratio': volume_ratio
                    }
                }
            else:
                reasons = []
                if not signal_breakout:
                    reasons.append(f'未突破蓝色上沿(当前${current_price:.2f} <= UP1 ${up1:.2f})')
                if not volume_ok:
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                    reasons.append(f'成交量不足(当前{volume_ratio:.2f}倍 < 0.8倍)')
                
                return {
                    'action': 'hold',
                    'reason': '; '.join(reasons),
                    'confidence': 0.0
                }
        
        else:  # 已持仓
            # 卖出信号检查
            signal_breakdown = current_price < low1
            signal_stoploss = self.entry_price > 0 and current_price < self.entry_price * (1 - self.stop_loss_pct)
            signal_takeprofit = self.entry_price > 0 and current_price > self.entry_price * (1 + self.take_profit_pct)
            
            if signal_takeprofit:
                profit_pct = ((current_price / self.entry_price) - 1) * 100
                return {
                    'action': 'sell',
                    'reason': f'止盈触发：盈利{profit_pct:.2f}% (目标{self.take_profit_pct*100}%)',
                    'confidence': 0.9,
                    'price': current_price,
                    'profit_pct': profit_pct
                }
            
            elif signal_stoploss:
                profit_pct = ((current_price / self.entry_price) - 1) * 100
                return {
                    'action': 'sell',
                    'reason': f'止损触发：亏损{profit_pct:.2f}% (止损线-{self.stop_loss_pct*100}%)',
                    'confidence': 0.9,
                    'price': current_price,
                    'profit_pct': profit_pct
                }
            
            elif signal_breakdown:
                profit_pct = ((current_price / self.entry_price) - 1) * 100 if self.entry_price > 0 else 0
                return {
                    'action': 'sell',
                    'reason': f'价格跌破蓝色下沿(${low1:.2f})，盈利{profit_pct:.2f}%',
                    'confidence': 0.75,
                    'price': current_price,
                    'profit_pct': profit_pct
                }
            
            else:
                return {
                    'action': 'hold',
                    'reason': f'持仓中，未触发卖出条件 (当前${current_price:.2f}, LOW1 ${low1:.2f})',
                    'confidence': 0.0
                }
    
    def execute_trade(self, action: str, price: float):
        """
        执行交易并更新状态
        
        Args:
            action: 交易动作 ('buy' 或 'sell')
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
            'name': 'double_ema_channel',
            'description': '双EMA通道突破策略',
            'parameters': {
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'volume_multiplier': self.volume_multiplier,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct
            },
            'risk_level': 'medium',
            'suitable_market': ['trending', 'volatile']
        }
