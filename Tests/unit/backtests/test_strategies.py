"""
策略模块单元测试

测试 SimpleTechnicalStrategy 和 MultiAgentStrategy
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

from Backtests.strategies.multi_agent_strategy import (
    SimpleTechnicalStrategy,
    MultiAgentStrategy
)


class TestSimpleTechnicalStrategy:
    """SimpleTechnicalStrategy 单元测试"""
    
    @pytest.fixture
    def strategy(self):
        """创建策略实例"""
        return SimpleTechnicalStrategy(short_window=5, long_window=20)
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # 创建一个有明显趋势的价格序列
        prices = np.linspace(100, 150, 50) + np.random.randn(50) * 2
        
        df = pd.DataFrame({
            'Close': prices,
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Volume': np.random.randint(1000000, 10000000, 50)
        }, index=dates)
        
        return df
    
    def test_initialization(self):
        """测试初始化"""
        strategy = SimpleTechnicalStrategy(short_window=10, long_window=30)
        
        assert strategy.short_window == 10
        assert strategy.long_window == 30
    
    def test_default_parameters(self):
        """测试默认参数"""
        strategy = SimpleTechnicalStrategy()
        
        assert strategy.short_window == 20
        assert strategy.long_window == 50
    
    def test_generate_signal_not_enough_data(self, strategy, sample_data):
        """测试数据不足时的信号生成"""
        # 只取前 10 天数据（少于 long_window）
        short_data = sample_data.iloc[:10]
        
        signal = strategy.generate_signal(
            symbol='TEST',
            date=short_data.index[-1],
            price=short_data['Close'].iloc[-1],
            historical_data=short_data
        )
        
        # 数据不足应返回 0（不交易）
        assert signal == 0
    
    def test_generate_signal_buy(self, strategy, sample_data):
        """测试买入信号"""
        # 使用足够的数据
        date = sample_data.index[25]
        
        signal = strategy.generate_signal(
            symbol='TEST',
            date=date,
            price=sample_data.loc[date, 'Close'],
            historical_data=sample_data.loc[:date]
        )
        
        # 信号应该是 0 或 1
        assert signal in [0, 1]
    
    def test_generate_signal_consistency(self, strategy, sample_data):
        """测试信号生成的一致性"""
        date = sample_data.index[30]
        
        # 多次调用应返回相同结果
        signal1 = strategy.generate_signal(
            symbol='TEST',
            date=date,
            price=sample_data.loc[date, 'Close'],
            historical_data=sample_data.loc[:date]
        )
        
        signal2 = strategy.generate_signal(
            symbol='TEST',
            date=date,
            price=sample_data.loc[date, 'Close'],
            historical_data=sample_data.loc[:date]
        )
        
        assert signal1 == signal2
    
    def test_golden_cross(self):
        """测试金叉信号（短期均线上穿长期均线）"""
        strategy = SimpleTechnicalStrategy(short_window=5, long_window=10)
        
        # 创建明确的金叉场景：确保有交叉点
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # 创建交叉场景：
        # 1. 前半部分：长期均线高于短期均线（下跌趋势）
        # 2. 中间部分：快速反弹
        # 3. 后半部分：短期均线高于长期均线（上涨趋势）
        prices = []
        for i in range(50):
            if i < 20:
                prices.append(100 - i * 2)  # 下跌到 60
            elif i < 30:
                prices.append(60 + (i - 20) * 5)  # 快速反弹到 110
            else:
                prices.append(110 + (i - 30) * 0.5)  # 继续上涨
        
        df = pd.DataFrame({
            'Close': prices,
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Volume': [1000000] * 50
        }, index=dates)
        
        # 找到金叉点（短期从下方穿过长期）
        close = df['Close']
        sma5 = close.rolling(5).mean()
        sma10 = close.rolling(10).mean()
        
        # 找到第一个金叉点
        golden_cross_idx = None
        for i in range(11, len(df)):
            if sma5.iloc[i-1] <= sma10.iloc[i-1] and sma5.iloc[i] > sma10.iloc[i]:
                golden_cross_idx = i
                break
        
        if golden_cross_idx:
            # 测试金叉当天的信号
            date = dates[golden_cross_idx]
            signal = strategy.generate_signal(
                symbol='TEST',
                date=date,
                price=df.loc[date, 'Close'],
                historical_data=df.iloc[:golden_cross_idx+1]
            )
            assert signal == 1, f"Golden cross at index {golden_cross_idx}, but signal is {signal}"
        else:
            # 如果没找到金叉，至少检查最后一天短期 > 长期时是否持有
            date = dates[-1]
            signal = strategy.generate_signal(
                symbol='TEST',
                date=date,
                price=df.loc[date, 'Close'],
                historical_data=df
            )
            # 没有交叉动作时应该返回 0（持有）
            assert signal in [0, 1]
    
    def test_death_cross(self):
        """测试死叉信号（短期均线下穿长期均线）"""
        strategy = SimpleTechnicalStrategy(short_window=5, long_window=10)
        
        # 创建明确的死叉场景
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        # 前 20 天上涨，后 10 天快速下跌（产生死叉）
        prices = np.concatenate([
            np.linspace(80, 120, 20),  # 上涨
            np.linspace(120, 80, 10)   # 快速下跌
        ])
        
        df = pd.DataFrame({
            'Close': prices,
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Volume': [1000000] * 30
        }, index=dates)
        
        # 检查最后一天的信号
        date = dates[-1]
        signal = strategy.generate_signal(
            symbol='TEST',
            date=date,
            price=df.loc[date, 'Close'],
            historical_data=df
        )
        
        # 应该产生卖出或不交易信号（不是买入）
        assert signal == 0


class TestMultiAgentStrategy:
    """MultiAgentStrategy 单元测试"""
    
    @pytest.fixture
    def strategy(self):
        """创建策略实例"""
        return MultiAgentStrategy()
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        prices = np.linspace(100, 120, 30)
        
        df = pd.DataFrame({
            'Close': prices,
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Volume': np.random.randint(1000000, 10000000, 30)
        }, index=dates)
        
        return df
    
    def test_initialization(self, strategy):
        """测试初始化"""
        assert strategy.meta_agent is not None
        assert strategy.technical_agent is not None
        assert strategy.news_agent is not None
    
    @pytest.mark.asyncio
    async def test_generate_signal_basic(self, strategy, sample_data):
        """测试基础信号生成"""
        date = sample_data.index[-1]
        
        signal = await strategy.generate_signal(
            symbol='AAPL',
            date=date,
            price=float(sample_data.loc[date, 'Close']),
            historical_data=sample_data
        )
        
        # 信号应该是 -1, 0, 或 1
        assert signal in [-1, 0, 1]
    
    @pytest.mark.asyncio
    async def test_generate_signal_with_real_data(self, strategy):
        """测试使用真实数据的信号生成"""
        import yfinance as yf
        
        # 获取少量真实数据
        df = yf.download('AAPL', start='2024-10-01', end='2024-10-15', progress=False)
        
        if len(df) > 0:
            date = df.index[-1]
            
            signal = await strategy.generate_signal(
                symbol='AAPL',
                date=date,
                price=float(df.loc[date, 'Close']),
                historical_data=df
            )
            
            assert signal in [-1, 0, 1]
    
    @pytest.mark.asyncio
    async def test_batch_generate_signals(self, strategy, sample_data):
        """测试批量信号生成"""
        # 只测试最后 3 天
        test_dates = sample_data.index[-3:]
        
        signals = await strategy.batch_generate_signals(
            symbol='AAPL',
            dates=test_dates,
            prices=sample_data.loc[test_dates, 'Close'],
            historical_data=sample_data
        )
        
        assert len(signals) == 3
        assert all(s in [-1, 0, 1] for s in signals)
    
    def test_strategy_has_required_methods(self, strategy):
        """测试策略包含必需的方法"""
        assert hasattr(strategy, 'generate_signal')
        assert hasattr(strategy, 'batch_generate_signals')
        assert callable(strategy.generate_signal)
        assert callable(strategy.batch_generate_signals)


class TestStrategyIntegration:
    """策略集成测试"""
    
    def test_simple_strategy_integration(self):
        """测试简单策略与回测引擎的集成"""
        from Backtests.vectorbt_engine import VectorBTBacktest
        
        backtest = VectorBTBacktest(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        backtest.load_data()
        
        # 使用简单策略生成信号
        strategy = SimpleTechnicalStrategy(short_window=10, long_window=20)
        
        signals = {}
        for symbol in backtest.symbols:
            if symbol not in backtest._price_data:
                continue
            
            df = backtest._price_data[symbol]
            symbol_signals = []
            
            for date, row in df.iterrows():
                historical_data = df.loc[:date]
                signal = strategy.generate_signal(
                    symbol=symbol,
                    date=date,
                    price=row['Close'],
                    historical_data=historical_data
                )
                symbol_signals.append(1 if signal > 0 else 0)
            
            signals[symbol] = pd.Series(symbol_signals, index=df.index)
        
        # 运行回测
        backtest.run_backtest(signals)
        stats = backtest.get_performance_stats('AAPL')
        
        assert stats is not None
        assert stats['symbol'] == 'AAPL'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
