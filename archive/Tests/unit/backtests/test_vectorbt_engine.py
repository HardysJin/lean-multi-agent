"""
VectorBT 回测引擎单元测试

测试增强版 VectorBTBacktest 类的核心功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

from Backtests.vectorbt_engine import (
    VectorBTBacktest,
    Signal,
    convert_layered_signals
)


class TestSignal:
    """测试 Signal 类"""
    
    def test_signal_creation(self):
        """测试信号创建"""
        signal = Signal(
            action='BUY',
            size=0.2,
            confidence=0.8,
            reason='Strong uptrend'
        )
        
        assert signal.action == 'BUY'
        assert signal.size == 0.2
        assert signal.confidence == 0.8
        assert signal.reason == 'Strong uptrend'
    
    def test_signal_size_limits(self):
        """测试仓位大小限制"""
        # 超过1.0应该被限制
        signal1 = Signal('BUY', size=1.5)
        assert signal1.size == 1.0
        
        # 小于0应该被限制
        signal2 = Signal('BUY', size=-0.1)
        assert signal2.size == 0.0
    
    def test_from_layered_strategy_signal(self):
        """测试从 LayeredStrategy 信号转换"""
        # BUY with high confidence
        signal_dict = {
            'action': 'BUY',
            'confidence': 0.9,
            'reason': 'Test'
        }
        signal = Signal.from_layered_strategy_signal(signal_dict)
        
        assert signal.action == 'BUY'
        assert signal.confidence == 0.9
        # confidence 0.9 -> size = 0.1 + (0.9-0.5)*0.4 = 0.26
        assert abs(signal.size - 0.26) < 0.01
    
    def test_hold_signal_has_zero_size(self):
        """测试 HOLD 信号的仓位为0"""
        signal_dict = {
            'action': 'HOLD',
            'confidence': 0.7
        }
        signal = Signal.from_layered_strategy_signal(signal_dict)
        
        assert signal.action == 'HOLD'
        assert signal.size == 0.0
    
    def test_sell_signal_has_full_size(self):
        """测试 SELL 信号的仓位为1"""
        signal_dict = {
            'action': 'SELL',
            'confidence': 0.8
        }
        signal = Signal.from_layered_strategy_signal(signal_dict)
        
        assert signal.action == 'SELL'
        assert signal.size == 1.0


class TestVectorBTBacktest:
    """VectorBTBacktest 类的单元测试"""
    
    @pytest.fixture
    def basic_backtest(self):
        """创建基础的回测实例"""
        return VectorBTBacktest(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_cash=100000,
            fees=0.001
        )
    
    @pytest.fixture
    def multi_symbol_backtest(self):
        """创建多股票的回测实例"""
        return VectorBTBacktest(
            symbols=['AAPL', 'MSFT'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_cash=100000
        )
    
    def test_initialization(self):
        """测试初始化"""
        backtest = VectorBTBacktest(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_cash=50000,
            fees=0.002,
            max_position_size=0.5
        )
        
        assert backtest.symbols == ['AAPL']
        assert backtest.start_date == '2024-01-01'
        assert backtest.end_date == '2024-01-31'
        assert backtest.initial_cash == 50000
        assert backtest.fees == 0.002
        assert backtest.max_position_size == 0.5
        assert isinstance(backtest._price_data, dict)
        assert isinstance(backtest._portfolios, dict)
    
    def test_load_data_single_symbol(self, basic_backtest):
        """测试加载单个股票数据"""
        basic_backtest.load_data()
        
        assert basic_backtest._price_data is not None
        assert 'AAPL' in basic_backtest._price_data
        
        df = basic_backtest._price_data['AAPL']
        assert isinstance(df, pd.DataFrame)
        assert 'Close' in df.columns
        assert len(df) > 0
    
    def test_load_data_multiple_symbols(self, multi_symbol_backtest):
        """测试加载多个股票数据"""
        multi_symbol_backtest.load_data()
        
        assert multi_symbol_backtest._price_data is not None
        assert len(multi_symbol_backtest._price_data) >= 1  # 至少有一个成功
        
        for symbol in multi_symbol_backtest._price_data.keys():
            df = multi_symbol_backtest._price_data[symbol]
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_run_backtest_with_sizing(self, basic_backtest):
        """测试运行带仓位大小的回测"""
        basic_backtest.load_data()
        
        df = basic_backtest._price_data['AAPL']
        
        # 创建测试信号
        signals = []
        for i in range(len(df)):
            if i % 5 == 0:
                signals.append(Signal('BUY', size=0.15, confidence=0.7))
            elif i % 7 == 0:
                signals.append(Signal('SELL', size=1.0, confidence=0.8))
            else:
                signals.append(Signal('HOLD', size=0.0, confidence=0.5))
        
        signal_dict = {'AAPL': signals}
        
        # 运行回测
        portfolios = basic_backtest.run_backtest_with_sizing(signal_dict)
        
        assert portfolios is not None
        assert 'AAPL' in portfolios
        assert portfolios['AAPL'] is not None
    
    def test_convert_signals_to_vectorbt(self, basic_backtest):
        """测试信号转换"""
        signals = [
            Signal('BUY', size=0.1, confidence=0.5),
            Signal('HOLD', size=0.0, confidence=0.5),
            Signal('SELL', size=1.0, confidence=0.8),
        ]
        
        entries, exits, sizes = basic_backtest._convert_signals_to_vectorbt(signals, 3)
        
        assert len(entries) == 3
        assert len(exits) == 3
        assert len(sizes) == 3
        
        # 第一个是BUY
        assert entries.iloc[0] == True
        assert exits.iloc[0] == False
        assert sizes.iloc[0] == 10.0  # 0.1 * 100
        
        # 第二个是HOLD
        assert entries.iloc[1] == False
        assert exits.iloc[1] == False
        assert sizes.iloc[1] == 0.0
        
        # 第三个是SELL
        assert entries.iloc[2] == False
        assert exits.iloc[2] == True
    
    def test_get_portfolio(self, basic_backtest):
        """测试获取 portfolio"""
        basic_backtest.load_data()
        
        df = basic_backtest._price_data['AAPL']
        signals = [Signal('BUY', 0.1, 0.5) for _ in range(len(df))]
        
        basic_backtest.run_backtest_with_sizing({'AAPL': signals})
        
        portfolio = basic_backtest.get_portfolio('AAPL')
        assert portfolio is not None
    
    def test_invalid_symbol(self):
        """测试无效的股票代码"""
        backtest = VectorBTBacktest(
            symbols=['INVALID_SYMBOL_XYZ'],
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        backtest.load_data()
        
        # 应该没有数据
        assert 'INVALID_SYMBOL_XYZ' not in backtest._price_data or \
               len(backtest._price_data.get('INVALID_SYMBOL_XYZ', [])) == 0


class TestConversionFunctions:
    """测试转换函数"""
    
    def test_convert_layered_signals(self):
        """测试 LayeredStrategy 信号转换"""
        layered_signals = {
            'AAPL': [
                {'action': 'BUY', 'confidence': 0.8, 'reason': 'Test1'},
                {'action': 'HOLD', 'confidence': 0.5, 'reason': 'Test2'},
                {'action': 'SELL', 'confidence': 0.9, 'reason': 'Test3'},
            ]
        }
        
        enhanced = convert_layered_signals(layered_signals)
        
        assert 'AAPL' in enhanced
        assert len(enhanced['AAPL']) == 3
        
        # 检查第一个信号
        signal1 = enhanced['AAPL'][0]
        assert isinstance(signal1, Signal)
        assert signal1.action == 'BUY'
        assert signal1.confidence == 0.8
        
        # 检查第二个信号
        signal2 = enhanced['AAPL'][1]
        assert signal2.action == 'HOLD'
        assert signal2.size == 0.0
        
        # 检查第三个信号
        signal3 = enhanced['AAPL'][2]
        assert signal3.action == 'SELL'
        assert signal3.size == 1.0


class TestIntegration:
    """集成测试"""
    
    def test_full_backtest_flow(self):
        """测试完整的回测流程"""
        # 1. 创建回测实例
        backtest = VectorBTBacktest(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_cash=100000,
            fees=0.001
        )
        
        # 2. 加载数据
        backtest.load_data()
        assert 'AAPL' in backtest._price_data
        
        # 3. 创建信号
        df = backtest._price_data['AAPL']
        layered_signals = {
            'AAPL': [
                {'action': 'BUY' if i % 3 == 0 else 'HOLD', 
                 'confidence': 0.6 + (i % 5) * 0.05,
                 'reason': f'Signal {i}'}
                for i in range(len(df))
            ]
        }
        
        # 4. 转换信号
        enhanced_signals = convert_layered_signals(layered_signals)
        
        # 5. 运行回测
        portfolios = backtest.run_backtest_with_sizing(enhanced_signals)
        
        # 6. 验证结果
        assert 'AAPL' in portfolios
        portfolio = portfolios['AAPL']
        assert portfolio is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
