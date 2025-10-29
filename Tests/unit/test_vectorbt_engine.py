"""
VectorBT 回测引擎单元测试

测试 VectorBTBacktest 类的各个核心功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

from Backtests.vectorbt_engine import VectorBTBacktest


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
            symbols=['AAPL', 'MSFT', 'GOOGL'],
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
            fees=0.002
        )
        
        assert backtest.symbols == ['AAPL']
        assert backtest.start_date == '2024-01-01'
        assert backtest.end_date == '2024-01-31'
        assert backtest.initial_cash == 50000
        assert backtest.fees == 0.002
        assert backtest._price_data is None
        assert backtest._portfolios is None
    
    def test_load_data_single_symbol(self, basic_backtest):
        """测试加载单个股票数据"""
        basic_backtest.load_data()
        
        assert basic_backtest._price_data is not None
        assert 'AAPL' in basic_backtest._price_data
        
        df = basic_backtest._price_data['AAPL']
        assert isinstance(df, pd.DataFrame)
        assert 'Close' in df.columns
        assert len(df) > 0
        assert df.index[0].date() >= pd.to_datetime('2024-01-01').date()
        assert df.index[-1].date() <= pd.to_datetime('2024-01-31').date()
    
    def test_load_data_multiple_symbols(self, multi_symbol_backtest):
        """测试加载多个股票数据"""
        multi_symbol_backtest.load_data()
        
        assert multi_symbol_backtest._price_data is not None
        assert len(multi_symbol_backtest._price_data) == 3
        
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            assert symbol in multi_symbol_backtest._price_data
            df = multi_symbol_backtest._price_data[symbol]
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    @pytest.mark.asyncio
    async def test_precompute_signals_without_meta_agent(self, basic_backtest):
        """测试预计算信号（不使用 MetaAgent）"""
        basic_backtest.load_data()
        
        # 使用简单移动平均策略
        signals = await basic_backtest.precompute_signals(use_meta_agent=False)
        
        assert signals is not None
        assert 'AAPL' in signals
        assert isinstance(signals['AAPL'], pd.Series)
        assert len(signals['AAPL']) == len(basic_backtest._price_data['AAPL'])
        
        # 信号应该是 0 或 1
        assert all(s in [0, 1] for s in signals['AAPL'].values)
    
    @pytest.mark.asyncio
    async def test_run_backtest(self, basic_backtest):
        """测试运行回测"""
        basic_backtest.load_data()
        signals = await basic_backtest.precompute_signals(use_meta_agent=False)
        
        # 运行回测
        basic_backtest.run_backtest(signals)
        
        assert basic_backtest._portfolios is not None
        assert 'AAPL' in basic_backtest._portfolios
        
        # 检查 Portfolio 对象
        portfolio = basic_backtest._portfolios['AAPL']
        assert portfolio is not None
        assert hasattr(portfolio, 'value')
        assert hasattr(portfolio, 'returns')
    
    @pytest.mark.asyncio
    async def test_get_performance_stats(self, basic_backtest):
        """测试获取性能统计"""
        basic_backtest.load_data()
        signals = await basic_backtest.precompute_signals(use_meta_agent=False)
        basic_backtest.run_backtest(signals)
        
        stats = basic_backtest.get_performance_stats('AAPL')
        
        # 检查所有必需的字段
        assert 'symbol' in stats
        assert 'start_date' in stats
        assert 'end_date' in stats
        assert 'initial_cash' in stats
        assert 'final_value' in stats
        assert 'total_return' in stats
        assert 'total_return_pct' in stats
        assert 'total_trades' in stats
        assert 'sharpe_ratio' in stats
        
        # 检查数据类型
        assert stats['symbol'] == 'AAPL'
        assert isinstance(stats['initial_cash'], (int, float))
        assert isinstance(stats['final_value'], (int, float))
        assert isinstance(stats['total_trades'], int)
    
    @pytest.mark.asyncio
    async def test_generate_report(self, basic_backtest, tmp_path):
        """测试生成报告"""
        basic_backtest.load_data()
        signals = await basic_backtest.precompute_signals(use_meta_agent=False)
        basic_backtest.run_backtest(signals)
        
        # 使用临时目录
        output_dir = str(tmp_path / "test_results")
        reports = basic_backtest.generate_report(output_dir=output_dir)
        
        assert 'AAPL' in reports
        assert 'summary' in reports
        
        # 检查文件是否存在
        assert reports['AAPL'] is None or os.path.exists(reports['AAPL'])
        assert os.path.exists(reports['summary'])
        
        # 检查 JSON 内容
        import json
        with open(reports['summary'], 'r') as f:
            summary = json.load(f)
        
        assert 'timestamp' in summary
        assert 'config' in summary
        assert 'results' in summary
        assert 'AAPL' in summary['results']
    
    def test_custom_signals(self, basic_backtest):
        """测试使用自定义信号"""
        basic_backtest.load_data()
        
        df = basic_backtest._price_data['AAPL']
        
        # 创建自定义信号：前半部分买入，后半部分不买入
        n = len(df)
        custom_signals = pd.Series([1] * (n // 2) + [0] * (n - n // 2), index=df.index)
        
        signals = {'AAPL': custom_signals}
        basic_backtest.run_backtest(signals)
        
        stats = basic_backtest.get_performance_stats('AAPL')
        assert stats['total_trades'] >= 0
    
    def test_invalid_symbol(self):
        """测试无效的股票代码"""
        backtest = VectorBTBacktest(
            symbols=['INVALID_SYMBOL_XYZ'],
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        backtest.load_data()
        
        # 应该没有数据或者数据为空
        assert backtest._price_data is None or \
               'INVALID_SYMBOL_XYZ' not in backtest._price_data or \
               len(backtest._price_data.get('INVALID_SYMBOL_XYZ', [])) == 0
    
    @pytest.mark.slow
    def test_date_range_validation(self):
        """测试日期范围验证"""
        # 未来日期应该返回空数据或者抛出错误
        backtest = VectorBTBacktest(
            symbols=['AAPL'],
            start_date='2030-01-01',
            end_date='2030-12-31'
        )
        
        backtest.load_data()
        
        # 数据应该为空或 None
        assert backtest._price_data is None or \
               'AAPL' not in backtest._price_data or \
               len(backtest._price_data.get('AAPL', [])) == 0
    
    @pytest.mark.asyncio
    async def test_zero_fees(self):
        """测试零手续费"""
        backtest = VectorBTBacktest(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            fees=0.0
        )
        
        assert backtest.fees == 0.0
        
        backtest.load_data()
        signals = await backtest.precompute_signals(use_meta_agent=False)
        backtest.run_backtest(signals)
        
        stats = backtest.get_performance_stats('AAPL')
        assert stats is not None


class TestQuickBacktest:
    """测试 quick_backtest 便捷函数"""
    
    @pytest.mark.asyncio
    async def test_quick_backtest_basic(self):
        """测试基础的快速回测"""
        from Backtests.vectorbt_engine import quick_backtest
        
        backtest = await quick_backtest(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            use_meta_agent=False
        )
        
        assert backtest is not None
        assert backtest._portfolios is not None
        assert 'AAPL' in backtest._portfolios
    
    @pytest.mark.asyncio
    async def test_quick_backtest_multiple_symbols(self):
        """测试多股票快速回测"""
        from Backtests.vectorbt_engine import quick_backtest
        
        backtest = await quick_backtest(
            symbols=['AAPL', 'MSFT'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            initial_cash=200000,
            use_meta_agent=False
        )
        
        assert backtest is not None
        assert len(backtest._portfolios) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
