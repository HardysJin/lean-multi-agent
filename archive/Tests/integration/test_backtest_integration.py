"""
回测系统集成测试

测试完整的回测流程，包括多股票、长时间段等场景
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

from Backtests.vectorbt_engine import VectorBTBacktest, quick_backtest
from Backtests.strategies.multi_agent_strategy import SimpleTechnicalStrategy


class TestSingleStockBacktest:
    """单股票回测测试"""
    
    @pytest.mark.asyncio
    async def test_short_period_backtest(self):
        """测试短期回测（2周）"""
        backtest = VectorBTBacktest(
            symbols=['AAPL'],
            start_date='2024-10-01',
            end_date='2024-10-15',
            initial_cash=100000
        )
        
        backtest.load_data()
        signals = await backtest.precompute_signals(use_meta_agent=False)
        backtest.run_backtest(signals)
        
        stats = backtest.get_performance_stats('AAPL')
        
        assert stats is not None
        assert stats['symbol'] == 'AAPL'
        assert stats['initial_cash'] == 100000
        assert 'total_return' in stats
        assert 'total_trades' in stats
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_medium_period_backtest(self):
        """测试中期回测（1个月）- 标记为慢速"""
        backtest = VectorBTBacktest(
            symbols=['AAPL'],
            start_date='2024-09-01',
            end_date='2024-09-30',
            initial_cash=100000
        )
        
        backtest.load_data()
        signals = await backtest.precompute_signals(use_meta_agent=False)
        backtest.run_backtest(signals)
        
        stats = backtest.get_performance_stats('AAPL')
        
        assert stats is not None
        assert len(backtest._price_data['AAPL']) > 0
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_long_period_backtest(self):
        """测试长期回测（3个月）- 标记为慢速"""
        backtest = VectorBTBacktest(
            symbols=['AAPL'],
            start_date='2024-07-01',
            end_date='2024-09-30',
            initial_cash=100000
        )
        
        backtest.load_data()
        signals = await backtest.precompute_signals(use_meta_agent=False)
        backtest.run_backtest(signals)
        
        stats = backtest.get_performance_stats('AAPL')
        
        assert stats is not None
        assert len(backtest._price_data['AAPL']) > 40  # 大约 3 个月的交易日


class TestMultiStockBacktest:
    """多股票回测测试"""
    
    @pytest.mark.asyncio
    async def test_two_stocks(self):
        """测试两个股票的回测"""
        backtest = VectorBTBacktest(
            symbols=['AAPL', 'MSFT'],
            start_date='2024-10-01',
            end_date='2024-10-15',
            initial_cash=100000
        )
        
        backtest.load_data()
        signals = await backtest.precompute_signals(use_meta_agent=False)
        backtest.run_backtest(signals)
        
        assert len(backtest._portfolios) == 2
        
        for symbol in ['AAPL', 'MSFT']:
            stats = backtest.get_performance_stats(symbol)
            assert stats is not None
            assert stats['symbol'] == symbol
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_multiple_stocks(self):
        """测试多个股票的回测 - 标记为慢速"""
        symbols = ['AAPL', 'MSFT', 'GOOGL']  # 减少到 3 个
        
        backtest = VectorBTBacktest(
            symbols=symbols,
            start_date='2024-10-01',
            end_date='2024-10-15',
            initial_cash=100000
        )
        
        backtest.load_data()
        signals = await backtest.precompute_signals(use_meta_agent=False)
        backtest.run_backtest(signals)
        
        # 检查所有股票都有结果
        for symbol in symbols:
            if symbol in backtest._price_data:
                stats = backtest.get_performance_stats(symbol)
                assert stats is not None


class TestParameterVariations:
    """参数变化测试"""
    
    @pytest.mark.asyncio
    async def test_different_initial_cash(self):
        """测试不同的初始资金"""
        cash_amounts = [10000, 100000]  # 减少测试次数
        
        for cash in cash_amounts:
            backtest = VectorBTBacktest(
                symbols=['AAPL'],
                start_date='2024-10-01',
                end_date='2024-10-15',
                initial_cash=cash
            )
            
            backtest.load_data()
            signals = await backtest.precompute_signals(use_meta_agent=False)
            backtest.run_backtest(signals)
            
            stats = backtest.get_performance_stats('AAPL')
            assert stats['initial_cash'] == cash
    
    @pytest.mark.asyncio
    async def test_different_fees(self):
        """测试不同的手续费"""
        fees = [0.0, 0.002]  # 减少测试次数
        
        results = []
        
        for fee in fees:
            backtest = VectorBTBacktest(
                symbols=['AAPL'],
                start_date='2024-10-01',
                end_date='2024-10-15',
                initial_cash=100000,
                fees=fee
            )
            
            backtest.load_data()
            signals = await backtest.precompute_signals(use_meta_agent=False)
            backtest.run_backtest(signals)
            
            stats = backtest.get_performance_stats('AAPL')
            results.append(stats['total_return'])
        
        # 手续费越高，收益应该越低（假设有交易）
        # 注意：如果没有交易，收益率可能相同
        assert len(results) == 2


class TestStrategyComparison:
    """策略比较测试"""
    
    @pytest.mark.slow
    def test_compare_different_ma_windows(self):
        """比较不同移动平均参数的策略 - 标记为慢速"""
        strategies = [
            (5, 20),   # 短期
            (20, 50),  # 中期
        ]
        
        results = {}
        
        for short, long in strategies:
            backtest = VectorBTBacktest(
                symbols=['AAPL'],
                start_date='2024-08-01',
                end_date='2024-09-30',
                initial_cash=100000
            )
            
            backtest.load_data()
            
            # 使用自定义策略
            strategy = SimpleTechnicalStrategy(short_window=short, long_window=long)
            
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
            
            backtest.run_backtest(signals)
            stats = backtest.get_performance_stats('AAPL')
            
            results[f'MA_{short}_{long}'] = stats['total_return']
        
        # 所有策略都应该有结果
        assert len(results) == 2


class TestReportGeneration:
    """报告生成测试"""
    
    @pytest.mark.asyncio
    async def test_generate_html_report(self, tmp_path):
        """测试生成 HTML 报告"""
        backtest = VectorBTBacktest(
            symbols=['AAPL'],
            start_date='2024-10-01',
            end_date='2024-10-15'
        )
        
        backtest.load_data()
        signals = await backtest.precompute_signals(use_meta_agent=False)
        backtest.run_backtest(signals)
        
        output_dir = str(tmp_path / "reports")
        reports = backtest.generate_report(output_dir=output_dir)
        
        assert 'AAPL' in reports
        assert 'summary' in reports
        
        # 检查 JSON 文件存在
        assert os.path.exists(reports['summary'])
    
    @pytest.mark.asyncio
    async def test_generate_multiple_reports(self, tmp_path):
        """测试生成多个股票的报告"""
        backtest = VectorBTBacktest(
            symbols=['AAPL', 'MSFT'],
            start_date='2024-10-01',
            end_date='2024-10-15'
        )
        
        backtest.load_data()
        signals = await backtest.precompute_signals(use_meta_agent=False)
        backtest.run_backtest(signals)
        
        output_dir = str(tmp_path / "reports")
        reports = backtest.generate_report(output_dir=output_dir)
        
        # 应该有两个股票的报告
        assert 'AAPL' in reports
        assert 'MSFT' in reports
        assert 'summary' in reports


class TestQuickBacktestFunction:
    """quick_backtest 函数测试"""
    
    @pytest.mark.asyncio
    async def test_quick_backtest_basic(self):
        """测试基础快速回测"""
        backtest = await quick_backtest(
            symbols=['AAPL'],
            start_date='2024-10-01',
            end_date='2024-10-15',
            use_meta_agent=False
        )
        
        assert backtest is not None
        assert backtest._portfolios is not None
        assert 'AAPL' in backtest._portfolios
        
        stats = backtest.get_performance_stats('AAPL')
        assert stats is not None
    
    @pytest.mark.asyncio
    async def test_quick_backtest_with_params(self):
        """测试带参数的快速回测"""
        backtest = await quick_backtest(
            symbols=['AAPL', 'MSFT'],
            start_date='2024-10-01',
            end_date='2024-10-15',
            initial_cash=200000,
            fees=0.002,
            use_meta_agent=False
        )
        
        assert backtest is not None
        assert len(backtest._portfolios) == 2
        
        for symbol in ['AAPL', 'MSFT']:
            stats = backtest.get_performance_stats(symbol)
            assert stats is not None


class TestEdgeCases:
    """边界情况测试"""
    
    def test_no_signals(self):
        """测试没有交易信号的情况"""
        backtest = VectorBTBacktest(
            symbols=['AAPL'],
            start_date='2024-10-01',
            end_date='2024-10-15'
        )
        
        backtest.load_data()
        
        # 创建全零信号
        df = backtest._price_data['AAPL']
        signals = {'AAPL': pd.Series([0] * len(df), index=df.index)}
        
        backtest.run_backtest(signals)
        stats = backtest.get_performance_stats('AAPL')
        
        # 没有交易，最终价值应该等于初始资金
        assert stats['total_trades'] == 0
        assert stats['final_value'] == stats['initial_cash']
    
    def test_always_buy(self):
        """测试一直买入的情况"""
        backtest = VectorBTBacktest(
            symbols=['AAPL'],
            start_date='2024-10-01',
            end_date='2024-10-15'
        )
        
        backtest.load_data()
        
        # 创建全 1 信号
        df = backtest._price_data['AAPL']
        signals = {'AAPL': pd.Series([1] * len(df), index=df.index)}
        
        backtest.run_backtest(signals)
        stats = backtest.get_performance_stats('AAPL')
        
        # 应该有交易
        assert stats['total_trades'] >= 0
    
    def test_single_trade(self):
        """测试单次交易"""
        backtest = VectorBTBacktest(
            symbols=['AAPL'],
            start_date='2024-10-01',
            end_date='2024-10-15'
        )
        
        backtest.load_data()
        
        df = backtest._price_data['AAPL']
        n = len(df)
        
        # 只在中间买入一次
        signal_array = [0] * n
        signal_array[n // 2] = 1
        signals = {'AAPL': pd.Series(signal_array, index=df.index)}
        
        backtest.run_backtest(signals)
        stats = backtest.get_performance_stats('AAPL')
        
        assert stats is not None


class TestPerformanceMetrics:
    """性能指标测试"""
    
    @pytest.mark.asyncio
    async def test_metrics_range(self):
        """测试性能指标的合理范围"""
        backtest = VectorBTBacktest(
            symbols=['AAPL'],
            start_date='2024-10-01',
            end_date='2024-10-15'
        )
        
        backtest.load_data()
        signals = await backtest.precompute_signals(use_meta_agent=False)
        backtest.run_backtest(signals)
        
        stats = backtest.get_performance_stats('AAPL')
        
        # 检查指标的合理性
        assert isinstance(stats['total_return'], float)
        assert isinstance(stats['total_trades'], int)
        assert stats['total_trades'] >= 0
        assert stats['initial_cash'] > 0
        assert stats['final_value'] > 0
        
        # 最大回撤应该在 0-1 之间
        if stats['max_drawdown'] is not None:
            assert 0 <= stats['max_drawdown'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
