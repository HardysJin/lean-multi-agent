"""
LayeredStrategy 端到端回测示例

演示如何使用 LayeredStrategy 进行完整的回测流程。
这个示例展示了分层决策架构在实际交易中的应用。

运行方式：
    python examples/layered_strategy_backtest.py

或者使用不同的参数：
    python examples/layered_strategy_backtest.py --symbols AAPL MSFT GOOGL --days 30
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import argparse
import logging

# 导入回测组件
from Backtests.vectorbt_engine import (
    VectorBTBacktest,
    Signal,
    convert_layered_signals
)
from Backtests.strategies.layered_strategy import LayeredStrategy, estimate_decision_frequency

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LayeredStrategyBacktest:
    """
    LayeredStrategy 回测包装器
    
    简化使用 LayeredStrategy 进行回测的流程
    """
    
    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_cash: float = 100000.0,
        use_mock_llm: bool = True,
        enable_memory: bool = False,
        enable_escalation: bool = True
    ):
        """
        初始化回测
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            initial_cash: 初始资金
            use_mock_llm: 是否使用 MockLLM (True=快速测试, False=真实LLM)
            enable_memory: 是否启用内存系统
            enable_escalation: 是否启用升级机制
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        
        # 初始化 LayeredStrategy
        logger.info(f"Initializing LayeredStrategy (mock_llm={use_mock_llm}, memory={enable_memory}, escalation={enable_escalation})")
        self.strategy = LayeredStrategy(
            use_mock_llm=use_mock_llm,
            enable_memory=enable_memory,
            enable_escalation=enable_escalation
        )
        
        # 初始化 VectorBT 回测引擎
        self.backtest = VectorBTBacktest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash
        )
        
        # 存储结果
        self.signals = None
        self.results = None
    
    async def run(self, show_progress: bool = True):
        """
        运行完整的回测流程
        
        Args:
            show_progress: 是否显示进度信息
        
        Returns:
            回测结果字典
        """
        logger.info("=" * 80)
        logger.info("🚀 Starting LayeredStrategy Backtest")
        logger.info("=" * 80)
        
        # Step 1: 加载历史数据
        logger.info("\n📊 Step 1: Loading historical data...")
        self.backtest.load_data()
        
        # 显示数据统计
        for symbol in self.symbols:
            if symbol in self.backtest._price_data:
                days = len(self.backtest._price_data[symbol])
                logger.info(f"  ✓ {symbol}: {days} trading days")
        
        # Step 2: 估算决策频率
        total_days = (pd.Timestamp(self.end_date) - pd.Timestamp(self.start_date)).days
        decision_estimates = estimate_decision_frequency(total_days, enable_escalation=True)
        
        logger.info(f"\n🎯 Step 2: Decision frequency estimation (for {total_days} days):")
        logger.info(f"  • Strategic decisions: ~{decision_estimates['strategic']} (quarterly)")
        logger.info(f"  • Campaign decisions: ~{decision_estimates['campaign']} (weekly)")
        logger.info(f"  • Tactical decisions: ~{decision_estimates['tactical']} (daily)")
        logger.info(f"  • Total decisions: ~{decision_estimates['total']}")
        
        # Step 3: 预计算交易信号
        logger.info("\n🧠 Step 3: Precomputing trading signals with LayeredStrategy...")
        
        def progress_callback(symbol, current, total, message):
            if show_progress and current % 5 == 0:  # 每5天显示一次
                logger.info(f"  {message}")
        
        # 创建异步任务来预计算信号
        signals_dict = {}
        for symbol in self.symbols:
            if symbol not in self.backtest._price_data:
                continue
            
            df = self.backtest._price_data[symbol]
            close_prices = df['Close']
            
            logger.info(f"Computing signals for {symbol} ({len(close_prices)} days)...")
            
            signal_list = []
            for idx, (date, price) in enumerate(close_prices.items()):
                if show_progress and (idx + 1) % 5 == 0:
                    logger.info(f"  {symbol}: {idx + 1}/{len(close_prices)} days")
                
                # 获取历史数据
                historical_data = df.loc[:date]
                
                if len(historical_data) < 20:
                    signal_list.append({'action': 'HOLD', 'confidence': 0.5, 'reason': 'Insufficient data'})
                    continue
                
                # 调用策略
                try:
                    signal = await self.strategy.generate_signal(
                        symbol=symbol,
                        date=date.strftime('%Y-%m-%d'),
                        price_data=historical_data,
                        context={'current_price': price, 'backtest_mode': True}
                    )
                    signal_list.append(signal)
                except Exception as e:
                    logger.error(f"Error generating signal for {symbol} on {date}: {e}")
                    signal_list.append({'action': 'HOLD', 'confidence': 0.5, 'reason': f'Error: {e}'})
            
            signals_dict[symbol] = signal_list
            
            # 统计信号
            buy_count = sum(1 for s in signal_list if s['action'] == 'BUY')
            sell_count = sum(1 for s in signal_list if s['action'] == 'SELL')
            hold_count = sum(1 for s in signal_list if s['action'] == 'HOLD')
            logger.info(f"✅ {symbol}: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}")
        
        self.signals = signals_dict
        
        # 显示信号统计
        logger.info("\n📈 Signal statistics:")
        for symbol, signal_list in self.signals.items():
            buy_count = sum(1 for s in signal_list if s['action'] == 'BUY')
            total = len(signal_list)
            logger.info(f"  • {symbol}: {buy_count} BUY signals / {total} days ({buy_count/total*100:.1f}%)")
        
        # Step 4: 转换信号格式
        logger.info("\n🔄 Step 4: Converting signals...")
        converted_signals = convert_layered_signals(self.signals)
        
        # 显示仓位信息
        for symbol, sig_list in converted_signals.items():
            total_position = sum(s.size for s in sig_list if s.action == 'BUY')
            avg_confidence = sum(s.confidence for s in sig_list) / len(sig_list) if sig_list else 0
            logger.info(f"  • {symbol}: Total position={total_position:.2f}, Avg confidence={avg_confidence:.2f}")
        
        # Step 5: 运行回测
        logger.info("\n💰 Step 5: Running backtest with position sizing...")
        self.backtest.run_backtest_with_sizing(converted_signals)
        
        # Step 6: 获取性能统计
        logger.info("\n📊 Step 6: Performance summary:")
        self.results = {}
        for symbol in self.symbols:
            if symbol in self.signals:
                portfolio = self.backtest.get_portfolio(symbol)
                if portfolio:
                    self.results[symbol] = portfolio
        
        # Step 7: 显示决策历史
        logger.info("\n🎯 Step 7: Decision history summary:")
        summary = self.strategy.get_decision_summary()
        logger.info(f"  • Total decisions made: {summary['total_decisions']}")
        logger.info(f"  • Strategic: {summary['by_level']['strategic']}")
        logger.info(f"  • Campaign: {summary['by_level']['campaign']}")
        logger.info(f"  • Tactical: {summary['by_level']['tactical']}")
        logger.info(f"  • Escalation rate: {summary['escalation_rate']*100:.1f}%")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Backtest Complete!")
        logger.info("=" * 80)
        
        return self.results
    
    def print_results(self):
        """打印回测结果"""
        if not self.results:
            logger.warning("No results available. Run backtest first.")
            return
        
        print("\n" + "=" * 80)
        print("📊 BACKTEST RESULTS")
        print("=" * 80)
        
        for symbol, portfolio in self.results.items():
            # 使用 VectorBT 的 stats() 方法获取统计
            stats = portfolio.stats()
            
            print(f"\n{symbol}:")
            print(f"  Start Date: {stats['Start']}")
            print(f"  End Date: {stats['End']}")
            print(f"  Initial Cash: ${self.initial_cash:,.2f}")
            print(f"  Final Value: ${stats['End Value']:,.2f}")
            print(f"  Total Return: {stats['Total Return [%]']:.2f}%")
            print(f"  Total Trades: {int(stats['Total Trades'])}")
            
            if stats['Total Trades'] > 0:
                print(f"  Win Rate: {stats['Win Rate [%]']:.2f}%")
                print(f"  Max Drawdown: {stats['Max Drawdown [%]']:.2f}%")
                print(f"  Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
                print(f"  Total Fees: ${stats['Total Fees Paid']:,.2f}")
        
        print("\n" + "=" * 80)
    
    def save_report(self, output_path: str = None):
        """
        保存回测报告
        
        Args:
            output_path: 输出路径（如果不提供，自动生成）
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbols_str = '_'.join(self.symbols)
            output_path = f"Results/layered_strategy_{symbols_str}_{timestamp}.html"
        
        logger.info(f"\n💾 Saving backtest report to: {output_path}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存报告
        try:
            self.backtest.generate_report(output_path)
            logger.info(f"✓ Report saved successfully!")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LayeredStrategy Backtest')
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['AAPL'],
        help='Stock symbols to backtest (default: AAPL)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to backtest (default: 30)'
    )
    parser.add_argument(
        '--cash',
        type=float,
        default=100000.0,
        help='Initial cash (default: 100000)'
    )
    parser.add_argument(
        '--real-llm',
        action='store_true',
        help='Use real LLM instead of MockLLM (slower, requires API key)'
    )
    parser.add_argument(
        '--enable-memory',
        action='store_true',
        help='Enable memory system (requires database setup)'
    )
    parser.add_argument(
        '--save-report',
        action='store_true',
        help='Save HTML report after backtest'
    )
    
    args = parser.parse_args()
    
    # 计算日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    logger.info(f"Configuration:")
    logger.info(f"  Symbols: {args.symbols}")
    logger.info(f"  Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({args.days} days)")
    logger.info(f"  Initial Cash: ${args.cash:,.2f}")
    logger.info(f"  Use Mock LLM: {not args.real_llm}")
    logger.info(f"  Enable Memory: {args.enable_memory}")
    
    # 创建并运行回测
    backtest = LayeredStrategyBacktest(
        symbols=args.symbols,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        initial_cash=args.cash,
        use_mock_llm=not args.real_llm,
        enable_memory=args.enable_memory
    )
    
    # 运行回测
    await backtest.run(show_progress=True)
    
    # 打印结果
    backtest.print_results()
    
    # 保存报告
    if args.save_report:
        backtest.save_report()
    
    logger.info("\n✨ Done!")


if __name__ == '__main__':
    asyncio.run(main())
