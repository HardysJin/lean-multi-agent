"""
VectorBT 回测系统测试

测试 VectorBT 集成和 Multi-Agent 策略
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

from Backtests.vectorbt_engine import VectorBTBacktest, quick_backtest
from Backtests.strategies.multi_agent_strategy import SimpleTechnicalStrategy
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_simple_strategy():
    """
    测试1: 简单技术策略（不使用 LLM）
    
    这个测试快速，用于验证回测引擎是否正常工作
    """
    print("\n" + "="*60)
    print("测试 1: 简单技术策略回测")
    print("="*60)
    
    # 创建回测引擎
    backtest = VectorBTBacktest(
        symbols=['AAPL'],
        start_date='2024-01-01',
        end_date='2024-10-28',
        initial_cash=100000,
        fees=0.001
    )
    
    # 加载数据
    print("\n📊 Step 1: 加载股票数据...")
    backtest.load_data()
    
    # 使用简单策略生成信号
    print("\n🤖 Step 2: 生成交易信号（简单移动平均策略）...")
    
    strategy = SimpleTechnicalStrategy(short_window=20, long_window=50)
    
    signals = {}
    for symbol in backtest.symbols:
        if symbol not in backtest._price_data:
            continue
        
        df = backtest._price_data[symbol]
        symbol_signals = []
        
        for idx, (date, row) in enumerate(df.iterrows()):
            historical_data = df.loc[:date]
            signal = strategy.generate_signal(
                symbol=symbol,
                date=date,
                price=row['Close'],
                historical_data=historical_data
            )
            symbol_signals.append(1 if signal > 0 else 0)  # 转换为 VectorBT 格式
        
        signals[symbol] = pd.Series(symbol_signals, index=df.index)
        print(f"  ✅ {symbol}: 生成 {sum(symbol_signals)} 个买入信号")
    
    # 运行回测
    print("\n📈 Step 3: 运行回测...")
    backtest.run_backtest(signals)
    
    # 获取统计
    print("\n📊 Step 4: 性能分析...")
    stats = backtest.get_performance_stats('AAPL')
    
    print("\n" + "="*60)
    print("回测结果")
    print("="*60)
    print(f"股票代码: {stats['symbol']}")
    print(f"回测周期: {stats['start_date']} 到 {stats['end_date']}")
    print(f"初始资金: ${stats['initial_cash']:,.2f}")
    print(f"最终价值: ${stats['final_value']:,.2f}")
    print(f"总收益: ${stats['profit_loss']:,.2f}")
    print(f"收益率: {stats['total_return_pct']}")
    print(f"总交易次数: {stats['total_trades']}")
    if stats['win_rate']:
        print(f"胜率: {stats['win_rate']:.2%}")
    print("="*60)
    
    # 生成报告
    print("\n📄 Step 5: 生成报告...")
    reports = backtest.generate_report()
    print(f"  ✅ 报告已保存:")
    for name, path in reports.items():
        print(f"     - {name}: {path}")
    
    return backtest


async def test_multi_agent_strategy_sample():
    """
    测试2: Multi-Agent 策略（采样测试）
    
    只测试几天，验证 AI Agent 是否能正常工作
    """
    print("\n" + "="*60)
    print("测试 2: Multi-Agent 策略（采样测试）")
    print("="*60)
    
    from Backtests.strategies.multi_agent_strategy import MultiAgentStrategy
    import pandas as pd
    
    # 创建策略
    strategy = MultiAgentStrategy()
    
    # 获取少量数据测试
    print("\n📊 获取测试数据...")
    import yfinance as yf
    df = yf.download('AAPL', start='2024-10-01', end='2024-10-28')
    
    # 只测试最后 3 天
    test_dates = df.index[-3:]
    print(f"\n🤖 测试 AI Agent 决策（最近 3 天）...")
    
    for date in test_dates:
        # 确保 price 是标量值
        price_value = float(df.loc[date, 'Close'])
        historical_data = df.loc[:date]
        
        print(f"\n📅 {date.date()}")
        print(f"   价格: ${price_value:.2f}")
        
        signal = await strategy.generate_signal(
            symbol='AAPL',
            date=date,
            price=price_value,
            historical_data=historical_data
        )
        
        action = "BUY" if signal == 1 else ("SELL" if signal == -1 else "HOLD")
        print(f"   决策: {action}")
    
    print("\n✅ Multi-Agent 策略测试完成！")


async def main():
    """主测试函数"""
    print("\n🚀 VectorBT 回测系统测试")
    print("="*60)
    
    # 测试 1: 简单策略（快速）
    try:
        backtest = await test_simple_strategy()
        print("\n✅ 测试 1 通过！")
    except Exception as e:
        print(f"\n❌ 测试 1 失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试 2: Multi-Agent 策略采样（需要 LLM API）
    print("\n" + "="*60)
    user_input = input("\n是否测试 Multi-Agent 策略？(需要 LLM API，耗时较长) [y/N]: ")
    
    if user_input.lower() == 'y':
        try:
            await test_multi_agent_strategy_sample()
            print("\n✅ 测试 2 通过！")
        except Exception as e:
            print(f"\n❌ 测试 2 失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⏭️  跳过 Multi-Agent 测试")
    
    print("\n" + "="*60)
    print("🎉 所有测试完成！")
    print("="*60)
    print("\n下一步:")
    print("1. 查看生成的 HTML 报告（在 Results/ 目录）")
    print("2. 运行完整的 Multi-Agent 回测（可能需要几小时）")
    print("3. 创建自定义策略并测试")


if __name__ == "__main__":
    import pandas as pd  # 需要导入以便脚本使用
    asyncio.run(main())
