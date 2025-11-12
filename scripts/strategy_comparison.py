"""
策略对比测试脚本
对比LLM多Agent决策 vs 简单Buy & Hold策略

用法：
    python tmp/strategy_comparison.py SPY
    python tmp/strategy_comparison.py QQQ
    python tmp/strategy_comparison.py AAPL

特性：
- 使用config.yaml中的统一参数
- 自动生成对比报告
- 详细的性能指标对比
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.backtest_engine.llm_backtest import LLMBacktestEngine
from backend.backtest_engine.simple_backtest import BacktestEngine
from backend.strategies.buy_and_hold import BuyAndHoldStrategy
from backend.data_collectors.market_data import MarketDataCollector
from backend.config.config_loader import get_config
from backend.utils.logger import get_logger
from tabulate import tabulate

logger = get_logger(__name__)


class StrategyComparison:
    """策略对比类"""
    
    def __init__(self, symbol: str):
        """
        初始化
        
        Args:
            symbol: 股票代码
        """
        self.symbol = symbol
        self.config = get_config()
        
        # 从config读取参数
        self.start_date = datetime.strptime(self.config.system.backtest_start, "%Y-%m-%d")
        self.end_date = datetime.strptime(self.config.system.backtest_end, "%Y-%m-%d")
        self.initial_capital = self.config.system.initial_capital
        self.commission = self.config.system.commission
        self.slippage = self.config.system.slippage
        
        # 数据收集器
        self.data_collector = MarketDataCollector()
        
        logger.info("=" * 80)
        logger.info(f"策略对比测试：{symbol}")
        logger.info("=" * 80)
        logger.info(f"回测期间: {self.start_date.date()} 到 {self.end_date.date()}")
        logger.info(f"初始资金: ${self.initial_capital:,.2f}")
        logger.info(f"手续费: {self.commission*100:.2f}%")
        logger.info(f"滑点: {self.slippage*100:.2f}%")
        logger.info("")
    
    def run_llm_strategy(self) -> Dict[str, Any]:
        """运行LLM多Agent策略"""
        logger.info("-" * 80)
        logger.info("运行策略1: LLM多Agent决策")
        logger.info("-" * 80)
        
        engine = LLMBacktestEngine()
        results = engine.run(symbol=self.symbol)
        
        return results
    
    def run_buy_hold_strategy(self) -> Dict[str, Any]:
        """运行Buy & Hold策略"""
        logger.info("-" * 80)
        logger.info("运行策略2: Buy & Hold")
        logger.info("-" * 80)
        
        # 创建策略实例
        strategy = BuyAndHoldStrategy()
        
        # 创建回测引擎
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=self.commission,
            slippage=self.slippage
        )
        
        # 收集市场数据（需要包含lookback期间）
        lookback_days = self.config.system.lookback_days
        from datetime import timedelta
        data_start = self.start_date - timedelta(days=lookback_days)
        
        logger.info(f"下载市场数据: {data_start.date()} 到 {self.end_date.date()}")
        market_data_dict = self.data_collector.collect(
            symbol=self.symbol,
            start_date=data_start,
            end_date=self.end_date
        )
        
        # MarketDataCollector返回的是{symbol: {"ohlcv": [...], ...}}字典
        if market_data_dict is None:
            raise ValueError(f"无法获取{self.symbol}的市场数据")
        
        # 从字典中提取DataFrame
        symbol_data = market_data_dict.get(self.symbol)
        if symbol_data is None:
            raise ValueError(f"无法获取{self.symbol}的市场数据，字典中没有该symbol")
        
        # 从ohlcv记录转换为DataFrame
        ohlcv_records = symbol_data.get("ohlcv")
        if not ohlcv_records:
            raise ValueError(f"{self.symbol}的市场数据为空")
        
        market_data = pd.DataFrame(ohlcv_records)
        # 设置Date为index
        if 'Date' in market_data.columns:
            market_data['Date'] = pd.to_datetime(market_data['Date'])
            market_data.set_index('Date', inplace=True)
        
        logger.info(f"数据点数: {len(market_data)}")
        
        # 运行回测
        results = engine.run(
            strategy=strategy,
            market_data=market_data,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        return results
    
    def compare_results(self, llm_results: Dict[str, Any], bh_results: Dict[str, Any]) -> Dict[str, Any]:
        """对比两个策略的结果"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("策略对比结果")
        logger.info("=" * 80)
        
        # LLM backtest返回格式: {'summary': {...}, 'trades': [...], ...}
        llm_summary = llm_results.get('summary', {})

        # 处理LLM指标（统一使用百分比字段 *_pct 表示百分比数值，例如 1.63 表示 1.63%）
        llm_total_return_pct = llm_summary.get('total_return', 0) * 100
        llm_annual_return_pct = llm_summary.get('annual_return', 0) * 100
        llm_max_drawdown_pct = llm_summary.get('max_drawdown', 0) * 100
        llm_win_rate_pct = llm_summary.get('win_rate', 0) * 100
        llm_benchmark_pct = llm_summary.get('benchmark_return', 0) * 100
        llm_alpha_pct = llm_summary.get('alpha', 0) * 100

        # --- 交易明细：处理每笔买卖交易，计算持仓、盈亏等 ---
        llm_trades_raw = llm_results.get('trades', [])
        
        # 追踪持仓状态
        position = 0  # 当前持仓数量
        avg_cost = 0.0  # 持仓均价
        total_cost = 0.0  # 总成本
        cumulative_pnl = 0.0  # 累计已实现盈亏
        
        trade_records = []
        
        for t in llm_trades_raw:
            action = t.get('action', '').lower()
            if action not in ['buy', 'sell']:
                continue
                
            date = t.get('date')
            price = float(t.get('price', 0))
            shares = int(t.get('shares', 0))
            cost = float(t.get('cost', 0))
            
            trade_pnl = 0.0
            trade_type = ""
            
            if action == 'buy':
                if position == 0:
                    trade_type = "开仓(多)"
                else:
                    trade_type = "加仓(多)"
                
                # 更新持仓
                old_total_cost = total_cost
                total_cost += cost
                position += shares
                avg_cost = total_cost / position if position > 0 else 0
                
            elif action == 'sell':
                if position > 0:
                    # 平仓
                    close_qty = min(abs(shares), position)
                    proceeds = float(t.get('proceeds', close_qty * price))
                    
                    # 计算这笔交易的盈亏
                    trade_pnl = proceeds - (close_qty * avg_cost)
                    cumulative_pnl += trade_pnl
                    
                    if close_qty == position:
                        trade_type = "平仓(多)"
                    else:
                        trade_type = "减仓(多)"
                    
                    # 更新持仓
                    position -= close_qty
                    if position > 0:
                        total_cost = position * avg_cost
                    else:
                        total_cost = 0
                        avg_cost = 0
            
            # 构建交易记录
            date_str = str(date)[:10] if date else '-'
            action_cn = "买入" if action == 'buy' else "卖出"
            price_str = f"${price:.2f}"
            shares_str = f"{abs(shares)}"  # 显示绝对值
            
            # 金额列：买入显示成本，卖出显示收益
            if action == 'buy':
                amount_str = f"${cost:.2f}"
            else:
                proceeds_val = float(t.get('proceeds', abs(shares) * price))
                amount_str = f"${proceeds_val:.2f}"
            position_str = f"{position}"
            avg_cost_str = f"${avg_cost:.2f}" if position > 0 else "-"
            
            # 交易盈亏（只有卖出才显示）
            if action == 'sell' and trade_pnl != 0:
                trade_pnl_str = f"${trade_pnl:+,.2f}"
                cumulative_pnl_str = f"${cumulative_pnl:+,.2f}"
            else:
                trade_pnl_str = "-"
                cumulative_pnl_str = "-" if cumulative_pnl == 0 else f"${cumulative_pnl:+,.2f}"
            
            trade_records.append([
                date_str,
                self.symbol,
                action_cn,
                trade_type,
                shares_str,
                price_str,
                amount_str,
                position_str,
                avg_cost_str,
                trade_pnl_str,
                cumulative_pnl_str
            ])

        # 如果有持仓，添加当前持仓状态（浮动盈亏）
        if position > 0:
            try:
                from datetime import timedelta
                lookback_days = getattr(self.config.system, 'lookback_days', 150)
                data_start = self.start_date - timedelta(days=lookback_days)
                market_data_dict = self.data_collector.collect(
                    symbol=self.symbol,
                    start_date=data_start,
                    end_date=self.end_date
                )
                symbol_data = market_data_dict.get(self.symbol) if isinstance(market_data_dict, dict) else None
                last_price = None
                if symbol_data:
                    ohlcv = symbol_data.get('ohlcv')
                    if ohlcv:
                        import pandas as pd
                        df_temp = pd.DataFrame(ohlcv)
                        if 'Date' in df_temp.columns:
                            df_temp['Date'] = pd.to_datetime(df_temp['Date'])
                            df_temp.set_index('Date', inplace=True)
                        if 'Close' in df_temp.columns:
                            last_price = float(df_temp['Close'].iloc[-1])
                
                if last_price is not None and avg_cost > 0:
                    # 计算浮动盈亏
                    unrealized_pnl = (last_price - avg_cost) * position
                    unrealized_pnl_total = cumulative_pnl + unrealized_pnl
                    
                    # 添加当前持仓状态行
                    trade_records.append([
                        "-",
                        self.symbol,
                        "持仓",
                        "持仓中",
                        f"{position}",
                        f"${last_price:.2f}(当前)",
                        "-",
                        f"{position}",
                        f"${avg_cost:.2f}",
                        f"${unrealized_pnl:+,.2f}(浮)",
                        f"${unrealized_pnl_total:+,.2f}(含浮)"
                    ])
            except Exception as e:
                logger.warning(f"获取最新价格失败: {e}")

        # 计算盈亏比（profit factor）: sum(wins)/abs(sum(losses))
        # 从已实现交易中计算
        llm_win_sum = 0.0
        llm_loss_sum = 0.0
        
        for rec in trade_records:
            action = rec[2]  # 操作列
            if action == "卖出":
                pnl_str = rec[9]  # 交易盈亏列
                if pnl_str != "-" and "(浮)" not in pnl_str:
                    try:
                        # 提取数值，去除 $ 和 ,
                        pnl_value = float(pnl_str.replace('$', '').replace(',', '').replace('+', ''))
                        if pnl_value > 0:
                            llm_win_sum += pnl_value
                        else:
                            llm_loss_sum += abs(pnl_value)
                    except:
                        pass

        if llm_loss_sum > 0:
            llm_profit_factor = llm_win_sum / llm_loss_sum
        else:
            llm_profit_factor = float('inf') if llm_win_sum > 0 else 0.0

        # 打印详细交易记录（最近10笔）
        if trade_records:
            print(f"\n【交易明细】(最近{min(10, len(trade_records))}/{len(trade_records)}笔)")
            headers = ["时间", "股票", "操作", "类型", "数量", "价格", "金额", "持仓", "持仓均价", "交易盈亏", "累计收益"]
            print(tabulate(trade_records[-10:], headers=headers, tablefmt='simple',
                         colalign=('left', 'left', 'center', 'center', 'right', 'right', 'right', 'right', 'right', 'right', 'right')))


        # 处理 Buy & Hold 回测（simple_backtest 返回百分比字段）
        bh_final = bh_results.get('final_equity', bh_results.get('final_value', 0))
        bh_total_return_pct = bh_results.get('total_return', 0)  # already percent

        # 计算Buy&Hold年化收益（如果可能）
        days = (self.end_date - self.start_date).days
        years = days / 365.25 if days > 0 else 0
        try:
            bh_annual_return_pct = ((1 + bh_total_return_pct / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
        except Exception:
            bh_annual_return_pct = 0

        bh_max_drawdown_pct = abs(bh_results.get('max_drawdown', 0))
        bh_sharpe = bh_results.get('sharpe_ratio', 0)
        bh_num_trades = bh_results.get('num_trades', 0)
        bh_win_rate_pct = bh_results.get('win_rate', 0)

        comparison = {
            'symbol': self.symbol,
            'period': {
                'start': self.start_date.strftime('%Y-%m-%d'),
                'end': self.end_date.strftime('%Y-%m-%d')
            },
            'llm_strategy': {
                'name': 'LLM Multi-Agent',
                'final_value': llm_summary.get('final_value', 0),
                'total_return_pct': llm_total_return_pct,
                'annual_return_pct': llm_annual_return_pct,
                'max_drawdown_pct': llm_max_drawdown_pct,
                'sharpe_ratio': llm_summary.get('sharpe_ratio', 0),
                'num_trades': llm_summary.get('total_trades', 0),
                'win_rate_pct': llm_win_rate_pct,
                'benchmark_return_pct': llm_benchmark_pct,
                'alpha_pct': llm_alpha_pct
            },
            'buy_hold_strategy': {
                'name': 'Buy & Hold',
                'final_value': bh_final,
                'total_return_pct': bh_total_return_pct,
                'annual_return_pct': bh_annual_return_pct,
                'max_drawdown_pct': bh_max_drawdown_pct,
                'sharpe_ratio': bh_sharpe,
                'num_trades': bh_num_trades,
                'win_rate_pct': bh_win_rate_pct
            }
        }

        # 计算 Buy&Hold 的盈亏比（profit factor）
        bh_trades = bh_results.get('trades', [])
        bh_win_sum = 0.0
        bh_loss_sum = 0.0
        for t in bh_trades:
            # simple_backtest的trade使用'profit'字段记录盈亏（卖出）
            pnl = t.get('profit')
            if pnl is None:
                continue
            if pnl > 0:
                bh_win_sum += pnl
            else:
                bh_loss_sum += abs(pnl)

        if bh_loss_sum > 0:
            bh_profit_factor = bh_win_sum / bh_loss_sum
        else:
            bh_profit_factor = float('inf') if bh_win_sum > 0 else 0.0

        # 将盈亏比加入comparison
        comparison['llm_strategy']['profit_factor'] = llm_profit_factor
        comparison['buy_hold_strategy']['profit_factor'] = bh_profit_factor

        # 差异（以百分比或原始差异表示）
        comparison['difference'] = {
            'return_diff_pct': comparison['llm_strategy']['total_return_pct'] - comparison['buy_hold_strategy']['total_return_pct'],
            'annual_return_diff_pct': comparison['llm_strategy']['annual_return_pct'] - comparison['buy_hold_strategy']['annual_return_pct'],
            'sharpe_diff': comparison['llm_strategy']['sharpe_ratio'] - comparison['buy_hold_strategy']['sharpe_ratio'],
            'drawdown_diff_pct': comparison['llm_strategy']['max_drawdown_pct'] - comparison['buy_hold_strategy']['max_drawdown_pct']
        }

        # 打印对比表格（使用百分比字段）
        print("\n")
        table_data = [
            ['总收益率⇧',
             f"{comparison['llm_strategy']['total_return_pct']:.2f}%",
             f"{comparison['buy_hold_strategy']['total_return_pct']:.2f}%",
             f"{comparison['difference']['return_diff_pct']:.2f}%"],

            ['年化收益⇧',
             f"{comparison['llm_strategy']['annual_return_pct']:.2f}%",
             f"{comparison['buy_hold_strategy']['annual_return_pct']:.2f}%",
             f"{comparison['difference']['annual_return_diff_pct']:.2f}%"],

            ['最终价值⇧',
             f"${comparison['llm_strategy']['final_value']:,.2f}",
             f"${comparison['buy_hold_strategy']['final_value']:,.2f}",
             f"${comparison['llm_strategy']['final_value'] - comparison['buy_hold_strategy']['final_value']:,.2f}"],

            ['最大回撤⇩',
             f"{comparison['llm_strategy']['max_drawdown_pct']:.2f}%",
             f"{comparison['buy_hold_strategy']['max_drawdown_pct']:.2f}%",
             f"{comparison['difference']['drawdown_diff_pct']:.2f}%"],

            ['夏普比率⇧',
             f"{comparison['llm_strategy']['sharpe_ratio']:.2f}",
             f"{comparison['buy_hold_strategy']['sharpe_ratio']:.2f}",
             f"{comparison['difference']['sharpe_diff']:.2f}"],

            ['交易次数⇩',
             f"{comparison['llm_strategy']['num_trades']}",
             f"{comparison['buy_hold_strategy']['num_trades']}",
             f"{comparison['llm_strategy']['num_trades'] - comparison['buy_hold_strategy']['num_trades']}"],

            ['胜率⇧',
             f"{comparison['llm_strategy']['win_rate_pct']:.1f}%",
             f"{comparison['buy_hold_strategy']['win_rate_pct']:.1f}%",
             f"{comparison['llm_strategy']['win_rate_pct'] - comparison['buy_hold_strategy']['win_rate_pct']:.1f}%"],

            ['盈亏比⇧',
             f"{comparison['llm_strategy'].get('profit_factor', 0):.2f}",
             f"{comparison['buy_hold_strategy'].get('profit_factor', 0):.2f}",
             f"{(comparison['llm_strategy'].get('profit_factor', 0) - comparison['buy_hold_strategy'].get('profit_factor', 0)):.2f}"],
        ]

        print(tabulate(table_data,
                      headers=['指标', 'LLM多Agent', 'Buy & Hold', '差异'],
                      tablefmt='presto'))
        print("\n")

        # Alpha（仅LLM策略有）
        if 'alpha_pct' in comparison['llm_strategy']:
            print(f"LLM策略 Alpha: {comparison['llm_strategy']['alpha_pct']:.2f}%")
            print(f"  (相对基准收益: {comparison['llm_strategy']['benchmark_return_pct']:.2f}%)")
            print("\n")

        # 结论
        print("结论:")
        if comparison['difference']['return_diff_pct'] > 0:
            print(f"  ✓ LLM多Agent策略表现更好，超额收益: {comparison['difference']['return_diff_pct']:.2f}%")
        elif comparison['difference']['return_diff_pct'] < 0:
            print(f"  ✗ Buy & Hold策略表现更好，超额收益: {-comparison['difference']['return_diff_pct']:.2f}%")
        else:
            print("  = 两个策略收益相当")

        if comparison['difference']['sharpe_diff'] > 0:
            print(f"  ✓ LLM策略风险调整后收益更好 (夏普比率高 {comparison['difference']['sharpe_diff']:.2f})")

        if comparison['difference']['drawdown_diff_pct'] < 0:
            print(f"  ✓ LLM策略回撤控制更好 (少回撤 {-comparison['difference']['drawdown_diff_pct']:.2f}%)")

        print("\n")

        return comparison
    
    def run(self) -> Dict[str, Any]:
        """运行完整对比测试"""
        try:
            # 运行两个策略
            bh_results = self.run_buy_hold_strategy()
            llm_results = self.run_llm_strategy()
            
            # 对比结果
            comparison = self.compare_results(llm_results, bh_results)
            
            # 保存结果
            out_dir = Path(__file__).parent / f"results"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_file = out_dir / f"comparison_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            
            logger.info(f"对比结果已保存到: {output_file}")
            
            return comparison
            
        except Exception as e:
            logger.error(f"对比测试失败: {e}", exc_info=True)
            raise


def main():
    """主函数"""
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python tmp/strategy_comparison.py <SYMBOL>")
        print("\n示例:")
        print("  python tmp/strategy_comparison.py SPY")
        print("  python tmp/strategy_comparison.py QQQ")
        print("  python tmp/strategy_comparison.py AAPL")
        print("  python tmp/strategy_comparison.py TSLA")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    
    # 运行对比测试
    comparison = StrategyComparison(symbol)
    results = comparison.run()
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
