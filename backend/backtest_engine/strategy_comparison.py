"""
策略对比工具
用于对比多个策略的回测表现
"""

import pandas as pd
from typing import Dict, Any, List
from tabulate import tabulate

from .simple_backtest import BacktestEngine
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class StrategyComparison:
    """
    策略对比工具
    
    可以同时回测多个策略并生成对比报告
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """初始化"""
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.results = {}
    
    def add_strategy(
        self,
        name: str,
        strategy,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        添加并回测一个策略
        
        Args:
            name: 策略名称
            strategy: 策略实例
            market_data: 市场数据
        
        Returns:
            该策略的回测结果
        """
        logger.info(f"\n回测策略: {name}")
        logger.info("-" * 70)
        
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=self.commission,
            slippage=self.slippage
        )
        
        result = engine.run(strategy, market_data)
        result['strategy_name'] = name
        result['engine'] = engine
        
        self.results[name] = result
        
        return result
    
    def get_comparison_table(self) -> str:
        """
        生成对比表格
        
        Returns:
            格式化的对比表格字符串
        """
        if not self.results:
            return "没有回测结果"
        
        table_data = []
        
        for name, result in self.results.items():
            table_data.append([
                name,
                f"${result.get('final_equity', 0):,.0f}",
                f"{result.get('total_return', 0):.2f}%",
                f"{result.get('buy_hold_return', 0):.2f}%",
                f"{result.get('excess_return', 0):.2f}%",
                result.get('num_trades', 0),
                f"{result.get('win_rate', 0):.1f}%",
                f"{result.get('max_drawdown', 0):.2f}%",
                f"{result.get('sharpe_ratio', 0):.2f}"
            ])
        
        headers = [
            '策略',
            '最终权益',
            '总收益率',
            'B&H收益',
            '超额收益',
            '交易次数',
            '胜率',
            '最大回撤',
            'Sharpe'
        ]
        
        return tabulate(table_data, headers=headers, tablefmt='grid')
    
    def print_summary(self):
        """打印汇总报告"""
        print("\n" + "=" * 80)
        print("策略对比报告")
        print("=" * 80)
        
        if not self.results:
            print("没有回测结果")
            return
        
        print(f"\n初始资金: ${self.initial_capital:,.2f}")
        print(f"手续费率: {self.commission:.2%}")
        print(f"滑点率: {self.slippage:.2%}")
        
        print("\n" + self.get_comparison_table())
        
        # 找出最佳策略
        best_return = max(self.results.items(), key=lambda x: x[1].get('total_return', 0))
        best_sharpe = max(self.results.items(), key=lambda x: x[1].get('sharpe_ratio', 0))
        
        print(f"\n最高收益策略: {best_return[0]} ({best_return[1].get('total_return', 0):.2f}%)")
        print(f"最优风险调整收益: {best_sharpe[0]} (Sharpe: {best_sharpe[1].get('sharpe_ratio', 0):.2f})")
        
        print("\n" + "=" * 80)
    
    def get_detailed_report(self, strategy_name: str) -> str:
        """
        获取特定策略的详细报告
        
        Args:
            strategy_name: 策略名称
        
        Returns:
            详细报告字符串
        """
        if strategy_name not in self.results:
            return f"未找到策略: {strategy_name}"
        
        result = self.results[strategy_name]
        
        report = f"\n{'=' * 80}\n"
        report += f"策略详细报告: {strategy_name}\n"
        report += f"{'=' * 80}\n\n"
        
        # 基本信息
        report += "基本信息:\n"
        report += f"  初始资金: ${result.get('initial_capital', 0):,.2f}\n"
        report += f"  最终权益: ${result.get('final_equity', 0):,.2f}\n"
        report += f"  总收益: ${result.get('final_equity', 0) - result.get('initial_capital', 0):,.2f}\n"
        report += f"  总收益率: {result.get('total_return', 0):.2f}%\n\n"
        
        # 对比基准
        report += "基准对比:\n"
        report += f"  买入持有收益: {result.get('buy_hold_return', 0):.2f}%\n"
        report += f"  超额收益: {result.get('excess_return', 0):.2f}%\n\n"
        
        # 交易统计
        report += "交易统计:\n"
        report += f"  总交易次数: {result.get('num_trades', 0)}\n"
        report += f"  胜率: {result.get('win_rate', 0):.2f}%\n"
        report += f"  平均盈利: ${result.get('avg_win', 0):,.2f}\n"
        report += f"  平均亏损: ${result.get('avg_loss', 0):,.2f}\n"
        report += f"  盈亏比: {result.get('profit_factor', 0):.2f}\n\n"
        
        # 风险指标
        report += "风险指标:\n"
        report += f"  最大回撤: {result.get('max_drawdown', 0):.2f}%\n"
        report += f"  夏普比率: {result.get('sharpe_ratio', 0):.2f}\n\n"
        
        # 交易记录
        trades = result.get('trades', [])
        if trades:
            report += "最近交易记录:\n"
            trade_table = []
            for trade in trades[-10:]:  # 最后10笔交易
                trade_table.append([
                    trade.get('date', ''),
                    trade.get('action', ''),
                    f"${trade.get('price', 0):.2f}",
                    trade.get('shares', 0),
                    f"${trade.get('profit', 0):,.2f}" if 'profit' in trade else '-',
                    f"{trade.get('profit_pct', 0):.2f}%" if 'profit_pct' in trade else '-',
                    trade.get('reason', '')[:30]
                ])
            
            report += tabulate(
                trade_table,
                headers=['日期', '动作', '价格', '股数', '盈亏', '盈亏%', '原因'],
                tablefmt='grid'
            )
            report += "\n"
        
        report += "=" * 80 + "\n"
        
        return report
