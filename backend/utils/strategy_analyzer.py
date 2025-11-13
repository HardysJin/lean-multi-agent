"""
策略表现分析工具

分析每个策略的交易表现，包括：
- 总收益/亏损
- 胜率
- 平均盈亏
- 最大盈利/亏损
- 交易次数
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict


class StrategyPerformanceAnalyzer:
    """策略表现分析器"""
    
    def __init__(self, trades: List[Dict[str, Any]]):
        """
        初始化分析器
        
        Args:
            trades: 交易记录列表，每条记录包含strategy字段
        """
        self.trades = trades
        self.trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    def analyze_by_strategy(self) -> Dict[str, Dict[str, Any]]:
        """
        按策略分析表现
        
        Returns:
            Dict: {strategy_name: {metrics}}
        """
        if self.trades_df.empty:
            return {}
        
        # 只分析有profit字段的交易（卖出交易）
        profitable_trades = self.trades_df[self.trades_df['profit'].notna()].copy()
        
        if profitable_trades.empty:
            return {}
        
        results = {}
        
        # 按策略分组
        for strategy_name, group in profitable_trades.groupby('strategy'):
            metrics = self._calculate_strategy_metrics(group)
            results[strategy_name] = metrics
        
        return results
    
    def _calculate_strategy_metrics(self, trades_group: pd.DataFrame) -> Dict[str, Any]:
        """
        计算单个策略的指标
        
        Args:
            trades_group: 该策略的所有交易记录
        
        Returns:
            Dict: 策略指标
        """
        profits = trades_group['profit']
        
        total_trades = len(trades_group)
        winning_trades = len(profits[profits > 0])
        losing_trades = len(profits[profits < 0])
        
        total_profit = profits.sum()
        avg_profit = profits.mean()
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        max_profit = profits.max() if not profits.empty else 0
        max_loss = profits.min() if not profits.empty else 0
        
        # 平均盈利和平均亏损
        avg_win = profits[profits > 0].mean() if winning_trades > 0 else 0
        avg_loss = profits[profits < 0].mean() if losing_trades > 0 else 0
        
        # 盈亏比
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # 计算sharpe ratio (简化版，假设无风险利率为0)
        sharpe_ratio = (avg_profit / profits.std()) if profits.std() != 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'sharpe_ratio': sharpe_ratio
        }
    
    def get_summary_report(self) -> str:
        """
        生成汇总报告
        
        Returns:
            str: 格式化的报告文本
        """
        analysis = self.analyze_by_strategy()
        
        if not analysis:
            return "没有可分析的交易数据"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("策略表现分析报告")
        report_lines.append("=" * 80)
        
        # 按总收益排序
        sorted_strategies = sorted(
            analysis.items(),
            key=lambda x: x[1]['total_profit'],
            reverse=True
        )
        
        for strategy_name, metrics in sorted_strategies:
            report_lines.append(f"\n📊 策略: {strategy_name.upper()}")
            report_lines.append("-" * 80)
            report_lines.append(f"  交易次数: {metrics['total_trades']}")
            report_lines.append(f"  胜率: {metrics['win_rate']:.2%} ({metrics['winning_trades']}胜 / {metrics['losing_trades']}败)")
            report_lines.append(f"  总盈亏: ${metrics['total_profit']:,.2f}")
            report_lines.append(f"  平均盈亏: ${metrics['avg_profit']:,.2f}")
            report_lines.append(f"  平均盈利: ${metrics['avg_win']:,.2f}")
            report_lines.append(f"  平均亏损: ${metrics['avg_loss']:,.2f}")
            report_lines.append(f"  盈亏比: {metrics['profit_loss_ratio']:.2f}")
            report_lines.append(f"  最大盈利: ${metrics['max_profit']:,.2f}")
            report_lines.append(f"  最大亏损: ${metrics['max_loss']:,.2f}")
            report_lines.append(f"  Sharpe比率: {metrics['sharpe_ratio']:.2f}")
        
        report_lines.append("\n" + "=" * 80)
        
        # 汇总统计
        total_profit = sum(m['total_profit'] for m in analysis.values())
        total_trades = sum(m['total_trades'] for m in analysis.values())
        
        report_lines.append("📈 总体统计")
        report_lines.append("-" * 80)
        report_lines.append(f"  总交易次数: {total_trades}")
        report_lines.append(f"  总盈亏: ${total_profit:,.2f}")
        
        # 最佳策略
        best_strategy = sorted_strategies[0]
        report_lines.append(f"  最佳策略: {best_strategy[0].upper()} (${best_strategy[1]['total_profit']:,.2f})")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def get_strategy_comparison_table(self) -> pd.DataFrame:
        """
        生成策略对比表
        
        Returns:
            pd.DataFrame: 策略对比数据表
        """
        analysis = self.analyze_by_strategy()
        
        if not analysis:
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(analysis).T
        
        # 重命名列
        df.columns = [
            '交易次数', '盈利次数', '亏损次数', '胜率',
            '总盈亏', '平均盈亏', '平均盈利', '平均亏损',
            '盈亏比', '最大盈利', '最大亏损', 'Sharpe比率'
        ]
        
        # 按总盈亏排序
        df = df.sort_values('总盈亏', ascending=False)
        
        return df
    
    def get_strategy_trades(self, strategy_name: str) -> pd.DataFrame:
        """
        获取特定策略的所有交易记录
        
        Args:
            strategy_name: 策略名称
        
        Returns:
            pd.DataFrame: 该策略的交易记录
        """
        if self.trades_df.empty:
            return pd.DataFrame()
        
        strategy_trades = self.trades_df[self.trades_df['strategy'] == strategy_name]
        return strategy_trades.sort_values('date')
    
    def plot_strategy_performance(self, save_path: str = None):
        """
        绘制策略表现对比图
        
        Args:
            save_path: 保存路径，如果为None则显示图表
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            analysis = self.analyze_by_strategy()
            
            if not analysis:
                print("没有可分析的数据")
                return
            
            # 准备数据
            strategies = list(analysis.keys())
            total_profits = [analysis[s]['total_profit'] for s in strategies]
            win_rates = [analysis[s]['win_rate'] * 100 for s in strategies]
            
            # 创建图表
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # 子图1: 总盈亏对比
            colors = ['green' if p > 0 else 'red' for p in total_profits]
            axes[0].bar(strategies, total_profits, color=colors, alpha=0.7)
            axes[0].set_title('策略总盈亏对比', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('策略', fontsize=12)
            axes[0].set_ylabel('总盈亏 ($)', fontsize=12)
            axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            axes[0].grid(axis='y', alpha=0.3)
            
            # 子图2: 胜率对比
            axes[1].bar(strategies, win_rates, color='steelblue', alpha=0.7)
            axes[1].set_title('策略胜率对比', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('策略', fontsize=12)
            axes[1].set_ylabel('胜率 (%)', fontsize=12)
            axes[1].axhline(y=50, color='red', linestyle='--', linewidth=0.5, label='50%基准线')
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"图表已保存到: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("需要安装matplotlib和seaborn: pip install matplotlib seaborn")


def analyze_backtest_results(results: Dict[str, Any]) -> str:
    """
    分析回测结果中的策略表现
    
    Args:
        results: 回测结果字典，包含trades字段
    
    Returns:
        str: 分析报告
    """
    trades = results.get('trades', [])
    
    if not trades:
        return "没有交易记录"
    
    analyzer = StrategyPerformanceAnalyzer(trades)
    return analyzer.get_summary_report()
