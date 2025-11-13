"""
简单回测引擎
基于pandas实现的向量化回测，支持多种策略
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from pathlib import Path
import json

from backend.config.config_loader import get_config
from backend.portfolio.portfolio_manager import PortfolioManager
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """
    回测引擎
    
    特性：
    - 向量化回测（高性能）
    - 支持多种策略
    - 详细的交易记录
    - 完整的性能指标
    """
    
    def __init__(
        self,
        initial_capital: float = None,
        commission: float = None,
        slippage: float = None,
    ):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金（None则使用config）
            commission: 手续费率（None则使用config）
            slippage: 滑点率（None则使用config）
        """
        # 加载配置
        config = get_config()
        
        # 使用参数或配置中的值
        self.initial_capital = initial_capital if initial_capital is not None else config.system.initial_capital
        self.commission = commission if commission is not None else config.system.commission
        self.slippage = slippage if slippage is not None else config.system.slippage
        
        # Portfolio Manager - 统一管理组合状态
        self.portfolio = PortfolioManager(initial_capital=self.initial_capital)
        
        # 回测结果（额外记录）
        self.equity_curve = []
        self.positions_history = []
        
    def run(
        self,
        strategy,
        market_data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            strategy: 策略实例（需要有generate_signals方法）
            market_data: 市场数据DataFrame，包含OHLCV列
                注意：market_data应该包含回测期间之前的lookback数据
            start_date: 回测开始日期（可选）
            end_date: 回测结束日期（可选）
        
        Returns:
            回测结果字典
        """
        logger.info("=" * 70)
        logger.info("开始回测")
        logger.info("=" * 70)
        
        # 准备数据 - market_data应该已经包含lookback数据
        # 我们使用所有可用数据，但只在start_date到end_date之间进行交易
        full_data = market_data.copy()
        
        # 确定回测的实际交易范围
        if start_date is None:
            start_date = full_data.index[0]
        if end_date is None:
            end_date = full_data.index[-1]
        
        # 过滤出截止到end_date的数据（避免看到未来）
        df = full_data[full_data.index <= end_date].copy()
        
        if len(df) == 0:
            raise ValueError("没有数据用于回测")
        
        logger.info(f"数据范围: {df.index[0]} 到 {df.index[-1]}")
        logger.info(f"回测期间: {start_date.date() if hasattr(start_date, 'date') else start_date} 到 {end_date.date() if hasattr(end_date, 'date') else end_date}")
        logger.info(f"总数据点数: {len(df)}")
        logger.info(f"初始资金: ${self.initial_capital:,.2f}")
        
        # 重置 Portfolio Manager
        self.portfolio.reset()
        
        # 回测状态追踪变量
        symbol = 'SYMBOL'  # 简单回测假设单一标的
        self.equity_curve = []
        self.positions_history = []
        
        # 逐日回测
        for i in range(len(df)):
            current_data = df.iloc[:i+1]  # 策略能看到从开始到当前的所有历史数据
            current_bar = df.iloc[i]
            current_price = current_bar['Close']
            current_date = df.index[i]
            
            # 只在回测期间内进行交易和记录
            in_backtest_period = current_date >= start_date
            
            # 从 Portfolio Manager 获取当前状态
            position = self.portfolio.get_position_shares(symbol)
            entry_price = self.portfolio.get_position_entry_price(symbol)
            
            # 准备上下文信息传递给策略
            context = {
                'position': 1 if position > 0 else 0,
                'entry_price': entry_price,
                'shares': position,
                'positions': {}  # 网格策略可能需要
            }
            
            # 生成信号（新架构：通过context传递状态）
            signal = strategy.generate_signals(current_data, **context)
            action = signal.get('action', 'hold')
            
            # 只在回测期间内执行交易
            if not in_backtest_period:
                # 在回测期间之前，只更新策略状态，不执行交易
                continue
            
            # 执行交易
            if action == 'buy' and position == 0:
                # 买入
                buy_price = current_price * (1 + self.slippage)
                shares = int(self.portfolio.cash * 0.99 / buy_price)  # 99%仓位
                
                if shares > 0:
                    # 使用 Portfolio Manager 执行买入
                    success = self.portfolio.execute_buy(
                        symbol=symbol,
                        shares=shares,
                        price=buy_price,
                        commission=self.commission,
                        date=current_date.strftime('%Y-%m-%d'),
                        strategy='simple_backtest'
                    )
            
            elif action == 'sell' and position > 0:
                # 卖出
                sell_price = current_price * (1 - self.slippage)
                
                # 使用 Portfolio Manager 执行卖出
                success = self.portfolio.execute_sell(
                    symbol=symbol,
                    shares=position,
                    price=sell_price,
                    commission=self.commission,
                    date=current_date.strftime('%Y-%m-%d'),
                    strategy='simple_backtest'
                )
            
            # 更新 Portfolio Manager 的组合价值快照
            position = self.portfolio.get_position_shares(symbol)
            equity = self.portfolio.get_portfolio_value({symbol: current_price})
            
            self.equity_curve.append({
                'date': current_date,
                'equity': equity,
                'cash': self.portfolio.cash,
                'position_value': position * current_price,
                'position': position
            })
            
            self.positions_history.append({
                'date': current_date,
                'position': position,
                'price': current_price
            })
        
        # 最后如果还持仓，强制平仓
        final_position = self.portfolio.get_position_shares(symbol)
        if final_position > 0:
            final_price = df.iloc[-1]['Close']
            final_date = df.index[-1]
            
            self.portfolio.execute_sell(
                symbol=symbol,
                shares=final_position,
                price=final_price,
                commission=self.commission,
                date=final_date.strftime('%Y-%m-%d'),
                strategy='simple_backtest'
            )
        
        # 计算性能指标
        final_equity = self.portfolio.cash
        performance = self._calculate_performance(df)
        
        logger.info("=" * 70)
        logger.info("回测完成")
        logger.info(f"最终权益: ${final_equity:,.2f}")
        logger.info(f"总收益: ${final_equity - self.initial_capital:,.2f}")
        logger.info(f"收益率: {performance['total_return']:.2f}%")
        logger.info("=" * 70)
        
        return performance
    
    def _calculate_performance(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """计算性能指标"""
        equity_df = pd.DataFrame(self.equity_curve)
        
        if len(equity_df) == 0:
            return {}
        
        equity_df.set_index('date', inplace=True)
        
        # 基本指标
        initial = self.initial_capital
        final = equity_df['equity'].iloc[-1]
        total_return = ((final / initial) - 1) * 100
        
        # 交易统计（从 Portfolio Manager 获取）
        all_trades = self.portfolio.get_trades()
        sell_trades = [t for t in all_trades if t.action == 'sell' and t.profit is not None]
        trades_df = pd.DataFrame([{'profit': t.profit} for t in sell_trades])
        num_trades = len(sell_trades)
        
        if num_trades > 0:
            winning_trades = trades_df[trades_df['profit'] > 0]
            losing_trades = trades_df[trades_df['profit'] < 0]
            
            win_rate = len(winning_trades) / num_trades * 100
            avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0
            
            profit_factor = abs(winning_trades['profit'].sum() / losing_trades['profit'].sum()) \
                if len(losing_trades) > 0 and losing_trades['profit'].sum() != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # 最大回撤
        equity_series = equity_df['equity']
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # 夏普比率（简化版，假设无风险利率为0）
        returns = equity_series.pct_change().dropna()
        if len(returns) > 0 and returns.std() != 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # 年化
        else:
            sharpe_ratio = 0
        
        # 买入持有基准
        buy_hold_return = ((market_data['Close'].iloc[-1] / market_data['Close'].iloc[0]) - 1) * 100
        
        return {
            'initial_capital': initial,
            'final_equity': final,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': [t.to_dict() for t in self.portfolio.get_trades()],
            'equity_curve': self.equity_curve
        }
    
    def save_results(self, filepath: str):
        """保存回测结果到JSON文件"""
        if not self.equity_curve:
            logger.warning("没有回测结果可保存")
            return
        
        results = {
            'trades': [t.to_dict() for t in self.portfolio.get_trades()],
            'equity_curve': self.equity_curve,
            'positions_history': self.positions_history
        }
        
        # 转换datetime为字符串
        def convert_dates(obj):
            if isinstance(obj, list):
                return [convert_dates(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_dates(v) for k, v in obj.items()}
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return obj.strftime('%Y-%m-%d')
            else:
                return obj
        
        results = convert_dates(results)
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"回测结果已保存到: {filepath}")
