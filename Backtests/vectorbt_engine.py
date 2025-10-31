"""
VectorBT 回测引擎

功能：
1. 仓位管理 - 动态调整持仓大小
2. 加减仓 - 不只是买入/卖出，还有增加/减少仓位
3. LLM决策仓位大小 - 根据conviction调整
4. 完整的性能指标 - 持仓、PnL、费用等
"""

import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def convert_layered_signals(signals_dict: Dict[str, List[Dict]]) -> Dict[str, List['Signal']]:
    """
    转换 LayeredStrategy 信号字典到 Signal 对象
    
    Args:
        signals_dict: {symbol: [signal_dict, ...]}
        
    Returns:
        {symbol: [Signal, ...]}
    """
    result = {}
    for symbol, signals in signals_dict.items():
        result[symbol] = [Signal.from_layered_strategy_signal(s) for s in signals]
    return result


class Signal:
    """
    交易信号
    
    Attributes:
        action: BUY, SELL, HOLD, ADD (加仓), REDUCE (减仓)
        size: 仓位大小 (0.0-1.0，表示资金的百分比)
        confidence: 信心水平 (0.0-1.0)
        reason: 决策原因
    """
    def __init__(
        self,
        action: str,
        size: float = 0.0,
        confidence: float = 0.5,
        reason: str = ""
    ):
        self.action = action.upper()
        self.size = max(0.0, min(1.0, size))  # 限制在 0-1
        self.confidence = max(0.0, min(1.0, confidence))
        self.reason = reason
    
    @classmethod
    def from_layered_strategy_signal(cls, signal: Dict):
        """从 LayeredStrategy 信号创建 Signal"""
        action = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0.5)
        
        # 根据 confidence 计算仓位大小
        # 高信心 = 大仓位，低信心 = 小仓位
        if action == 'BUY':
            # confidence 0.5-1.0 映射到 size 0.1-0.3 (保守策略)
            size = 0.1 + (confidence - 0.5) * 0.4  # 10%-30%
        elif action == 'SELL':
            size = 1.0  # 卖出全部
        else:
            size = 0.0  # HOLD
        
        return cls(
            action=action,
            size=size,
            confidence=confidence,
            reason=signal.get('reason', '')
        )


class VectorBTBacktest:
    """
    VectorBT 回测引擎
    
    支持仓位管理和详细的性能分析
    """
    
    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_cash: float = 100000.0,
        fees: float = 0.001,
        max_position_size: float = 0.3  # 单个股票最大仓位 30%
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.fees = fees
        self.max_position_size = max_position_size
        
        self._price_data = {}
        self._signals = {}  # Signal 对象
        self._portfolios = {}
        
        self.logger = logging.getLogger(__name__)
    
    def load_data(self):
        """加载历史数据"""
        import yfinance as yf
        
        self.logger.info(f"Loading data for {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.start_date, end=self.end_date)
                
                if df.empty:
                    self.logger.warning(f"⚠️  No data for {symbol}")
                    continue
                
                self._price_data[symbol] = df
                self.logger.info(f"✅ Loaded {symbol}: {len(df)} bars")
            except Exception as e:
                self.logger.error(f"❌ Failed to load {symbol}: {e}")
        
        return self._price_data
    
    def run_backtest_with_sizing(
        self,
        signals: Dict[str, List[Signal]]
    ) -> Dict[str, vbt.Portfolio]:
        """
        运行支持仓位大小的回测
        
        Args:
            signals: Dict[symbol, List[Signal]]
        
        Returns:
            Dict[symbol, Portfolio]
        """
        self.logger.info("Running enhanced backtest with position sizing...")
        
        self._portfolios = {}
        
        for symbol in self.symbols:
            if symbol not in signals or symbol not in self._price_data:
                continue
            
            df = self._price_data[symbol]
            signal_list = signals[symbol]
            
            # 转换 Signal 为 VectorBT 格式
            entries, exits, size_array = self._convert_signals_to_vectorbt(
                signal_list,
                len(df)
            )
            
            # 运行回测
            try:
                # VectorBT 支持的 size_type:
                # 'amount' - 股票数量
                # 'value' - 金额
                # 'percent' - 当前资金的百分比
                portfolio = vbt.Portfolio.from_signals(
                    close=df['Close'],
                    entries=entries,
                    exits=exits,
                    size=size_array,  # 动态仓位大小
                    size_type='percent',  # 使用百分比方式
                    init_cash=self.initial_cash,
                    fees=self.fees,
                    freq='1D'
                )
                
                self._portfolios[symbol] = portfolio
                
                # 打印详细统计
                stats = self._get_detailed_stats(portfolio, symbol)
                self._log_stats(symbol, stats)
                
            except Exception as e:
                self.logger.error(f"Failed to run backtest for {symbol}: {e}")
        
        return self._portfolios
    
    def _convert_signals_to_vectorbt(
        self,
        signals: List[Signal],
        length: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        转换 Signal 为 VectorBT 格式
        
        Returns:
            (entries, exits, sizes)
        """
        entries = []
        exits = []
        sizes = []
        
        for signal in signals:
            if signal.action == 'BUY':
                entries.append(True)
                exits.append(False)
                # 转换为百分比 (0.1 -> 10%)
                sizes.append(signal.size * 100)
            elif signal.action == 'SELL':
                entries.append(False)
                exits.append(True)
                sizes.append(0.0)
            else:  # HOLD
                entries.append(False)
                exits.append(False)
                sizes.append(0.0)
        
        # 确保长度匹配
        while len(entries) < length:
            entries.append(False)
            exits.append(False)
            sizes.append(0.0)
        
        return (
            pd.Series(entries[:length]),
            pd.Series(exits[:length]),
            pd.Series(sizes[:length])
        )
    
    def _get_detailed_stats(
        self,
        portfolio: vbt.Portfolio,
        symbol: str
    ) -> Dict:
        """获取详细的回测统计"""
        try:
            # VectorBT返回Series，需要提取标量值
            def extract_value(val):
                """提取标量值"""
                if isinstance(val, pd.Series):
                    return val.iloc[0] if len(val) > 0 else 0.0
                return val
            
            # 使用stats()方法获取完整统计
            full_stats = portfolio.stats()
            
            stats = {
                'symbol': symbol,
                'initial_cash': self.initial_cash,
                'final_value': extract_value(portfolio.final_value()),
                'total_return': extract_value(portfolio.total_return()),
                'total_trades': extract_value(full_stats.get('Total Trades', 0)),
                'win_rate': extract_value(full_stats.get('Win Rate [%]', 0)) / 100.0,
                'max_drawdown': extract_value(full_stats.get('Max Drawdown [%]', 0)) / 100.0,
                'sharpe_ratio': extract_value(full_stats.get('Sharpe Ratio', 0)),
                'total_fees': extract_value(full_stats.get('Total Fees Paid', 0)),
            }
        except Exception as e:
            self.logger.warning(f"Failed to compute some stats: {e}")
            stats = {
                'symbol': symbol,
                'initial_cash': self.initial_cash,
                'final_value': self.initial_cash,
                'total_return': 0.0,
                'total_trades': 0,
            }
        
        return stats
    
    def _log_stats(self, symbol: str, stats: Dict):
        """打印统计信息"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 {symbol} Performance:")
        self.logger.info(f"  Initial Cash: ${stats['initial_cash']:,.2f}")
        self.logger.info(f"  Final Value: ${stats['final_value']:,.2f}")
        self.logger.info(f"  Total Return: {stats['total_return']:.2%}")
        self.logger.info(f"  Total Trades: {stats['total_trades']}")
        
        if stats['total_trades'] > 0:
            self.logger.info(f"  Win Rate: {stats.get('win_rate', 0):.2%}")
            self.logger.info(f"  Max Drawdown: {stats.get('max_drawdown', 0):.2%}")
            self.logger.info(f"  Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")
            
            # 计算净利润 (Final Value - Initial Cash)
            net_pnl = stats['final_value'] - stats['initial_cash']
            self.logger.info(f"  Net PnL: ${net_pnl:,.2f}")
            self.logger.info(f"  Total Fees: ${stats.get('total_fees', 0):,.2f}")
        
        self.logger.info(f"{'='*60}\n")
    
    def get_portfolio(self, symbol: str) -> Optional[vbt.Portfolio]:
        """获取特定股票的 portfolio"""
        return self._portfolios.get(symbol)
    
    def generate_report(self, output_path: str):
        """生成HTML报告"""
        if not self._portfolios:
            self.logger.warning("No portfolios to report")
            return
        
        # 使用第一个 portfolio 生成报告
        symbol = list(self._portfolios.keys())[0]
        portfolio = self._portfolios[symbol]
        
        try:
            # VectorBT 可以生成详细的HTML报告
            fig = portfolio.plot()
            fig.write_html(output_path)
            self.logger.info(f"✅ Report saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")


def convert_signals(
    layered_signals: Dict[str, List[Dict]]
) -> Dict[str, List[Signal]]:
    """
    转换 LayeredStrategy 信号为 Signal
    
    Args:
        layered_signals: Dict[symbol, List[signal_dict]]
    
    Returns:
        Dict[symbol, List[Signal]]
    """
    enhanced = {}
    
    for symbol, signals in layered_signals.items():
        enhanced[symbol] = [
            Signal.from_layered_strategy_signal(sig)
            for sig in signals
        ]
    
    return enhanced
