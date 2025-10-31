"""
VectorBT 回测引擎

提供基于 VectorBT 的高性能回测功能
"""

import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import asyncio
import logging

from Agents.orchestration import MetaAgent


class VectorBTBacktest:
    """
    VectorBT 回测引擎
    
    功能：
    - 批量信号预计算（避免实时 LLM 调用）
    - 多股票回测
    - 性能分析和报告生成
    """
    
    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_cash: float = 100000.0,
        fees: float = 0.001  # 0.1% 手续费
    ):
        """
        初始化回测引擎
        
        Args:
            symbols: 股票代码列表，如 ['AAPL', 'MSFT']
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            initial_cash: 初始资金
            fees: 交易手续费比例
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.fees = fees
        
        self.logger = logging.getLogger(__name__)
        
        # 数据缓存
        self._price_data = None
        self._signals = None
        self._portfolios = None
        
        # Meta Agent (用于生成信号)
        self.meta_agent = None
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        加载股票数据
        
        Returns:
            Dict[symbol, DataFrame]: 每个股票的 OHLCV 数据
        """
        self.logger.info(f"Loading data for {len(self.symbols)} symbols...")
        
        self._price_data = {}
        
        for symbol in self.symbols:
            try:
                # 使用 yfinance 下载数据
                data = vbt.YFData.download(
                    symbol,
                    start=self.start_date,
                    end=self.end_date
                )
                
                self._price_data[symbol] = data.get()
                self.logger.info(f"✅ Loaded {symbol}: {len(data.get())} bars")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to load {symbol}: {e}")
                continue
        
        return self._price_data
    
    async def precompute_signals(
        self,
        strategy_func=None,
        use_meta_agent: bool = True,
        progress_callback=None
    ) -> Dict[str, pd.Series]:
        """
        预计算所有交易信号
        
        Args:
            strategy_func: 自定义策略函数 (symbol, date, price) -> signal
            use_meta_agent: 是否使用 MetaAgent 生成信号
            progress_callback: 进度回调函数 (symbol, current, total, message)
        
        Returns:
            Dict[symbol, Series]: 每个股票的信号序列 (1=BUY, 0=SELL/HOLD)
        """
        self.logger.info("Precomputing trading signals...")
        
        if not self._price_data:
            self.load_data()
        
        # 初始化 Meta Agent
        if use_meta_agent and self.meta_agent is None:
            self.meta_agent = MetaAgent()
            if progress_callback:
                progress_callback(None, 0, 0, "初始化 Meta Agent...")
        
        self._signals = {}
        
        for symbol in self.symbols:
            if symbol not in self._price_data:
                continue
            
            df = self._price_data[symbol]
            close_prices = df['Close']
            
            self.logger.info(f"Computing signals for {symbol} ({len(close_prices)} days)...")
            if progress_callback:
                progress_callback(symbol, 0, len(close_prices), f"开始分析 {symbol}")
            
            signals = []
            
            for idx, (date, price) in enumerate(close_prices.items()):
                # 显示进度
                if (idx + 1) % 10 == 0 or idx == 0:
                    self.logger.info(f"  {symbol}: {idx + 1}/{len(close_prices)} days")
                    if progress_callback:
                        progress_callback(
                            symbol, 
                            idx + 1, 
                            len(close_prices), 
                            f"分析 {symbol} ({date.strftime('%Y-%m-%d')})"
                        )
                
                try:
                    if use_meta_agent:
                        # 使用 MetaAgent 生成信号
                        signal = await self._get_meta_agent_signal(symbol, date, price)
                    elif strategy_func:
                        # 使用自定义策略
                        signal = strategy_func(symbol, date, price, df.loc[:date])
                    else:
                        # 默认：简单移动平均策略
                        signal = self._simple_ma_strategy(df.loc[:date])
                    
                    signals.append(signal)
                
                except Exception as e:
                    self.logger.warning(f"Signal generation failed for {symbol} on {date}: {e}")
                    signals.append(0)  # 默认 HOLD
            
            # 转换为 pandas Series
            self._signals[symbol] = pd.Series(signals, index=close_prices.index)
            self.logger.info(f"✅ {symbol}: Generated {sum(signals)} BUY signals")
        
        return self._signals
    
    async def _get_meta_agent_signal(
        self,
        symbol: str,
        date: pd.Timestamp,
        price: float
    ) -> int:
        """
        使用 MetaAgent 生成交易信号
        
        Returns:
            1 = BUY, 0 = SELL/HOLD
        """
        decision = await self.meta_agent.analyze_and_decide(
            symbol=symbol,
            query=f"Analyze {symbol} on {date.date()} at price ${price:.2f}. Should I buy?"
        )
        
        return 1 if decision.action == "BUY" else 0
    
    def _simple_ma_strategy(self, historical_data: pd.DataFrame) -> int:
        """
        简单移动平均策略（备用）
        
        策略：当短期均线 > 长期均线时买入
        """
        if len(historical_data) < 50:
            return 0
        
        close = historical_data['Close']
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        
        return 1 if sma20 > sma50 else 0
    
    def run_backtest(
        self,
        signals: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, vbt.Portfolio]:
        """
        运行回测
        
        Args:
            signals: 预计算的信号，如果为 None 则使用已缓存的信号
        
        Returns:
            Dict[symbol, Portfolio]: 每个股票的回测结果
        """
        if signals is None:
            signals = self._signals
        
        if not signals:
            raise ValueError("No signals available. Run precompute_signals() first.")
        
        self.logger.info("Running backtest...")
        
        self._portfolios = {}
        
        for symbol in self.symbols:
            if symbol not in signals or symbol not in self._price_data:
                continue
            
            df = self._price_data[symbol]
            signal_series = signals[symbol]
            
            # 使用 VectorBT 运行回测
            portfolio = vbt.Portfolio.from_signals(
                close=df['Close'],
                entries=signal_series == 1,  # BUY signals
                exits=signal_series == 0,    # SELL/HOLD signals
                init_cash=self.initial_cash,
                fees=self.fees,
                freq='1D'
            )
            
            self._portfolios[symbol] = portfolio
            
            # 打印简单统计
            total_return = portfolio.total_return()
            self.logger.info(f"✅ {symbol}: Total Return = {total_return:.2%}")
        
        return self._portfolios
    
    def get_performance_stats(self, symbol: Optional[str] = None) -> Dict:
        """
        获取性能统计
        
        Args:
            symbol: 指定股票代码，如果为 None 则返回所有股票的统计
        
        Returns:
            性能指标字典
        """
        if symbol:
            if symbol not in self._portfolios:
                raise ValueError(f"No backtest results for {symbol}")
            
            portfolio = self._portfolios[symbol]
            return self._extract_stats(symbol, portfolio)
        else:
            # 返回所有股票的统计
            all_stats = {}
            for sym, portfolio in self._portfolios.items():
                all_stats[sym] = self._extract_stats(sym, portfolio)
            return all_stats
    
    def _extract_stats(self, symbol: str, portfolio: vbt.Portfolio) -> Dict:
        """提取详细统计指标"""
        stats = portfolio.stats()
        
        return {
            'symbol': symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_cash': self.initial_cash,
            
            # 收益指标
            'total_return': float(portfolio.total_return()),
            'total_return_pct': f"{portfolio.total_return():.2%}",
            'annualized_return': float(stats['Annualized Return [%]']) / 100 if 'Annualized Return [%]' in stats else None,
            
            # 风险指标
            'max_drawdown': float(stats['Max Gross Exposure [%]']) / 100 if 'Max Gross Exposure [%]' in stats else None,
            'sharpe_ratio': float(portfolio.sharpe_ratio()) if hasattr(portfolio, 'sharpe_ratio') else None,
            
            # 交易统计
            'total_trades': int(stats['Total Trades']) if 'Total Trades' in stats else 0,
            'win_rate': float(stats['Win Rate [%]']) / 100 if 'Win Rate [%]' in stats else None,
            
            # 最终值
            'final_value': float(portfolio.total_return() * self.initial_cash + self.initial_cash),
            'profit_loss': float(portfolio.total_return() * self.initial_cash),
            
            # 完整统计
            'full_stats': stats
        }
    
    def generate_report(self, output_dir: str = "Results") -> Dict[str, str]:
        """
        生成回测报告
        
        Args:
            output_dir: 输出目录
        
        Returns:
            报告文件路径字典
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        report_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for symbol, portfolio in self._portfolios.items():
            # 生成 HTML 报告
            report_path = f"{output_dir}/{symbol}_backtest_{timestamp}.html"
            
            try:
                # VectorBT 内置绘图 - 使用非交互式模式
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # 创建基本的性能图表（非 FigureWidget）
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Portfolio Value', 'Daily Returns'),
                    row_heights=[0.7, 0.3],
                    vertical_spacing=0.1
                )
                
                # 获取数据
                portfolio_value = portfolio.value()
                returns = portfolio.returns()
                
                # 添加 Portfolio Value 曲线
                fig.add_trace(
                    go.Scatter(
                        x=portfolio_value.index,
                        y=portfolio_value.values,
                        name='Portfolio Value',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
                
                # 添加 Returns 柱状图
                fig.add_trace(
                    go.Bar(
                        x=returns.index,
                        y=returns.values,
                        name='Daily Returns',
                        marker=dict(
                            color=returns.values,
                            colorscale='RdYlGn',
                            showscale=False
                        )
                    ),
                    row=2, col=1
                )
                
                # 设置布局
                fig.update_layout(
                    title=f'{symbol} Backtest Report',
                    height=800,
                    showlegend=True,
                    template='plotly_white'
                )
                
                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Value ($)", row=1, col=1)
                fig.update_yaxes(title_text="Returns", row=2, col=1)
                
                # 保存为静态 HTML
                fig.write_html(report_path)
                
                report_files[symbol] = report_path
                self.logger.info(f"📊 Report saved: {report_path}")
                
            except Exception as e:
                self.logger.warning(f"Could not generate HTML report for {symbol}: {e}")
                # 即使图表失败，也继续生成 JSON 报告
                report_files[symbol] = None
        
        # 生成汇总 JSON
        summary_path = f"{output_dir}/backtest_summary_{timestamp}.json"
        summary = {
            'timestamp': timestamp,
            'config': {
                'symbols': self.symbols,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_cash': self.initial_cash,
                'fees': self.fees
            },
            'results': self.get_performance_stats()
        }
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        report_files['summary'] = summary_path
        self.logger.info(f"📄 Summary saved: {summary_path}")
        
        return report_files
    
    def plot(self, symbol: Optional[str] = None):
        """
        绘制回测结果
        
        Args:
            symbol: 指定股票代码，如果为 None 则绘制所有
        """
        if symbol:
            if symbol not in self._portfolios:
                raise ValueError(f"No backtest results for {symbol}")
            
            self._portfolios[symbol].plot().show()
        else:
            # 绘制所有股票
            for sym, portfolio in self._portfolios.items():
                print(f"\n=== {sym} ===")
                portfolio.plot().show()


# 便捷函数
async def quick_backtest(
    symbols: List[str],
    start_date: str,
    end_date: str,
    initial_cash: float = 100000,
    fees: float = 0.001,
    use_meta_agent: bool = True
) -> VectorBTBacktest:
    """
    快速回测（一步完成）
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        initial_cash: 初始资金
        fees: 手续费率
        use_meta_agent: 是否使用 MetaAgent
    
    Returns:
        完成回测的 VectorBTBacktest 对象
    """
    backtest = VectorBTBacktest(
        symbols, 
        start_date, 
        end_date,
        initial_cash=initial_cash,
        fees=fees
    )
    
    # 加载数据
    backtest.load_data()
    
    # 预计算信号
    await backtest.precompute_signals(use_meta_agent=use_meta_agent)
    
    # 运行回测
    backtest.run_backtest()
    
    return backtest
