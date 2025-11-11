"""
LLM Multi-Agent Backtest Engine
完整的LLM多Agent回测引擎

流程：
1. 收集历史市场数据
2. 每周一次，调用所有Agent进行分析（使用as_of_date保证无时间泄漏）
3. Coordinator基于所有Agent分析结果，通过LLM做出决策
4. 执行交易并计算收益
5. 统计最终结果
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from backend.agents.technical_agent import TechnicalAgent
from backend.agents.sentiment_agent import SentimentAgent
from backend.agents.news_agent import NewsAgent
from backend.agents.coordinator import WeeklyCoordinator
from backend.data_collectors.market_data import MarketDataCollector
from backend.data_collectors.news_collector import NewsCollector
from backend.data_collectors.sentiment_analyzer import SentimentAnalyzer
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class LLMBacktestEngine:
    """LLM多Agent回测引擎"""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 0.1%
        decision_frequency: str = 'weekly'  # weekly
    ):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission: 手续费率
            decision_frequency: 决策频率（目前只支持weekly）
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.decision_frequency = decision_frequency
        
        # 初始化Agents
        logger.info("初始化Agents...")
        self.technical_agent = TechnicalAgent()
        self.sentiment_agent = SentimentAgent()
        self.news_agent = NewsAgent()
        self.coordinator = WeeklyCoordinator()
        
        # 初始化数据收集器
        logger.info("初始化数据收集器...")
        self.market_collector = MarketDataCollector(tickers=["SPY", "QQQ", "^VIX"])
        self.news_collector = NewsCollector()  # 自动加载API key
        self.sentiment_collector = SentimentAnalyzer()
        
        # 回测状态
        self.cash = initial_capital
        self.positions = {}  # {symbol: shares}
        self.trades = []
        self.decisions = []
        self.portfolio_values = []
        
    def run(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            symbol: 交易标的
            start_date: 回测开始日期
            end_date: 回测结束日期
            lookback_days: 每次决策时回看的历史天数
        
        Returns:
            回测结果
        """
        logger.info("=" * 80)
        logger.info("开始LLM多Agent回测")
        logger.info("=" * 80)
        logger.info(f"标的: {symbol}")
        logger.info(f"期间: {start_date.date()} 到 {end_date.date()}")
        logger.info(f"初始资金: ${self.initial_capital:,.2f}")
        logger.info(f"决策频率: {self.decision_frequency}")
        
        # 下载完整市场数据（包含lookback）
        data_start = start_date - timedelta(days=lookback_days + 10)  # 多留10天buffer
        logger.info(f"下载市场数据: {data_start.date()} 到 {end_date.date()}")
        
        full_market_data = self.market_collector.collect(
            start_date=data_start,
            end_date=end_date
        )
        
        if symbol not in full_market_data:
            raise ValueError(f"未找到{symbol}的市场数据")
        
        # 转换为DataFrame便于处理
        ohlcv = full_market_data[symbol]['ohlcv']
        price_df = pd.DataFrame(ohlcv)
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        price_df.set_index('Date', inplace=True)
        
        logger.info(f"市场数据: {len(price_df)}条记录")
        
        # 生成决策时间点（每周一次）
        decision_dates = self._generate_decision_dates(start_date, end_date)
        logger.info(f"决策时间点: {len(decision_dates)}次")
        
        # 逐个决策点执行
        for i, decision_date in enumerate(decision_dates):
            logger.info("")
            logger.info("-" * 80)
            logger.info(f"决策点 {i+1}/{len(decision_dates)}: {decision_date.date()}")
            logger.info("-" * 80)
            
            # 收集截至决策日的数据
            analysis_start = decision_date - timedelta(days=lookback_days)
            analysis_end = decision_date
            
            logger.info(f"分析期间: {analysis_start.date()} 到 {analysis_end.date()}")
            
            # 获取该时间段的数据
            period_data = self._collect_period_data(
                symbol=symbol,
                analysis_start=analysis_start,
                analysis_end=analysis_end,
                decision_date=decision_date
            )
            
            # 各Agent分析（传入as_of_date保证无时间泄漏）
            logger.info("运行Technical Agent...")
            technical_result = self.technical_agent.analyze(
                period_data['market_data'],
                as_of_date=decision_date
            )
            
            logger.info("运行Sentiment Agent...")
            sentiment_result = self.sentiment_agent.analyze(
                period_data['sentiment_data'],
                as_of_date=decision_date
            )
            
            logger.info("运行News Agent...")
            news_result = self.news_agent.analyze(
                period_data['news_data'],
                as_of_date=decision_date
            )
            
            # Coordinator综合决策（调用LLM）
            logger.info("运行Coordinator (LLM决策)...")
            coordinator_input = {
                'analysis_start_date': analysis_start.strftime('%Y-%m-%d'),
                'analysis_end_date': analysis_end.strftime('%Y-%m-%d'),
                'forecast_start_date': decision_date.strftime('%Y-%m-%d'),
                'forecast_end_date': (decision_date + timedelta(days=7)).strftime('%Y-%m-%d'),
                'lookback_days': lookback_days,
                'forecast_days': 7,
                'market_data': period_data['market_data'],
                'technical_analysis': technical_result,
                'sentiment_analysis': sentiment_result,
                'news_analysis': news_result,
                'current_portfolio': self._get_portfolio_snapshot(symbol, price_df, decision_date),
                'last_period_pnl': self._calculate_last_period_pnl(),
                'decision_history': self.decisions[-3:] if len(self.decisions) >= 3 else self.decisions
            }
            
            decision = self.coordinator.analyze(coordinator_input, as_of_date=decision_date)
            
            logger.info(f"决策: {decision.get('recommended_strategy', 'N/A')}")
            logger.info(f"信心: {decision.get('confidence', 0):.2f}")
            logger.info(f"推理: {decision.get('reasoning', 'N/A')[:100]}...")
            
            # 执行决策
            execution_result = self._execute_decision(
                symbol=symbol,
                decision=decision,
                price_df=price_df,
                decision_date=decision_date
            )
            
            # 记录决策和交易
            self.decisions.append({
                'date': decision_date.strftime('%Y-%m-%d'),
                'decision': decision,
                'execution': execution_result
            })
            
            # 更新组合价值
            portfolio_value = self._calculate_portfolio_value(symbol, price_df, decision_date)
            self.portfolio_values.append({
                'date': decision_date.strftime('%Y-%m-%d'),
                'value': portfolio_value,
                'cash': self.cash,
                'positions': dict(self.positions)
            })
            
            logger.info(f"组合价值: ${portfolio_value:,.2f}")
            logger.info(f"现金: ${self.cash:,.2f}")
            logger.info(f"持仓: {self.positions}")
        
        # 计算最终结果
        results = self._calculate_results(symbol, price_df, start_date, end_date)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("回测完成")
        logger.info("=" * 80)
        
        return results
    
    def _generate_decision_dates(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """生成决策时间点（每周）"""
        dates = []
        current = start_date
        
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=7)
        
        return dates
    
    def _collect_period_data(
        self,
        symbol: str,
        analysis_start: datetime,
        analysis_end: datetime,
        decision_date: datetime
    ) -> Dict[str, Any]:
        """收集特定时间段的数据（模拟真实环境）"""
        
        # 市场数据
        market_data = self.market_collector.collect(
            start_date=analysis_start,
            end_date=analysis_end
        )
        
        # 新闻数据
        news_data = self.news_collector.collect(
            start_date=analysis_start,
            end_date=analysis_end,
            symbol=symbol
        )
        
        # 情绪数据
        sentiment_data = self.sentiment_collector.collect(
            start_date=analysis_start,
            end_date=analysis_end
        )
        
        return {
            'market_data': market_data,
            'news_data': news_data,
            'sentiment_data': sentiment_data
        }
    
    def _get_portfolio_snapshot(self, symbol: str, price_df: pd.DataFrame, date: datetime) -> Dict[str, Any]:
        """获取当前组合快照"""
        current_price = self._get_price_at_date(price_df, date)
        
        holdings = {}
        if symbol in self.positions and self.positions[symbol] > 0:
            holdings[symbol] = {
                'shares': self.positions[symbol],
                'current_price': current_price,
                'market_value': self.positions[symbol] * current_price
            }
        
        total_value = self.cash + sum(h['market_value'] for h in holdings.values())
        
        return {
            'cash': self.cash,
            'holdings': holdings,
            'total_value': total_value
        }
    
    def _calculate_last_period_pnl(self) -> float:
        """计算上期盈亏"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        current = self.portfolio_values[-1]['value']
        previous = self.portfolio_values[-2]['value']
        return current - previous
    
    def _execute_decision(
        self,
        symbol: str,
        decision: Dict[str, Any],
        price_df: pd.DataFrame,
        decision_date: datetime
    ) -> Dict[str, Any]:
        """执行决策 - 修复：正确识别交易信号"""
        
        strategy = decision.get('recommended_strategy', '').lower()
        current_price = self._get_price_at_date(price_df, decision_date)
        
        if current_price is None:
            logger.warning(f"无法获取{decision_date.date()}的价格")
            return {'action': 'none', 'reason': 'no_price'}
        
        current_position = self.positions.get(symbol, 0)
        
        execution = {
            'date': decision_date.strftime('%Y-%m-%d'),
            'price': current_price,
            'action': 'none',
            'shares': 0,
            'cost': 0
        }
        
        # 修复：grid_trading, momentum, mean_reversion都是买入信号
        if strategy in ['grid_trading', 'momentum', 'mean_reversion', 'double_ema_channel', 'buy_and_hold']:
            if current_position == 0:
                # 买入（使用50%资金）
                available_cash = self.cash * 0.5
                shares = int(available_cash / (current_price * (1 + self.commission)))
                
                if shares > 0:
                    cost = shares * current_price * (1 + self.commission)
                    self.cash -= cost
                    self.positions[symbol] = self.positions.get(symbol, 0) + shares
                    
                    execution['action'] = 'buy'
                    execution['shares'] = shares
                    execution['cost'] = cost
                    
                    self.trades.append(execution.copy())
                    logger.info(f"✓ 买入 {shares} 股 @ ${current_price:.2f}, 成本: ${cost:.2f}")
            else:
                logger.info(f"已持仓 {current_position} 股，跳过买入")
        
        elif strategy == 'hold':
            if current_position > 0:
                # hold策略：如果有持仓，考虑卖出
                logger.info("策略建议持有现金，卖出现有持仓")
                shares = current_position
                proceeds = shares * current_price * (1 - self.commission)
                self.cash += proceeds
                self.positions[symbol] = 0
                
                execution['action'] = 'sell'
                execution['shares'] = -shares
                execution['cost'] = -proceeds
                
                self.trades.append(execution.copy())
                logger.info(f"✓ 卖出 {shares} 股 @ ${current_price:.2f}, 收入: ${proceeds:.2f}")
            else:
                execution['action'] = 'hold'
                logger.info("持有现金")
        
        return execution
    
    def _get_price_at_date(self, price_df: pd.DataFrame, date: datetime) -> float:
        """获取指定日期的价格（收盘价）"""
        try:
            # 找到最接近的日期（处理周末/假日）
            nearest_date = price_df.index.asof(date)
            if pd.isna(nearest_date):
                return None
            return float(price_df.loc[nearest_date, 'Close'])
        except Exception as e:
            logger.error(f"获取价格失败: {e}")
            return None
    
    def _calculate_portfolio_value(self, symbol: str, price_df: pd.DataFrame, date: datetime) -> float:
        """计算组合价值"""
        cash = self.cash
        position_value = 0
        
        if symbol in self.positions and self.positions[symbol] > 0:
            price = self._get_price_at_date(price_df, date)
            if price:
                position_value = self.positions[symbol] * price
        
        return cash + position_value
    
    def _calculate_results(
        self,
        symbol: str,
        price_df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """计算回测结果"""
        
        # 最终组合价值
        final_value = self.portfolio_values[-1]['value'] if self.portfolio_values else self.initial_capital
        
        # 收益相关
        total_return = (final_value - self.initial_capital) / self.initial_capital
        total_pnl = final_value - self.initial_capital
        
        # Buy & Hold基准
        start_price = self._get_price_at_date(price_df, start_date)
        end_price = self._get_price_at_date(price_df, end_date)
        bh_return = (end_price - start_price) / start_price if start_price else 0
        
        # 计算年化收益率
        days = (end_date - start_date).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 计算最大回撤
        values = [pv['value'] for pv in self.portfolio_values]
        max_drawdown = self._calculate_max_drawdown(values)
        
        # 胜率
        winning_trades = sum(1 for t in self.trades if t['action'] == 'sell' and t['cost'] < 0)
        total_trades = len([t for t in self.trades if t['action'] in ['buy', 'sell']])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        results = {
            'summary': {
                'symbol': symbol,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'days': days,
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'total_pnl': total_pnl,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': self._calculate_sharpe_ratio(values),
                'total_trades': total_trades,
                'win_rate': win_rate,
                'benchmark_return': bh_return,
                'alpha': total_return - bh_return
            },
            'trades': self.trades,
            'decisions': self.decisions,
            'portfolio_values': self.portfolio_values
        }
        
        # 打印汇总
        logger.info("")
        logger.info("回测结果汇总:")
        logger.info(f"  初始资金: ${self.initial_capital:,.2f}")
        logger.info(f"  最终价值: ${final_value:,.2f}")
        logger.info(f"  总收益: ${total_pnl:,.2f} ({total_return*100:.2f}%)")
        logger.info(f"  年化收益: {annual_return*100:.2f}%")
        logger.info(f"  最大回撤: {max_drawdown*100:.2f}%")
        logger.info(f"  夏普比率: {results['summary']['sharpe_ratio']:.2f}")
        logger.info(f"  交易次数: {total_trades}")
        logger.info(f"  胜率: {win_rate*100:.1f}%")
        logger.info(f"  基准收益: {bh_return*100:.2f}%")
        logger.info(f"  Alpha: {(total_return - bh_return)*100:.2f}%")
        
        return results
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """计算最大回撤"""
        if not values:
            return 0.0
        
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, values: List[float], risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(values) < 2:
            return 0.0
        
        # 计算周度收益率
        returns = []
        for i in range(1, len(values)):
            ret = (values[i] - values[i-1]) / values[i-1]
            returns.append(ret)
        
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # 年化（假设每周一次决策，52周/年）
        annual_mean = mean_return * 52
        annual_std = std_return * np.sqrt(52)
        
        sharpe = (annual_mean - risk_free_rate) / annual_std
        return sharpe
