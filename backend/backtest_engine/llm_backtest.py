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
from math import ceil
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
from backend.config.config_loader import get_config
from backend.portfolio.portfolio_manager import PortfolioManager
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class LLMBacktestEngine:
    """LLM多Agent回测引擎"""
    
    def __init__(
        self,
        config_path: str = None, # 默认使用backend/config/config.yaml
    ):
        """
        初始化回测引擎
        
        Args:
            config_path: 配置文件路径，默认使用backend/config/config.yaml
        """
        # 加载配置
        self.config = get_config(config_path)
        
        # 使用参数或配置中的值
        self.initial_capital = self.config.system.initial_capital
        self.commission = self.config.system.commission
        
        # 初始化Agents
        logger.info("初始化Agents...")
        self.technical_agent = TechnicalAgent()
        self.sentiment_agent = SentimentAgent()
        self.news_agent = NewsAgent()
        
        # 准备coordinator配置
        coordinator_config = {
            'can_suggest_positions': self.config.llm.can_suggest_positions,
            'require_approval': self.config.llm.require_approval,
            'available_strategies': self.config.strategies.available,
            'default_strategy': self.config.strategies.default,
            'prompt_version': getattr(self.config.system, 'prompt_version', 'v1')  # 从config读取版本
        }
        self.coordinator = WeeklyCoordinator(config=coordinator_config)
        
        # 初始化数据收集器
        logger.info("初始化数据收集器...")
        # 使用配置中的tickers列表
        tickers = self.config.data_sources.market_data.tickers if hasattr(self.config.data_sources.market_data, 'tickers') else ["SPY", "QQQ", "^VIX"]
        self.market_collector = MarketDataCollector(tickers=tickers)
        self.news_collector = NewsCollector()  # 自动加载API key
        self.sentiment_collector = SentimentAnalyzer()
        
        # Portfolio Manager - 统一管理组合状态
        self.portfolio = PortfolioManager(initial_capital=self.initial_capital)
        
        # 策略实例缓存 {strategy_name: strategy_instance}
        self.strategy_instances = {}
        
    def run(
        self,
        symbol: str,
        start_date: datetime = None,
        end_date: datetime = None,
        lookback_days: int = None
    ) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            symbol: 交易标的
            start_date: 回测开始日期（None则使用config中的配置）
            end_date: 回测结束日期（None则使用config中的配置）
            lookback_days: 每次决策时回看的历史日历天数（None则使用config中的lookback_days * 5，确保有足够交易日）
        
        Returns:
            回测结果
        """
        # 使用参数或配置中的值
        if start_date is None:
            start_date = datetime.strptime(self.config.system.backtest_start, '%Y-%m-%d')
        if end_date is None:
            end_date = datetime.strptime(self.config.system.backtest_end, '%Y-%m-%d')
        if lookback_days is None:
            # 配置中的lookback_days是交易日数，转换为日历天数（大约需要乘以1.4）
            lookback_days = int(self.config.system.lookback_days * 1.5)
        logger.info("=" * 80)
        logger.info("开始LLM多Agent回测")
        logger.info("=" * 80)
        logger.info(f"标的: {symbol}")
        logger.info(f"期间: {start_date.date()} 到 {end_date.date()}")
        logger.info(f"初始资金: ${self.initial_capital:,.2f}")
        logger.info(f"决策频率: {self.config.system.forecast_days}天")
        logger.info(f"Lookback天数: {lookback_days}天（约{int(lookback_days * 5/7)}个交易日）")
        
        # 下载完整市场数据（回测期间 + lookback期间）
        # 使用日历天数确保有足够的交易日
        warming_start_date = start_date - timedelta(days=lookback_days)
        logger.info(f"下载市场数据: {warming_start_date.date()} 到 {end_date.date()}")
        
        full_market_data = self.market_collector.collect(
            start_date=warming_start_date,
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
            # 准备decision history（转换格式以适配prompts.format_decision_history）
            all_decisions = self.portfolio.get_decisions()
            recent_decisions = all_decisions[-3:] if len(all_decisions) >= 3 else all_decisions
            formatted_history = self._format_decision_history_for_prompt(recent_decisions)
            
            coordinator_input = {
                'analysis_start_date': analysis_start.strftime('%Y-%m-%d'),
                'analysis_end_date': analysis_end.strftime('%Y-%m-%d'),
                'forecast_start_date': decision_date.strftime('%Y-%m-%d'),
                'forecast_end_date': (decision_date + timedelta(days=self.config.system.forecast_days)).strftime('%Y-%m-%d'),
                'lookback_days': self.config.system.lookback_days,
                'forecast_days': self.config.system.forecast_days,
                'market_data': period_data['market_data'],
                'technical_analysis': technical_result,
                'sentiment_analysis': sentiment_result,
                'news_analysis': news_result,
                'current_portfolio': self._get_portfolio_snapshot(symbol, price_df, decision_date),
                'last_period_pnl': self._calculate_last_period_pnl(),
                'decision_history': formatted_history
            }
            
            decision = self.coordinator.analyze(coordinator_input, as_of_date=decision_date)
            
            logger.info(f"LLM决策: {decision.get('recommended_strategy', 'N/A')}")
            logger.info(f"信心: {decision.get('confidence', 0):.2f}")
            logger.info(f"推理: {decision.get('reasoning', 'N/A')[:100]}...")
            
            # 记录LLM决策
            self.portfolio.record_decision(
                date=decision_date.strftime('%Y-%m-%d'),
                decision=decision
            )
            
            # 执行策略：在forecast期间（下一周）每天运行策略
            next_decision_date = decision_dates[i+1] if i+1 < len(decision_dates) else end_date
            forecast_start = decision_date
            forecast_end = min(next_decision_date, end_date)
            
            logger.info(f"执行期间: {forecast_start.date()} 到 {forecast_end.date()}")
            
            # 获取forecast期间的所有交易日
            forecast_days = price_df[
                (price_df.index > forecast_start) & 
                (price_df.index <= forecast_end)
            ].index
            
            logger.info(f"  将在 {len(forecast_days)} 个交易日内每日运行策略")
            
            # 每天运行策略
            for day_idx, trading_day in enumerate(forecast_days, 1):
                logger.info(f"  Day {day_idx}: {trading_day.date()}")
                
                # 执行策略（策略每天检查买卖点）
                execution_result = self._execute_decision(
                    symbol=symbol,
                    decision=decision,
                    price_df=price_df,
                    decision_date=trading_day
                )
                
                # 记录每日执行
                self.portfolio.add_daily_execution({
                    'date': trading_day.strftime('%Y-%m-%d'),
                    'execution': execution_result
                })
                
                # 每天更新组合价值
                current_price = self._get_price_at_date(price_df, trading_day)
                self.portfolio.record_portfolio_value(
                    date=trading_day.strftime('%Y-%m-%d'),
                    current_prices={symbol: current_price} if current_price else {}
                )
                
                if execution_result['action'] != 'hold':
                    logger.info(f"    → {execution_result['action'].upper()}: {execution_result.get('shares', 0)} 股")
            
            # 打印本周汇总
            portfolio_history = self.portfolio.get_portfolio_history()
            if portfolio_history:
                portfolio_value = portfolio_history[-1]['value']
                logger.info(f"组合价值: ${portfolio_value:,.2f}")
                logger.info(f"现金: ${self.portfolio.cash:,.2f}")
                logger.info(f"持仓: {self.portfolio.get_all_positions()}")
            else:
                # 如果没有交易日，手动计算当前组合价值
                current_price = self._get_price_at_date(price_df, decision_date)
                portfolio_value = self.portfolio.get_portfolio_value({symbol: current_price} if current_price else {})
                logger.info(f"组合价值: ${portfolio_value:,.2f} (无交易日)")
                logger.info(f"现金: ${self.portfolio.cash:,.2f}")
                logger.info(f"持仓: {self.portfolio.get_all_positions()}")
        
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
            current += timedelta(days=self.config.system.forecast_days)
        
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
        return self.portfolio.get_portfolio_snapshot(symbol, current_price if current_price else 0.0)
    
    def _calculate_last_period_pnl(self) -> float:
        """计算上期盈亏"""
        return self.portfolio.calculate_last_period_pnl()
    
    def _format_decision_history_for_prompt(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将回测中的decision格式转换为prompts.format_decision_history期望的格式
        
        Args:
            decisions: 回测中存储的决策列表
        
        Returns:
            转换后的历史决策列表
        """
        formatted = []
        
        for i, dec in enumerate(decisions):
            # dec 现在是 Decision 对象，需要从其属性中提取
            if hasattr(dec, 'decision'):
                decision = dec.decision
            else:
                decision = dec.get('decision', {}) if isinstance(dec, dict) else {}
            
            # 提取期间信息
            analysis_period = decision.get('analysis_period', {})
            period_str = f"{analysis_period.get('start', 'N/A')} to {analysis_period.get('end', 'N/A')}"
            
            # 提取regime信息（v2格式）
            regime_info = decision.get('regime_classification', {})
            predicted_regime = regime_info.get('primary_regime', 'N/A')
            confidence = regime_info.get('confidence', 0.0)
            
            # 提取策略信息（v2格式）
            strategy_rec = decision.get('strategy_recommendation', {})
            strategy_used = strategy_rec.get('primary_strategy', decision.get('recommended_strategy', 'N/A'))
            
            # 计算outcome（如果有下一个决策，可以计算实际结果）
            outcome = {}
            if i < len(decisions) - 1:
                # 计算本期的实际表现
                # TODO: 这里需要从portfolio_values中计算实际的market_return和portfolio_return
                # 暂时使用占位值
                outcome = {
                    'market_return': 0.0,  # 需要从价格数据计算
                    'portfolio_return': 0.0,  # 需要从组合价值计算
                    'correct_regime': None  # 需要验证regime预测是否正确
                }
            
            formatted.append({
                'period': period_str,
                'predicted_regime': predicted_regime,
                'confidence': confidence,
                'strategy_used': strategy_used,
                'outcome': outcome
            })
        
        return formatted
    
    def _execute_decision(
        self,
        symbol: str,
        decision: Dict[str, Any],
        price_df: pd.DataFrame,
        decision_date: datetime
    ) -> Dict[str, Any]:
        """执行决策 - 调用对应策略的execute方法"""
        
        strategy_name = decision.get('recommended_strategy', 'unknown').lower()
        current_price = self._get_price_at_date(price_df, decision_date)
        
        if current_price is None:
            logger.warning(f"无法获取{decision_date.date()}的价格")
            return {'action': 'none', 'reason': 'no_price'}
        
        # 导入策略工厂
        from backend.strategies.strategy_factory import StrategyFactory
        
        # 获取或创建策略实例（复用同一个实例以保持状态）
        if strategy_name not in self.strategy_instances:
            try:
                self.strategy_instances[strategy_name] = StrategyFactory.create_strategy(strategy_name)
                logger.info(f"创建新策略实例: {strategy_name}")
            except ValueError as e:
                logger.error(f"无法创建策略 {strategy_name}: {e}")
                return {'action': 'none', 'reason': 'invalid_strategy'}
        
        strategy = self.strategy_instances[strategy_name]
        
        # 准备策略所需的市场数据
        # 获取策略所需的最小数据点数
        required_data_points = strategy.get_required_data_points() if hasattr(strategy, 'get_required_data_points') else 50
        
        # 传递截止到当前决策日期的所有历史数据（避免看到未来）
        # 从数据开始到决策日期的所有数据
        period_prices = price_df[price_df.index <= decision_date].copy()
        
        if period_prices.empty:
            logger.error(f"无法获取截止到 {decision_date.date()} 的价格数据")
            return {'action': 'none', 'reason': 'no_data'}
        
        # 检查数据点是否足够
        if len(period_prices) < required_data_points:
            logger.warning(
                f"策略 {strategy_name} 需要至少 {required_data_points} 个数据点，"
                f"但只有 {len(period_prices)} 个数据点"
            )
            # 如果数据不足，仍然传递给策略，让策略自己决定如何处理
            # 策略内部会返回hold信号
        
        # 同步策略的持仓状态（从 Portfolio Manager 读取）
        if hasattr(strategy, 'position'):
            strategy.position = 1 if self.portfolio.has_position(symbol) else 0
        if hasattr(strategy, 'entry_price') and strategy.position == 1:
            # 获取入场价格
            entry_price = self.portfolio.get_position_entry_price(symbol)
            if entry_price > 0:
                strategy.entry_price = entry_price
            else:
                # 如果 Portfolio Manager 中没有记录，尝试从交易记录获取
                last_buy_price = self.portfolio.get_last_buy_price()
                if last_buy_price:
                    strategy.entry_price = last_buy_price
        
        # 调用策略的generate_signals方法
        strategy_result = strategy.generate_signals(period_prices)
        
        if not strategy_result or 'action' not in strategy_result:
            logger.warning(f"策略 {strategy_name} 未生成有效信号")
            return {'action': 'none', 'reason': 'no_signal'}
        
        # 获取策略返回的action
        action = strategy_result['action'].lower()
        reason = strategy_result.get('reason', 'N/A')
        confidence = strategy_result.get('confidence', 0.0)
        
        logger.info(f"策略 {strategy_name} 决策: {action}")
        logger.info(f"理由: {reason}")
        logger.info(f"信心: {confidence:.2f}")

        # 根据策略结果执行交易
        return self._execute_trade_from_strategy_result(
            symbol=symbol,
            strategy_result=strategy_result,
            current_price=current_price,
            decision_date=decision_date,
            strategy_name=strategy_name  # 传递策略名称
        )
    
    def _execute_trade_from_strategy_result(
        self,
        symbol: str,
        strategy_result: Dict[str, Any],
        current_price: float,
        decision_date: datetime,
        strategy_name: str = None
    ) -> Dict[str, Any]:
        """
        根据策略信号执行交易
        
        Args:
            symbol: 交易标的
            strategy_result: 策略返回的信号
            current_price: 当前价格
            decision_date: 决策日期
            strategy_name: 策略名称
        
        Returns:
            执行结果字典
        """
        action = strategy_result.get('action', 'hold').lower()
        current_position = self.portfolio.get_position_shares(symbol)
        
        # 初始化执行记录
        execution = {
            'date': decision_date.strftime('%Y-%m-%d'),
            'price': current_price,
            'action': 'hold',
            'shares': 0,
            'cost': 0,
            'strategy': strategy_name or 'unknown'
        }
        
        # 根据信号类型执行交易
        if action == 'buy':
            self._execute_buy(symbol, strategy_result, current_price, current_position, execution, decision_date)
        elif action == 'sell':
            self._execute_sell(symbol, strategy_result, current_price, current_position, execution)
        else:
            logger.info("策略信号: 持有")
        
        # 通知策略更新其内部状态（如果策略有execute_trade方法）
        if execution['action'] in ['buy', 'sell'] and strategy_name in self.strategy_instances:
            strategy = self.strategy_instances[strategy_name]
            if hasattr(strategy, 'execute_trade'):
                strategy.execute_trade(execution['action'], current_price)
        
        return execution
    
    def _execute_buy(
        self,
        symbol: str,
        strategy_result: Dict[str, Any],
        current_price: float,
        current_position: int,
        execution: Dict[str, Any],
        current_date: datetime = None
    ) -> None:
        """
        执行买入操作
        
        Args:
            symbol: 交易标的
            strategy_result: 策略信号
            current_price: 当前价格
            current_position: 当前持仓
            execution: 执行记录（会被修改）
            current_date: 当前日期（用于获取VIX）
        """
        # 检查现金充足性
        if self.portfolio.cash <= 0:
            logger.info(f"现金不足 (${self.portfolio.cash:.2f})，跳过买入")
            return
        
        # 风控参数（基础值）
        BASE_MAX_POSITION_PERCENT = 0.5  # 单股基础最大持仓比例
        
        # 检查市场特殊情况，动态调整风控限制
        max_position_percent, is_exceptional = self._get_dynamic_position_limit(
            symbol, 
            strategy_result,
            BASE_MAX_POSITION_PERCENT,
            current_date
        )
        
        # 计算当前资产和持仓比例
        total_assets = self.portfolio.cash + current_position * current_price
        current_position_value = current_position * current_price
        current_position_ratio = current_position_value / total_assets if total_assets > 0 else 0
        
        # 检查是否超过动态持仓限制
        if current_position_ratio >= max_position_percent:
            if is_exceptional:
                logger.info(
                    f"持仓比例 {current_position_ratio:.1%} 已达特殊情况上限 {max_position_percent:.1%}，"
                    f"跳过买入"
                )
            else:
                logger.info(f"持仓比例已达上限 {current_position_ratio:.1%}，跳过买入")
            return
        
        # 根据策略confidence计算买入比例
        # confidence 控制买入力度：0.5 = 买入到一半的max_position_percent
        # max_position_percent 是动态上限：正常50%，特殊情况可到60-90%
        confidence = strategy_result.get('confidence', 0.5)
        buy_percent = confidence * max_position_percent
        
        # 计算目标持仓价值和可用现金
        target_position_value = total_assets * buy_percent
        available_for_purchase = target_position_value - current_position_value
        available_cash = min(available_for_purchase, self.portfolio.cash) if available_for_purchase > 0 else 0
        
        if available_cash <= 0:
            logger.info("可用资金不足，跳过买入")
            return
        
        # 计算买入股数
        shares = int(available_cash / (current_price * (1 + self.commission)))
        
        if shares <= 0:
            logger.info(f"可用资金不足以买入1股 (${available_cash:.2f})，跳过买入")
            return
        
        # 使用 Portfolio Manager 执行买入（原子性保证）
        success = self.portfolio.execute_buy(
            symbol=symbol,
            shares=shares,
            price=current_price,
            commission=self.commission,
            date=execution['date'],
            strategy=execution['strategy']
        )
        
        if not success:
            logger.warning("买入失败")
            return
        
        # 更新执行记录
        cost = shares * current_price * (1 + self.commission)
        execution['action'] = 'buy'
        execution['shares'] = shares
        execution['cost'] = cost
        execution['cash'] = self.portfolio.cash
    
    def _get_dynamic_position_limit(
        self,
        symbol: str,
        strategy_result: Dict[str, Any],
        base_limit: float,
        current_date: datetime = None
    ) -> tuple[float, bool]:
        """
        根据市场情况动态调整持仓上限
        
        Args:
            symbol: 交易标的
            strategy_result: 策略信号（可能包含市场状态信息）
            base_limit: 基础持仓上限
            current_date: 当前日期（用于获取VIX数据）
        
        Returns:
            (调整后的上限, 是否为例外情况)
        """
        # 获取最新的市场数据（如果有）
        portfolio_history = self.portfolio.get_portfolio_history()
        if not portfolio_history:
            return base_limit, False
        
        # 直接从市场数据获取VIX
        vix_level = None
        
        if current_date:
            try:
                # 从market_collector获取VIX数据
                # 获取当前日期前后一周的数据以确保有数据
                vix_start = current_date - timedelta(days=7)
                vix_end = current_date
                
                vix_data = self.market_collector.collect(
                    start_date=vix_start,
                    end_date=vix_end
                )
                
                if '^VIX' in vix_data and vix_data['^VIX']['ohlcv']:
                    # 获取最新的VIX收盘价
                    latest_vix = vix_data['^VIX']['ohlcv'][-1]
                    vix_level = latest_vix['Close']
                    
            except Exception as e:
                logger.debug(f"获取VIX数据失败: {e}")
                vix_level = None
        
        # 检查策略信号中是否有特殊标记
        is_crash_protection = strategy_result.get('crash_protection', False)
        is_extreme_opportunity = strategy_result.get('extreme_opportunity', False)
        
        # 规则1：VIX极端高位（恐慌性抛售）- 允许更大仓位抄底
        if vix_level and vix_level > 40:  # VIX > 40 表示极端恐慌
            adjusted_limit = min(base_limit + 0.2, 0.9)  # 最多放宽到90%
            logger.info(f"⚠️  检测到VIX极端高位 ({vix_level:.1f})，放宽持仓上限至 {adjusted_limit:.1%}")
            return adjusted_limit, True
        
        elif vix_level and vix_level > 30:  # VIX > 30 表示高度恐慌
            adjusted_limit = min(base_limit + 0.1, 0.8)  # 放宽到80%
            logger.info(f"⚠️  检测到VIX高位 ({vix_level:.1f})，放宽持仓上限至 {adjusted_limit:.1%}")
            return adjusted_limit, True
        
        # 规则2：策略明确标记为崩盘保护或极端机会
        if is_crash_protection or is_extreme_opportunity:
            adjusted_limit = min(base_limit + 0.15, 0.85)
            reason = "崩盘保护" if is_crash_protection else "极端机会"
            logger.info(f"⚠️  策略标记为{reason}，放宽持仓上限至 {adjusted_limit:.1%}")
            return adjusted_limit, True
        
        # 规则3：策略confidence极高（>0.85）且持仓比例较低
        confidence = strategy_result.get('confidence', 0.5)
        if confidence > 0.85:
            adjusted_limit = min(base_limit + 0.1, 0.8)
            logger.info(f"📈 策略信心极高 ({confidence:.2f})，放宽持仓上限至 {adjusted_limit:.1%}")
            return adjusted_limit, True
        
        # 默认：使用基础限制
        return base_limit, False
    
    def _execute_sell(
        self,
        symbol: str,
        strategy_result: Dict[str, Any],
        current_price: float,
        current_position: int,
        execution: Dict[str, Any]
    ) -> None:
        """
        执行卖出操作
        
        Args:
            symbol: 交易标的
            strategy_result: 策略信号
            current_price: 当前价格
            current_position: 当前持仓
            execution: 执行记录（会被修改）
        """
        # 检查是否有持仓
        if current_position <= 0:
            logger.info("无持仓，跳过卖出")
            return
        
        # 根据confidence决定卖出数量
        confidence = strategy_result.get('confidence', 1.0)
        shares = ceil(current_position * confidence)
        shares = min(shares, current_position)  # 确保不超过持仓
        
        # 使用 Portfolio Manager 执行卖出（原子性保证）
        success = self.portfolio.execute_sell(
            symbol=symbol,
            shares=shares,
            price=current_price,
            commission=self.commission,
            date=execution['date'],
            strategy=execution['strategy']
        )
        
        if not success:
            logger.warning("卖出失败")
            return
        
        # 获取最后一笔交易记录（刚刚执行的卖出）
        trades = self.portfolio.get_trades()
        last_trade = trades[-1] if trades else None
        
        # 更新执行记录
        execution['action'] = 'sell'
        execution['shares'] = -shares
        execution['cost'] = last_trade.cost if last_trade else 0
        execution['proceeds'] = -last_trade.cost if last_trade else 0
        execution['profit'] = last_trade.profit if last_trade else 0
        execution['cash'] = self.portfolio.cash
    
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
        price = self._get_price_at_date(price_df, date)
        return self.portfolio.get_portfolio_value({symbol: price} if price else {})
    
    def _calculate_results(
        self,
        symbol: str,
        price_df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """计算回测结果"""
        
        # 最终组合价值
        portfolio_history = self.portfolio.get_portfolio_history()
        final_value = portfolio_history[-1]['value'] if portfolio_history else self.initial_capital
        
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
        values = [pv['value'] for pv in portfolio_history]
        max_drawdown = self._calculate_max_drawdown(values)
        
        # 胜率（只统计卖出交易）
        all_trades = self.portfolio.get_trades()
        sell_trades = [t for t in all_trades if t.action == 'sell' and t.profit is not None]
        winning_trades = sum(1 for t in sell_trades if t.profit > 0)
        total_sell_trades = len(sell_trades)
        win_rate = winning_trades / total_sell_trades if total_sell_trades > 0 else 0
        
        # 总交易次数（买入+卖出）
        total_trades = len([t for t in all_trades if t.action in ['buy', 'sell']])
        
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
            'trades': [t.to_dict() for t in self.portfolio.get_trades()],
            'decisions': [d.to_dict() for d in self.portfolio.get_decisions()],
            'portfolio_values': self.portfolio.get_portfolio_history()
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
        
        # 策略表现分析
        # self._print_strategy_performance(results)
        
        return results
    
    def _print_strategy_performance(self, results: Dict[str, Any]):
        """打印策略表现分析"""
        try:
            from backend.utils.strategy_analyzer import StrategyPerformanceAnalyzer
            
            trades = results.get('trades', [])
            if not trades:
                return
            
            analyzer = StrategyPerformanceAnalyzer(trades)
            report = analyzer.get_summary_report()
            
            logger.info("")
            logger.info(report)
            
        except Exception as e:
            logger.warning(f"策略表现分析失败: {e}")
    
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
