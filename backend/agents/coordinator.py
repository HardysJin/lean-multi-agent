"""
Coordinator - Main Decision Agent
主决策Agent，综合所有分析并通过LLM做出周度交易决策
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_agent import BaseAgent
from backend.utils.llm_client import LLMClient, create_llm_client
from backend.config import prompts
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class WeeklyCoordinator(BaseAgent):
    """
    周度决策协调器
    
    职责：
    1. 接收各Agent的分析结果
    2. 调用LLM进行综合决策
    3. 输出结构化决策JSON
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化Coordinator
        
        Args:
            llm_client: LLM客户端，如果为None则自动创建
            config: 配置字典
        """
        super().__init__("WeeklyCoordinator", config)
        
        # LLM客户端
        self.llm_client = llm_client or create_llm_client()
        
        # 决策配置
        self.can_suggest_positions = self.config.get('can_suggest_positions', True)
        self.require_approval = self.config.get('require_approval', True)
    
    def analyze(self, data: Dict[str, Any], as_of_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        综合分析并生成决策
        
        Args:
            data: 包含以下字段的字典
                - analysis_start_date: 分析开始日期
                - analysis_end_date: 分析结束日期（数据截止日期）
                - forecast_start_date: 预测开始日期
                - forecast_end_date: 预测结束日期
                - lookback_days: 回看天数
                - forecast_days: 预测天数
                - market_data: 市场数据汇总
                - technical_analysis: 技术分析结果
                - sentiment_analysis: 情绪分析结果
                - news_analysis: 新闻分析结果
                - current_portfolio: 当前持仓
                - last_period_pnl: 上期盈亏
                - decision_history: 历史决策记录
            as_of_date: 决策时间点（用于回测，默认None=当前时间）
        
        Returns:
            Dict: 决策结果
                {
                    "analysis_period": {"start": date, "end": date, "days": N},
                    "forecast_period": {"start": date, "end": date, "days": M},
                    "market_state": str,
                    "reasoning": str,
                    "recommended_strategy": str,
                    "suggested_positions": dict (可选),
                    "confidence": float,
                    "risk_assessment": str,
                    "timestamp": str
                }
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        if not self.validate_input(data):
            logger.warning("Invalid input data for Coordinator")
            return self._get_fallback_decision()
        
        logger.info("Generating weekly trading decision via LLM")
        
        try:
            # 构建prompt
            user_prompt = self._build_user_prompt(data, as_of_date)
            
            # 调用LLM
            decision = self._call_llm_for_decision(user_prompt)
            
            # 验证决策
            if not self._validate_decision(decision):
                logger.warning("Invalid decision from LLM, using fallback")
                return self._get_fallback_decision()
            
            # 添加时间戳和时间范围信息
            decision['timestamp'] = as_of_date.isoformat()
            decision['require_approval'] = self.require_approval
            
            # 添加分析和预测期间
            decision['analysis_period'] = {
                'start': data.get('analysis_start_date', ''),
                'end': data.get('analysis_end_date', ''),
                'days': data.get('lookback_days', 7)
            }
            decision['forecast_period'] = {
                'start': data.get('forecast_start_date', ''),
                'end': data.get('forecast_end_date', ''),
                'days': data.get('forecast_days', 7)
            }
            
            logger.info(
                f"Decision generated: {decision['recommended_strategy']} "
                f"(based on {decision['analysis_period']['days']} days, "
                f"forecasting {decision['forecast_period']['days']} days)"
            )
            
            return decision
            
        except Exception as e:
            import traceback
            logger.error(f"Error generating decision: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_fallback_decision()
    
    def _build_user_prompt(self, data: Dict[str, Any], as_of_date: datetime) -> str:
        """
        构建用户prompt
        
        Args:
            data: 输入数据
            as_of_date: 决策时间点
        
        Returns:
            str: 格式化的prompt
        """
        # 提取时间范围
        analysis_start = data.get('analysis_start_date', '')
        analysis_end = data.get('analysis_end_date', as_of_date.strftime('%Y-%m-%d'))
        forecast_start = data.get('forecast_start_date', '')
        forecast_end = data.get('forecast_end_date', '')
        lookback_days = data.get('lookback_days', 7)
        forecast_days = data.get('forecast_days', 7)
        
        # 提取各部分数据
        market_data = self._format_market_data(data.get('market_data', {}))
        technical_analysis = self._format_technical_analysis(data.get('technical_analysis', {}))
        sentiment_analysis = self._format_sentiment_analysis(data.get('sentiment_analysis', {}))
        news_analysis = self._format_news_analysis(data.get('news_analysis', {}))
        current_portfolio = self._format_portfolio(data.get('current_portfolio', {}))
        last_period_pnl = data.get('last_period_pnl', 0.0)
        decision_history = self._format_decision_history(data.get('decision_history', []))
        
        # 使用模板
        prompt = prompts.COORDINATOR_USER_PROMPT_TEMPLATE.format(
            analysis_start_date=analysis_start,
            analysis_end_date=analysis_end,
            forecast_start_date=forecast_start,
            forecast_end_date=forecast_end,
            lookback_days=lookback_days,
            forecast_days=forecast_days,
            market_data=market_data,
            technical_analysis=technical_analysis,
            sentiment_analysis=sentiment_analysis,
            news_summary=news_analysis,
            current_portfolio=current_portfolio,
            last_period_pnl=f"${last_period_pnl:,.2f}",
            decision_history=decision_history
        )
        
        return prompt
    
    def _format_market_data(self, market_data: Dict[str, Any]) -> str:
        """格式化市场数据"""
        if not market_data:
            return "Market data not available"
        
        lines = []
        for ticker, data in market_data.items():
            if ticker == '^VIX':
                ticker = 'VIX'
            
            stats = data.get('weekly_stats', {})
            price = data.get('latest_price', 0)
            week_return = stats.get('return', 0)
            
            lines.append(f"{ticker}: ${price:.2f} ({week_return:+.2%})")
        
        return ", ".join(lines)
    
    def _format_technical_analysis(self, technical: Dict[str, Any]) -> str:
        """格式化技术分析"""
        if not technical:
            return "Technical analysis not available"
        
        overall = technical.get('overall_signal', {})
        trend = technical.get('trend', {})
        momentum = technical.get('momentum', {})
        volatility = technical.get('volatility', {})
        
        lines = [
            f"Signal: {overall.get('signal', 'neutral').upper()}",
            f"Trend: {trend.get('direction', 'unknown')} (strength: {trend.get('strength', 0):.2f})",
            f"Momentum: {momentum.get('momentum', 'neutral')} (RSI: {momentum.get('rsi', 50):.1f})",
            f"Volatility: {volatility.get('level', 'unknown')}"
        ]
        
        return "\n".join(lines)
    
    def _format_sentiment_analysis(self, sentiment: Dict[str, Any]) -> str:
        """格式化情绪分析"""
        if not sentiment:
            return "Sentiment analysis not available"
        
        overall = sentiment.get('overall_sentiment', {})
        vix = sentiment.get('vix_sentiment', {})
        
        lines = [
            f"Overall: {overall.get('sentiment', 'neutral')} (score: {overall.get('score', 0.5):.2f})",
            f"VIX: {vix.get('vix_value', 'N/A')} ({vix.get('sentiment', 'neutral')})",
            f"Risk Level: {overall.get('risk_level', 'unknown')}"
        ]
        
        return "\n".join(lines)
    
    def _format_news_analysis(self, news: Dict[str, Any]) -> str:
        """格式化新闻分析"""
        if not news:
            return "News analysis not available"
        
        major_events = news.get('major_events', [])
        implications = news.get('trading_implications', '')
        risk_factors = news.get('risk_factors', [])
        
        lines = []
        
        if major_events:
            lines.append(f"Major Events: {len(major_events)} detected")
            for event in major_events[:3]:
                lines.append(f"  - {event.get('title', '')[:60]}...")
        else:
            lines.append("No major events")
        
        if risk_factors:
            lines.append(f"Risk Factors: {', '.join(risk_factors)}")
        
        lines.append(f"Implications: {implications}")
        
        return "\n".join(lines)
    
    def _format_portfolio(self, portfolio: Dict[str, Any]) -> str:
        """格式化持仓"""
        if not portfolio:
            return "100% cash"
        
        # 处理新格式的portfolio (包含holdings)
        if 'holdings' in portfolio:
            holdings = portfolio['holdings']
            total_value = portfolio.get('total_value', 100000)
            cash = portfolio.get('cash', 0)
            
            if not holdings:
                return f"100% cash (${cash:,.2f})"
            
            lines = []
            for symbol, info in holdings.items():
                weight = info['market_value'] / total_value if total_value > 0 else 0
                lines.append(f"{symbol}: {weight:.1%} (${info['market_value']:,.2f})")
            
            cash_weight = cash / total_value if total_value > 0 else 0
            lines.append(f"cash: {cash_weight:.1%} (${cash:,.2f})")
            return ", ".join(lines)
        
        # 处理旧格式的portfolio (symbol: weight)
        lines = []
        for symbol, weight in portfolio.items():
            if isinstance(weight, (int, float)):
                lines.append(f"{symbol}: {weight:.1%}")
            else:
                lines.append(f"{symbol}: {weight}")
        
        return ", ".join(lines) if lines else "100% cash"
    
    def _format_decision_history(self, history: List[Dict[str, Any]]) -> str:
        """格式化决策历史"""
        if not history:
            return "No previous decisions"
        
        lines = []
        for decision in history[-4:]:  # 最近4条
            # 支持新旧两种格式
            if 'analysis_period' in decision:
                period = f"{decision['analysis_period']['start']} to {decision['analysis_period']['end']}"
            else:
                period = decision.get('week', 'N/A')
            
            strategy = decision.get('strategy', decision.get('recommended_strategy', 'N/A'))
            pnl = decision.get('pnl', 0)
            lines.append(f"{period}: {strategy} ({pnl:+.2%})")
        
        return "\n".join(lines)
    
    def _call_llm_for_decision(self, user_prompt: str) -> Dict[str, Any]:
        """
        调用LLM生成决策
        
        Args:
            user_prompt: 用户prompt
        
        Returns:
            Dict: 决策JSON
        """
        messages = [
            {"role": "system", "content": prompts.COORDINATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # 添加few-shot examples（可选）
        # 这里简化，直接调用
        
        try:
            decision = self.llm_client.generate_json(messages)
            
            # 如果不允许建议仓位，移除该字段
            if not self.can_suggest_positions and 'suggested_positions' in decision:
                del decision['suggested_positions']
            
            return decision
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _validate_decision(self, decision: Dict[str, Any]) -> bool:
        """
        验证决策的完整性
        
        Args:
            decision: 决策字典
        
        Returns:
            bool: 是否有效
        """
        required_fields = [
            'market_state',
            'reasoning',
            'recommended_strategy',
            'confidence',
            'risk_assessment'
        ]
        
        for field in required_fields:
            if field not in decision:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # 验证策略是否在允许列表中
        valid_strategies = ['grid_trading', 'momentum', 'mean_reversion', 'hold']
        if decision['recommended_strategy'] not in valid_strategies:
            logger.warning(f"Invalid strategy: {decision['recommended_strategy']}")
            return False
        
        # 验证置信度范围
        if not (0 <= decision['confidence'] <= 1):
            logger.warning(f"Invalid confidence: {decision['confidence']}")
            return False
        
        return True
    
    def _get_fallback_decision(self) -> Dict[str, Any]:
        """
        获取后备决策（当LLM失败时）
        
        Returns:
            Dict: 保守的默认决策
        """
        logger.info("Using fallback decision")
        
        return {
            "market_state": "uncertain",
            "reasoning": "Unable to generate LLM decision. Using conservative fallback.",
            "recommended_strategy": "hold",
            "confidence": 0.3,
            "risk_assessment": "High uncertainty - preserve capital",
            "suggested_positions": {"cash": 1.0} if self.can_suggest_positions else None,
            "timestamp": datetime.now().isoformat(),
            "require_approval": True,
            "fallback": True
        }
