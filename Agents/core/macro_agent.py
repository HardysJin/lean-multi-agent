"""
Macro Agent - Core Business Logic

Pure business logic for macro economic environment analysis.
No MCP dependencies, easy to test and reuse.

职责：
- 分析宏观经济环境（GDP、失业率、通胀）
- 分析货币政策（利率、央行政策）
- 分析市场情绪（VIX、市场宽度）
- 判断市场regime（牛市/熊市/震荡）
- 提供全局约束条件（风险控制参数）
"""

from .base_agent import BaseAgent
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import json

from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class MacroContext:
    """
    宏观环境上下文
    
    包含所有宏观分析结果，供下游DecisionMaker使用
    """
    # 市场regime
    market_regime: str  # 'bull' | 'bear' | 'sideways' | 'transition'
    regime_confidence: float  # 0-1
    
    # 利率环境
    interest_rate_trend: str  # 'rising' | 'falling' | 'stable'
    current_rate: float  # 当前利率（百分比）
    
    # 风险水平
    risk_level: float  # 0-10，10表示极高风险
    volatility_level: str  # 'low' | 'medium' | 'high' | 'extreme'
    
    # 经济指标
    gdp_trend: str  # 'expanding' | 'contracting' | 'stable'
    inflation_level: str  # 'low' | 'moderate' | 'high'
    
    # 市场情绪
    market_sentiment: str  # 'extreme_fear' | 'fear' | 'neutral' | 'greed' | 'extreme_greed'
    vix_level: float  # VIX指数
    
    # 约束条件（供下游使用）
    constraints: Dict[str, Any]  # {max_risk, allow_long, allow_short, max_position_size, ...}
    
    # 元数据
    analysis_timestamp: datetime
    data_end_time: Optional[datetime]  # 回测模式下的数据截止时间
    confidence_score: float  # 整体分析置信度 0-1
    reasoning: str  # LLM推理过程
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（便于序列化）"""
        result = asdict(self)
        # 转换datetime为ISO格式
        result['analysis_timestamp'] = self.analysis_timestamp.isoformat()
        if self.data_end_time:
            result['data_end_time'] = self.data_end_time.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MacroContext':
        """从字典创建（便于反序列化）"""
        # 转换ISO格式为datetime
        data['analysis_timestamp'] = datetime.fromisoformat(data['analysis_timestamp'])
        if data.get('data_end_time'):
            data['data_end_time'] = datetime.fromisoformat(data['data_end_time'])
        return cls(**data)


class MacroAgent(BaseAgent):
    """
    宏观分析Agent - Pure Business Logic
    
    提供宏观环境分析，独立于个股分析。
    纯业务逻辑，无MCP依赖，易于测试。
    
    使用示例：
    ```python
    # 基本使用
    agent = MacroAgent()
    context = await agent.analyze_macro_environment()
    
    # 使用MockLLM测试
    from Agents.core.base_agent import MockLLM
    mock = MockLLM(response='{"market_regime": "bull", ...}')
    agent = MacroAgent(llm_client=mock)
    context = await agent.analyze_macro_environment()
    
    # 自定义配置
    agent = MacroAgent(
        llm_client=custom_llm,
        cache_ttl=7200  # 2小时cache
    )
    
    # 回测模式（防止Look-Ahead）
    context = await agent.analyze_macro_environment(
        visible_data_end=datetime(2023, 6, 1)
    )
    ```
    """
    
    def __init__(
        self,
        llm_client=None,
        cache_ttl: int = 3600,
        enable_cache: bool = True
    ):
        """
        初始化MacroAgent
        
        Args:
            llm_client: LLM客户端实例（可以是MockLLM用于测试）
            cache_ttl: 缓存有效期（秒），默认3600秒（1小时）
            enable_cache: 是否启用缓存，默认True
        """
        super().__init__(
            name="macro-agent",
            llm_client=llm_client,
            enable_cache=enable_cache,
            cache_ttl=cache_ttl
        )
        
        self.logger.info(f"MacroAgent initialized (cache_ttl={cache_ttl}s, enable_cache={enable_cache})")
    
    # ═══════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════
    
    async def analyze_macro_environment(
        self,
        visible_data_end: Optional[datetime] = None,
        force_refresh: bool = False
    ) -> MacroContext:
        """
        执行完整的宏观环境分析
        
        Args:
            visible_data_end: 回测模式下的数据截止时间（防止Look-Ahead）
            force_refresh: 强制刷新，忽略缓存
            
        Returns:
            MacroContext: 完整的宏观分析结果
        """
        # 检查缓存
        if not force_refresh:
            cache_key = self._get_cache_key(visible_data_end)
            cached = self._get_from_cache(cache_key)
            if cached:
                self.logger.info(f"Cache hit: {cache_key}")
                return cached
        
        # 执行分析
        self.logger.info(f"Performing macro analysis (visible_data_end={visible_data_end})")
        context = await self._perform_analysis(visible_data_end)
        
        # 缓存结果
        cache_key = self._get_cache_key(visible_data_end)
        self._put_to_cache(cache_key, context)
        
        return context
    
    async def get_market_regime(
        self,
        visible_data_end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        快速获取市场regime
        
        Args:
            visible_data_end: 回测模式下的数据截止时间
            
        Returns:
            {regime: str, confidence: float, reasoning: str}
        """
        # 先尝试从cache获取
        cache_key = self._get_cache_key(visible_data_end)
        cached_context = self._get_from_cache(cache_key)
        
        if cached_context:
            return {
                'regime': cached_context.market_regime,
                'confidence': cached_context.regime_confidence,
                'reasoning': f"From cached analysis"
            }
        
        # 没有cache，执行轻量级分析
        regime_info = await self._quick_regime_analysis(visible_data_end)
        return regime_info
    
    async def get_risk_constraints(
        self,
        visible_data_end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        获取风险约束条件
        
        Args:
            visible_data_end: 回测模式下的数据截止时间
            
        Returns:
            约束条件字典
        """
        # 先尝试从cache获取
        cache_key = self._get_cache_key(visible_data_end)
        cached_context = self._get_from_cache(cache_key)
        
        if cached_context:
            return cached_context.constraints
        
        # 没有cache，执行完整分析
        context = await self.analyze_macro_environment(visible_data_end)
        return context.constraints
    
    # ═══════════════════════════════════════════════
    # Private Methods - Analysis Logic
    # ═══════════════════════════════════════════════
    
    async def _perform_analysis(
        self,
        visible_data_end: Optional[datetime] = None
    ) -> MacroContext:
        """
        执行完整的宏观分析
        
        分析流程：
        1. 收集宏观数据
        2. 使用LLM分析和推理
        3. 生成约束条件
        4. 构建MacroContext
        """
        # 1. 收集宏观数据
        macro_data = await self._collect_macro_data(visible_data_end)
        
        # 2. 使用LLM分析
        analysis_result = await self._llm_analyze(macro_data)
        
        # 3. 生成约束条件
        constraints = self._generate_constraints(analysis_result)
        
        # 4. 构建MacroContext
        context = MacroContext(
            market_regime=analysis_result['market_regime'],
            regime_confidence=analysis_result['regime_confidence'],
            interest_rate_trend=analysis_result['interest_rate_trend'],
            current_rate=analysis_result['current_rate'],
            risk_level=analysis_result['risk_level'],
            volatility_level=analysis_result['volatility_level'],
            gdp_trend=analysis_result['gdp_trend'],
            inflation_level=analysis_result['inflation_level'],
            market_sentiment=analysis_result['market_sentiment'],
            vix_level=analysis_result['vix_level'],
            constraints=constraints,
            analysis_timestamp=datetime.now(),
            data_end_time=visible_data_end,
            confidence_score=analysis_result['confidence_score'],
            reasoning=analysis_result['reasoning']
        )
        
        return context
    
    async def _collect_macro_data(
        self,
        visible_data_end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        收集宏观数据
        
        TODO: 实现真实的数据收集逻辑
        当前：返回模拟数据
        """
        mock_data = {
            'date': visible_data_end.isoformat() if visible_data_end else datetime.now().isoformat(),
            'vix': 18.5,
            'fed_rate': 5.25,
            'gdp_growth': 2.1,
            'unemployment': 3.8,
            'inflation_cpi': 3.2,
            'sp500_trend': 'upward',
            'market_breadth': 0.65,
            'recent_events': [
                'Fed holds rates steady',
                'Strong employment report',
                'Moderate inflation readings'
            ]
        }
        
        self.logger.info(f"Collected macro data: VIX={mock_data['vix']}, Rate={mock_data['fed_rate']}%")
        return mock_data
    
    async def _llm_analyze(self, macro_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用LLM分析宏观数据
        """
        system_prompt = """You are a senior macro economist and market strategist.
Analyze the provided macro economic data and determine:
1. Market regime (bull/bear/sideways/transition)
2. Interest rate trend (rising/falling/stable)
3. Risk level (0-10 scale)
4. Volatility level (low/medium/high/extreme)
5. GDP trend (expanding/contracting/stable)
6. Inflation level (low/moderate/high)
7. Market sentiment (extreme_fear/fear/neutral/greed/extreme_greed)

Provide detailed reasoning for your conclusions."""

        user_prompt = f"""Analyze this macro environment:

Data:
{json.dumps(macro_data, indent=2)}

Please respond in JSON format with the following structure:
{{
    "market_regime": "bull|bear|sideways|transition",
    "regime_confidence": 0.0-1.0,
    "interest_rate_trend": "rising|falling|stable",
    "current_rate": {macro_data['fed_rate']},
    "risk_level": 0-10,
    "volatility_level": "low|medium|high|extreme",
    "gdp_trend": "expanding|contracting|stable",
    "inflation_level": "low|moderate|high",
    "market_sentiment": "extreme_fear|fear|neutral|greed|extreme_greed",
    "vix_level": {macro_data['vix']},
    "confidence_score": 0.0-1.0,
    "reasoning": "detailed explanation"
}}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = response.content
            
            # 提取JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            self.logger.info(f"LLM analysis: regime={result['market_regime']}, risk={result['risk_level']}")
            return result
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}, using fallback")
            return self._fallback_analysis(macro_data)
    
    def _fallback_analysis(self, macro_data: Dict[str, Any]) -> Dict[str, Any]:
        """LLM失败时的降级分析"""
        vix = macro_data['vix']
        
        if vix < 15:
            regime, volatility, risk, sentiment = 'bull', 'low', 3.0, 'greed'
        elif vix < 20:
            regime, volatility, risk, sentiment = 'sideways', 'medium', 5.0, 'neutral'
        elif vix < 30:
            regime, volatility, risk, sentiment = 'transition', 'high', 7.0, 'fear'
        else:
            regime, volatility, risk, sentiment = 'bear', 'extreme', 9.0, 'extreme_fear'
        
        return {
            'market_regime': regime,
            'regime_confidence': 0.6,
            'interest_rate_trend': 'stable',
            'current_rate': macro_data['fed_rate'],
            'risk_level': risk,
            'volatility_level': volatility,
            'gdp_trend': 'stable',
            'inflation_level': 'moderate',
            'market_sentiment': sentiment,
            'vix_level': vix,
            'confidence_score': 0.6,
            'reasoning': f"Fallback analysis based on VIX={vix}"
        }
    
    def _generate_constraints(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """根据宏观分析生成约束条件"""
        regime = analysis['market_regime']
        risk_level = analysis['risk_level']
        
        constraints = {
            'max_risk_per_trade': 0.02,
            'max_portfolio_risk': 0.10,
            'allow_long': True,
            'allow_short': False,
            'max_position_size': 0.20,
            'max_leverage': 1.0
        }
        
        # 根据regime调整
        if regime == 'bull':
            constraints['allow_long'] = True
            constraints['max_position_size'] = 0.25
            constraints['max_risk_per_trade'] = 0.025
        elif regime == 'bear':
            constraints['allow_long'] = False
            constraints['allow_short'] = True
            constraints['max_position_size'] = 0.10
            constraints['max_risk_per_trade'] = 0.015
        elif regime == 'sideways':
            constraints['max_position_size'] = 0.15
            constraints['max_risk_per_trade'] = 0.02
        
        # 根据风险级别调整
        if risk_level > 7:
            constraints['max_risk_per_trade'] *= 0.5
            constraints['max_position_size'] *= 0.5
            constraints['max_leverage'] = 0.5
        elif risk_level < 4:
            constraints['max_risk_per_trade'] *= 1.2
            constraints['max_position_size'] *= 1.2
        
        self.logger.info(f"Generated constraints: {constraints}")
        return constraints
    
    async def _quick_regime_analysis(
        self,
        visible_data_end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """快速regime分析（基于规则）"""
        macro_data = await self._collect_macro_data(visible_data_end)
        vix = macro_data['vix']
        
        if vix < 15:
            regime, confidence = 'bull', 0.7
        elif vix < 20:
            regime, confidence = 'sideways', 0.6
        elif vix < 30:
            regime, confidence = 'transition', 0.5
        else:
            regime, confidence = 'bear', 0.7
        
        return {
            'regime': regime,
            'confidence': confidence,
            'reasoning': f"Quick analysis based on VIX={vix}"
        }
    
    def _get_cache_key(self, visible_data_end: Optional[datetime] = None) -> str:
        """生成缓存键"""
        if visible_data_end:
            return f"backtest_{visible_data_end.date().isoformat()}"
        else:
            now = datetime.now()
            return f"live_{now.year}{now.month:02d}{now.day:02d}_{now.hour:02d}"
