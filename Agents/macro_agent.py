"""
Macro Agent - 宏观环境分析Agent

职责：
- 分析宏观经济环境（GDP、失业率、通胀）
- 分析货币政策（利率、央行政策）
- 分析市场情绪（VIX、市场宽度）
- 判断市场regime（牛市/熊市/震荡）
- 提供全局约束条件（风险控制参数）

特点：
- 不需要symbol参数（宏观分析独立于个股）
- 结果可被多个股票复用（性能优化）
- 支持时间控制（防止Look-Ahead Bias）
- 智能缓存（避免重复分析）

设计：
- 使用 Dependency Injection（非Singleton）
- 支持灵活配置（LLM、缓存策略）
- 易于测试（可注入Mock）
"""

from .base_mcp_agent import BaseMCPAgent
from mcp.types import Tool, Resource, TextContent
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import json
import hashlib

from Agents.llm_config import get_default_llm, LLMConfig
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


class MacroAgent(BaseMCPAgent):
    """
    宏观分析Agent
    
    提供宏观环境分析，独立于个股分析。
    采用Dependency Injection设计，易于测试和扩展。
    
    使用示例：
    ```python
    # 基本使用
    agent = MacroAgent()
    context = await agent.analyze_macro_environment()
    
    # 自定义配置
    agent = MacroAgent(
        llm_config=custom_config,
        cache_ttl=7200  # 2小时cache
    )
    
    # 回测模式（防止Look-Ahead）
    context = await agent.analyze_macro_environment(
        visible_data_end=datetime(2023, 6, 1)
    )
    
    # 测试模式（注入Mock）
    mock_llm = Mock(spec=LLMConfig)
    agent = MacroAgent(llm_config=mock_llm)
    ```
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        cache_ttl: int = 3600,  # cache有效期（秒），默认1小时
        enable_cache: bool = True
    ):
        """
        初始化MacroAgent
        
        Args:
            llm_config: LLM配置，如果为None则使用默认配置
            cache_ttl: 缓存有效期（秒），默认3600秒（1小时）
            enable_cache: 是否启用缓存，默认True
        """
        super().__init__(
            name="macro-agent",
            description="Analyzes macro economic environment, monetary policy, and market regime",
            version="1.0.0",
            llm_config=llm_config,
            enable_llm=True
        )
        
        # 缓存配置
        self.cache_ttl = cache_ttl
        self.enable_cache = enable_cache
        self._cache: Dict[str, tuple[MacroContext, datetime]] = {}
        
        self.logger.info(f"MacroAgent initialized (cache_ttl={cache_ttl}s, enable_cache={enable_cache})")
    
    # ═══════════════════════════════════════════════
    # MCP Protocol Implementation
    # ═══════════════════════════════════════════════
    
    def get_tools(self) -> List[Tool]:
        """
        返回MacroAgent提供的工具列表
        
        工具：
        1. analyze_macro_environment - 完整宏观分析
        2. get_market_regime - 快速获取市场regime
        3. get_risk_constraints - 获取风险约束条件
        """
        return [
            Tool(
                name="analyze_macro_environment",
                description=(
                    "Perform comprehensive macro environment analysis including "
                    "market regime, interest rates, economic indicators, and risk constraints. "
                    "This is the primary tool for macro analysis. "
                    "Returns detailed MacroContext with all analysis results."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "visible_data_end": {
                            "type": "string",
                            "description": "Optional ISO datetime string for backtest mode. Only use data before this time."
                        },
                        "force_refresh": {
                            "type": "boolean",
                            "description": "Force refresh analysis, ignore cache. Default false."
                        }
                    }
                }
            ),
            Tool(
                name="get_market_regime",
                description=(
                    "Quick analysis to determine current market regime (bull/bear/sideways). "
                    "Faster than full analysis, useful for quick checks. "
                    "Returns: {regime: str, confidence: float, reasoning: str}"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "visible_data_end": {
                            "type": "string",
                            "description": "Optional ISO datetime string for backtest mode"
                        }
                    }
                }
            ),
            Tool(
                name="get_risk_constraints",
                description=(
                    "Get risk management constraints based on current macro environment. "
                    "Returns constraints like max_risk, allow_long, allow_short, etc. "
                    "Used by downstream decision makers to enforce risk limits."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "visible_data_end": {
                            "type": "string",
                            "description": "Optional ISO datetime string for backtest mode"
                        }
                    }
                }
            )
        ]
    
    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        处理工具调用
        
        Args:
            name: 工具名称
            arguments: 工具参数
            
        Returns:
            工具执行结果
        """
        self.logger.info(f"Tool called: {name} with arguments: {arguments}")
        
        if name == "analyze_macro_environment":
            return await self._handle_analyze_macro_environment(arguments)
        elif name == "get_market_regime":
            return await self._handle_get_market_regime(arguments)
        elif name == "get_risk_constraints":
            return await self._handle_get_risk_constraints(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    def get_resources(self) -> List[Resource]:
        """
        返回MacroAgent提供的资源列表
        
        资源：
        1. macro://current - 当前宏观环境
        2. macro://cache-stats - 缓存统计
        """
        return [
            Resource(
                uri="macro://current",
                name="Current Macro Environment",
                description="Current macro economic environment analysis",
                mimeType="application/json"
            ),
            Resource(
                uri="macro://cache-stats",
                name="Cache Statistics",
                description="MacroAgent cache statistics and performance metrics",
                mimeType="application/json"
            )
        ]
    
    async def handle_resource_read(self, uri: str) -> str:
        """
        处理资源读取
        
        Args:
            uri: 资源URI
            
        Returns:
            资源内容（JSON字符串）
        """
        self.logger.info(f"Resource read: {uri}")
        
        if uri == "macro://current":
            # 返回当前宏观环境（从cache）
            context = await self.analyze_macro_environment()
            return json.dumps(context.to_dict(), indent=2)
        
        elif uri == "macro://cache-stats":
            # 返回缓存统计
            stats = {
                "cache_enabled": self.enable_cache,
                "cache_ttl": self.cache_ttl,
                "cached_items": len(self._cache),
                "cache_keys": list(self._cache.keys())
            }
            return json.dumps(stats, indent=2)
        
        else:
            raise ValueError(f"Unknown resource URI: {uri}")
    
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
        
        这是MacroAgent的主要方法，提供完整的宏观分析。
        
        Args:
            visible_data_end: 回测模式下的数据截止时间（防止Look-Ahead）
            force_refresh: 强制刷新，忽略缓存
            
        Returns:
            MacroContext: 完整的宏观分析结果
            
        Example:
            >>> agent = MacroAgent()
            >>> context = await agent.analyze_macro_environment()
            >>> print(context.market_regime)  # 'bull'
            >>> print(context.constraints)    # {'max_risk': 0.02, ...}
        """
        # 检查缓存
        if not force_refresh and self.enable_cache:
            cache_key = self._get_cache_key(visible_data_end)
            if cache_key in self._cache:
                cached_context, cached_time = self._cache[cache_key]
                age = (datetime.now() - cached_time).total_seconds()
                if age < self.cache_ttl:
                    self.logger.info(f"Cache hit: {cache_key} (age={age:.1f}s)")
                    return cached_context
                else:
                    self.logger.info(f"Cache expired: {cache_key} (age={age:.1f}s)")
        
        # 执行分析
        self.logger.info(f"Performing macro analysis (visible_data_end={visible_data_end})")
        context = await self._perform_analysis(visible_data_end)
        
        # 缓存结果
        if self.enable_cache:
            cache_key = self._get_cache_key(visible_data_end)
            self._cache[cache_key] = (context, datetime.now())
            self.logger.info(f"Cached result: {cache_key}")
        
        return context
    
    async def get_market_regime(
        self,
        visible_data_end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        快速获取市场regime（不执行完整分析）
        
        Args:
            visible_data_end: 回测模式下的数据截止时间
            
        Returns:
            {regime: str, confidence: float, reasoning: str}
        """
        # 先尝试从cache获取
        cache_key = self._get_cache_key(visible_data_end)
        if cache_key in self._cache:
            cached_context, cached_time = self._cache[cache_key]
            age = (datetime.now() - cached_time).total_seconds()
            if age < self.cache_ttl:
                return {
                    'regime': cached_context.market_regime,
                    'confidence': cached_context.regime_confidence,
                    'reasoning': f"From cached analysis ({age:.1f}s ago)"
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
        if cache_key in self._cache:
            cached_context, cached_time = self._cache[cache_key]
            age = (datetime.now() - cached_time).total_seconds()
            if age < self.cache_ttl:
                return cached_context.constraints
        
        # 没有cache，执行完整分析
        context = await self.analyze_macro_environment(visible_data_end)
        return context.constraints
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'cache_enabled': self.enable_cache,
            'cache_ttl': self.cache_ttl,
            'cached_items': len(self._cache),
            'cache_keys': list(self._cache.keys())
        }
    
    # ═══════════════════════════════════════════════
    # Private Methods - Tool Handlers
    # ═══════════════════════════════════════════════
    
    async def _handle_analyze_macro_environment(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """处理 analyze_macro_environment 工具调用"""
        visible_data_end = arguments.get('visible_data_end')
        if visible_data_end:
            visible_data_end = datetime.fromisoformat(visible_data_end)
        
        force_refresh = arguments.get('force_refresh', False)
        
        context = await self.analyze_macro_environment(visible_data_end, force_refresh)
        return context.to_dict()
    
    async def _handle_get_market_regime(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """处理 get_market_regime 工具调用"""
        visible_data_end = arguments.get('visible_data_end')
        if visible_data_end:
            visible_data_end = datetime.fromisoformat(visible_data_end)
        
        return await self.get_market_regime(visible_data_end)
    
    async def _handle_get_risk_constraints(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """处理 get_risk_constraints 工具调用"""
        visible_data_end = arguments.get('visible_data_end')
        if visible_data_end:
            visible_data_end = datetime.fromisoformat(visible_data_end)
        
        return await self.get_risk_constraints(visible_data_end)
    
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
        1. 收集宏观数据（经济指标、市场数据）
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
        - 经济指标（GDP、失业率、CPI）
        - 利率数据（Fed Funds Rate）
        - 市场数据（VIX、市场宽度）
        - 新闻和政策事件
        
        当前：返回模拟数据
        """
        # 模拟数据（后续需要连接真实数据源）
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
        
        LLM会：
        1. 综合分析所有宏观指标
        2. 判断市场regime
        3. 评估风险水平
        4. 提供详细推理
        """
        # 构建prompt
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

        # 调用LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = await self._llm_client.ainvoke(messages)
            
            # 解析响应
            content = response.content
            
            # 尝试提取JSON（如果被包裹在markdown中）
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            self.logger.info(f"LLM analysis: regime={result['market_regime']}, risk={result['risk_level']}")
            return result
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}, using fallback")
            # 降级：使用简单规则
            return self._fallback_analysis(macro_data)
    
    def _fallback_analysis(self, macro_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM失败时的降级分析（使用简单规则）
        """
        vix = macro_data['vix']
        
        # 简单规则判断
        if vix < 15:
            regime = 'bull'
            volatility = 'low'
            risk = 3.0
            sentiment = 'greed'
        elif vix < 20:
            regime = 'sideways'
            volatility = 'medium'
            risk = 5.0
            sentiment = 'neutral'
        elif vix < 30:
            regime = 'transition'
            volatility = 'high'
            risk = 7.0
            sentiment = 'fear'
        else:
            regime = 'bear'
            volatility = 'extreme'
            risk = 9.0
            sentiment = 'extreme_fear'
        
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
        """
        根据宏观分析生成约束条件
        
        约束条件会被下游DecisionMaker使用，强制执行风险控制
        """
        regime = analysis['market_regime']
        risk_level = analysis['risk_level']
        
        constraints = {
            'max_risk_per_trade': 0.02,  # 默认每笔最大风险2%
            'max_portfolio_risk': 0.10,  # 默认组合最大风险10%
            'allow_long': True,
            'allow_short': False,
            'max_position_size': 0.20,  # 默认单仓位最大20%
            'max_leverage': 1.0
        }
        
        # 根据regime调整
        if regime == 'bull':
            constraints['allow_long'] = True
            constraints['max_position_size'] = 0.25
            constraints['max_risk_per_trade'] = 0.025
        elif regime == 'bear':
            constraints['allow_long'] = False  # 熊市禁止做多
            constraints['allow_short'] = True
            constraints['max_position_size'] = 0.10
            constraints['max_risk_per_trade'] = 0.015
        elif regime == 'sideways':
            constraints['max_position_size'] = 0.15
            constraints['max_risk_per_trade'] = 0.02
        
        # 根据风险级别调整
        if risk_level > 7:  # 高风险环境
            constraints['max_risk_per_trade'] *= 0.5
            constraints['max_position_size'] *= 0.5
            constraints['max_leverage'] = 0.5
        elif risk_level < 4:  # 低风险环境
            constraints['max_risk_per_trade'] *= 1.2
            constraints['max_position_size'] *= 1.2
        
        self.logger.info(f"Generated constraints: {constraints}")
        return constraints
    
    async def _quick_regime_analysis(
        self,
        visible_data_end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        快速regime分析（不使用LLM，基于规则）
        """
        macro_data = await self._collect_macro_data(visible_data_end)
        vix = macro_data['vix']
        
        if vix < 15:
            regime = 'bull'
            confidence = 0.7
        elif vix < 20:
            regime = 'sideways'
            confidence = 0.6
        elif vix < 30:
            regime = 'transition'
            confidence = 0.5
        else:
            regime = 'bear'
            confidence = 0.7
        
        return {
            'regime': regime,
            'confidence': confidence,
            'reasoning': f"Quick analysis based on VIX={vix}"
        }
    
    def _get_cache_key(self, visible_data_end: Optional[datetime] = None) -> str:
        """
        生成缓存键
        
        缓存键规则：
        - 如果是回测模式（有visible_data_end），按日期缓存
        - 如果是实时模式，按小时缓存（避免频繁调用）
        """
        if visible_data_end:
            # 回测模式：按天缓存
            return f"backtest_{visible_data_end.date().isoformat()}"
        else:
            # 实时模式：按小时缓存
            now = datetime.now()
            return f"live_{now.year}{now.month:02d}{now.day:02d}_{now.hour:02d}"
