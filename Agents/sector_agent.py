"""
Sector Agent - 行业分析Agent

职责：
- 分析行业趋势和轮动
- 评估行业相对强度
- 判断行业景气度
- 提供行业层面的约束和建议

特点：
- 介于宏观和个股之间的中观分析
- 支持多个行业并行分析
- 智能缓存行业数据
- 与MacroAgent协同工作
"""

from .base_mcp_agent import BaseMCPAgent
from mcp.types import Tool, Resource
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import logging
import json

from Agents.llm_config import get_default_llm, LLMConfig
from langchain_core.messages import HumanMessage, SystemMessage


# 行业分类映射
SECTOR_MAPPING = {
    'AAPL': 'Technology',
    'GOOGL': 'Technology',
    'MSFT': 'Technology',
    'AMZN': 'Consumer Cyclical',
    'TSLA': 'Consumer Cyclical',
    'JPM': 'Financial Services',
    'BAC': 'Financial Services',
    'JNJ': 'Healthcare',
    'PFE': 'Healthcare',
    'XOM': 'Energy',
    'CVX': 'Energy',
}


@dataclass
class SectorContext:
    """
    行业分析上下文
    """
    sector: str  # 行业名称
    trend: str  # 'bullish' | 'bearish' | 'neutral'
    relative_strength: float  # 相对大盘强度 -1 to 1
    momentum: str  # 'accelerating' | 'decelerating' | 'stable'
    
    # 行业指标
    sector_rotation_signal: str  # 'rotating_in' | 'rotating_out' | 'neutral'
    avg_pe_ratio: Optional[float]  # 行业平均PE
    avg_growth_rate: Optional[float]  # 行业平均增长率
    
    # 行业情绪
    sentiment: str  # 'bullish' | 'bearish' | 'neutral'
    confidence: float  # 0-1
    
    # 建议
    recommendation: str  # 'overweight' | 'neutral' | 'underweight'
    reasoning: str
    
    # 元数据
    analysis_timestamp: datetime
    data_end_time: Optional[datetime]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['analysis_timestamp'] = self.analysis_timestamp.isoformat()
        if self.data_end_time:
            result['data_end_time'] = self.data_end_time.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SectorContext':
        data['analysis_timestamp'] = datetime.fromisoformat(data['analysis_timestamp'])
        if data.get('data_end_time'):
            data['data_end_time'] = datetime.fromisoformat(data['data_end_time'])
        return cls(**data)


class SectorAgent(BaseMCPAgent):
    """
    行业分析Agent
    
    分析特定行业的趋势、轮动和相对强度
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        cache_ttl: int = 1800,  # 30分钟缓存
        enable_cache: bool = True
    ):
        super().__init__(
            name="sector-agent",
            description="Analyzes sector trends, rotation, and relative strength",
            version="1.0.0",
            llm_config=llm_config,
            enable_llm=True
        )
        
        self.cache_ttl = cache_ttl
        self.enable_cache = enable_cache
        self._cache: Dict[str, tuple[SectorContext, datetime]] = {}
        
        self.logger.info(f"SectorAgent initialized (cache_ttl={cache_ttl}s)")
    
    # ═══════════════════════════════════════════════
    # MCP Protocol Implementation
    # ═══════════════════════════════════════════════
    
    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="analyze_sector",
                description=(
                    "Analyze a specific sector including trend, rotation, and relative strength. "
                    "Returns comprehensive sector analysis."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sector": {
                            "type": "string",
                            "description": "Sector name (e.g., 'Technology', 'Healthcare', 'Financial Services')"
                        },
                        "visible_data_end": {
                            "type": "string",
                            "description": "Optional ISO datetime for backtest mode"
                        }
                    },
                    "required": ["sector"]
                }
            ),
            Tool(
                name="get_sector_for_symbol",
                description=(
                    "Get the sector classification for a stock symbol. "
                    "Returns sector name."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., 'AAPL', 'GOOGL')"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="compare_sectors",
                description=(
                    "Compare multiple sectors and rank them by relative strength. "
                    "Returns ranked list of sectors."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sectors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of sector names to compare"
                        },
                        "visible_data_end": {
                            "type": "string",
                            "description": "Optional ISO datetime for backtest mode"
                        }
                    },
                    "required": ["sectors"]
                }
            )
        ]
    
    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Any:
        self.logger.info(f"Tool called: {name}")
        
        if name == "analyze_sector":
            return await self._handle_analyze_sector(arguments)
        elif name == "get_sector_for_symbol":
            return self._handle_get_sector_for_symbol(arguments)
        elif name == "compare_sectors":
            return await self._handle_compare_sectors(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    def get_resources(self) -> List[Resource]:
        return [
            Resource(
                uri="sector://sectors",
                name="Available Sectors",
                description="List of all available sectors",
                mimeType="application/json"
            ),
            Resource(
                uri="sector://cache-stats",
                name="Cache Statistics",
                description="SectorAgent cache statistics",
                mimeType="application/json"
            )
        ]
    
    async def handle_resource_read(self, uri: str) -> str:
        if uri == "sector://sectors":
            sectors = list(set(SECTOR_MAPPING.values()))
            return json.dumps({"sectors": sectors}, indent=2)
        elif uri == "sector://cache-stats":
            stats = self.get_cache_stats()
            return json.dumps(stats, indent=2)
        else:
            raise ValueError(f"Unknown resource URI: {uri}")
    
    # ═══════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════
    
    async def analyze_sector(
        self,
        sector: str,
        visible_data_end: Optional[datetime] = None
    ) -> SectorContext:
        """分析特定行业"""
        # 检查缓存
        if self.enable_cache:
            cache_key = self._get_cache_key(sector, visible_data_end)
            if cache_key in self._cache:
                cached_context, cached_time = self._cache[cache_key]
                age = (datetime.now() - cached_time).total_seconds()
                if age < self.cache_ttl:
                    self.logger.info(f"Cache hit: {cache_key} (age={age:.1f}s)")
                    return cached_context
        
        # 执行分析
        self.logger.info(f"Analyzing sector: {sector}")
        context = await self._perform_sector_analysis(sector, visible_data_end)
        
        # 缓存结果
        if self.enable_cache:
            cache_key = self._get_cache_key(sector, visible_data_end)
            self._cache[cache_key] = (context, datetime.now())
            self.logger.info(f"Cached result: {cache_key}")
        
        return context
    
    def get_sector_for_symbol(self, symbol: str) -> str:
        """获取股票所属行业"""
        return SECTOR_MAPPING.get(symbol.upper(), "Unknown")
    
    async def compare_sectors(
        self,
        sectors: List[str],
        visible_data_end: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """比较多个行业，按相对强度排序"""
        results = []
        
        for sector in sectors:
            context = await self.analyze_sector(sector, visible_data_end)
            results.append({
                'sector': sector,
                'relative_strength': context.relative_strength,
                'trend': context.trend,
                'recommendation': context.recommendation
            })
        
        # 按相对强度排序
        results.sort(key=lambda x: x['relative_strength'], reverse=True)
        return results
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            'cache_enabled': self.enable_cache,
            'cache_ttl': self.cache_ttl,
            'cached_items': len(self._cache),
            'cache_keys': list(self._cache.keys())
        }
    
    # ═══════════════════════════════════════════════
    # Private Methods - Tool Handlers
    # ═══════════════════════════════════════════════
    
    async def _handle_analyze_sector(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        sector = arguments['sector']
        visible_data_end = arguments.get('visible_data_end')
        if visible_data_end:
            visible_data_end = datetime.fromisoformat(visible_data_end)
        
        context = await self.analyze_sector(sector, visible_data_end)
        return context.to_dict()
    
    def _handle_get_sector_for_symbol(self, arguments: Dict[str, Any]) -> Dict[str, str]:
        symbol = arguments['symbol']
        sector = self.get_sector_for_symbol(symbol)
        return {'symbol': symbol, 'sector': sector}
    
    async def _handle_compare_sectors(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        sectors = arguments['sectors']
        visible_data_end = arguments.get('visible_data_end')
        if visible_data_end:
            visible_data_end = datetime.fromisoformat(visible_data_end)
        
        return await self.compare_sectors(sectors, visible_data_end)
    
    # ═══════════════════════════════════════════════
    # Private Methods - Analysis Logic
    # ═══════════════════════════════════════════════
    
    async def _perform_sector_analysis(
        self,
        sector: str,
        visible_data_end: Optional[datetime] = None
    ) -> SectorContext:
        """执行行业分析"""
        # 1. 收集行业数据
        sector_data = await self._collect_sector_data(sector, visible_data_end)
        
        # 2. LLM分析
        analysis_result = await self._llm_analyze_sector(sector, sector_data)
        
        # 3. 构建SectorContext
        context = SectorContext(
            sector=sector,
            trend=analysis_result['trend'],
            relative_strength=analysis_result['relative_strength'],
            momentum=analysis_result['momentum'],
            sector_rotation_signal=analysis_result['sector_rotation_signal'],
            avg_pe_ratio=analysis_result.get('avg_pe_ratio'),
            avg_growth_rate=analysis_result.get('avg_growth_rate'),
            sentiment=analysis_result['sentiment'],
            confidence=analysis_result['confidence'],
            recommendation=analysis_result['recommendation'],
            reasoning=analysis_result['reasoning'],
            analysis_timestamp=datetime.now(),
            data_end_time=visible_data_end
        )
        
        return context
    
    async def _collect_sector_data(
        self,
        sector: str,
        visible_data_end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """收集行业数据（TODO: 连接真实数据源）"""
        # 模拟数据
        mock_data = {
            'sector': sector,
            'date': visible_data_end.isoformat() if visible_data_end else datetime.now().isoformat(),
            'price_performance_1m': 5.2,  # 1个月涨跌幅
            'price_performance_3m': 12.5,
            'relative_to_sp500': 3.8,  # 相对大盘表现
            'avg_volume_ratio': 1.15,  # 成交量比率
            'top_performers': ['AAPL', 'MSFT', 'GOOGL'] if sector == 'Technology' else [],
            'recent_news_sentiment': 0.65  # -1 to 1
        }
        
        self.logger.info(f"Collected sector data for {sector}")
        return mock_data
    
    async def _llm_analyze_sector(
        self,
        sector: str,
        sector_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用LLM分析行业数据"""
        system_prompt = f"""You are a sector rotation analyst.
Analyze the provided sector data and determine:
1. Trend: bullish/bearish/neutral
2. Relative strength: -1.0 to 1.0 (vs market)
3. Momentum: accelerating/decelerating/stable
4. Rotation signal: rotating_in/rotating_out/neutral
5. Sentiment: bullish/bearish/neutral
6. Recommendation: overweight/neutral/underweight

Respond in JSON format."""

        user_prompt = f"""Analyze {sector} sector:

Data:
{json.dumps(sector_data, indent=2)}

Response format:
{{
    "trend": "bullish|bearish|neutral",
    "relative_strength": -1.0 to 1.0,
    "momentum": "accelerating|decelerating|stable",
    "sector_rotation_signal": "rotating_in|rotating_out|neutral",
    "sentiment": "bullish|bearish|neutral",
    "confidence": 0.0-1.0,
    "recommendation": "overweight|neutral|underweight",
    "reasoning": "detailed explanation"
}}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = await self._llm_client.ainvoke(messages)
            content = response.content
            
            # 提取JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            self.logger.info(f"LLM analysis: {sector} - {result['trend']}, RS={result['relative_strength']}")
            return result
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}, using fallback")
            return self._fallback_sector_analysis(sector_data)
    
    def _fallback_sector_analysis(self, sector_data: Dict[str, Any]) -> Dict[str, Any]:
        """降级分析（基于规则）"""
        rel_perf = sector_data.get('relative_to_sp500', 0)
        
        if rel_perf > 5:
            trend = 'bullish'
            recommendation = 'overweight'
            rotation = 'rotating_in'
        elif rel_perf < -5:
            trend = 'bearish'
            recommendation = 'underweight'
            rotation = 'rotating_out'
        else:
            trend = 'neutral'
            recommendation = 'neutral'
            rotation = 'neutral'
        
        return {
            'trend': trend,
            'relative_strength': rel_perf / 10.0,  # 归一化到-1到1
            'momentum': 'stable',
            'sector_rotation_signal': rotation,
            'sentiment': trend,
            'confidence': 0.6,
            'recommendation': recommendation,
            'reasoning': f"Fallback analysis: relative performance = {rel_perf}%"
        }
    
    def _get_cache_key(
        self,
        sector: str,
        visible_data_end: Optional[datetime] = None
    ) -> str:
        """生成缓存键"""
        if visible_data_end:
            return f"backtest_{sector}_{visible_data_end.date().isoformat()}"
        else:
            now = datetime.now()
            return f"live_{sector}_{now.year}{now.month:02d}{now.day:02d}_{now.hour:02d}"
