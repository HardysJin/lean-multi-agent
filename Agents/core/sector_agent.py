"""
Sector Agent - Core Business Logic

Pure business logic for sector/industry analysis.
No MCP dependencies, easy to test and reuse.

职责：
- 分析行业趋势和轮动
- 评估行业相对强度
- 判断行业景气度
- 提供行业层面的约束和建议
"""

from .base_agent import BaseAgent
from Agents.utils.tool_registry import tool
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import json

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


class SectorAgent(BaseAgent):
    """
    行业分析Agent - Pure Business Logic
    
    分析特定行业的趋势、轮动和相对强度。
    纯业务逻辑，无MCP依赖，易于测试。
    
    使用示例：
    ```python
    # 基本使用
    agent = SectorAgent()
    context = await agent.analyze_sector('Technology')
    
    # 使用 MockLLM 测试
    from Agents.utils import MockLLM
    mock = MockLLM(response='{"trend": "bullish", ...}')
    agent = SectorAgent(llm_client=mock)
    
    # 获取股票所属行业
    sector = agent.get_sector_for_symbol('AAPL')  # 'Technology'
    ```
    """
    
    def __init__(
        self,
        llm_client=None,
        cache_ttl: int = 1800,  # 30分钟缓存
        enable_cache: bool = True
    ):
        """
        初始化 SectorAgent
        
        Args:
            llm_client: LLM客户端实例（可以是MockLLM用于测试）
            cache_ttl: 缓存有效期（秒），默认1800秒（30分钟）
            enable_cache: 是否启用缓存，默认True
        """
        super().__init__(
            name="sector-agent",
            llm_client=llm_client,
            enable_cache=enable_cache,
            cache_ttl=cache_ttl
        )
        
        self.logger.info(f"SectorAgent initialized (cache_ttl={cache_ttl}s)")
    
    # ═══════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════
    
    def get_sector_for_symbol(self, symbol: str) -> str:
        """
        获取股票所属行业
        
        Args:
            symbol: 股票代码
            
        Returns:
            行业名称
        """
        return SECTOR_MAPPING.get(symbol, 'Unknown')
    
    @tool(description="Analyze specific sector performance and trends")
    async def analyze_sector(
        self,
        sector: str,
        visible_data_end: Optional[datetime] = None,
        force_refresh: bool = False
    ) -> SectorContext:
        """
        分析特定行业
        
        Args:
            sector: 行业名称（如 'Technology', 'Healthcare'）
            visible_data_end: 回测模式下的数据截止时间
            force_refresh: 强制刷新，忽略缓存
            
        Returns:
            SectorContext: 行业分析结果
        """
        # 检查缓存
        if not force_refresh:
            cache_key = self._get_cache_key(sector, visible_data_end)
            cached = self._get_from_cache(cache_key)
            if cached:
                self.logger.info(f"Cache hit: {cache_key}")
                return cached
        
        # 执行分析
        self.logger.info(f"Analyzing sector: {sector} (visible_data_end={visible_data_end})")
        context = await self._perform_analysis(sector, visible_data_end)
        
        # 缓存结果
        cache_key = self._get_cache_key(sector, visible_data_end)
        self._put_to_cache(cache_key, context)
        
        return context
    
    async def analyze_symbol_sector(
        self,
        symbol: str,
        visible_data_end: Optional[datetime] = None
    ) -> SectorContext:
        """
        分析股票所属行业
        
        Args:
            symbol: 股票代码
            visible_data_end: 回测模式下的数据截止时间
            
        Returns:
            SectorContext: 行业分析结果
        """
        sector = self.get_sector_for_symbol(symbol)
        return await self.analyze_sector(sector, visible_data_end)
    
    async def compare_sectors(
        self,
        sectors: List[str],
        visible_data_end: Optional[datetime] = None
    ) -> Dict[str, SectorContext]:
        """
        比较多个行业
        
        Args:
            sectors: 行业名称列表
            visible_data_end: 回测模式下的数据截止时间
            
        Returns:
            行业分析结果字典 {sector_name: SectorContext}
        """
        results = {}
        for sector in sectors:
            results[sector] = await self.analyze_sector(sector, visible_data_end)
        return results
    
    # ═══════════════════════════════════════════════
    # Private Methods
    # ═══════════════════════════════════════════════
    
    async def _perform_analysis(
        self,
        sector: str,
        visible_data_end: Optional[datetime] = None
    ) -> SectorContext:
        """执行完整的行业分析"""
        # 1. 收集行业数据
        sector_data = await self._collect_sector_data(sector, visible_data_end)
        
        # 2. LLM 分析
        analysis_result = await self._llm_analyze(sector, sector_data)
        
        # 3. 构建 SectorContext
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
        """
        收集行业数据
        
        TODO: 实现真实数据收集
        当前：返回模拟数据
        """
        mock_data = {
            'sector': sector,
            'date': visible_data_end.isoformat() if visible_data_end else datetime.now().isoformat(),
            'price_performance_1m': 5.2,  # 1月表现
            'price_performance_3m': 12.5,  # 3月表现
            'relative_to_sp500': 2.3,  # 相对S&P500表现
            'avg_volume_ratio': 1.15,  # 成交量比率
            'top_performers': ['AAPL', 'MSFT', 'GOOGL'] if sector == 'Technology' else [],
            'sector_etf_flow': 'positive',  # ETF资金流向
        }
        
        self.logger.info(f"Collected sector data: {sector}")
        return mock_data
    
    async def _llm_analyze(
        self,
        sector: str,
        sector_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用 LLM 分析行业数据"""
        system_prompt = """You are a sector/industry analyst with deep expertise in market dynamics.
Analyze the provided sector data and determine:
1. Trend (bullish/bearish/neutral)
2. Relative strength (-1 to 1, relative to market)
3. Momentum (accelerating/decelerating/stable)
4. Sector rotation signal (rotating_in/rotating_out/neutral)
5. Sentiment (bullish/bearish/neutral)
6. Recommendation (overweight/neutral/underweight)

Provide detailed reasoning."""

        user_prompt = f"""Analyze this sector:

Sector: {sector}
Data:
{json.dumps(sector_data, indent=2)}

Respond in JSON format:
{{
    "trend": "bullish|bearish|neutral",
    "relative_strength": -1.0 to 1.0,
    "momentum": "accelerating|decelerating|stable",
    "sector_rotation_signal": "rotating_in|rotating_out|neutral",
    "avg_pe_ratio": 25.0,
    "avg_growth_rate": 0.15,
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
            response = await self.llm.ainvoke(messages)
            content = response.content
            
            # 提取 JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            self.logger.info(f"LLM analysis: {sector} - {result['trend']}, strength={result['relative_strength']}")
            return result
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}, using fallback")
            return self._fallback_analysis(sector, sector_data)
    
    def _fallback_analysis(
        self,
        sector: str,
        sector_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """LLM 失败时的降级分析"""
        perf_1m = sector_data.get('price_performance_1m', 0)
        relative = sector_data.get('relative_to_sp500', 0)
        
        # 简单规则
        if perf_1m > 5 and relative > 2:
            trend, sentiment, rec = 'bullish', 'bullish', 'overweight'
            strength = 0.7
        elif perf_1m < -3 or relative < -2:
            trend, sentiment, rec = 'bearish', 'bearish', 'underweight'
            strength = -0.5
        else:
            trend, sentiment, rec = 'neutral', 'neutral', 'neutral'
            strength = 0.0
        
        return {
            'trend': trend,
            'relative_strength': strength,
            'momentum': 'stable',
            'sector_rotation_signal': 'neutral',
            'avg_pe_ratio': 20.0,
            'avg_growth_rate': 0.10,
            'sentiment': sentiment,
            'confidence': 0.6,
            'recommendation': rec,
            'reasoning': f"Fallback analysis: 1M={perf_1m}%, Relative={relative}%"
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
