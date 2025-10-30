"""
Tests for SectorAgent
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime
import json

from Agents.core import SectorAgent, SectorContext, SECTOR_MAPPING, MockLLM


# Pytest fixture for MockLLM
@pytest.fixture
def mock_llm():
    """提供标准的 MockLLM 响应"""
    return MockLLM(response=json.dumps({
        'trend': 'bullish',
        'relative_strength': 0.7,
        'momentum': 'accelerating',
        'sector_rotation_signal': 'rotating_in',
        'avg_pe_ratio': 25.0,
        'avg_growth_rate': 0.15,
        'sentiment': 'bullish',
        'confidence': 0.85,
        'recommendation': 'overweight',
        'reasoning': 'Mock LLM analysis for testing'
    }))


class TestSectorAgentBasics:
    """基本功能测试"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_llm):
        agent = SectorAgent(llm_client=mock_llm)
        assert agent.name == "sector-agent"
        assert agent.enable_cache is True
        assert agent.cache_ttl == 1800
    
    @pytest.mark.asyncio
    async def test_analyze_sector(self, mock_llm):
        agent = SectorAgent(llm_client=mock_llm)
        context = await agent.analyze_sector("Technology")
        
        assert isinstance(context, SectorContext)
        assert context.sector == "Technology"
        assert context.trend in ['bullish', 'bearish', 'neutral']
        assert -1 <= context.relative_strength <= 1
        assert context.recommendation in ['overweight', 'neutral', 'underweight']
    
    @pytest.mark.asyncio
    async def test_get_sector_for_symbol(self, mock_llm):
        agent = SectorAgent(llm_client=mock_llm)
        
        sector = agent.get_sector_for_symbol('AAPL')
        assert sector == 'Technology'
        
        sector = agent.get_sector_for_symbol('JPM')
        assert sector == 'Financial Services'
        
        sector = agent.get_sector_for_symbol('UNKNOWN')
        assert sector == 'Unknown'
    
    @pytest.mark.asyncio
    async def test_compare_sectors(self, mock_llm):
        agent = SectorAgent(llm_client=mock_llm)
        
        sectors = ['Technology', 'Healthcare', 'Financial Services']
        results = await agent.compare_sectors(sectors)
        
        # 返回字典 {sector_name: SectorContext}
        assert len(results) == 3
        assert 'Technology' in results
        assert 'Healthcare' in results
        assert 'Financial Services' in results
        
        # 验证每个结果都是 SectorContext
        for sector_name, context in results.items():
            assert isinstance(context, SectorContext)
            assert context.sector == sector_name


class TestSectorAgentCache:
    """缓存测试"""
    
    @pytest.mark.asyncio
    async def test_cache_works(self, mock_llm):
        agent = SectorAgent(llm_client=mock_llm, cache_ttl=3600)
        
        context1 = await agent.analyze_sector("Technology")
        context2 = await agent.analyze_sector("Technology")
        
        # MockLLM 只被调用一次（第二次走 cache）
        assert mock_llm.call_count == 1
        assert context1.analysis_timestamp == context2.analysis_timestamp
        assert len(agent._cache) > 0
    
    @pytest.mark.asyncio
    async def test_cache_disabled(self, mock_llm):
        agent = SectorAgent(llm_client=mock_llm, enable_cache=False)
        
        context1 = await agent.analyze_sector("Technology")
        context2 = await agent.analyze_sector("Technology")
        
        # MockLLM 应该被调用两次（cache 禁用）
        assert mock_llm.call_count == 2
        assert len(agent._cache) == 0
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, mock_llm):
        agent = SectorAgent(llm_client=mock_llm)
        
        await agent.analyze_sector("Technology")
        assert len(agent._cache) > 0
        
        agent.clear_cache()
        assert len(agent._cache) == 0


class TestSectorAgentTimeControl:
    """时间控制测试"""
    
    @pytest.mark.asyncio
    async def test_backtest_mode(self, mock_llm):
        agent = SectorAgent(llm_client=mock_llm)
        
        backtest_time = datetime(2023, 6, 1)
        context = await agent.analyze_sector("Technology", visible_data_end=backtest_time)
        
        assert context.data_end_time == backtest_time
    
    @pytest.mark.asyncio
    async def test_different_times_different_cache(self, mock_llm):
        agent = SectorAgent(llm_client=mock_llm)
        
        time1 = datetime(2023, 6, 1)
        time2 = datetime(2023, 7, 1)
        
        context1 = await agent.analyze_sector("Technology", visible_data_end=time1)
        context2 = await agent.analyze_sector("Technology", visible_data_end=time2)
        
        assert context1.data_end_time != context2.data_end_time
        assert len(agent._cache) == 2


@pytest.mark.skip(reason="MCP protocol tests - moved to wrapper tests")
class TestSectorAgentTools:
    """MCP工具测试 - 这些测试应该在 MCP wrapper 中进行"""
    
    def test_get_tools(self):
        agent = SectorAgent(llm_client=mock_llm)
        tools = agent.get_tools()
        
        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert 'analyze_sector' in tool_names
        assert 'get_sector_for_symbol' in tool_names
        assert 'compare_sectors' in tool_names
    
    @pytest.mark.asyncio
    async def test_handle_analyze_sector(self, mock_llm):
        agent = SectorAgent(llm_client=mock_llm)
        
        result = await agent.handle_tool_call(
            'analyze_sector',
            {'sector': 'Technology'}
        )
        
        assert isinstance(result, dict)
        assert result['sector'] == 'Technology'
        assert 'trend' in result
    
    @pytest.mark.asyncio
    async def test_handle_get_sector_for_symbol(self, mock_llm):
        agent = SectorAgent(llm_client=mock_llm)
        
        result = await agent.handle_tool_call(
            'get_sector_for_symbol',
            {'symbol': 'AAPL'}
        )
        
        assert result['symbol'] == 'AAPL'
        assert result['sector'] == 'Technology'
    
    @pytest.mark.asyncio
    async def test_handle_compare_sectors(self, mock_llm):
        agent = SectorAgent(llm_client=mock_llm)
        
        result = await agent.handle_tool_call(
            'compare_sectors',
            {'sectors': ['Technology', 'Healthcare']}
        )
        
        assert isinstance(result, list)
        assert len(result) == 2


@pytest.mark.skip(reason="MCP protocol tests - moved to wrapper tests")
class TestSectorAgentResources:
    """MCP资源测试 - 这些测试应该在 MCP wrapper 中进行"""
    
    def test_get_resources(self):
        agent = SectorAgent(llm_client=mock_llm)
        resources = agent.get_resources()
        
        assert len(resources) == 2
        uris = [str(r.uri) for r in resources]
        assert 'sector://sectors' in uris
        assert 'sector://cache-stats' in uris
    
    @pytest.mark.asyncio
    async def test_read_sectors_resource(self, mock_llm):
        agent = SectorAgent(llm_client=mock_llm)
        
        content = await agent.handle_resource_read('sector://sectors')
        data = json.loads(content)
        
        assert 'sectors' in data
        assert isinstance(data['sectors'], list)
    
    @pytest.mark.asyncio
    async def test_read_cache_stats_resource(self, mock_llm):
        agent = SectorAgent(llm_client=mock_llm)
        
        content = await agent.handle_resource_read('sector://cache-stats')
        stats = json.loads(content)
        
        assert 'cache_enabled' in stats
        assert 'cached_items' in stats


class TestSectorContext:
    """SectorContext测试"""
    
    def test_to_dict(self):
        context = SectorContext(
            sector='Technology',
            trend='bullish',
            relative_strength=0.5,
            momentum='accelerating',
            sector_rotation_signal='rotating_in',
            avg_pe_ratio=25.0,
            avg_growth_rate=15.0,
            sentiment='bullish',
            confidence=0.85,
            recommendation='overweight',
            reasoning='Test',
            analysis_timestamp=datetime(2023, 6, 1, 12, 0, 0),
            data_end_time=None
        )
        
        data = context.to_dict()
        assert data['sector'] == 'Technology'
        assert data['analysis_timestamp'] == '2023-06-01T12:00:00'
    
    def test_from_dict(self):
        data = {
            'sector': 'Technology',
            'trend': 'bullish',
            'relative_strength': 0.5,
            'momentum': 'accelerating',
            'sector_rotation_signal': 'rotating_in',
            'avg_pe_ratio': 25.0,
            'avg_growth_rate': 15.0,
            'sentiment': 'bullish',
            'confidence': 0.85,
            'recommendation': 'overweight',
            'reasoning': 'Test',
            'analysis_timestamp': '2023-06-01T12:00:00',
            'data_end_time': None
        }
        
        context = SectorContext.from_dict(data)
        assert context.sector == 'Technology'
        assert isinstance(context.analysis_timestamp, datetime)


class TestSectorMapping:
    """行业映射测试"""
    
    def test_sector_mapping_exists(self):
        assert 'AAPL' in SECTOR_MAPPING
        assert SECTOR_MAPPING['AAPL'] == 'Technology'
    
    def test_multiple_sectors(self):
        tech_stocks = [k for k, v in SECTOR_MAPPING.items() if v == 'Technology']
        assert len(tech_stocks) > 0
        
        finance_stocks = [k for k, v in SECTOR_MAPPING.items() if v == 'Financial Services']
        assert len(finance_stocks) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
