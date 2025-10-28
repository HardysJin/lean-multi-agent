"""
Test News Agent - 测试新闻分析Agent

测试覆盖：
1. NewsAgent初始化
2. 新闻获取（模拟数据）
3. LLM情绪分析（mock）
4. 情绪报告生成
5. 关键词搜索
6. 资源读取
7. 工具调用

性能优化：
- 使用mock LLM避免真实API调用
- 使用模拟新闻数据避免真实API调用
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import json
from datetime import datetime, timedelta

from Agents.news_agent import (
    NewsAgent, NewsArticle, NewsSentimentReport
)
from Agents.llm_config import get_mock_llm, LLMConfig


def run_async(coro):
    """Helper function to run async code in tests"""
    return asyncio.run(coro)


@pytest.fixture
def mock_llm_config():
    """创建mock LLM配置"""
    mock_llm = get_mock_llm()
    llm_config = LLMConfig(model="mock", api_key="test-key")
    # 注意：get_mock_llm返回的是MockLLM，LLMConfig会用它
    return llm_config


@pytest.fixture
def news_agent_with_mock_llm(mock_llm_config):
    """创建带mock LLM的NewsAgent"""
    # 不使用真实API
    agent = NewsAgent(api_key=None, llm_config=mock_llm_config)
    # 确保没有真实的news client
    agent.news_client = None
    return agent


class TestNewsAgentInitialization:
    """测试NewsAgent初始化"""
    
    def test_initialization_without_api_key(self, mock_llm_config):
        """测试无API key的初始化"""
        agent = NewsAgent(llm_config=mock_llm_config)
        
        assert agent.name == "news-agent"
        assert "news" in agent.description.lower()
        assert agent.has_llm()  # 应该有LLM
    
    def test_initialization_with_api_key(self, mock_llm_config):
        """测试带API key的初始化"""
        agent = NewsAgent(api_key="test-key-123", llm_config=mock_llm_config)
        
        assert agent.api_key == "test-key-123"
        assert agent.has_llm()
    
    def test_initialization_with_llm_config(self):
        """测试带LLM配置的初始化"""
        llm_config = LLMConfig(model="mock", api_key="test")
        agent = NewsAgent(llm_config=llm_config)
        
        assert agent.has_llm()
        assert agent.llm is not None


class TestNewsAgentTools:
    """测试NewsAgent工具"""
    
    def test_get_tools(self, news_agent_with_mock_llm):
        """测试获取工具列表"""
        tools = news_agent_with_mock_llm.get_tools()
        
        assert len(tools) == 4
        tool_names = [tool.name for tool in tools]
        assert "get_latest_news" in tool_names
        assert "analyze_news_sentiment" in tool_names
        assert "get_news_summary" in tool_names
        assert "search_news_by_keyword" in tool_names
    
    def test_tool_schemas(self, news_agent_with_mock_llm):
        """测试工具schema"""
        tools = news_agent_with_mock_llm.get_tools()
        
        for tool in tools:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'inputSchema')
            assert 'type' in tool.inputSchema
            assert 'properties' in tool.inputSchema


class TestNewsAgentResources:
    """测试NewsAgent资源"""
    
    def test_get_resources(self, news_agent_with_mock_llm):
        """测试获取资源列表"""
        resources = news_agent_with_mock_llm.get_resources()
        
        assert len(resources) == 2
        # URI中的{symbol}会被URL编码为%7Bsymbol%7D
        uris_str = [str(res.uri) for res in resources]
        assert any('recent' in uri and 'symbol' in uri for uri in uris_str)
        assert any('sentiment' in uri and 'symbol' in uri for uri in uris_str)


class TestNewsFetching:
    """测试新闻获取"""
    
    def test_fetch_news_without_api(self, news_agent_with_mock_llm):
        """测试无API时返回模拟数据"""
        agent = news_agent_with_mock_llm
        
        articles = run_async(agent._fetch_news("AAPL", limit=5))
        
        assert len(articles) > 0
        assert all(isinstance(a, NewsArticle) for a in articles)
        assert all(hasattr(a, 'title') for a in articles)
        assert all(hasattr(a, 'description') for a in articles)
    
    def test_mock_news_generation(self, news_agent_with_mock_llm):
        """测试模拟新闻生成"""
        agent = news_agent_with_mock_llm
        
        mock_articles = agent._get_mock_news("TSLA", limit=3)
        
        assert len(mock_articles) == 3
        assert all(isinstance(a, NewsArticle) for a in mock_articles)
        assert all("TSLA" in a.title for a in mock_articles)
    
    def test_news_caching(self, news_agent_with_mock_llm):
        """测试新闻缓存"""
        agent = news_agent_with_mock_llm
        
        # 第一次获取
        articles1 = run_async(agent._fetch_news("AAPL", limit=5))
        
        # 第二次获取（应该使用缓存）
        articles2 = run_async(agent._fetch_news("AAPL", limit=5))
        
        # 应该返回相同的对象（缓存）
        assert len(articles1) == len(articles2)


class TestSentimentAnalysis:
    """测试情绪分析"""
    
    def test_sentiment_parsing(self, news_agent_with_mock_llm):
        """测试情绪响应解析"""
        agent = news_agent_with_mock_llm
        
        response = """SENTIMENT: positive
SCORE: 0.8
REASONING: Strong earnings beat expectations with positive guidance."""
        
        result = agent._parse_sentiment_response(response)
        
        assert result['sentiment'] == 'positive'
        assert result['score'] == 0.8
        assert 'earnings' in result['reasoning'].lower()
    
    def test_sentiment_parsing_negative(self, news_agent_with_mock_llm):
        """测试负面情绪解析"""
        agent = news_agent_with_mock_llm
        
        response = """SENTIMENT: negative
SCORE: -0.6
REASONING: Regulatory concerns and declining market share."""
        
        result = agent._parse_sentiment_response(response)
        
        assert result['sentiment'] == 'negative'
        assert result['score'] == -0.6
    
    def test_sentiment_parsing_neutral(self, news_agent_with_mock_llm):
        """测试中性情绪解析"""
        agent = news_agent_with_mock_llm
        
        response = """SENTIMENT: neutral
SCORE: 0.0
REASONING: Routine business update with no significant impact."""
        
        result = agent._parse_sentiment_response(response)
        
        assert result['sentiment'] == 'neutral'
        assert result['score'] == 0.0
    
    def test_analyze_sentiment_without_llm(self):
        """测试无LLM时的情绪分析"""
        # 创建没有LLM的agent
        agent = NewsAgent(llm_config=None)
        agent._llm_client = None
        agent.news_client = None
        
        articles = [
            NewsArticle(
                title="Test Article",
                description="Test description",
                source="Test Source",
                url="http://example.com",
                published_at=datetime.now()
            )
        ]
        
        result = run_async(agent._analyze_sentiment_with_llm(articles))
        
        # 应该返回中性情绪
        assert result[0].sentiment == "neutral"
        assert result[0].sentiment_score == 0.0


class TestSentimentReport:
    """测试情绪报告"""
    
    def test_generate_sentiment_report(self, news_agent_with_mock_llm):
        """测试生成情绪报告"""
        agent = news_agent_with_mock_llm
        
        report = run_async(agent._generate_sentiment_report("AAPL"))
        
        assert isinstance(report, NewsSentimentReport)
        assert report.symbol == "AAPL"
        assert report.overall_sentiment in ['positive', 'negative', 'neutral', 'mixed']
        assert -1.0 <= report.sentiment_score <= 1.0
        assert report.articles_analyzed >= 0
        assert isinstance(report.key_themes, list)
    
    def test_report_sentiment_counts(self, news_agent_with_mock_llm):
        """测试报告中的情绪统计"""
        agent = news_agent_with_mock_llm
        
        report = run_async(agent._generate_sentiment_report("MSFT"))
        
        # 验证计数
        total = report.positive_count + report.negative_count + report.neutral_count
        assert total == report.articles_analyzed


class TestKeyThemeExtraction:
    """测试关键主题提取"""
    
    def test_extract_key_themes(self, news_agent_with_mock_llm):
        """测试关键主题提取"""
        agent = news_agent_with_mock_llm
        
        articles = [
            NewsArticle(
                title="Apple Reports Record iPhone Sales",
                description="",
                source="",
                url="",
                published_at=datetime.now()
            ),
            NewsArticle(
                title="Apple iPhone Demand Surges in China",
                description="",
                source="",
                url="",
                published_at=datetime.now()
            ),
            NewsArticle(
                title="Record Revenue from iPhone Sales",
                description="",
                source="",
                url="",
                published_at=datetime.now()
            ),
        ]
        
        themes = agent._extract_key_themes(articles)
        
        assert isinstance(themes, list)
        assert len(themes) <= 5
        # "iphone" 和 "sales" 应该出现（如果大小写不敏感）
        themes_lower = [t.lower() for t in themes]
        assert any('iphone' in t or 'sales' in t or 'record' in t for t in themes_lower)


class TestToolCalls:
    """测试工具调用"""
    
    def test_get_latest_news_tool(self, news_agent_with_mock_llm):
        """测试get_latest_news工具"""
        agent = news_agent_with_mock_llm
        
        result = run_async(agent.handle_tool_call(
            "get_latest_news",
            {"symbol": "AAPL", "limit": 5}
        ))
        
        assert "symbol" in result
        assert result["symbol"] == "AAPL"
        assert "articles" in result
        assert "count" in result
        assert isinstance(result["articles"], list)
    
    def test_analyze_news_sentiment_tool(self, news_agent_with_mock_llm):
        """测试analyze_news_sentiment工具"""
        agent = news_agent_with_mock_llm
        
        test_articles = [
            {
                "title": "Company Reports Strong Earnings",
                "description": "Quarterly results exceed expectations"
            },
            {
                "title": "Stock Price Drops on Regulatory Concerns",
                "description": "New regulations may impact revenue"
            }
        ]
        
        result = run_async(agent.handle_tool_call(
            "analyze_news_sentiment",
            {"articles": test_articles}
        ))
        
        assert "articles" in result
        assert "count" in result
        assert len(result["articles"]) == 2
    
    def test_get_news_summary_tool(self, news_agent_with_mock_llm):
        """测试get_news_summary工具"""
        agent = news_agent_with_mock_llm
        
        result = run_async(agent.handle_tool_call(
            "get_news_summary",
            {"symbol": "TSLA", "days_back": 7}
        ))
        
        assert "symbol" in result
        assert result["symbol"] == "TSLA"
        assert "overall_sentiment" in result
        assert "sentiment_score" in result
        assert "articles_analyzed" in result
    
    def test_search_news_by_keyword_tool(self, news_agent_with_mock_llm):
        """测试search_news_by_keyword工具（无API）"""
        agent = news_agent_with_mock_llm
        
        result = run_async(agent.handle_tool_call(
            "search_news_by_keyword",
            {"keyword": "AI technology", "limit": 5}
        ))
        
        assert "keyword" in result
        assert result["keyword"] == "AI technology"
        assert "articles" in result
    
    def test_unknown_tool(self, news_agent_with_mock_llm):
        """测试未知工具"""
        agent = news_agent_with_mock_llm
        
        result = run_async(agent.handle_tool_call(
            "unknown_tool",
            {}
        ))
        
        assert "error" in result


class TestResourceReading:
    """测试资源读取"""
    
    def test_read_recent_news_resource(self, news_agent_with_mock_llm):
        """测试读取recent news资源"""
        agent = news_agent_with_mock_llm
        
        result = run_async(agent.handle_resource_read("news://recent/AAPL"))
        
        assert result is not None
        data = json.loads(result)
        assert isinstance(data, list)
    
    def test_read_sentiment_resource(self, news_agent_with_mock_llm):
        """测试读取sentiment资源"""
        agent = news_agent_with_mock_llm
        
        result = run_async(agent.handle_resource_read("news://sentiment/MSFT"))
        
        assert result is not None
        data = json.loads(result)
        assert "symbol" in data
        assert "overall_sentiment" in data
    
    def test_read_unknown_resource(self, news_agent_with_mock_llm):
        """测试读取未知资源"""
        agent = news_agent_with_mock_llm
        
        result = run_async(agent.handle_resource_read("news://unknown/AAPL"))
        
        data = json.loads(result)
        assert "error" in data


class TestCompanyNameMapping:
    """测试公司名称映射"""
    
    def test_get_company_name(self, news_agent_with_mock_llm):
        """测试获取公司名称"""
        agent = news_agent_with_mock_llm
        
        assert agent._get_company_name("AAPL") == "Apple"
        assert agent._get_company_name("MSFT") == "Microsoft"
        assert agent._get_company_name("TSLA") == "Tesla"
        
        # 未知symbol返回symbol本身
        assert agent._get_company_name("UNKNOWN") == "UNKNOWN"


class TestErrorHandling:
    """测试错误处理"""
    
    def test_tool_call_error_handling(self, news_agent_with_mock_llm):
        """测试工具调用错误处理"""
        agent = news_agent_with_mock_llm
        
        # 模拟一个会抛出异常的场景
        with patch.object(agent, '_fetch_news', side_effect=Exception("Test error")):
            result = run_async(agent.handle_tool_call(
                "get_latest_news",
                {"symbol": "AAPL"}
            ))
            
            assert "error" in result
    
    def test_resource_read_error_handling(self, news_agent_with_mock_llm):
        """测试资源读取错误处理"""
        agent = news_agent_with_mock_llm
        
        # 模拟错误
        with patch.object(agent, '_fetch_news', side_effect=Exception("Test error")):
            result = run_async(agent.handle_resource_read("news://recent/AAPL"))
            
            data = json.loads(result)
            assert "error" in data


class TestIntegration:
    """测试集成场景"""
    
    def test_end_to_end_news_analysis(self, news_agent_with_mock_llm):
        """测试端到端新闻分析流程"""
        agent = news_agent_with_mock_llm
        
        # 1. 获取新闻
        news_result = run_async(agent.handle_tool_call(
            "get_latest_news",
            {"symbol": "AAPL", "limit": 3}
        ))
        
        assert news_result["count"] > 0
        
        # 2. 生成报告
        summary_result = run_async(agent.handle_tool_call(
            "get_news_summary",
            {"symbol": "AAPL"}
        ))
        
        assert "overall_sentiment" in summary_result
        assert "key_themes" in summary_result
    
    def test_news_agent_with_meta_agent_integration(self, mock_llm_config):
        """测试与MetaAgent的集成（结构测试）"""
        from Agents.meta_agent import MetaAgent
        
        # 创建agents (都使用mock LLM)
        meta = MetaAgent(llm_config=mock_llm_config)
        news = NewsAgent(llm_config=mock_llm_config)
        news.news_client = None  # 不使用真实API
        
        # 连接
        run_async(meta.connect_to_agent(
            agent_name="news",
            agent_instance=news,
            description="News and sentiment analysis specialist"
        ))
        
        # 验证连接
        assert "news" in meta.agents
        assert len(meta.agents["news"].tools) == 4
        
        # 验证可以调用工具
        result = run_async(meta.execute_tool(
            agent_name="news",
            tool_name="get_latest_news",
            arguments={"symbol": "AAPL", "limit": 3}
        ))
        
        assert "symbol" in result
        assert result["symbol"] == "AAPL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
