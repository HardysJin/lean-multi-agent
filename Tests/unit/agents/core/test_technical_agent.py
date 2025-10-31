"""
Test Technical Analysis Agent - 测试技术分析Agent

测试覆盖：
1. Agent初始化
2. 指标计算（模拟数据）
3. 信号生成
4. 形态检测
5. 支撑阻力位识别

性能优化：
- 使用mock数据避免真实API调用
- TechnicalAgent不需要LLM
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock
import json

from Agents.core import TechnicalAnalysisAgent


def run_async(coro):
    """Helper function to run async code in tests"""
    return asyncio.run(coro)


class TestTechnicalAnalysisAgent:
    """测试技术分析Agent基础功能"""
    
    def test_initialization(self):
        """测试初始化"""
        agent = TechnicalAnalysisAgent()
        
        assert agent.name == "technical-analysis-agent"
        assert agent._llm_client is None  # TechnicalAgent不使用LLM
    
    def test_initialization_with_cache_ttl(self):
        """测试带缓存TTL的初始化"""
        agent = TechnicalAnalysisAgent(cache_ttl=7200, price_cache_ttl=1800)
        
        assert agent.cache_ttl == 7200
        assert agent.price_cache_ttl == 1800


class TestIndicatorCalculation:
    """测试指标计算"""
    
    @pytest.fixture
    def agent(self):
        """创建Agent实例"""
        return TechnicalAnalysisAgent()
    
    def test_calculate_mock_indicators(self, agent):
        """测试计算模拟指标"""
        result = agent._calculate_mock_indicators("AAPL")
        
        assert result['symbol'] == "AAPL"
        assert 'current_price' in result
        assert 'indicators' in result
        assert 'rsi' in result['indicators']
        assert 'macd' in result['indicators']
        assert 'sma20' in result['indicators']
        assert 'sma50' in result['indicators']
        assert 'sma200' in result['indicators']
    
    def test_calculate_indicators_api(self, agent):
        """测试calculate_indicators API"""
        result = agent.calculate_indicators("AAPL")
        
        assert result['symbol'] == "AAPL"
        assert 'indicators' in result
        assert 'timestamp' in result
    
    def test_calculate_indicators_with_timeframe(self, agent):
        """测试带时间尺度的指标计算"""
        result = agent.calculate_indicators("AAPL", timeframe="1h")
        
        assert result['symbol'] == "AAPL"
    
    def test_rsi_interpretation(self, agent):
        """测试RSI解读"""
        assert agent._interpret_rsi(25) == "oversold"
        assert agent._interpret_rsi(75) == "overbought"
        assert agent._interpret_rsi(50) == "neutral"
    
    def test_calculate_distance(self, agent):
        """测试距离计算"""
        distance = agent._calculate_distance(105, 100)
        assert abs(distance - 0.05) < 0.001
        
        distance = agent._calculate_distance(95, 100)
        assert abs(distance - (-0.05)) < 0.001
        
        distance = agent._calculate_distance(100, 0)
        assert distance == 0


class TestSignalGeneration:
    """测试信号生成"""
    
    @pytest.fixture
    def agent(self):
        """创建Agent实例"""
        return TechnicalAnalysisAgent()
    
    def test_generate_signals_api(self, agent):
        """测试generate_signals API"""
        result = agent.generate_signals("AAPL")
        
        assert result['symbol'] == "AAPL"
        assert result['action'] in ['BUY', 'SELL', 'HOLD']
        assert 1 <= result['conviction'] <= 10
        assert 'reasoning' in result
        assert 'score_breakdown' in result
    
    def test_signal_has_score_breakdown(self, agent):
        """测试信号包含分数明细"""
        result = agent.generate_signals("TSLA")
        
        breakdown = result['score_breakdown']
        assert 'bullish' in breakdown
        assert 'bearish' in breakdown
        assert 'total' in breakdown
    
    def test_signal_includes_indicators(self, agent):
        """测试信号包含指标摘要"""
        result = agent.generate_signals("MSFT")
        
        assert 'indicators_summary' in result
        assert 'current_price' in result


class TestPatternDetection:
    """测试形态检测"""
    
    @pytest.fixture
    def agent(self):
        """创建Agent实例"""
        return TechnicalAnalysisAgent()
    
    def test_detect_patterns_api(self, agent):
        """测试detect_patterns API"""
        result = agent.detect_patterns("AAPL")
        
        assert result['symbol'] == "AAPL"
        assert 'patterns' in result
        assert 'lookback_days' in result
    
    def test_detect_patterns_with_lookback(self, agent):
        """测试带回溯期的形态检测"""
        result = agent.detect_patterns("AAPL", lookback_days=30)
        
        assert result['lookback_days'] == 30
    
    def test_patterns_have_confidence(self, agent):
        """测试形态包含信心度"""
        result = agent.detect_patterns("AAPL", 60)
        
        if result['patterns']:
            for pattern in result['patterns']:
                assert 'pattern' in pattern
                assert 'confidence' in pattern
                assert 0 <= pattern['confidence'] <= 1


class TestSupportResistance:
    """测试支撑阻力位"""
    
    @pytest.fixture
    def agent(self):
        """创建Agent实例"""
        return TechnicalAnalysisAgent()
    
    def test_find_support_resistance_api(self, agent):
        """测试find_support_resistance API"""
        result = agent.find_support_resistance("AAPL")
        
        assert result['symbol'] == "AAPL"
        assert 'current_price' in result
        assert 'levels' in result
    
    def test_levels_have_properties(self, agent):
        """测试支撑阻力位包含必要属性"""
        result = agent.find_support_resistance("AAPL")
        
        if result['levels']:
            for level in result['levels']:
                assert 'level' in level
                assert 'type' in level
                assert level['type'] in ['support', 'resistance']


class TestCaching:
    """测试缓存机制"""
    
    @pytest.fixture
    def agent(self):
        """创建Agent实例"""
        return TechnicalAnalysisAgent()
    
    def test_indicators_use_cache(self, agent):
        """测试指标计算使用缓存"""
        # 第一次调用
        result1 = agent.calculate_indicators("AAPL")
        
        # 第二次调用应该使用缓存
        result2 = agent.calculate_indicators("AAPL")
        
        # 应该返回相同的数据（从缓存）
        assert result1['symbol'] == result2['symbol']
        assert result1['timestamp'] == result2['timestamp']


@pytest.mark.skip(reason="MCP protocol tests - moved to wrapper")
class TestResourceReading:
    """测试资源读取"""
    
    @pytest.fixture
    def agent(self):
        """创建Agent实例"""
        return TechnicalAnalysisAgent()
    
    def test_read_indicators_resource_with_symbol(self, agent):
        """测试读取指定symbol的指标资源"""
        content = run_async(agent.handle_resource_read("indicators://AAPL"))
        
        assert content['symbol'] == "AAPL"
        assert 'indicators' in content
    
    def test_read_signals_resource_with_symbol(self, agent):
        """测试读取指定symbol的信号资源"""
        content = run_async(agent.handle_resource_read("signals://AAPL"))
        
        assert content['symbol'] == "AAPL"
        assert content['action'] in ['BUY', 'SELL', 'HOLD']
    
    def test_read_indicators_resource_without_symbol(self, agent):
        """测试读取所有指标资源（无algorithm会返回错误）"""
        content = run_async(agent.handle_resource_read("indicators://"))
        
        assert 'error' in content
    
    def test_read_unknown_resource(self, agent):
        """测试读取未知资源"""
        content = run_async(agent.handle_resource_read("unknown://"))
        
        assert isinstance(content, dict)


@pytest.mark.skip(reason="LEAN integration tests - deprecated")
class TestLEANIntegration:
    """测试LEAN集成（已弃用，现在使用mock）"""
    
    def test_calculate_from_lean_with_indicators(self):
        """测试从LEAN获取指标（现在返回mock数据）"""
        # 创建mock algorithm
        mock_algo = Mock()
        
        # Mock indicators（这些现在会被忽略）
        mock_rsi = Mock()
        mock_rsi.Current.Value = 55.0
        
        mock_macd = Mock()
        mock_macd.Current.Value = 1.5
        mock_macd.Signal.Current.Value = 1.2
        mock_macd.Histogram.Current.Value = 0.3
        
        mock_sma200 = Mock()
        mock_sma200.Current.Value = 145.0
        
        mock_algo.indicators = {
            'AAPL': {
                'RSI': mock_rsi,
                'MACD': mock_macd,
                'SMA200': mock_sma200
            }
        }
        
        # Mock Securities
        mock_security = Mock()
        mock_security.Price = 150.0
        mock_algo.Securities = {'AAPL': mock_security}
        mock_algo.Time = Mock()
        mock_algo.Time.isoformat.return_value = "2025-10-28T12:00:00"
        
        # 创建agent（这个功能已废弃）
        from Agents.mcp.technical_agent_mcp_wrapper import TechnicalAnalysisAgent as TechnicalAgentMCP
        agent = TechnicalAgentMCP(algorithm=mock_algo)
        
        # 计算指标（现在会返回mock数据，忽略LEAN indicators）
        result = agent._calculate_from_lean('AAPL')
        
        # 验证返回了mock数据
        assert result['symbol'] == 'AAPL'
        assert 'current_price' in result
        assert 'indicators' in result
        assert 'rsi' in result['indicators']
        assert 'note' in result
        assert 'Mock data' in result['note']
    
    def test_fallback_to_mock_when_no_lean_indicators(self):
        """测试当LEAN无指标时fallback到mock"""
        mock_algo = Mock()
        mock_algo.indicators = {}  # 空indicators
        
        agent = TechnicalAnalysisAgent(algorithm=mock_algo)
        result = agent._calculate_from_lean('AAPL')
        
        # 应该fallback到mock
        assert 'note' in result
        assert 'Mock data' in result['note']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
