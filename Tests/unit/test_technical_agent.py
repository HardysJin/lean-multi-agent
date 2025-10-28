"""
Test Technical Analysis Agent - 测试技术分析Agent

测试覆盖：
1. Agent初始化（有/无LEAN algorithm）
2. 指标计算（模拟数据）
3. 信号生成
4. 形态检测
5. 支撑阻力位识别
6. 资源读取
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock
import json

from Agents.technical_agent import TechnicalAnalysisAgent


def run_async(coro):
    """Helper function to run async code in tests"""
    return asyncio.run(coro)


class TestTechnicalAnalysisAgent:
    """测试技术分析Agent基础功能"""
    
    def test_initialization_without_algorithm(self):
        """测试无LEAN algorithm的初始化"""
        agent = TechnicalAnalysisAgent(algorithm=None)
        
        assert agent.name == "technical-analysis-agent"
        assert agent.description is not None
        assert agent.algorithm is None
    
    def test_initialization_with_algorithm(self):
        """测试有LEAN algorithm的初始化"""
        mock_algo = Mock()
        agent = TechnicalAnalysisAgent(algorithm=mock_algo)
        
        assert agent.algorithm == mock_algo
    
    def test_get_tools(self):
        """测试获取工具列表"""
        agent = TechnicalAnalysisAgent()
        tools = agent.get_tools()
        
        assert len(tools) == 4
        tool_names = [tool.name for tool in tools]
        assert "calculate_indicators" in tool_names
        assert "generate_signals" in tool_names
        assert "detect_patterns" in tool_names
        assert "find_support_resistance" in tool_names
    
    def test_get_resources(self):
        """测试获取资源列表"""
        agent = TechnicalAnalysisAgent()
        resources = agent.get_resources()
        
        assert len(resources) == 2
        resource_uris = [str(res.uri) for res in resources]
        assert "indicators://" in resource_uris
        assert "signals://" in resource_uris


class TestIndicatorCalculation:
    """测试指标计算"""
    
    @pytest.fixture
    def agent(self):
        """创建Agent实例（无LEAN）"""
        return TechnicalAnalysisAgent(algorithm=None)
    
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
    
    def test_calculate_indicators_tool(self, agent):
        """测试calculate_indicators工具"""
        result = run_async(agent.handle_tool_call("calculate_indicators", {
            "symbol": "AAPL"
        }))
        
        assert result['symbol'] == "AAPL"
        assert 'indicators' in result
        assert 'timestamp' in result
    
    def test_calculate_indicators_with_timeframe(self, agent):
        """测试带时间尺度的指标计算"""
        result = run_async(agent.handle_tool_call("calculate_indicators", {
            "symbol": "AAPL",
            "timeframe": "1h"
        }))
        
        assert result['symbol'] == "AAPL"
    
    def test_rsi_interpretation(self, agent):
        """测试RSI解读"""
        assert agent._interpret_rsi(25) == "oversold"
        assert agent._interpret_rsi(75) == "overbought"
        assert agent._interpret_rsi(50) == "neutral"
        assert agent._interpret_rsi(60) == "normal"
    
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
        return TechnicalAnalysisAgent(algorithm=None)
    
    def test_generate_signals_tool(self, agent):
        """测试generate_signals工具"""
        result = run_async(agent.handle_tool_call("generate_signals", {
            "symbol": "AAPL"
        }))
        
        assert result['symbol'] == "AAPL"
        assert result['action'] in ['BUY', 'SELL', 'HOLD']
        assert 1 <= result['conviction'] <= 10
        assert 'reasoning' in result
        assert 'score_breakdown' in result
    
    def test_signal_has_score_breakdown(self, agent):
        """测试信号包含分数明细"""
        result = run_async(agent.handle_tool_call("generate_signals", {
            "symbol": "TSLA"
        }))
        
        breakdown = result['score_breakdown']
        assert 'bullish' in breakdown
        assert 'bearish' in breakdown
        assert 'total' in breakdown
    
    def test_signal_includes_indicators(self, agent):
        """测试信号包含指标摘要"""
        result = run_async(agent.handle_tool_call("generate_signals", {
            "symbol": "MSFT"
        }))
        
        assert 'indicators_summary' in result
        assert 'current_price' in result


class TestPatternDetection:
    """测试形态检测"""
    
    @pytest.fixture
    def agent(self):
        """创建Agent实例"""
        return TechnicalAnalysisAgent(algorithm=None)
    
    def test_detect_patterns_tool(self, agent):
        """测试detect_patterns工具"""
        result = run_async(agent.handle_tool_call("detect_patterns", {
            "symbol": "AAPL"
        }))
        
        assert result['symbol'] == "AAPL"
        assert 'patterns' in result
        assert 'lookback_days' in result
    
    def test_detect_patterns_with_lookback(self, agent):
        """测试带回溯期的形态检测"""
        result = run_async(agent.handle_tool_call("detect_patterns", {
            "symbol": "AAPL",
            "lookback_days": 90
        }))
        
        assert result['lookback_days'] == 90
    
    def test_patterns_have_confidence(self, agent):
        """测试形态包含信心度"""
        result = agent._detect_patterns("AAPL", 60)
        
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
        return TechnicalAnalysisAgent(algorithm=None)
    
    def test_find_support_resistance_tool(self, agent):
        """测试find_support_resistance工具"""
        result = run_async(agent.handle_tool_call("find_support_resistance", {
            "symbol": "AAPL"
        }))
        
        assert result['symbol'] == "AAPL"
        assert 'current_price' in result
        assert 'support_levels' in result
        assert 'resistance_levels' in result
    
    def test_levels_have_properties(self, agent):
        """测试支撑阻力位包含必要属性"""
        result = agent._find_support_resistance("AAPL")
        
        for level in result['support_levels']:
            assert 'level' in level
            assert 'type' in level
            assert 'distance_pct' in level
            assert 'strength' in level
    
    def test_level_strength_calculation(self, agent):
        """测试强度计算"""
        assert agent._calculate_level_strength('sma20') == 0.5
        assert agent._calculate_level_strength('sma50') == 0.7
        assert agent._calculate_level_strength('sma200') == 0.9
        assert agent._calculate_level_strength('unknown') == 0.5


class TestResourceReading:
    """测试资源读取"""
    
    @pytest.fixture
    def agent(self):
        """创建Agent实例"""
        return TechnicalAnalysisAgent(algorithm=None)
    
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


class TestCaching:
    """测试缓存机制"""
    
    @pytest.fixture
    def agent(self):
        """创建Agent实例"""
        return TechnicalAnalysisAgent(algorithm=None)
    
    def test_cache_validity(self, agent):
        """测试缓存有效性检查"""
        # 新key应该无效
        assert not agent._is_cache_valid("test_key")
        
        # 添加到缓存
        from datetime import datetime
        agent._cache_timestamp["test_key"] = datetime.now()
        
        # 应该有效
        assert agent._is_cache_valid("test_key", max_age_seconds=60)
    
    def test_indicators_use_cache(self, agent):
        """测试指标计算使用缓存"""
        # 第一次调用
        result1 = agent._calculate_indicators("AAPL")
        
        # 第二次调用应该使用缓存
        result2 = agent._calculate_indicators("AAPL")
        
        # 应该是同一个对象（从缓存返回）
        assert result1 is result2


class TestErrorHandling:
    """测试错误处理"""
    
    @pytest.fixture
    def agent(self):
        """创建Agent实例"""
        return TechnicalAnalysisAgent(algorithm=None)
    
    def test_missing_symbol(self, agent):
        """测试缺少symbol参数"""
        with pytest.raises(ValueError) as exc_info:
            run_async(agent.handle_tool_call("calculate_indicators", {}))
        
        assert "Symbol is required" in str(exc_info.value)
    
    def test_unknown_tool(self, agent):
        """测试未知工具"""
        with pytest.raises(ValueError) as exc_info:
            run_async(agent.handle_tool_call("unknown_tool", {"symbol": "AAPL"}))
        
        assert "Unknown tool" in str(exc_info.value)


class TestLEANIntegration:
    """测试LEAN集成（使用mock）"""
    
    def test_calculate_from_lean_with_indicators(self):
        """测试从LEAN获取指标"""
        # 创建mock algorithm
        mock_algo = Mock()
        
        # Mock indicators
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
        
        # 创建agent
        agent = TechnicalAnalysisAgent(algorithm=mock_algo)
        
        # 计算指标
        result = agent._calculate_from_lean('AAPL')
        
        assert result['symbol'] == 'AAPL'
        assert result['current_price'] == 150.0
        assert result['indicators']['rsi']['value'] == 55.0
    
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
