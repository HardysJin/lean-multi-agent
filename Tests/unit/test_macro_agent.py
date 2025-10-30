"""
Tests for MacroAgent

测试覆盖：
1. 基本功能测试
2. 缓存机制测试
3. 时间控制测试（防止Look-Ahead）
4. Dependency Injection测试
5. 多实例隔离测试
6. LLM集成测试
7. 工具调用测试
8. 资源读取测试
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json

from Agents.macro_agent import MacroAgent, MacroContext
from Agents.llm_config import LLMConfig


class TestMacroAgentBasics:
    """基本功能测试"""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """测试初始化"""
        agent = MacroAgent()
        
        assert agent.name == "macro-agent"
        assert agent.enable_cache is True
        assert agent.cache_ttl == 3600
        assert len(agent._cache) == 0
    
    @pytest.mark.asyncio
    async def test_initialization_with_custom_config(self):
        """测试自定义配置"""
        agent = MacroAgent(
            cache_ttl=7200,
            enable_cache=False
        )
        
        assert agent.cache_ttl == 7200
        assert agent.enable_cache is False
    
    @pytest.mark.asyncio
    async def test_analyze_macro_environment_basic(self):
        """测试基本宏观分析"""
        agent = MacroAgent()
        
        context = await agent.analyze_macro_environment()
        
        # 验证返回类型
        assert isinstance(context, MacroContext)
        
        # 验证必要字段
        assert context.market_regime in ['bull', 'bear', 'sideways', 'transition']
        assert 0 <= context.regime_confidence <= 1
        assert context.interest_rate_trend in ['rising', 'falling', 'stable']
        assert context.risk_level >= 0
        assert context.volatility_level in ['low', 'medium', 'high', 'extreme']
        assert isinstance(context.constraints, dict)
        assert 'max_risk_per_trade' in context.constraints
        assert 'allow_long' in context.constraints
    
    @pytest.mark.asyncio
    async def test_get_market_regime(self):
        """测试快速regime获取"""
        agent = MacroAgent()
        
        result = await agent.get_market_regime()
        
        assert 'regime' in result
        assert 'confidence' in result
        assert 'reasoning' in result
        assert result['regime'] in ['bull', 'bear', 'sideways', 'transition']
        assert 0 <= result['confidence'] <= 1
    
    @pytest.mark.asyncio
    async def test_get_risk_constraints(self):
        """测试风险约束获取"""
        agent = MacroAgent()
        
        constraints = await agent.get_risk_constraints()
        
        assert isinstance(constraints, dict)
        assert 'max_risk_per_trade' in constraints
        assert 'max_portfolio_risk' in constraints
        assert 'allow_long' in constraints
        assert 'allow_short' in constraints
        assert 'max_position_size' in constraints


class TestMacroAgentCache:
    """缓存机制测试"""
    
    @pytest.mark.asyncio
    async def test_cache_works(self):
        """测试缓存生效"""
        agent = MacroAgent(cache_ttl=3600)
        
        # 第一次调用
        context1 = await agent.analyze_macro_environment()
        
        # 第二次调用（应该从cache返回）
        context2 = await agent.analyze_macro_environment()
        
        # 应该是同一个对象（从cache返回）
        assert context1.analysis_timestamp == context2.analysis_timestamp
        
        # 验证cache中有数据
        assert len(agent._cache) > 0
    
    @pytest.mark.asyncio
    async def test_cache_disabled(self):
        """测试禁用缓存"""
        agent = MacroAgent(enable_cache=False)
        
        context1 = await agent.analyze_macro_environment()
        await asyncio.sleep(0.1)
        context2 = await agent.analyze_macro_environment()
        
        # 应该是不同的分析（时间戳不同）
        assert context1.analysis_timestamp != context2.analysis_timestamp
        
        # cache应该为空
        assert len(agent._cache) == 0
    
    @pytest.mark.asyncio
    async def test_force_refresh_ignores_cache(self):
        """测试强制刷新忽略缓存"""
        agent = MacroAgent()
        
        # 第一次调用
        context1 = await agent.analyze_macro_environment()
        
        await asyncio.sleep(0.1)
        
        # 强制刷新
        context2 = await agent.analyze_macro_environment(force_refresh=True)
        
        # 应该是新的分析
        assert context1.analysis_timestamp != context2.analysis_timestamp
    
    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """测试清空缓存"""
        agent = MacroAgent()
        
        await agent.analyze_macro_environment()
        assert len(agent._cache) > 0
        
        agent.clear_cache()
        assert len(agent._cache) == 0
    
    @pytest.mark.asyncio
    async def test_get_cache_stats(self):
        """测试缓存统计"""
        agent = MacroAgent()
        
        await agent.analyze_macro_environment()
        
        stats = agent.get_cache_stats()
        
        assert stats['cache_enabled'] is True
        assert stats['cache_ttl'] == 3600
        assert stats['cached_items'] > 0
        assert isinstance(stats['cache_keys'], list)


class TestMacroAgentTimeControl:
    """时间控制测试（防止Look-Ahead）"""
    
    @pytest.mark.asyncio
    async def test_backtest_mode_with_time(self):
        """测试回测模式（指定时间）"""
        agent = MacroAgent()
        
        backtest_time = datetime(2023, 6, 1, 0, 0, 0)
        context = await agent.analyze_macro_environment(visible_data_end=backtest_time)
        
        # 验证记录了数据截止时间
        assert context.data_end_time == backtest_time
    
    @pytest.mark.asyncio
    async def test_different_times_different_cache(self):
        """测试不同时间点使用不同缓存"""
        agent = MacroAgent()
        
        time1 = datetime(2023, 6, 1)
        time2 = datetime(2023, 7, 1)
        
        context1 = await agent.analyze_macro_environment(visible_data_end=time1)
        context2 = await agent.analyze_macro_environment(visible_data_end=time2)
        
        # 应该是不同的分析结果
        assert context1.data_end_time != context2.data_end_time
        
        # cache中应该有2个条目
        assert len(agent._cache) == 2


class TestMacroAgentDependencyInjection:
    """Dependency Injection测试"""
    
    @pytest.mark.asyncio
    async def test_multiple_instances_isolated(self):
        """测试多实例隔离"""
        agent1 = MacroAgent()
        agent2 = MacroAgent()
        
        # 两个实例应该有独立的cache
        await agent1.analyze_macro_environment()
        agent1._cache['test_key'] = ('test_value', datetime.now())
        
        assert 'test_key' in agent1._cache
        assert 'test_key' not in agent2._cache
    
    @pytest.mark.asyncio
    async def test_custom_llm_config_injection(self):
        """测试自定义LLM配置注入"""
        # 创建mock LLM config
        mock_llm_config = Mock(spec=LLMConfig)
        mock_llm = AsyncMock()
        mock_llm_config.get_llm.return_value = mock_llm
        
        # 模拟LLM响应
        mock_response = Mock()
        mock_response.content = json.dumps({
            'market_regime': 'bull',
            'regime_confidence': 0.9,
            'interest_rate_trend': 'rising',
            'current_rate': 5.25,
            'risk_level': 4.0,
            'volatility_level': 'medium',
            'gdp_trend': 'expanding',
            'inflation_level': 'moderate',
            'market_sentiment': 'greed',
            'vix_level': 15.0,
            'confidence_score': 0.85,
            'reasoning': 'Test reasoning'
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        agent = MacroAgent(llm_config=mock_llm_config)
        agent._llm_client = mock_llm  # 直接注入mock LLM
        
        context = await agent.analyze_macro_environment()
        
        # 验证使用了mock LLM
        assert mock_llm.ainvoke.called
        assert context.market_regime == 'bull'
        assert context.reasoning == 'Test reasoning'


class TestMacroAgentTools:
    """MCP工具调用测试"""
    
    def test_get_tools(self):
        """测试工具列表"""
        agent = MacroAgent()
        
        tools = agent.get_tools()
        
        assert len(tools) == 3
        tool_names = [tool.name for tool in tools]
        assert 'analyze_macro_environment' in tool_names
        assert 'get_market_regime' in tool_names
        assert 'get_risk_constraints' in tool_names
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_analyze(self):
        """测试analyze工具调用"""
        agent = MacroAgent()
        
        result = await agent.handle_tool_call(
            'analyze_macro_environment',
            {'force_refresh': False}
        )
        
        assert isinstance(result, dict)
        assert 'market_regime' in result
        assert 'constraints' in result
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_regime(self):
        """测试regime工具调用"""
        agent = MacroAgent()
        
        result = await agent.handle_tool_call(
            'get_market_regime',
            {}
        )
        
        assert isinstance(result, dict)
        assert 'regime' in result
        assert 'confidence' in result
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_constraints(self):
        """测试constraints工具调用"""
        agent = MacroAgent()
        
        result = await agent.handle_tool_call(
            'get_risk_constraints',
            {}
        )
        
        assert isinstance(result, dict)
        assert 'max_risk_per_trade' in result
        assert 'allow_long' in result
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_with_time(self):
        """测试工具调用（带时间参数）"""
        agent = MacroAgent()
        
        result = await agent.handle_tool_call(
            'analyze_macro_environment',
            {'visible_data_end': '2023-06-01T00:00:00'}
        )
        
        assert result['data_end_time'] == '2023-06-01T00:00:00'


class TestMacroAgentResources:
    """MCP资源读取测试"""
    
    def test_get_resources(self):
        """测试资源列表"""
        agent = MacroAgent()
        
        resources = agent.get_resources()
        
        assert len(resources) == 2
        uris = [str(res.uri) for res in resources]  # 转换为字符串进行比较
        assert 'macro://current' in uris
        assert 'macro://cache-stats' in uris
    
    @pytest.mark.asyncio
    async def test_read_current_resource(self):
        """测试读取当前宏观环境资源"""
        agent = MacroAgent()
        
        content = await agent.handle_resource_read('macro://current')
        
        data = json.loads(content)
        assert 'market_regime' in data
        assert 'constraints' in data
    
    @pytest.mark.asyncio
    async def test_read_cache_stats_resource(self):
        """测试读取缓存统计资源"""
        agent = MacroAgent()
        
        content = await agent.handle_resource_read('macro://cache-stats')
        
        stats = json.loads(content)
        assert 'cache_enabled' in stats
        assert 'cache_ttl' in stats
        assert 'cached_items' in stats


class TestMacroContext:
    """MacroContext数据结构测试"""
    
    def test_to_dict(self):
        """测试转换为字典"""
        context = MacroContext(
            market_regime='bull',
            regime_confidence=0.8,
            interest_rate_trend='rising',
            current_rate=5.25,
            risk_level=4.0,
            volatility_level='medium',
            gdp_trend='expanding',
            inflation_level='moderate',
            market_sentiment='greed',
            vix_level=15.0,
            constraints={'max_risk': 0.02},
            analysis_timestamp=datetime(2023, 6, 1, 12, 0, 0),
            data_end_time=datetime(2023, 6, 1, 0, 0, 0),
            confidence_score=0.85,
            reasoning='Test reasoning'
        )
        
        data = context.to_dict()
        
        assert data['market_regime'] == 'bull'
        assert data['analysis_timestamp'] == '2023-06-01T12:00:00'
        assert data['data_end_time'] == '2023-06-01T00:00:00'
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'market_regime': 'bull',
            'regime_confidence': 0.8,
            'interest_rate_trend': 'rising',
            'current_rate': 5.25,
            'risk_level': 4.0,
            'volatility_level': 'medium',
            'gdp_trend': 'expanding',
            'inflation_level': 'moderate',
            'market_sentiment': 'greed',
            'vix_level': 15.0,
            'constraints': {'max_risk': 0.02},
            'analysis_timestamp': '2023-06-01T12:00:00',
            'data_end_time': '2023-06-01T00:00:00',
            'confidence_score': 0.85,
            'reasoning': 'Test reasoning'
        }
        
        context = MacroContext.from_dict(data)
        
        assert context.market_regime == 'bull'
        assert isinstance(context.analysis_timestamp, datetime)
        assert isinstance(context.data_end_time, datetime)


class TestMacroAgentIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """测试完整工作流"""
        agent = MacroAgent(cache_ttl=60)
        
        # 1. 执行完整分析
        context = await agent.analyze_macro_environment()
        assert isinstance(context, MacroContext)
        
        # 2. 快速获取regime（应该从cache）
        regime = await agent.get_market_regime()
        assert regime['regime'] == context.market_regime
        
        # 3. 获取约束（应该从cache）
        constraints = await agent.get_risk_constraints()
        assert constraints == context.constraints
        
        # 4. 验证cache
        stats = agent.get_cache_stats()
        assert stats['cached_items'] > 0
        
        # 5. 清空cache
        agent.clear_cache()
        stats_after_clear = agent.get_cache_stats()
        assert stats_after_clear['cached_items'] == 0
    
    @pytest.mark.asyncio
    async def test_backtest_scenario(self):
        """测试回测场景"""
        agent = MacroAgent()
        
        # 模拟回测：分析2023年多个时间点
        dates = [
            datetime(2023, 1, 1),
            datetime(2023, 3, 1),
            datetime(2023, 6, 1),
            datetime(2023, 9, 1),
        ]
        
        contexts = []
        for date in dates:
            context = await agent.analyze_macro_environment(visible_data_end=date)
            contexts.append(context)
            assert context.data_end_time == date
        
        # 验证产生了4个不同的分析
        assert len(contexts) == 4
        
        # 验证cache中有4个条目
        stats = agent.get_cache_stats()
        assert stats['cached_items'] == 4


class TestMacroAgentConstraints:
    """约束条件生成测试"""
    
    @pytest.mark.asyncio
    async def test_bull_market_constraints(self):
        """测试牛市约束"""
        agent = MacroAgent()
        
        # Mock分析结果为牛市
        with patch.object(agent, '_llm_analyze', return_value={
            'market_regime': 'bull',
            'regime_confidence': 0.9,
            'interest_rate_trend': 'stable',
            'current_rate': 5.25,
            'risk_level': 3.0,
            'volatility_level': 'low',
            'gdp_trend': 'expanding',
            'inflation_level': 'moderate',
            'market_sentiment': 'greed',
            'vix_level': 12.0,
            'confidence_score': 0.9,
            'reasoning': 'Bull market'
        }):
            context = await agent.analyze_macro_environment()
            
            # 牛市应该允许做多，仓位更大
            assert context.constraints['allow_long'] is True
            assert context.constraints['max_position_size'] >= 0.20
    
    @pytest.mark.asyncio
    async def test_bear_market_constraints(self):
        """测试熊市约束"""
        agent = MacroAgent()
        
        # Mock分析结果为熊市
        with patch.object(agent, '_llm_analyze', return_value={
            'market_regime': 'bear',
            'regime_confidence': 0.9,
            'interest_rate_trend': 'rising',
            'current_rate': 5.25,
            'risk_level': 8.0,
            'volatility_level': 'high',
            'gdp_trend': 'contracting',
            'inflation_level': 'high',
            'market_sentiment': 'fear',
            'vix_level': 35.0,
            'confidence_score': 0.85,
            'reasoning': 'Bear market'
        }):
            context = await agent.analyze_macro_environment()
            
            # 熊市应该禁止做多
            assert context.constraints['allow_long'] is False
            assert context.constraints['allow_short'] is True
            assert context.constraints['max_position_size'] <= 0.15


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
