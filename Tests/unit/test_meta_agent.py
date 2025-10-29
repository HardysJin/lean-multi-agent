"""
Test Meta Agent - 测试元Agent (MCP Client)

测试覆盖：
1. Meta Agent初始化
2. 连接specialist agents
3. 工具发现和执行
4. 资源读取
5. LLM决策流程（mock）
6. 记忆系统集成
7. 决策解析
8. 历史记录管理
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import json
from datetime import datetime

from Agents.meta_agent import (
    MetaAgent, MetaDecision, ToolCall, AgentConnection,
    create_meta_agent_with_technical
)
from Agents.technical_agent import TechnicalAnalysisAgent
from Agents.llm_config import LLMConfig, get_mock_llm
from Memory.state_manager import MultiTimeframeStateManager


def run_async(coro):
    """Helper function to run async code in tests"""
    return asyncio.run(coro)


class TestMetaAgentInitialization:
    """测试Meta Agent初始化"""
    
    def test_initialization_without_dependencies(self):
        """测试无依赖的基础初始化（禁用memory）"""
        meta = MetaAgent(enable_memory=False)
        
        # 使用默认LLM (from get_default_llm)
        assert meta.llm_client is not None
        assert meta.state_manager is None
        assert len(meta.agents) == 0
        assert len(meta.tool_call_history) == 0
        assert len(meta.decision_history) == 0
    
    def test_initialization_with_llm_config(self):
        """测试带LLM配置的初始化"""
        # 使用Mock LLM避免真实API调用
        mock_llm = get_mock_llm()
        llm_config = LLMConfig(model="mock", api_key="test-key")
        
        # 直接设置LLM实例
        meta = MetaAgent(llm_config=llm_config)
        
        # 验证LLM client类型（LLMConfig可能返回ChatOpenAI包装的mock）
        assert meta.llm_client is not None
    
    def test_initialization_with_state_manager(self):
        """测试带StateManager的初始化"""
        mock_state = Mock(spec=MultiTimeframeStateManager)
        meta = MetaAgent(state_manager=mock_state)
        
        assert meta.state_manager == mock_state


class TestAgentConnection:
    """测试Agent连接"""
    
    def test_connect_to_technical_agent(self):
        """测试连接到Technical Agent"""
        meta = MetaAgent()
        technical = TechnicalAnalysisAgent(algorithm=None)
        
        run_async(meta.connect_to_agent(
            agent_name="technical",
            agent_instance=technical,
            description="Technical analysis specialist"
        ))
        
        assert "technical" in meta.agents
        assert meta.agents["technical"].name == "technical"
        assert len(meta.agents["technical"].tools) == 4  # 4 tools
        assert len(meta.agents["technical"].resources) == 2  # 2 resources
    
    def test_connect_multiple_agents(self):
        """测试连接多个agents"""
        meta = MetaAgent()
        technical1 = TechnicalAnalysisAgent(algorithm=None)
        technical2 = TechnicalAnalysisAgent(algorithm=None)
        
        run_async(meta.connect_to_agent("technical1", technical1))
        run_async(meta.connect_to_agent("technical2", technical2))
        
        assert len(meta.agents) == 2
        assert "technical1" in meta.agents
        assert "technical2" in meta.agents
    
    def test_list_agents(self):
        """测试列出所有agents"""
        meta = MetaAgent()
        technical = TechnicalAnalysisAgent()
        
        run_async(meta.connect_to_agent("technical", technical))
        
        agents = meta.list_agents()
        assert agents == ["technical"]
    
    def test_get_agent_info(self):
        """测试获取agent信息"""
        meta = MetaAgent()
        technical = TechnicalAnalysisAgent()
        
        run_async(meta.connect_to_agent("technical", technical, "Tech analysis"))
        
        info = meta.get_agent_info("technical")
        assert info is not None
        assert info["name"] == "technical"
        assert info["description"] == "Tech analysis"
        assert len(info["tools"]) == 4
        assert len(info["resources"]) == 2
    
    def test_get_nonexistent_agent_info(self):
        """测试获取不存在的agent信息"""
        meta = MetaAgent()
        
        info = meta.get_agent_info("nonexistent")
        assert info is None


class TestToolDiscovery:
    """测试工具发现"""
    
    @pytest.fixture
    def meta_with_agent(self):
        """创建已连接agent的Meta实例"""
        meta = MetaAgent()
        technical = TechnicalAnalysisAgent()
        run_async(meta.connect_to_agent("technical", technical))
        return meta
    
    def test_get_all_tools(self, meta_with_agent):
        """测试获取所有工具"""
        tools = meta_with_agent.get_all_tools()
        
        assert len(tools) == 4  # Technical agent has 4 tools
        
        # 验证工具包含agent_name
        for tool in tools:
            assert 'agent_name' in tool
            assert tool['agent_name'] == 'technical'
            assert 'name' in tool
            assert 'description' in tool
            assert 'inputSchema' in tool
    
    def test_tools_have_required_fields(self, meta_with_agent):
        """测试工具包含必要字段"""
        tools = meta_with_agent.get_all_tools()
        
        tool_names = [t['name'] for t in tools]
        assert 'calculate_indicators' in tool_names
        assert 'generate_signals' in tool_names
        assert 'detect_patterns' in tool_names
        assert 'find_support_resistance' in tool_names


class TestToolExecution:
    """测试工具执行"""
    
    @pytest.fixture
    def meta_with_agent(self):
        """创建已连接agent的Meta实例"""
        meta = MetaAgent()
        technical = TechnicalAnalysisAgent()
        run_async(meta.connect_to_agent("technical", technical))
        return meta
    
    def test_execute_calculate_indicators(self, meta_with_agent):
        """测试执行calculate_indicators工具"""
        result = run_async(meta_with_agent.execute_tool(
            agent_name="technical",
            tool_name="calculate_indicators",
            arguments={"symbol": "AAPL"}
        ))
        
        assert result['symbol'] == "AAPL"
        assert 'indicators' in result
        assert 'timestamp' in result
    
    def test_execute_generate_signals(self, meta_with_agent):
        """测试执行generate_signals工具"""
        result = run_async(meta_with_agent.execute_tool(
            agent_name="technical",
            tool_name="generate_signals",
            arguments={"symbol": "TSLA"}
        ))
        
        assert result['symbol'] == "TSLA"
        assert result['action'] in ['BUY', 'SELL', 'HOLD']
        assert 1 <= result['conviction'] <= 10
    
    def test_tool_execution_recorded_in_history(self, meta_with_agent):
        """测试工具执行被记录到历史"""
        initial_count = len(meta_with_agent.tool_call_history)
        
        run_async(meta_with_agent.execute_tool(
            agent_name="technical",
            tool_name="calculate_indicators",
            arguments={"symbol": "AAPL"}
        ))
        
        assert len(meta_with_agent.tool_call_history) == initial_count + 1
        
        last_call = meta_with_agent.tool_call_history[-1]
        assert last_call.agent_name == "technical"
        assert last_call.tool_name == "calculate_indicators"
        assert last_call.arguments == {"symbol": "AAPL"}
        assert last_call.execution_time_ms > 0
    
    def test_execute_nonexistent_agent(self, meta_with_agent):
        """测试执行不存在的agent的工具"""
        with pytest.raises(ValueError) as exc_info:
            run_async(meta_with_agent.execute_tool(
                agent_name="nonexistent",
                tool_name="some_tool",
                arguments={}
            ))
        
        assert "not connected" in str(exc_info.value)
    
    def test_execute_nonexistent_tool(self, meta_with_agent):
        """测试执行不存在的工具"""
        with pytest.raises(ValueError) as exc_info:
            run_async(meta_with_agent.execute_tool(
                agent_name="technical",
                tool_name="nonexistent_tool",
                arguments={}
            ))
        
        assert "not found" in str(exc_info.value)
    
    def test_tool_execution_error_handling(self, meta_with_agent):
        """测试工具执行错误处理"""
        # calculate_indicators需要symbol参数
        with pytest.raises(ValueError):
            run_async(meta_with_agent.execute_tool(
                agent_name="technical",
                tool_name="calculate_indicators",
                arguments={}  # 缺少symbol
            ))
        
        # 错误的调用应该也被记录
        last_call = meta_with_agent.tool_call_history[-1]
        assert 'error' in last_call.result


class TestResourceReading:
    """测试资源读取"""
    
    @pytest.fixture
    def meta_with_agent(self):
        """创建已连接agent的Meta实例"""
        meta = MetaAgent()
        technical = TechnicalAnalysisAgent()
        run_async(meta.connect_to_agent("technical", technical))
        return meta
    
    def test_read_indicators_resource(self, meta_with_agent):
        """测试读取indicators资源"""
        result = run_async(meta_with_agent.read_resource(
            agent_name="technical",
            resource_uri="indicators://AAPL"
        ))
        
        assert result['symbol'] == "AAPL"
        assert 'indicators' in result
    
    def test_read_signals_resource(self, meta_with_agent):
        """测试读取signals资源"""
        result = run_async(meta_with_agent.read_resource(
            agent_name="technical",
            resource_uri="signals://AAPL"
        ))
        
        assert result['symbol'] == "AAPL"
        assert result['action'] in ['BUY', 'SELL', 'HOLD']
    
    def test_read_resource_from_nonexistent_agent(self, meta_with_agent):
        """测试从不存在的agent读取资源"""
        with pytest.raises(ValueError) as exc_info:
            run_async(meta_with_agent.read_resource(
                agent_name="nonexistent",
                resource_uri="some://uri"
            ))
        
        assert "not connected" in str(exc_info.value)


class TestMemoryIntegration:
    """测试记忆系统集成"""
    
    def test_retrieve_memory_without_state_manager(self):
        """测试无StateManager时的记忆检索"""
        meta = MetaAgent(state_manager=None, enable_memory=False)
        
        context = meta._retrieve_memory_context("AAPL")
        
        assert 'note' in context
        assert context['recent_decisions'] == []
    
    def test_retrieve_memory_with_state_manager(self):
        """测试有StateManager时的记忆检索"""
        # Mock StateManager and sql_store
        mock_state = Mock(spec=MultiTimeframeStateManager)
        mock_sql_store = Mock()
        mock_sql_store.get_recent_decisions.return_value = []
        mock_state.sql_store = mock_sql_store
        
        meta = MetaAgent(state_manager=mock_state)
        
        context = meta._retrieve_memory_context("AAPL", lookback_hours=24)
        
        # 验证调用了正确的方法
        mock_sql_store.get_recent_decisions.assert_called_once_with(symbol="AAPL", limit=5)
        
        assert 'recent_decisions' in context
    
    def test_memory_retrieval_error_handling(self):
        """测试记忆检索错误处理"""
        mock_state = Mock(spec=MultiTimeframeStateManager)
        mock_sql_store = Mock()
        mock_sql_store.get_recent_decisions.side_effect = Exception("Database error")
        mock_state.sql_store = mock_sql_store
        
        meta = MetaAgent(state_manager=mock_state)
        
        context = meta._retrieve_memory_context("AAPL")
        
        assert 'error' in context
        assert 'Database error' in context['error']


class TestDecisionParsing:
    """测试决策解析"""
    
    def test_parse_buy_decision(self):
        """测试解析BUY决策"""
        meta = MetaAgent()
        
        response = """Based on the analysis:
ACTION: BUY
CONVICTION: 8
REASONING: Strong technical indicators with RSI showing oversold conditions and MACD showing bullish crossover."""
        
        decision = meta._parse_decision(
            symbol="AAPL",
            response=response,
            tool_calls=[]
        )
        
        assert decision.symbol == "AAPL"
        assert decision.action == "BUY"
        assert decision.conviction == 8
        assert "RSI" in decision.reasoning
        assert "MACD" in decision.reasoning
    
    def test_parse_sell_decision(self):
        """测试解析SELL决策"""
        meta = MetaAgent()
        
        response = """ACTION: SELL
CONVICTION: 9
REASONING: Overbought conditions detected."""
        
        decision = meta._parse_decision(
            symbol="TSLA",
            response=response,
            tool_calls=[]
        )
        
        assert decision.action == "SELL"
        assert decision.conviction == 9
    
    def test_parse_hold_decision(self):
        """测试解析HOLD决策"""
        meta = MetaAgent()
        
        response = """ACTION: HOLD
CONVICTION: 5
REASONING: Mixed signals, need more data."""
        
        decision = meta._parse_decision(
            symbol="MSFT",
            response=response,
            tool_calls=[]
        )
        
        assert decision.action == "HOLD"
        assert decision.conviction == 5
    
    def test_parse_decision_with_default_values(self):
        """测试使用默认值解析不完整的响应"""
        meta = MetaAgent()
        
        response = "Some analysis without proper format"
        
        decision = meta._parse_decision(
            symbol="AAPL",
            response=response,
            tool_calls=[]
        )
        
        assert decision.action == "HOLD"  # 默认
        assert decision.conviction == 5  # 默认
        assert decision.reasoning == response
    
    def test_parse_decision_with_tool_calls(self):
        """测试解析包含工具调用的决策"""
        meta = MetaAgent()
        
        tool_call = ToolCall(
            agent_name="technical",
            tool_name="calculate_indicators",
            arguments={"symbol": "AAPL"},
            result={"rsi": 45},
            timestamp=datetime.now(),
            execution_time_ms=100
        )
        
        response = """ACTION: BUY
CONVICTION: 7
REASONING: Based on technical analysis"""
        
        decision = meta._parse_decision(
            symbol="AAPL",
            response=response,
            tool_calls=[tool_call]
        )
        
        assert len(decision.tool_calls) == 1
        assert decision.evidence['tools_used'][0]['agent'] == 'technical'
    
    def test_conviction_boundary_handling(self):
        """测试conviction边界处理"""
        meta = MetaAgent()
        
        # 测试超出范围的值
        response = """ACTION: BUY
CONVICTION: 15
REASONING: Test"""
        
        decision = meta._parse_decision("AAPL", response, [])
        assert decision.conviction == 10  # 应该被限制在10
        
        response = """ACTION: BUY
CONVICTION: -5
REASONING: Test"""
        
        decision = meta._parse_decision("AAPL", response, [])
        assert decision.conviction == 1  # 应该被限制在1


class TestDecisionToDecisionRecord:
    """测试MetaDecision转换为DecisionRecord"""
    
    def test_conversion_to_decision_record(self):
        """测试转换到DecisionRecord"""
        from Memory.schemas import Timeframe
        
        decision = MetaDecision(
            symbol="AAPL",
            action="BUY",
            conviction=8,
            reasoning="Strong buy signal",
            evidence={"test": "data"},
            tool_calls=[],
            timestamp=datetime.now()
        )
        
        decision_record = decision.to_decision_record()
        
        assert decision_record.symbol == "AAPL"
        assert decision_record.action == "BUY"
        assert decision_record.conviction == 8.0
        assert decision_record.reasoning == "Strong buy signal"
        assert decision_record.timeframe == Timeframe.TACTICAL
        assert decision_record.agent_name == "meta_agent"
        assert 'evidence' in decision_record.metadata


class TestLLMIntegration:
    """测试LLM集成（使用mock）"""
    
    def test_build_system_prompt(self):
        """测试构建系统提示"""
        meta = MetaAgent()
        technical = TechnicalAnalysisAgent()
        run_async(meta.connect_to_agent("technical", technical, "Tech specialist"))
        
        system_prompt = meta._build_system_prompt()
        
        assert "Meta Agent" in system_prompt
        assert "technical" in system_prompt
        assert "BUY" in system_prompt
        assert "SELL" in system_prompt
        assert "HOLD" in system_prompt
    
    def test_format_tools_for_llm(self):
        """测试格式化工具给LLM"""
        meta = MetaAgent()
        technical = TechnicalAnalysisAgent()
        run_async(meta.connect_to_agent("technical", technical))
        
        tools = meta._format_tools_for_llm()
        
        assert len(tools) == 4
        
        # 验证工具名称包含agent前缀
        tool_names = [t['name'] for t in tools]
        assert "technical__calculate_indicators" in tool_names
        assert "technical__generate_signals" in tool_names
        
        # 验证格式符合Anthropic要求
        for tool in tools:
            assert 'name' in tool
            assert 'description' in tool
            assert 'input_schema' in tool


class TestHistoryManagement:
    """测试历史记录管理"""
    
    def test_get_tool_call_history(self):
        """测试获取工具调用历史"""
        meta = MetaAgent()
        technical = TechnicalAnalysisAgent()
        run_async(meta.connect_to_agent("technical", technical))
        
        # 执行一些工具调用
        run_async(meta.execute_tool("technical", "calculate_indicators", {"symbol": "AAPL"}))
        run_async(meta.execute_tool("technical", "generate_signals", {"symbol": "AAPL"}))
        
        history = meta.get_tool_call_history()
        assert len(history) == 2
        
        # 测试限制数量
        limited = meta.get_tool_call_history(limit=1)
        assert len(limited) == 1
        assert limited[0].tool_name == "generate_signals"  # 最后一个
    
    def test_get_decision_history(self):
        """测试获取决策历史"""
        meta = MetaAgent()
        
        # 手动添加一些决策
        decision1 = MetaDecision(
            symbol="AAPL",
            action="BUY",
            conviction=7,
            reasoning="Test",
            evidence={},
            tool_calls=[],
            timestamp=datetime.now()
        )
        decision2 = MetaDecision(
            symbol="TSLA",
            action="SELL",
            conviction=8,
            reasoning="Test",
            evidence={},
            tool_calls=[],
            timestamp=datetime.now()
        )
        
        meta.decision_history = [decision1, decision2]
        
        history = meta.get_decision_history()
        assert len(history) == 2
        
        limited = meta.get_decision_history(limit=1)
        assert len(limited) == 1
        assert limited[0].symbol == "TSLA"
    
    def test_clear_history(self):
        """测试清空历史"""
        meta = MetaAgent()
        technical = TechnicalAnalysisAgent()
        run_async(meta.connect_to_agent("technical", technical))
        
        # 添加一些历史
        run_async(meta.execute_tool("technical", "calculate_indicators", {"symbol": "AAPL"}))
        
        decision = MetaDecision(
            symbol="AAPL",
            action="BUY",
            conviction=7,
            reasoning="Test",
            evidence={},
            tool_calls=[],
            timestamp=datetime.now()
        )
        meta.decision_history.append(decision)
        
        assert len(meta.tool_call_history) > 0
        assert len(meta.decision_history) > 0
        
        # 清空
        meta.clear_history()
        
        assert len(meta.tool_call_history) == 0
        assert len(meta.decision_history) == 0


class TestConvenienceFunction:
    """测试便捷函数"""
    
    def test_create_meta_agent_with_technical(self):
        """测试便捷创建函数"""
        meta = run_async(create_meta_agent_with_technical(
            llm_config=None,  # 使用默认LLM配置
            state_manager=None,
            algorithm=None
        ))
        
        assert isinstance(meta, MetaAgent)
        assert "technical" in meta.agents
        assert len(meta.get_all_tools()) == 4


class TestEndToEndWorkflow:
    """测试端到端工作流（不使用真实LLM）"""
    
    def test_complete_workflow_without_llm(self):
        """测试完整工作流（手动工具调用）"""
        # 创建Meta Agent
        meta = MetaAgent(llm_config=None)  # 使用默认LLM配置
        technical = TechnicalAnalysisAgent()
        run_async(meta.connect_to_agent("technical", technical))
        
        # 1. 获取可用工具
        tools = meta.get_all_tools()
        assert len(tools) > 0
        
        # 2. 执行工具获取数据
        indicators = run_async(meta.execute_tool(
            "technical",
            "calculate_indicators",
            {"symbol": "AAPL"}
        ))
        assert 'indicators' in indicators
        
        signals = run_async(meta.execute_tool(
            "technical",
            "generate_signals",
            {"symbol": "AAPL"}
        ))
        assert 'action' in signals
        
        # 3. 读取资源
        resource_data = run_async(meta.read_resource(
            "technical",
            "indicators://AAPL"
        ))
        assert 'indicators' in resource_data
        
        # 4. 验证历史记录
        assert len(meta.tool_call_history) == 2
        
        # 5. 手动创建决策（模拟LLM输出）
        decision = meta._parse_decision(
            symbol="AAPL",
            response=f"""ACTION: {signals['action']}
CONVICTION: {signals['conviction']}
REASONING: Based on technical analysis showing RSI at {indicators['indicators']['rsi']['value']}""",
            tool_calls=meta.tool_call_history.copy()
        )
        
        assert decision.symbol == "AAPL"
        assert decision.action == signals['action']
        assert len(decision.tool_calls) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
