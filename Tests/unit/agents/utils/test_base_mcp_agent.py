"""
Test Base MCP Agent - 测试MCP Agent基类

测试覆盖：
1. Agent初始化
2. Tools注册和调用
3. Resources注册和读取
4. Prompts注册和获取（可选）
5. ExampleAgent示例实现
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json

from Agents.base_mcp_agent import BaseMCPAgent, ExampleAgent
from mcp.types import Tool, Resource


def run_async(coro):
    """Helper function to run async code in tests"""
    return asyncio.run(coro)


class TestBaseMCPAgent:
    """测试BaseMCPAgent基类"""
    
    def test_initialization(self):
        """测试Agent初始化"""
        
        # 创建一个简单的子类用于测试
        class SimpleAgent(BaseMCPAgent):
            def get_tools(self):
                return []
            
            async def handle_tool_call(self, name, arguments):
                return {}
        
        agent = SimpleAgent("test-agent", "Test Agent", "1.0.0")
        
        assert agent.name == "test-agent"
        assert agent.description == "Test Agent"
        assert agent.version == "1.0.0"
        assert agent.server is not None
    
    def test_get_tools_not_implemented(self):
        """测试未实现get_tools会抛出异常"""
        
        class IncompleteAgent(BaseMCPAgent):
            async def handle_tool_call(self, name, arguments):
                return {}
        
        agent = IncompleteAgent("incomplete", "Incomplete Agent")
        
        with pytest.raises(NotImplementedError):
            agent.get_tools()
    
    def test_handle_tool_call_not_implemented(self):
        """测试未实现handle_tool_call会抛出异常"""
        
        class IncompleteAgent(BaseMCPAgent):
            def get_tools(self):
                return []
        
        agent = IncompleteAgent("incomplete", "Incomplete Agent")
        
        with pytest.raises(NotImplementedError):
            run_async(agent.handle_tool_call("test", {}))
    
    def test_validate_arguments(self):
        """测试参数验证"""
        
        class SimpleAgent(BaseMCPAgent):
            def get_tools(self):
                return []
            
            async def handle_tool_call(self, name, arguments):
                return {}
        
        agent = SimpleAgent("test", "Test")
        
        # 正常情况
        agent._validate_arguments({"key1": "value1", "key2": "value2"}, ["key1", "key2"])
        
        # 缺少参数
        with pytest.raises(ValueError) as exc_info:
            agent._validate_arguments({"key1": "value1"}, ["key1", "key2"])
        
        assert "Missing required arguments" in str(exc_info.value)
        assert "key2" in str(exc_info.value)
    
    def test_create_tool_schema(self):
        """测试创建Tool schema"""
        
        class SimpleAgent(BaseMCPAgent):
            def get_tools(self):
                return []
            
            async def handle_tool_call(self, name, arguments):
                return {}
        
        agent = SimpleAgent("test", "Test")
        
        tool = agent._create_tool_schema(
            name="test_tool",
            description="A test tool",
            properties={
                "param1": {"type": "string"},
                "param2": {"type": "number"}
            },
            required=["param1"]
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.inputSchema["type"] == "object"
        assert "param1" in tool.inputSchema["properties"]
        assert tool.inputSchema["required"] == ["param1"]
    
    def test_create_resource_uri(self):
        """测试创建资源URI"""
        
        class SimpleAgent(BaseMCPAgent):
            def get_tools(self):
                return []
            
            async def handle_tool_call(self, name, arguments):
                return {}
        
        agent = SimpleAgent("test", "Test")
        
        # 无标识符
        uri1 = agent._create_resource_uri("data")
        assert uri1 == "data://"
        
        # 有标识符
        uri2 = agent._create_resource_uri("news", "latest")
        assert uri2 == "news://latest"
        
        uri3 = agent._create_resource_uri("indicators", "AAPL")
        assert uri3 == "indicators://AAPL"
    
    def test_get_resources_default(self):
        """测试默认资源列表为空"""
        
        class SimpleAgent(BaseMCPAgent):
            def get_tools(self):
                return []
            
            async def handle_tool_call(self, name, arguments):
                return {}
        
        agent = SimpleAgent("test", "Test")
        resources = agent.get_resources()
        
        assert resources == []
    
    def test_get_prompts_default(self):
        """测试默认prompts列表为空"""
        
        class SimpleAgent(BaseMCPAgent):
            def get_tools(self):
                return []
            
            async def handle_tool_call(self, name, arguments):
                return {}
        
        agent = SimpleAgent("test", "Test")
        prompts = agent.get_prompts()
        
        assert prompts == []


class TestExampleAgent:
    """测试ExampleAgent示例实现"""
    
    @pytest.fixture
    def agent(self):
        """创建ExampleAgent实例"""
        return ExampleAgent()
    
    def test_initialization(self, agent):
        """测试初始化"""
        assert agent.name == "example-agent"
        assert agent.description == "A simple example agent that echoes input"
        assert agent.version == "1.0.0"
    
    def test_get_tools(self, agent):
        """测试获取工具列表"""
        tools = agent.get_tools()
        
        assert len(tools) == 2
        tool_names = [tool.name for tool in tools]
        assert "echo" in tool_names
        assert "greet" in tool_names
    
    def test_echo_tool(self, agent):
        """测试echo工具"""
        result = run_async(agent.handle_tool_call("echo", {"message": "Hello World"}))
        
        assert isinstance(result, dict)
        assert result["echo"] == "Hello World"
        assert "timestamp" in result
    
    def test_echo_missing_argument(self, agent):
        """测试echo缺少参数"""
        with pytest.raises(ValueError) as exc_info:
            run_async(agent.handle_tool_call("echo", {}))
        
        assert "Missing required arguments" in str(exc_info.value)
    
    def test_greet_tool_english(self, agent):
        """测试greet工具（英语）"""
        result = run_async(agent.handle_tool_call("greet", {
            "name": "Alice",
            "language": "en"
        }))
        
        assert result["greeting"] == "Hello, Alice!"
        assert result["language"] == "en"
    
    def test_greet_tool_chinese(self, agent):
        """测试greet工具（中文）"""
        result = run_async(agent.handle_tool_call("greet", {
            "name": "张三",
            "language": "zh"
        }))
        
        assert result["greeting"] == "你好，张三！"
        assert result["language"] == "zh"
    
    def test_greet_tool_default_language(self, agent):
        """测试greet工具（默认语言）"""
        result = run_async(agent.handle_tool_call("greet", {"name": "Bob"}))
        
        assert result["greeting"] == "Hello, Bob!"
        assert result["language"] == "en"
    
    def test_unknown_tool(self, agent):
        """测试调用未知工具"""
        with pytest.raises(ValueError) as exc_info:
            run_async(agent.handle_tool_call("unknown_tool", {}))
        
        assert "Unknown tool" in str(exc_info.value)
    
    def test_get_resources(self, agent):
        """测试获取资源列表"""
        resources = agent.get_resources()
        
        assert len(resources) == 1
        assert str(resources[0].uri) == "status://"
        assert resources[0].name == "Agent Status"
    
    def test_read_status_resource(self, agent):
        """测试读取status资源"""
        content = run_async(agent.handle_resource_read("status://"))
        
        assert isinstance(content, dict)
        assert content["status"] == "running"
        assert content["version"] == "1.0.0"
    
    def test_read_unknown_resource(self, agent):
        """测试读取未知资源"""
        content = run_async(agent.handle_resource_read("unknown://"))
        
        assert isinstance(content, dict)
        assert "error" in content


class TestToolSchemaCreation:
    """测试Tool schema创建的详细场景"""
    
    def test_tool_with_enum(self):
        """测试带enum的tool"""
        
        class TestAgent(BaseMCPAgent):
            def get_tools(self):
                return [
                    self._create_tool_schema(
                        name="analyze",
                        description="Analyze data",
                        properties={
                            "method": {
                                "type": "string",
                                "enum": ["quick", "deep", "full"],
                                "description": "Analysis method"
                            }
                        },
                        required=["method"]
                    )
                ]
            
            async def handle_tool_call(self, name, arguments):
                return {}
        
        agent = TestAgent("test", "Test")
        tools = agent.get_tools()
        
        assert len(tools) == 1
        assert tools[0].name == "analyze"
        assert "enum" in tools[0].inputSchema["properties"]["method"]
    
    def test_tool_with_nested_properties(self):
        """测试带嵌套属性的tool"""
        
        class TestAgent(BaseMCPAgent):
            def get_tools(self):
                return [
                    self._create_tool_schema(
                        name="process",
                        description="Process data",
                        properties={
                            "config": {
                                "type": "object",
                                "properties": {
                                    "threshold": {"type": "number"},
                                    "mode": {"type": "string"}
                                }
                            }
                        },
                        required=["config"]
                    )
                ]
            
            async def handle_tool_call(self, name, arguments):
                return {}
        
        agent = TestAgent("test", "Test")
        tools = agent.get_tools()
        
        assert len(tools) == 1
        config_schema = tools[0].inputSchema["properties"]["config"]
        assert config_schema["type"] == "object"
        assert "threshold" in config_schema["properties"]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
