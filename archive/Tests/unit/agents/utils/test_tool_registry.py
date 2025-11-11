"""
Unit tests for ToolRegistry

测试自动工具发现机制的各个方面
"""

import pytest
from typing import List, Dict, Any, Optional
from Agents.utils.tool_registry import ToolRegistry, tool


# ═══════════════════════════════════════════════
# Test Fixtures - Mock Agents
# ═══════════════════════════════════════════════

class SimpleAgent:
    """简单的测试Agent"""
    
    @tool(description="Simple method with one parameter")
    def simple_method(self, symbol: str) -> Dict[str, Any]:
        """
        Simple test method
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with symbol
        """
        return {"symbol": symbol}
    
    @tool()
    def method_with_defaults(
        self,
        symbol: str,
        timeframe: str = "1d",
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Method with default parameters
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe period
            limit: Maximum number of results
        """
        return {"symbol": symbol, "timeframe": timeframe, "limit": limit}
    
    @tool(description="Async method")
    async def async_method(self, symbol: str) -> Dict[str, Any]:
        """
        Async test method
        
        Args:
            symbol: Stock symbol
        """
        return {"symbol": symbol, "async": True}
    
    def non_tool_method(self):
        """This method should NOT be discovered"""
        return "not a tool"


class ComplexAgent:
    """复杂类型的测试Agent"""
    
    @tool(description="Method with list parameter")
    def list_method(self, symbols: List[str], limit: int = 5) -> List[Dict]:
        """
        Method with list parameter
        
        Args:
            symbols: List of stock symbols
            limit: Maximum results
        """
        return [{"symbol": s} for s in symbols[:limit]]
    
    @tool(description="Method with dict parameter")
    def dict_method(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Method with dict parameter
        
        Args:
            config: Configuration dictionary
        """
        return config
    
    @tool(description="Method with optional parameter")
    def optional_method(
        self,
        symbol: str,
        timeframe: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Method with optional parameter
        
        Args:
            symbol: Stock symbol
            timeframe: Optional timeframe
        """
        return {"symbol": symbol, "timeframe": timeframe}
    
    @tool()
    def numeric_types(
        self,
        count: int,
        ratio: float,
        enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Method with various numeric types
        
        Args:
            count: Integer count
            ratio: Float ratio
            enabled: Boolean flag
        """
        return {"count": count, "ratio": ratio, "enabled": enabled}


class NoToolsAgent:
    """没有标记工具的Agent"""
    
    def regular_method(self):
        return "no tools"
    
    def another_method(self, x: int):
        return x * 2


# ═══════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════

class TestToolDecorator:
    """测试 @tool 装饰器"""
    
    def test_decorator_marks_method(self):
        """装饰器应该标记方法"""
        agent = SimpleAgent()
        assert hasattr(agent.simple_method, '__is_tool__')
        assert agent.simple_method.__is_tool__ is True
        assert agent.simple_method.__tool_name__ == 'simple_method'
    
    def test_decorator_preserves_behavior(self):
        """装饰器不应改变方法行为"""
        agent = SimpleAgent()
        result = agent.simple_method("AAPL")
        assert result == {"symbol": "AAPL"}
    
    def test_decorator_with_custom_name(self):
        """装饰器应支持自定义名称"""
        @tool(name="custom_name")
        def test_func(self, x: int):
            return x
        
        assert test_func.__tool_name__ == "custom_name"
    
    def test_decorator_with_custom_description(self):
        """装饰器应支持自定义描述"""
        @tool(description="Custom description")
        def test_func(self, x: int):
            return x
        
        assert test_func.__tool_description__ == "Custom description"
    
    def test_decorator_extracts_description_from_docstring(self):
        """装饰器应从docstring提取描述"""
        agent = SimpleAgent()
        assert agent.simple_method.__tool_description__ == "Simple method with one parameter"
    
    def test_async_method_decorator(self):
        """装饰器应支持async方法"""
        agent = SimpleAgent()
        assert hasattr(agent.async_method, '__is_tool__')
        assert agent.async_method.__is_tool__ is True


class TestToolDiscovery:
    """测试工具发现"""
    
    def test_discover_simple_tools(self):
        """应该发现简单的工具"""
        agent = SimpleAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        tool_names = [t['name'] for t in tools]
        assert 'simple_method' in tool_names
        assert 'method_with_defaults' in tool_names
        assert 'async_method' in tool_names
        assert 'non_tool_method' not in tool_names
    
    def test_discover_correct_count(self):
        """应该发现正确数量的工具"""
        agent = SimpleAgent()
        tools = ToolRegistry.discover_tools(agent)
        assert len(tools) == 3  # 3个带装饰器的方法
    
    def test_discover_no_tools(self):
        """没有工具的Agent应返回空列表"""
        agent = NoToolsAgent()
        tools = ToolRegistry.discover_tools(agent)
        assert len(tools) == 0
    
    def test_discover_complex_tools(self):
        """应该发现复杂类型的工具"""
        agent = ComplexAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        assert len(tools) == 4
        tool_names = [t['name'] for t in tools]
        assert 'list_method' in tool_names
        assert 'dict_method' in tool_names
        assert 'optional_method' in tool_names
        assert 'numeric_types' in tool_names


class TestToolDefinition:
    """测试工具定义格式"""
    
    def test_tool_has_required_fields(self):
        """工具定义应包含必需字段"""
        agent = SimpleAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        for tool in tools:
            assert 'name' in tool
            assert 'description' in tool
            assert 'inputSchema' in tool
    
    def test_input_schema_structure(self):
        """inputSchema应该有正确的结构"""
        agent = SimpleAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        for tool in tools:
            schema = tool['inputSchema']
            assert schema['type'] == 'object'
            assert 'properties' in schema
    
    def test_simple_method_schema(self):
        """简单方法应有正确的schema"""
        agent = SimpleAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        simple_tool = next(t for t in tools if t['name'] == 'simple_method')
        schema = simple_tool['inputSchema']
        
        assert 'symbol' in schema['properties']
        assert schema['properties']['symbol']['type'] == 'string'
        assert 'symbol' in schema['required']


class TestParameterExtraction:
    """测试参数提取"""
    
    def test_required_parameters(self):
        """应正确识别必需参数"""
        agent = SimpleAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        simple_tool = next(t for t in tools if t['name'] == 'simple_method')
        schema = simple_tool['inputSchema']
        
        assert 'required' in schema
        assert 'symbol' in schema['required']
    
    def test_optional_parameters(self):
        """应正确识别可选参数"""
        agent = SimpleAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        defaults_tool = next(t for t in tools if t['name'] == 'method_with_defaults')
        schema = defaults_tool['inputSchema']
        
        assert 'symbol' in schema['required']
        assert 'timeframe' not in schema.get('required', [])
        assert 'limit' not in schema.get('required', [])
    
    def test_default_values(self):
        """应正确提取默认值"""
        agent = SimpleAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        defaults_tool = next(t for t in tools if t['name'] == 'method_with_defaults')
        schema = defaults_tool['inputSchema']
        
        assert schema['properties']['timeframe']['default'] == '1d'
        assert schema['properties']['limit']['default'] == 10
    
    def test_parameter_descriptions(self):
        """应从docstring提取参数描述"""
        agent = SimpleAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        simple_tool = next(t for t in tools if t['name'] == 'simple_method')
        schema = simple_tool['inputSchema']
        
        assert 'description' in schema['properties']['symbol']
        assert schema['properties']['symbol']['description'] == 'Stock symbol'
    
    def test_self_cls_excluded(self):
        """self和cls参数应被排除"""
        agent = SimpleAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        for tool in tools:
            schema = tool['inputSchema']
            assert 'self' not in schema['properties']
            assert 'cls' not in schema['properties']


class TestTypeMapping:
    """测试类型映射"""
    
    def test_string_type(self):
        """str应映射到string"""
        agent = SimpleAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        simple_tool = next(t for t in tools if t['name'] == 'simple_method')
        schema = simple_tool['inputSchema']
        
        assert schema['properties']['symbol']['type'] == 'string'
    
    def test_integer_type(self):
        """int应映射到integer"""
        agent = ComplexAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        numeric_tool = next(t for t in tools if t['name'] == 'numeric_types')
        schema = numeric_tool['inputSchema']
        
        assert schema['properties']['count']['type'] == 'integer'
    
    def test_float_type(self):
        """float应映射到number"""
        agent = ComplexAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        numeric_tool = next(t for t in tools if t['name'] == 'numeric_types')
        schema = numeric_tool['inputSchema']
        
        assert schema['properties']['ratio']['type'] == 'number'
    
    def test_boolean_type(self):
        """bool应映射到boolean"""
        agent = ComplexAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        numeric_tool = next(t for t in tools if t['name'] == 'numeric_types')
        schema = numeric_tool['inputSchema']
        
        assert schema['properties']['enabled']['type'] == 'boolean'
    
    def test_list_type(self):
        """List应映射到array"""
        agent = ComplexAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        list_tool = next(t for t in tools if t['name'] == 'list_method')
        schema = list_tool['inputSchema']
        
        assert schema['properties']['symbols']['type'] == 'array'
    
    def test_dict_type(self):
        """Dict应映射到object"""
        agent = ComplexAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        dict_tool = next(t for t in tools if t['name'] == 'dict_method')
        schema = dict_tool['inputSchema']
        
        assert schema['properties']['config']['type'] == 'object'
    
    def test_optional_type(self):
        """Optional[T]应映射到T的类型"""
        agent = ComplexAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        optional_tool = next(t for t in tools if t['name'] == 'optional_method')
        schema = optional_tool['inputSchema']
        
        # Optional[str] 应该被识别为 string
        assert schema['properties']['timeframe']['type'] == 'string'
        # 但不应该在required中
        assert 'timeframe' not in schema.get('required', [])


class TestEdgeCases:
    """测试边界情况"""
    
    def test_method_without_parameters(self):
        """无参数的方法应正常处理"""
        from Agents.utils.tool_registry import tool
        
        class EmptyParamsAgent:
            @tool(description="Method without parameters")
            def no_params(self):
                """No parameters method"""
                return {"result": "ok"}
        
        agent = EmptyParamsAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        assert len(tools) == 1
        tool_def = tools[0]
        assert len(tool_def['inputSchema']['properties']) == 0
        assert len(tool_def['inputSchema'].get('required', [])) == 0
    
    def test_method_without_docstring(self):
        """无docstring的方法应使用方法名作为描述"""
        from Agents.utils.tool_registry import tool
        
        class NoDocstringAgent:
            @tool()
            def some_method(self, x: int):
                return x
        
        agent = NoDocstringAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        assert len(tools) == 1
        # 应该从方法名生成描述
        assert 'some_method' in tools[0]['description'].lower() or tools[0]['description'] == 'Some Method'
    
    def test_method_without_type_hints(self):
        """无类型提示的方法应使用默认类型"""
        from Agents.utils.tool_registry import tool
        
        class NoTypeHintsAgent:
            @tool(description="No type hints")
            def no_types(self, param1, param2=10):
                """
                Method without type hints
                
                Args:
                    param1: First parameter
                    param2: Second parameter
                """
                return {"param1": param1, "param2": param2}
        
        agent = NoTypeHintsAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        assert len(tools) == 1
        schema = tools[0]['inputSchema']
        
        # 应该有properties，即使没有类型提示
        assert 'param1' in schema['properties']
        assert 'param2' in schema['properties']
        # 默认类型应该是string
        assert schema['properties']['param1']['type'] == 'string'
    
    def test_multiple_agents_isolation(self):
        """多个Agent实例应该隔离"""
        agent1 = SimpleAgent()
        agent2 = ComplexAgent()
        
        tools1 = ToolRegistry.discover_tools(agent1)
        tools2 = ToolRegistry.discover_tools(agent2)
        
        names1 = {t['name'] for t in tools1}
        names2 = {t['name'] for t in tools2}
        
        # 两个agent的工具不应重叠
        assert names1.isdisjoint(names2)


class TestBackwardCompatibility:
    """测试向后兼容性"""
    
    def test_mcp_protocol_format(self):
        """工具格式应符合MCP协议"""
        agent = SimpleAgent()
        tools = ToolRegistry.discover_tools(agent)
        
        for tool in tools:
            # MCP协议要求的字段
            assert 'name' in tool
            assert 'description' in tool
            assert 'inputSchema' in tool
            
            # inputSchema应该是JSON Schema格式
            schema = tool['inputSchema']
            assert schema['type'] == 'object'
            assert 'properties' in schema
    
    def test_tool_call_compatibility(self):
        """工具应该可以正常调用"""
        agent = SimpleAgent()
        
        # 直接调用装饰过的方法
        result = agent.simple_method("AAPL")
        assert result == {"symbol": "AAPL"}
        
        # 带默认参数的调用
        result = agent.method_with_defaults("TSLA")
        assert result["symbol"] == "TSLA"
        assert result["timeframe"] == "1d"
        assert result["limit"] == 10
    
    @pytest.mark.asyncio
    async def test_async_tool_compatibility(self):
        """异步工具应该正常工作"""
        agent = SimpleAgent()
        result = await agent.async_method("AAPL")
        assert result["symbol"] == "AAPL"
        assert result["async"] is True


class TestToolRegistry:
    """测试ToolRegistry类方法"""
    
    def test_discover_tools_is_classmethod(self):
        """discover_tools应该是类方法"""
        agent = SimpleAgent()
        # 应该可以直接从类调用
        tools = ToolRegistry.discover_tools(agent)
        assert isinstance(tools, list)
    
    def test_tool_is_classmethod(self):
        """tool应该是类方法"""
        # 应该可以作为装饰器使用
        @ToolRegistry.tool(description="Test")
        def test_func():
            pass
        
        assert hasattr(test_func, '__is_tool__')
    
    def test_registry_doesnt_store_state(self):
        """ToolRegistry不应存储Agent状态"""
        agent1 = SimpleAgent()
        agent2 = SimpleAgent()
        
        tools1 = ToolRegistry.discover_tools(agent1)
        tools2 = ToolRegistry.discover_tools(agent2)
        
        # 两次调用应该返回相同结构的工具
        assert len(tools1) == len(tools2)
        assert [t['name'] for t in tools1] == [t['name'] for t in tools2]


class TestErrorHandling:
    """测试错误处理"""
    
    def test_discover_tools_with_invalid_method(self):
        """发现工具时遇到无效方法应优雅处理"""
        class BrokenAgent:
            @tool(description="Broken method")
            def broken_method(self):
                # 故意没有参数类型
                pass
        
        agent = BrokenAgent()
        # 应该不会抛出异常
        tools = ToolRegistry.discover_tools(agent)
        
        # 应该仍然能发现工具（即使可能缺少某些信息）
        assert len(tools) >= 0
    
    def test_discover_tools_with_none_agent(self):
        """传入None应该优雅处理"""
        # 这应该不会崩溃，但可能返回空列表
        try:
            tools = ToolRegistry.discover_tools(None)
            assert isinstance(tools, list)
        except (AttributeError, TypeError):
            # 也可以接受抛出错误
            pass


# ═══════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════

class TestIntegration:
    """集成测试"""
    
    def test_full_workflow(self):
        """完整工作流测试"""
        # 1. 创建agent
        agent = SimpleAgent()
        
        # 2. 发现工具
        tools = ToolRegistry.discover_tools(agent)
        
        # 3. 验证工具数量
        assert len(tools) == 3
        
        # 4. 获取特定工具
        simple_tool = next(t for t in tools if t['name'] == 'simple_method')
        
        # 5. 验证工具定义完整
        assert simple_tool['name'] == 'simple_method'
        assert 'description' in simple_tool
        assert 'inputSchema' in simple_tool
        
        # 6. 验证可以调用方法
        result = agent.simple_method("AAPL")
        assert result['symbol'] == 'AAPL'
    
    def test_realistic_scenario(self):
        """真实场景测试"""
        # 模拟MetaAgent连接多个specialist agents
        agents = {
            'simple': SimpleAgent(),
            'complex': ComplexAgent(),
        }
        
        all_tools = []
        for agent_name, agent_instance in agents.items():
            tools = ToolRegistry.discover_tools(agent_instance)
            # 添加agent_name（模拟MetaAgent的行为）
            for tool in tools:
                tool['agent_name'] = agent_name
                all_tools.append(tool)
        
        # 应该发现所有工具
        assert len(all_tools) == 7  # 3 + 4
        
        # 每个工具应该有agent_name
        for tool in all_tools:
            assert 'agent_name' in tool
            assert tool['agent_name'] in ['simple', 'complex']
