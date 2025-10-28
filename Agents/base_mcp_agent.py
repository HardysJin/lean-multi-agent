"""
Base MCP Agent - MCP Agent基类

所有专家Agent的基础类，封装MCP Server的通用逻辑

核心职责：
1. 创建和管理MCP Server实例
2. 注册标准handlers (list_tools, call_tool, list_resources, read_resource)
3. 提供抽象方法供子类实现具体功能

设计模式：Template Method Pattern
- 基类定义框架和通用逻辑
- 子类实现具体的工具和资源

MCP协议三大组件：
- Tools: 可调用的函数（类似API endpoints）
- Resources: 可读取的数据源（类似REST resources）
- Prompts: 预定义的prompt模板（可选）
"""

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, Resource, TextContent, Prompt
from typing import List, Dict, Any, Optional
import asyncio
import logging
import json


class BaseMCPAgent:
    """
    MCP Agent基类
    
    所有专家Agent继承此类，实现自己的工具集
    
    子类需要实现的方法：
    - get_tools() -> List[Tool]: 返回该Agent提供的工具列表
    - handle_tool_call(name, arguments) -> Any: 处理工具调用
    
    可选实现的方法：
    - get_resources() -> List[Resource]: 返回资源列表
    - handle_resource_read(uri) -> str: 处理资源读取
    - get_prompts() -> List[Prompt]: 返回prompt模板列表
    - handle_prompt_get(name, arguments) -> str: 获取prompt
    
    使用示例：
    ```python
    class MyAgent(BaseMCPAgent):
        def get_tools(self):
            return [
                Tool(name="my_tool", description="...", inputSchema={...})
            ]
        
        async def handle_tool_call(self, name, arguments):
            if name == "my_tool":
                return self._do_something(arguments)
    
    # 运行Agent
    agent = MyAgent("my-agent", "My Agent Description")
    asyncio.run(agent.run())
    ```
    """
    
    def __init__(self, name: str, description: str, version: str = "1.0.0"):
        """
        初始化MCP Agent
        
        Args:
            name: Agent名称（标识符，应使用kebab-case）
            description: Agent功能描述
            version: Agent版本号
        """
        self.name = name
        self.description = description
        self.version = version
        
        # 创建MCP Server实例
        self.server = Server(name)
        
        # 配置日志
        self.logger = logging.getLogger(f"MCP.{name}")
        self.logger.setLevel(logging.INFO)
        
        # 注册MCP handlers
        self._register_handlers()
        
        self.logger.info(f"Initialized {name} v{version}")
    
    def _register_handlers(self):
        """
        注册MCP协议的标准handlers
        
        这些handler会被MCP Client调用
        """
        
        # ═══════════════════════════════════════════════
        # Tools Handler
        # ═══════════════════════════════════════════════
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """
            列出该Agent提供的所有工具
            
            MCP Client会调用此方法来发现可用的工具
            """
            try:
                tools = self.get_tools()
                self.logger.info(f"Listed {len(tools)} tools")
                return tools
            except Exception as e:
                self.logger.error(f"Error listing tools: {e}")
                return []
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """
            处理工具调用
            
            Args:
                name: 工具名称
                arguments: 工具参数
                
            Returns:
                List[TextContent]: 工具执行结果
            """
            self.logger.info(f"Tool call: {name} with args: {arguments}")
            
            try:
                # 调用子类实现的handler
                result = await self.handle_tool_call(name, arguments)
                
                # 格式化返回结果
                if isinstance(result, str):
                    result_text = result
                elif isinstance(result, dict) or isinstance(result, list):
                    result_text = json.dumps(result, indent=2, ensure_ascii=False)
                else:
                    result_text = str(result)
                
                self.logger.info(f"Tool {name} succeeded")
                
                return [TextContent(
                    type="text",
                    text=result_text
                )]
                
            except Exception as e:
                self.logger.error(f"Tool {name} failed: {e}", exc_info=True)
                
                # 返回错误信息
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "tool": name,
                        "agent": self.name
                    }, indent=2)
                )]
        
        # ═══════════════════════════════════════════════
        # Resources Handler
        # ═══════════════════════════════════════════════
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """
            列出该Agent提供的所有资源
            
            资源是可读取的数据源，类似REST API的resources
            """
            try:
                resources = self.get_resources()
                self.logger.info(f"Listed {len(resources)} resources")
                return resources
            except Exception as e:
                self.logger.error(f"Error listing resources: {e}")
                return []
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """
            读取指定资源
            
            Args:
                uri: 资源URI（例如: "news://latest", "indicators://AAPL"）
                
            Returns:
                资源内容（通常是JSON字符串）
            """
            self.logger.info(f"Read resource: {uri}")
            
            try:
                content = await self.handle_resource_read(uri)
                
                if isinstance(content, dict) or isinstance(content, list):
                    content = json.dumps(content, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Resource {uri} read successfully")
                return str(content)
                
            except Exception as e:
                self.logger.error(f"Failed to read resource {uri}: {e}")
                return json.dumps({"error": str(e), "uri": uri})
        
        # ═══════════════════════════════════════════════
        # Prompts Handler (Optional)
        # ═══════════════════════════════════════════════
        
        @self.server.list_prompts()
        async def list_prompts() -> List[Prompt]:
            """
            列出该Agent提供的prompt模板
            
            Prompts是预定义的prompt模板，可以被LLM使用
            """
            try:
                prompts = self.get_prompts()
                self.logger.info(f"Listed {len(prompts)} prompts")
                return prompts
            except Exception as e:
                self.logger.error(f"Error listing prompts: {e}")
                return []
        
        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: Optional[Dict[str, str]] = None) -> str:
            """
            获取指定的prompt模板
            
            Args:
                name: Prompt名称
                arguments: Prompt参数（用于模板填充）
                
            Returns:
                填充后的prompt文本
            """
            self.logger.info(f"Get prompt: {name} with args: {arguments}")
            
            try:
                prompt_text = await self.handle_prompt_get(name, arguments or {})
                return prompt_text
                
            except Exception as e:
                self.logger.error(f"Failed to get prompt {name}: {e}")
                return f"Error: {e}"
    
    # ═══════════════════════════════════════════════
    # 抽象方法 - 子类必须实现
    # ═══════════════════════════════════════════════
    
    def get_tools(self) -> List[Tool]:
        """
        返回该Agent提供的工具列表
        
        子类必须实现此方法
        
        Returns:
            List[Tool]: 工具列表
            
        示例:
        ```python
        return [
            Tool(
                name="analyze_data",
                description="Analyze data and return insights",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"},
                        "method": {"type": "string", "enum": ["quick", "deep"]}
                    },
                    "required": ["data"]
                }
            )
        ]
        ```
        """
        raise NotImplementedError(f"{self.name} must implement get_tools()")
    
    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        处理工具调用
        
        子类必须实现此方法
        
        Args:
            name: 工具名称
            arguments: 工具参数
            
        Returns:
            Any: 工具执行结果（可以是dict, list, str等）
            
        示例:
        ```python
        if name == "analyze_data":
            data = arguments['data']
            method = arguments.get('method', 'quick')
            return self._analyze(data, method)
        else:
            raise ValueError(f"Unknown tool: {name}")
        ```
        """
        raise NotImplementedError(f"{self.name} must implement handle_tool_call()")
    
    # ═══════════════════════════════════════════════
    # 可选方法 - 子类可以实现
    # ═══════════════════════════════════════════════
    
    def get_resources(self) -> List[Resource]:
        """
        返回该Agent提供的资源列表
        
        子类可选实现（默认返回空列表）
        
        Returns:
            List[Resource]: 资源列表
            
        示例:
        ```python
        return [
            Resource(
                uri="data://latest",
                name="Latest Data",
                description="Real-time data feed",
                mimeType="application/json"
            )
        ]
        ```
        """
        return []
    
    async def handle_resource_read(self, uri: str) -> Any:
        """
        读取指定资源
        
        子类可选实现（默认返回空）
        
        Args:
            uri: 资源URI
            
        Returns:
            资源内容
            
        示例:
        ```python
        if uri == "data://latest":
            return self._get_latest_data()
        else:
            raise ValueError(f"Unknown resource: {uri}")
        ```
        """
        return {"error": f"Resource not implemented: {uri}"}
    
    def get_prompts(self) -> List[Prompt]:
        """
        返回该Agent提供的prompt模板列表
        
        子类可选实现（默认返回空列表）
        
        Returns:
            List[Prompt]: Prompt模板列表
        """
        return []
    
    async def handle_prompt_get(self, name: str, arguments: Dict[str, str]) -> str:
        """
        获取指定的prompt模板
        
        子类可选实现（默认返回空）
        
        Args:
            name: Prompt名称
            arguments: Prompt参数
            
        Returns:
            Prompt文本
        """
        return f"Prompt not implemented: {name}"
    
    # ═══════════════════════════════════════════════
    # 运行方法
    # ═══════════════════════════════════════════════
    
    async def run(self):
        """
        启动MCP Server
        
        这会启动stdio通信，等待MCP Client的连接
        
        使用方式:
        ```python
        agent = MyAgent("my-agent", "My Agent")
        asyncio.run(agent.run())
        ```
        """
        self.logger.info(f"Starting MCP Server: {self.name}")
        self.logger.info(f"Description: {self.description}")
        self.logger.info(f"Version: {self.version}")
        
        try:
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )
        except Exception as e:
            self.logger.error(f"Server error: {e}", exc_info=True)
            raise
    
    # ═══════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════
    
    def _validate_arguments(self, arguments: Dict, required_keys: List[str]) -> None:
        """
        验证参数是否包含必需的键
        
        Args:
            arguments: 参数字典
            required_keys: 必需的键列表
            
        Raises:
            ValueError: 如果缺少必需参数
        """
        missing = [key for key in required_keys if key not in arguments]
        if missing:
            raise ValueError(f"Missing required arguments: {missing}")
    
    def _create_tool_schema(self,
                           name: str,
                           description: str,
                           properties: Dict[str, Any],
                           required: List[str]) -> Tool:
        """
        便捷方法：创建Tool对象
        
        Args:
            name: 工具名称
            description: 工具描述
            properties: 参数属性定义
            required: 必需参数列表
            
        Returns:
            Tool对象
        """
        return Tool(
            name=name,
            description=description,
            inputSchema={
                "type": "object",
                "properties": properties,
                "required": required
            }
        )
    
    def _create_resource_uri(self, resource_type: str, identifier: str = "") -> str:
        """
        便捷方法：创建资源URI
        
        Args:
            resource_type: 资源类型（如"data", "news", "indicators"）
            identifier: 资源标识符（可选）
            
        Returns:
            格式化的URI字符串
            
        示例:
            _create_resource_uri("news", "latest") -> "news://latest"
            _create_resource_uri("indicators", "AAPL") -> "indicators://AAPL"
        """
        if identifier:
            return f"{resource_type}://{identifier}"
        return f"{resource_type}://"


# ═══════════════════════════════════════════════
# 示例Agent实现
# ═══════════════════════════════════════════════

class ExampleAgent(BaseMCPAgent):
    """
    示例Agent - 演示如何继承BaseMCPAgent
    
    这是一个简单的echo agent，返回接收到的参数
    """
    
    def __init__(self):
        super().__init__(
            name="example-agent",
            description="A simple example agent that echoes input",
            version="1.0.0"
        )
    
    def get_tools(self) -> List[Tool]:
        """定义工具"""
        return [
            self._create_tool_schema(
                name="echo",
                description="Echo back the input message",
                properties={
                    "message": {
                        "type": "string",
                        "description": "Message to echo"
                    }
                },
                required=["message"]
            ),
            
            self._create_tool_schema(
                name="greet",
                description="Greet a person by name",
                properties={
                    "name": {
                        "type": "string",
                        "description": "Name of the person to greet"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["en", "zh", "es"],
                        "description": "Language for greeting"
                    }
                },
                required=["name"]
            )
        ]
    
    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """处理工具调用"""
        
        if name == "echo":
            self._validate_arguments(arguments, ["message"])
            return {
                "echo": arguments["message"],
                "timestamp": "2025-10-28T12:00:00Z"
            }
        
        elif name == "greet":
            self._validate_arguments(arguments, ["name"])
            person_name = arguments["name"]
            language = arguments.get("language", "en")
            
            greetings = {
                "en": f"Hello, {person_name}!",
                "zh": f"你好，{person_name}！",
                "es": f"¡Hola, {person_name}!"
            }
            
            return {
                "greeting": greetings.get(language, greetings["en"]),
                "language": language
            }
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    def get_resources(self) -> List[Resource]:
        """定义资源"""
        return [
            Resource(
                uri=self._create_resource_uri("status"),
                name="Agent Status",
                description="Current status of the example agent",
                mimeType="application/json"
            )
        ]
    
    async def handle_resource_read(self, uri: str) -> Any:
        """读取资源"""
        if uri == "status://":
            return {
                "status": "running",
                "version": self.version,
                "uptime": "N/A"
            }
        else:
            return await super().handle_resource_read(uri)


# ═══════════════════════════════════════════════
# 主程序入口
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建并运行示例Agent
    agent = ExampleAgent()
    
    print("Starting Example MCP Agent...")
    print("Use MCP Inspector to test:")
    print("  npx @modelcontextprotocol/inspector python Agents/base_mcp_agent.py")
    print()
    
    asyncio.run(agent.run())
