"""
Meta Agent模块
实现MetaAgent类，协调多个specialist agents进行综合决策

Meta Agent作为协调器（Orchestrator），不是专家（Specialist）：
- 直接调用 core agents (in-process)
- 使用 LLM 进行智能决策
- 整合 Memory System
- 无需 MCP 协议（未来如需要可以创建 MCP wrapper）
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from Agents.utils.llm_config import get_default_llm, LLMConfig
from Memory.state_manager import MultiTimeframeStateManager
from Memory.schemas import DecisionRecord, Timeframe


@dataclass
class AgentConnection:
    """Specialist agent连接信息（in-process）"""
    name: str
    instance: Any  # Agent instance (直接引用)
    tools: List[Dict[str, Any]]
    resources: List[Dict[str, Any]]
    description: str


@dataclass
class ToolCall:
    """工具调用记录"""
    agent_name: str
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    timestamp: datetime
    execution_time_ms: float


@dataclass
class MetaDecision:
    """Meta Agent的最终决策"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    conviction: int  # 1-10
    reasoning: str
    evidence: Dict[str, Any]  # 来自各个agent的证据
    tool_calls: List[ToolCall]
    timestamp: datetime
    
    def to_decision_record(self, timeframe: Timeframe = Timeframe.TACTICAL) -> DecisionRecord:
        """转换为DecisionRecord用于存储到Memory System"""
        return DecisionRecord(
            id=f"META_{self.symbol}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}",
            timestamp=self.timestamp,
            timeframe=timeframe,
            symbol=self.symbol,
            action=self.action,
            quantity=0,  # 需要根据conviction计算
            price=0.0,  # 需要从evidence中提取
            reasoning=self.reasoning,
            agent_name="meta_agent",
            conviction=float(self.conviction),
            metadata={
                'evidence': self.evidence,
                'tool_calls_count': len(self.tool_calls),
                'agents_consulted': list(set(tc.agent_name for tc in self.tool_calls))
            }
        )


class MetaAgent:
    """
    Meta Agent - 协调器/编排器（Orchestrator）
    
    直接调用 specialist agents (in-process)，协调工具调用，
    使用 LLM 进行智能决策，集成 Memory System。
    
    不使用 MCP 协议（如需要可创建 MCP wrapper）。
    """
    
    def __init__(
        self,
        llm_client=None,
        state_manager: Optional[MultiTimeframeStateManager] = None,
        enable_memory: bool = True
    ):
        """
        初始化Meta Agent
        
        Args:
            llm_client: LLM客户端（如果为None，使用默认）
            state_manager: StateManager instance for memory integration
            enable_memory: 是否启用Memory System（默认True）
        """
        # Memory System - 默认启用
        if enable_memory and state_manager is None:
            # 自动创建默认的state_manager
            self.state_manager = MultiTimeframeStateManager(
                sql_db_path="Data/sql/trading_memory.db",
                vector_db_path="Data/vector_db/chroma"
            )
            print("✓ Memory System自动启用 (Data/sql/trading_memory.db)")
        else:
            self.state_manager = state_manager
        
        # 连接的agents
        self.agents: Dict[str, AgentConnection] = {}
        
        # LLM client
        self.llm_client = llm_client if llm_client else get_default_llm()
        
        # 工具调用历史
        self.tool_call_history: List[ToolCall] = []
        
        # 决策历史
        self.decision_history: List[MetaDecision] = []
    
    async def connect_to_agent(
        self,
        agent_name: str,
        agent_instance: Any,
        description: str = ""
    ) -> None:
        """
        连接到specialist agent (in-process)
        
        直接使用 core agents，无需 MCP 协议。
        
        Args:
            agent_name: Agent名称
            agent_instance: Core agent实例（如 MacroAgent, TechnicalAnalysisAgent）
            description: Agent描述
        """
        # 根据agent类型自动发现工具
        tools_dict = []
        resources_dict = []
        
        # 通过introspection获取public方法作为工具
        agent_class_name = agent_instance.__class__.__name__
        
        if agent_class_name == "TechnicalAnalysisAgent":
            tools_dict = [
                {
                    'name': 'calculate_indicators',
                    'description': 'Calculate technical indicators (RSI, MACD, etc.) for a symbol',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'symbol': {'type': 'string', 'description': 'Stock symbol'},
                            'timeframe': {'type': 'string', 'description': 'Timeframe (5min, 1h, 1d)', 'default': '1d'}
                        },
                        'required': ['symbol']
                    }
                },
                {
                    'name': 'generate_signals',
                    'description': 'Generate buy/sell signals based on technical indicators',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'symbol': {'type': 'string', 'description': 'Stock symbol'}
                        },
                        'required': ['symbol']
                    }
                },
                {
                    'name': 'detect_patterns',
                    'description': 'Detect chart patterns (head-shoulders, double-top, etc.)',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'symbol': {'type': 'string', 'description': 'Stock symbol'},
                            'timeframe': {'type': 'string', 'description': 'Timeframe', 'default': '1d'}
                        },
                        'required': ['symbol']
                    }
                },
                {
                    'name': 'find_support_resistance',
                    'description': 'Find key support and resistance levels',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'symbol': {'type': 'string', 'description': 'Stock symbol'}
                        },
                        'required': ['symbol']
                    }
                }
            ]
            resources_dict = [
                {
                    'uri': f'technical://{agent_name}/cache',
                    'name': 'Cache Status',
                    'description': 'View cached technical data',
                    'mimeType': 'application/json'
                },
                {
                    'uri': f'technical://{agent_name}/capabilities',
                    'name': 'Capabilities',
                    'description': 'Available indicators and patterns',
                    'mimeType': 'application/json'
                }
            ]
        elif agent_class_name == "NewsAgent":
            tools_dict = [
                {
                    'name': 'fetch_news',
                    'description': 'Fetch news articles for symbols',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'symbols': {'type': 'array', 'items': {'type': 'string'}},
                            'max_articles': {'type': 'integer', 'default': 10}
                        },
                        'required': ['symbols']
                    }
                },
                {
                    'name': 'analyze_sentiment',
                    'description': 'Analyze sentiment of news articles',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'articles': {'type': 'array'}
                        },
                        'required': ['articles']
                    }
                }
            ]
        elif agent_class_name == "MacroAgent":
            tools_dict = [
                {
                    'name': 'analyze_macro_environment',
                    'description': 'Analyze macroeconomic environment',
                    'inputSchema': {'type': 'object', 'properties': {}}
                },
                {
                    'name': 'get_market_regime',
                    'description': 'Determine current market regime',
                    'inputSchema': {'type': 'object', 'properties': {}}
                }
            ]
        elif agent_class_name == "SectorAgent":
            tools_dict = [
                {
                    'name': 'analyze_sector',
                    'description': 'Analyze specific sector performance',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'sector': {'type': 'string'}
                        },
                        'required': ['sector']
                    }
                }
            ]
        
        # 创建连接（直接引用agent实例）
        connection = AgentConnection(
            name=agent_name,
            instance=agent_instance,  # 直接存储agent实例
            tools=tools_dict,
            resources=resources_dict,
            description=description or getattr(agent_instance, 'description', agent_name)
        )
        
        self.agents[agent_name] = connection
        print(f"✓ Connected to agent: {agent_name} ({len(tools_dict)} tools, {len(resources_dict)} resources)")
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        获取所有可用工具
        
        Returns:
            所有agents的工具列表，每个工具包含agent_name
        """
        all_tools = []
        for agent_name, connection in self.agents.items():
            for tool in connection.tools:
                tool_with_agent = tool.copy()
                tool_with_agent['agent_name'] = agent_name
                all_tools.append(tool_with_agent)
        return all_tools
    
    async def execute_tool(
        self,
        agent_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        执行specialist agent的工具
        
        Args:
            agent_name: Agent名称
            tool_name: 工具名称
            arguments: 工具参数
            
        Returns:
            工具执行结果
            
        Raises:
            ValueError: 如果agent或tool不存在
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not connected")
        
        connection = self.agents[agent_name]
        agent_instance = connection.instance  # 获取agent实例
        
        # 验证工具存在
        tool_exists = any(t['name'] == tool_name for t in connection.tools)
        if not tool_exists:
            raise ValueError(f"Tool '{tool_name}' not found in agent '{agent_name}'")
        
        # 执行工具
        start_time = datetime.now()
        try:
            # 直接调用agent的方法（不通过handle_tool_call）
            method = getattr(agent_instance, tool_name, None)
            if method is None:
                raise ValueError(f"Method '{tool_name}' not found in agent '{agent_name}'")
            
            # 调用方法
            result = method(**arguments)
            
            # 如果返回的是协程，则await
            if hasattr(result, '__await__'):
                result = await result
                
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # 记录工具调用
            tool_call = ToolCall(
                agent_name=agent_name,
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                timestamp=start_time,
                execution_time_ms=execution_time
            )
            self.tool_call_history.append(tool_call)
            
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            error_result = {"error": str(e)}
            
            # 记录失败的调用
            tool_call = ToolCall(
                agent_name=agent_name,
                tool_name=tool_name,
                arguments=arguments,
                result=error_result,
                timestamp=start_time,
                execution_time_ms=execution_time
            )
            self.tool_call_history.append(tool_call)
            
            raise
    
    async def read_resource(
        self,
        agent_name: str,
        resource_uri: str
    ) -> Any:
        """
        读取specialist agent的资源
        
        Args:
            agent_name: Agent名称
            resource_uri: 资源URI
            
        Returns:
            资源内容
            
        Raises:
            ValueError: 如果agent不存在
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not connected")
        
        connection = self.agents[agent_name]
        agent_instance = connection.instance
        
        # 简单实现：返回agent状态信息
        # 实际应用中可以根据URI返回不同资源
        if 'cache' in resource_uri:
            return {
                'uri': resource_uri,
                'agent': agent_name,
                'cache_info': getattr(agent_instance, '_cache', {})
            }
        elif 'capabilities' in resource_uri:
            return {
                'uri': resource_uri,
                'agent': agent_name,
                'tools': [t['name'] for t in connection.tools],
                'description': connection.description
            }
        else:
            return {'uri': resource_uri, 'agent': agent_name, 'data': 'Resource not found'}

    
    def _retrieve_memory_context(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        从Memory System检索上下文
        
        Args:
            symbol: 交易标的
            lookback_hours: 回溯时间（小时）
            
        Returns:
            记忆上下文字典
        """
        if not self.state_manager:
            return {
                "note": "No state manager available",
                "recent_decisions": []
            }
        
        try:
            # 获取近期决策（直接从sql_store）
            recent_decisions = self.state_manager.sql_store.get_recent_decisions(
                symbol=symbol,
                limit=5
            )
            
            # TODO: 实现从向量存储检索相似市场决策
            # similar_events = self.state_manager.get_similar_past_decisions(...)
            
            return {
                "recent_decisions": [
                    {
                        "action": d.action,
                        "confidence": d.conviction,
                        "reasoning": d.reasoning,
                        "timestamp": d.timestamp.isoformat(),
                        "timeframe": str(d.timeframe)
                    }
                    for d in recent_decisions
                ]
            }
        except Exception as e:
            return {
                "error": f"Failed to retrieve memory: {str(e)}",
                "recent_decisions": []
            }
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        agents_info = "\n".join([
            f"- {name}: {conn.description} ({len(conn.tools)} tools)"
            for name, conn in self.agents.items()
        ])
        
        return f"""You are a Meta Agent coordinating multiple specialist agents for quantitative trading.

Connected Agents:
{agents_info}

Your role:
1. Analyze the current situation including macro and sector context
2. Respect all constraints provided (especially allow_long/allow_short)
3. Decide which specialist agents to consult
4. Call appropriate tools to gather information
5. Synthesize all inputs into a final trading decision
6. Provide clear reasoning for your decision

Context Priority:
1. **Constraints** (MUST follow): Risk limits, trading restrictions from macro environment
2. **Macro Context**: Market regime, interest rates, overall risk level
3. **Sector Context**: Industry trends, rotation signals, relative strength
4. **Memory**: Historical decisions and patterns
5. **Technical/News**: Individual stock analysis

Constraint Enforcement:
- If allow_long=False: DO NOT recommend BUY
- If allow_short=False: DO NOT recommend short positions
- If max_position_size specified: Consider position sizing
- If max_risk_per_trade specified: Adjust conviction accordingly

Trading Actions:
- BUY: Strong evidence to enter long position (only if constraints allow)
- SELL: Strong evidence to exit or enter short position  
- HOLD: Insufficient evidence, conflicting signals, or constraints prohibit action

Conviction Score (1-10):
- 1-3: Low conviction, weak signals
- 4-6: Moderate conviction, some supporting evidence
- 7-8: High conviction, strong evidence from multiple sources
- 9-10: Very high conviction, overwhelming evidence

Always consider:
- Multiple timeframes and perspectives
- Risk management principles (from constraints)
- Macro environment alignment
- Sector trends and rotation
- Historical context from memory
- Confluence of signals from different agents"""
    
    def _format_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        将工具格式化为Anthropic tool calling格式
        
        Returns:
            Anthropic工具定义列表
        """
        tools = []
        for agent_name, connection in self.agents.items():
            for tool in connection.tools:
                # Anthropic tool格式
                anthropic_tool = {
                    "name": f"{agent_name}__{tool['name']}",  # 加上agent前缀避免冲突
                    "description": f"[{agent_name}] {tool['description']}",
                    "input_schema": tool['inputSchema']
                }
                tools.append(anthropic_tool)
        return tools
    
    async def _call_llm_with_tools(
        self,
        messages: List[Dict[str, Any]],
        max_iterations: int = 5
    ) -> Tuple[str, List[ToolCall]]:
        """
        调用LLM，支持工具调用 (使用LangChain的tool binding)
        
        Args:
            messages: 对话消息
            max_iterations: 最大迭代次数（防止无限循环）
            
        Returns:
            (最终响应文本, 工具调用列表)
        """
        if not self.llm_client:
            return "No LLM client available. Please configure LLM.", []
        
        # 使用 LangChain 的 tool calling 支持
        from langchain_core.tools import tool
        from langchain_core.messages import AIMessage, ToolMessage
        
        tool_calls_made = []
        
        # 将我们的工具转换为LangChain工具格式
        langchain_tools = self._create_langchain_tools()
        
        if not langchain_tools:
            # 如果没有工具，直接调用LLM
            langchain_messages = [SystemMessage(content=self._build_system_prompt())]
            for msg in messages:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
            
            try:
                response = self.llm_client.invoke(langchain_messages)
                return response.content, []
            except Exception as e:
                return f"LLM call failed: {str(e)}", []
        
        # 绑定工具到LLM
        try:
            llm_with_tools = self.llm_client.bind_tools(langchain_tools)
        except AttributeError:
            # 如果LLM不支持bind_tools（如MockLLM），回退到简单模式
            langchain_messages = [SystemMessage(content=self._build_system_prompt())]
            for msg in messages:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
            
            try:
                response = self.llm_client.invoke(langchain_messages)
                return response.content, []
            except Exception as e:
                return f"LLM call failed: {str(e)}", []
        
        # 构建初始消息
        langchain_messages = [SystemMessage(content=self._build_system_prompt())]
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
        
        # 迭代调用，支持多轮工具调用
        for iteration in range(max_iterations):
            try:
                # 调用LLM
                response = llm_with_tools.invoke(langchain_messages)
                
                # 检查是否有工具调用
                if not hasattr(response, 'tool_calls') or not response.tool_calls:
                    # 没有工具调用，返回最终响应
                    return response.content, tool_calls_made
                
                # 添加AI响应到消息历史
                langchain_messages.append(response)
                
                # 执行工具调用
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    tool_id = tool_call.get('id', f'call_{iteration}')
                    
                    # 解析agent_name和actual_tool_name
                    if "__" in tool_name:
                        agent_name, actual_tool_name = tool_name.split("__", 1)
                    else:
                        # 尝试查找工具所属的agent
                        agent_name = None
                        for name, conn in self.agents.items():
                            if any(t['name'] == tool_name for t in conn.tools):
                                agent_name = name
                                actual_tool_name = tool_name
                                break
                        
                        if not agent_name:
                            # 工具未找到
                            error_msg = f"Tool '{tool_name}' not found"
                            langchain_messages.append(
                                ToolMessage(
                                    content=json.dumps({"error": error_msg}),
                                    tool_call_id=tool_id
                                )
                            )
                            continue
                    
                    # 执行工具
                    try:
                        result = await self.execute_tool(
                            agent_name=agent_name,
                            tool_name=actual_tool_name,
                            arguments=tool_args
                        )
                        
                        # 添加工具结果到消息
                        langchain_messages.append(
                            ToolMessage(
                                content=json.dumps(result, default=str),
                                tool_call_id=tool_id
                            )
                        )
                        
                        # 记录到tool_calls_made
                        tool_calls_made.append(self.tool_call_history[-1])
                        
                    except Exception as e:
                        # 工具执行失败
                        error_msg = f"Tool execution failed: {str(e)}"
                        langchain_messages.append(
                            ToolMessage(
                                content=json.dumps({"error": error_msg}),
                                tool_call_id=tool_id
                            )
                        )
            
            except Exception as e:
                return f"Error during LLM tool calling: {str(e)}", tool_calls_made
        
        # 达到最大迭代次数，最后再调用一次获取最终答案
        try:
            final_response = llm_with_tools.invoke(langchain_messages)
            return final_response.content, tool_calls_made
        except Exception as e:
            return f"Max iterations reached. Last error: {str(e)}", tool_calls_made
    
    def _create_langchain_tools(self) -> List[Any]:
        """
        将MCP工具转换为LangChain工具格式
        
        Returns:
            LangChain工具列表
        """
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field, create_model
        
        langchain_tools = []
        
        for agent_name, connection in self.agents.items():
            for tool in connection.tools:
                tool_name = f"{agent_name}__{tool['name']}"
                tool_description = f"[{agent_name}] {tool['description']}"
                
                # 创建输入模型
                input_schema = tool['inputSchema']
                properties = input_schema.get('properties', {})
                required = input_schema.get('required', [])
                
                # 构建Pydantic字段
                fields = {}
                for prop_name, prop_schema in properties.items():
                    field_type = str  # 默认类型
                    if prop_schema.get('type') == 'integer':
                        field_type = int
                    elif prop_schema.get('type') == 'number':
                        field_type = float
                    elif prop_schema.get('type') == 'boolean':
                        field_type = bool
                    
                    # 设置默认值
                    default = ... if prop_name in required else None
                    
                    fields[prop_name] = (
                        field_type,
                        Field(
                            default=default,
                            description=prop_schema.get('description', '')
                        )
                    )
                
                # 如果没有参数，使用空模型
                if not fields:
                    fields = {'__dummy__': (str, Field(default='', description='No parameters'))}
                
                # 创建输入模型
                InputModel = create_model(
                    f"{tool_name}_input",
                    **fields
                )
                
                # 创建工具执行函数
                def make_tool_func(agent_name, tool_name):
                    async def tool_func(**kwargs):
                        # 移除dummy参数
                        kwargs.pop('__dummy__', None)
                        result = await self.execute_tool(
                            agent_name=agent_name,
                            tool_name=tool_name,
                            arguments=kwargs
                        )
                        return json.dumps(result, default=str)
                    return tool_func
                
                # 创建LangChain工具
                lc_tool = StructuredTool(
                    name=tool_name,
                    description=tool_description,
                    func=make_tool_func(agent_name, tool['name']),
                    args_schema=InputModel,
                    coroutine=make_tool_func(agent_name, tool['name'])
                )
                
                langchain_tools.append(lc_tool)
        
        return langchain_tools
    
    async def analyze_and_decide(
        self,
        symbol: str,
        query: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        macro_context: Optional[Dict[str, Any]] = None,
        sector_context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> MetaDecision:
        """
        分析并做出交易决策
        
        这是Meta Agent的核心方法：
        1. 从Memory System检索上下文
        2. 接收宏观和行业背景
        3. 应用约束条件
        4. 使用LLM分析situation
        5. LLM决定调用哪些specialist工具
        6. 综合所有信息形成最终决策
        
        Args:
            symbol: 交易标的
            query: 可选的具体问题（如"Should I buy AAPL?"）
            additional_context: 额外的上下文信息
            macro_context: 宏观环境背景（来自MacroAgent）
            sector_context: 行业分析背景（来自SectorAgent）
            constraints: 约束条件（风险控制参数）
            
        Returns:
            MetaDecision对象
        """
        # 0. 检查约束条件（优先级最高）
        if constraints:
            # 检查是否允许交易
            if not constraints.get('allow_long', True) and not constraints.get('allow_short', False):
                # 禁止做多又禁止做空 = 只能HOLD
                return MetaDecision(
                    symbol=symbol,
                    action='HOLD',
                    conviction=10,
                    reasoning='Market constraints prohibit all trading (熊市禁止做多)',
                    evidence={'constraints': constraints},
                    tool_calls=[],
                    timestamp=datetime.now()
                )
        
        # 1. 检索记忆上下文
        memory_context = self._retrieve_memory_context(symbol)
        
        # 2. 构建初始消息（包含宏观和行业背景）
        context_str = json.dumps({
            "symbol": symbol,
            "memory": memory_context,
            "additional": additional_context or {},
            "macro": macro_context or {},
            "sector": sector_context or {},
            "constraints": constraints or {}
        }, indent=2, default=str)
        
        user_message = f"""Analyze the trading opportunity for {symbol}.

Context:
{context_str}

{'Question: ' + query if query else ''}

Please:
1. Use available tools to gather current market data and analysis
2. Consider historical context from memory
3. Synthesize all information
4. Provide a clear trading decision (BUY/SELL/HOLD) with conviction (1-10)
5. Explain your reasoning

Format your final decision as:
ACTION: [BUY/SELL/HOLD]
CONVICTION: [1-10]
REASONING: [detailed explanation]"""
        
        messages = [{"role": "user", "content": user_message}]
        
        # 3. 调用LLM（可能包含多轮工具调用）
        final_response, tool_calls = await self._call_llm_with_tools(messages)
        
        # 4. 解析决策
        decision = self._parse_decision(
            symbol=symbol,
            response=final_response,
            tool_calls=tool_calls
        )
        
        # 5. 存储到记忆系统
        if self.state_manager:
            try:
                decision_record = decision.to_decision_record()
                self.state_manager.store_decision(decision_record)
            except Exception as e:
                print(f"Warning: Failed to store decision in memory: {e}")
        
        # 6. 记录到历史
        self.decision_history.append(decision)
        
        return decision
    
    def _parse_decision(
        self,
        symbol: str,
        response: str,
        tool_calls: List[ToolCall]
    ) -> MetaDecision:
        """
        从LLM响应中解析决策
        
        Args:
            symbol: 交易标的
            response: LLM响应文本
            tool_calls: 执行的工具调用
            
        Returns:
            MetaDecision对象
        """
        # 提取ACTION
        action = "HOLD"  # 默认
        for line in response.split('\n'):
            if line.strip().startswith("ACTION:"):
                action_text = line.split("ACTION:", 1)[1].strip()
                if "BUY" in action_text.upper():
                    action = "BUY"
                elif "SELL" in action_text.upper():
                    action = "SELL"
                else:
                    action = "HOLD"
                break
        
        # 提取CONVICTION
        conviction = 5  # 默认
        for line in response.split('\n'):
            if line.strip().startswith("CONVICTION:"):
                conviction_text = line.split("CONVICTION:", 1)[1].strip()
                try:
                    conviction = int(conviction_text.split()[0])
                    conviction = max(1, min(10, conviction))  # 限制在1-10
                except (ValueError, IndexError):
                    conviction = 5
                break
        
        # 提取REASONING
        reasoning_lines = []
        in_reasoning = False
        for line in response.split('\n'):
            if line.strip().startswith("REASONING:"):
                reasoning_lines.append(line.split("REASONING:", 1)[1].strip())
                in_reasoning = True
            elif in_reasoning:
                if line.strip() and not line.strip().startswith(("ACTION:", "CONVICTION:")):
                    reasoning_lines.append(line.strip())
        
        reasoning = " ".join(reasoning_lines) if reasoning_lines else response
        
        # 收集证据
        evidence = {
            "raw_response": response,
            "tools_used": [
                {
                    "agent": tc.agent_name,
                    "tool": tc.tool_name,
                    "result_summary": str(tc.result)[:200] + "..." if len(str(tc.result)) > 200 else str(tc.result)
                }
                for tc in tool_calls
            ]
        }
        
        return MetaDecision(
            symbol=symbol,
            action=action,
            conviction=conviction,
            reasoning=reasoning,
            evidence=evidence,
            tool_calls=tool_calls,
            timestamp=datetime.now()
        )
    
    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        获取agent信息
        
        Args:
            agent_name: Agent名称
            
        Returns:
            Agent信息字典，如果不存在返回None
        """
        if agent_name not in self.agents:
            return None
        
        conn = self.agents[agent_name]
        return {
            "name": conn.name,
            "description": conn.description,
            "tools": conn.tools,
            "resources": conn.resources
        }
    
    def list_agents(self) -> List[str]:
        """获取所有连接的agent名称"""
        return list(self.agents.keys())
    
    def get_tool_call_history(self, limit: Optional[int] = None) -> List[ToolCall]:
        """
        获取工具调用历史
        
        Args:
            limit: 返回的最大数量
            
        Returns:
            工具调用列表
        """
        if limit:
            return self.tool_call_history[-limit:]
        return self.tool_call_history
    
    def get_decision_history(self, limit: Optional[int] = None) -> List[MetaDecision]:
        """
        获取决策历史
        
        Args:
            limit: 返回的最大数量
            
        Returns:
            决策列表
        """
        if limit:
            return self.decision_history[-limit:]
        return self.decision_history
    
    def clear_history(self) -> None:
        """清空历史记录"""
        self.tool_call_history.clear()
        self.decision_history.clear()


# 便捷函数
async def create_meta_agent_with_technical(
    llm_config: Optional[LLMConfig] = None,
    state_manager: Optional[MultiTimeframeStateManager] = None,
    algorithm: Any = None
) -> MetaAgent:
    """
    创建Meta Agent并连接Technical Agent
    
    这是一个便捷函数，用于快速设置基础配置。
    
    Args:
        llm_config: LLM configuration (uses default if None) - DEPRECATED, use llm_client
        state_manager: StateManager instance
        algorithm: LEAN algorithm instance (optional) - DEPRECATED, not used in core agents
        
    Returns:
        配置好的MetaAgent实例
    """
    from Agents.core import TechnicalAnalysisAgent
    
    # 创建Meta Agent（使用新API）
    meta = MetaAgent(
        llm_client=llm_config.get_llm() if llm_config else None,
        state_manager=state_manager,
        enable_memory=state_manager is not None
    )
    
    # 创建并连接Technical Agent（不需要algorithm参数）
    technical = TechnicalAnalysisAgent()
    await meta.connect_to_agent(
        agent_name="technical",
        agent_instance=technical,
        description="Technical analysis specialist providing indicators, signals, patterns, and support/resistance levels"
    )
    
    return meta


if __name__ == "__main__":
    # 示例用法
    async def main():
        # 创建Meta Agent
        meta = MetaAgent(
            enable_memory=False  # 使用新API
        )
        
        # 连接Technical Agent
        from Agents.core import TechnicalAnalysisAgent
        technical = TechnicalAnalysisAgent()
        await meta.connect_to_agent(
            agent_name="technical",
            agent_instance=technical,
            description="Technical analysis specialist"
        )
        
        # 查看可用工具
        print("Available tools:")
        for tool in meta.get_all_tools():
            print(f"  - {tool['agent_name']}.{tool['name']}: {tool['description']}")
        
        # 执行工具
        result = await meta.execute_tool(
            agent_name="technical",
            tool_name="calculate_indicators",
            arguments={"symbol": "AAPL"}
        )
        print(f"\nTool result: {json.dumps(result, indent=2, default=str)}")
        
        # 如果有API key，可以进行完整决策
        # decision = await meta.analyze_and_decide(
        #     symbol="AAPL",
        #     query="Should I buy AAPL based on technical analysis?"
        # )
        # print(f"\nDecision: {decision.action} (conviction: {decision.conviction})")
        # print(f"Reasoning: {decision.reasoning}")
    
    asyncio.run(main())
