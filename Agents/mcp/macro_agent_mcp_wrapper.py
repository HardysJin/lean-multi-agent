"""
Macro Agent - MCP Server Wrapper (Backward Compatibility)

这是一个兼容性包装器，保持原有的MCP Server接口。
核心业务逻辑已移至 Agents/core/macro_agent.py

新代码应该直接使用:
    from Agents.core import MacroAgent

这个文件保留是为了向后兼容，逐步迁移。
"""

from .base_mcp_agent import BaseMCPAgent
from .core.macro_agent import MacroAgent as CoreMacroAgent, MacroContext
from mcp.types import Tool, Resource
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from Agents.utils.llm_config import LLMConfig


# 重新导出 MacroContext 以保持兼容性
__all__ = ['MacroAgent', 'MacroContext']


class MacroAgent(BaseMCPAgent):
    """
    MacroAgent MCP Server Wrapper
    
    包装 CoreMacroAgent，提供 MCP Server 接口。
    所有业务逻辑委托给 core agent。
    
    使用示例（兼容旧代码）：
    ```python
    agent = MacroAgent()
    context = await agent.analyze_macro_environment()
    ```
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        cache_ttl: int = 3600,
        enable_cache: bool = True
    ):
        """初始化 MCP Server 包装器"""
        super().__init__(
            name="macro-agent",
            description="Analyzes macro economic environment, monetary policy, and market regime",
            version="1.0.0",
            llm_config=llm_config,
            enable_llm=True
        )
        
        # 创建 core agent（委托所有业务逻辑）
        self._core = CoreMacroAgent(
            llm_client=self._llm_client if hasattr(self, '_llm_client') else None,
            llm_config=llm_config,
            cache_ttl=cache_ttl,
            enable_cache=enable_cache
        )
        
        self.logger.info("MacroAgent MCP wrapper initialized")
    
    # ═══════════════════════════════════════════════
    # Properties - 暴露 core agent 属性以保持兼容性
    # ═══════════════════════════════════════════════
    
    @property
    def cache_ttl(self) -> int:
        """获取缓存TTL"""
        return self._core.cache_ttl
    
    @property
    def enable_cache(self) -> bool:
        """是否启用缓存"""
        return self._core.enable_cache
    
    @property
    def _cache(self) -> Dict:
        """获取缓存（用于测试）"""
        return self._core._cache
    
    # ═══════════════════════════════════════════════
    # MCP Protocol Implementation - 委托给 core agent
    # ═══════════════════════════════════════════════
    
    def get_tools(self) -> List[Tool]:
        """返回 MCP 工具列表"""
        return [
            Tool(
                name="analyze_macro_environment",
                description=(
                    "Perform comprehensive macro environment analysis including "
                    "market regime, interest rates, economic indicators, and risk constraints."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "visible_data_end": {
                            "type": "string",
                            "description": "Optional ISO datetime string for backtest mode"
                        },
                        "force_refresh": {
                            "type": "boolean",
                            "description": "Force refresh analysis, ignore cache"
                        }
                    }
                }
            ),
            Tool(
                name="get_market_regime",
                description="Quick analysis to determine current market regime",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "visible_data_end": {
                            "type": "string",
                            "description": "Optional ISO datetime string for backtest mode"
                        }
                    }
                }
            ),
            Tool(
                name="get_risk_constraints",
                description="Get risk management constraints based on current macro environment",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "visible_data_end": {
                            "type": "string",
                            "description": "Optional ISO datetime string for backtest mode"
                        }
                    }
                }
            )
        ]
    
    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """处理 MCP 工具调用 - 委托给 core agent"""
        self.logger.info(f"MCP Tool called: {name}")
        
        # 解析 datetime 参数
        visible_data_end = arguments.get('visible_data_end')
        if visible_data_end:
            visible_data_end = datetime.fromisoformat(visible_data_end)
        
        if name == "analyze_macro_environment":
            force_refresh = arguments.get('force_refresh', False)
            context = await self._core.analyze_macro_environment(visible_data_end, force_refresh)
            return context.to_dict()
        
        elif name == "get_market_regime":
            return await self._core.get_market_regime(visible_data_end)
        
        elif name == "get_risk_constraints":
            return await self._core.get_risk_constraints(visible_data_end)
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    def get_resources(self) -> List[Resource]:
        """返回 MCP 资源列表"""
        return [
            Resource(
                uri="macro://current",
                name="Current Macro Environment",
                description="Current macro economic environment analysis",
                mimeType="application/json"
            ),
            Resource(
                uri="macro://cache-stats",
                name="Cache Statistics",
                description="Cache statistics and performance metrics",
                mimeType="application/json"
            )
        ]
    
    async def handle_resource_read(self, uri: str) -> str:
        """处理 MCP 资源读取 - 委托给 core agent"""
        self.logger.info(f"MCP Resource read: {uri}")
        
        if uri == "macro://current":
            context = await self._core.analyze_macro_environment()
            return json.dumps(context.to_dict(), indent=2)
        
        elif uri == "macro://cache-stats":
            stats = self._core.get_cache_stats()
            return json.dumps(stats, indent=2)
        
        else:
            raise ValueError(f"Unknown resource URI: {uri}")
    
    # ═══════════════════════════════════════════════
    # Public API - 委托给 core agent
    # ═══════════════════════════════════════════════
    
    async def analyze_macro_environment(
        self,
        visible_data_end: Optional[datetime] = None,
        force_refresh: bool = False
    ) -> MacroContext:
        """委托给 core agent"""
        return await self._core.analyze_macro_environment(visible_data_end, force_refresh)
    
    async def get_market_regime(
        self,
        visible_data_end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """委托给 core agent"""
        return await self._core.get_market_regime(visible_data_end)
    
    async def get_risk_constraints(
        self,
        visible_data_end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """委托给 core agent"""
        return await self._core.get_risk_constraints(visible_data_end)
    
    def clear_cache(self):
        """委托给 core agent"""
        self._core.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """委托给 core agent"""
        return self._core.get_cache_stats()
