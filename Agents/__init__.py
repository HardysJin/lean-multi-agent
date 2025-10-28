"""
Agents Module - MCP-based Multi-Agent System

包含：
- base_mcp_agent: MCP Agent基类
- technical_agent: 技术分析Agent (MCP Server)
- news_agent: 新闻分析Agent (MCP Server, 可选)
- meta_agent: Meta Agent (MCP Client)
- prompt_builder: Prompt构建工具

Legacy (Old Implementation):
- base_agent: Agent基类 (旧版)
- multi_agent_system: 现有的Multi-Agent系统 (旧版)
"""

from .base_mcp_agent import BaseMCPAgent, ExampleAgent
from .technical_agent import TechnicalAnalysisAgent

__all__ = [
    # MCP-based Agents
    'BaseMCPAgent',
    'ExampleAgent',
    'TechnicalAnalysisAgent',
]
