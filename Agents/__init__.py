"""
Agents Module - MCP-based Multi-Agent System

包含：
- base_mcp_agent: MCP Agent基类
- technical_agent: 技术分析Agent (MCP Server)
- news_agent: 新闻分析Agent (MCP Server, 可选)
- macro_agent: 宏观分析Agent (MCP Server) - NEW
- meta_agent: Meta Agent (MCP Client)
- prompt_builder: Prompt构建工具
- llm_config: 统一LLM配置管理

Legacy (Old Implementation):
- base_agent: Agent基类 (旧版)
- multi_agent_system: 现有的Multi-Agent系统 (旧版)
"""

from .base_mcp_agent import BaseMCPAgent, ExampleAgent
from .technical_agent import TechnicalAnalysisAgent
from .news_agent import NewsAgent, NewsArticle, NewsSentimentReport
from .macro_agent import MacroAgent, MacroContext
from .sector_agent import SectorAgent, SectorContext
from .meta_agent import MetaAgent, MetaDecision, ToolCall, create_meta_agent_with_technical
from .llm_config import (
    LLMConfig, LLMProvider,
    get_default_llm, get_default_llm_config, set_default_llm_config,
    create_llm, get_available_providers, get_mock_llm
)

__all__ = [
    # MCP-based Agents
    'BaseMCPAgent',
    'ExampleAgent',
    'TechnicalAnalysisAgent',
    'NewsAgent',
    'NewsArticle',
    'NewsSentimentReport',
    'MetaAgent',
    'MetaDecision',
    'ToolCall',
    'create_meta_agent_with_technical',
    
    # LLM Configuration
    'LLMConfig',
    'LLMProvider',
    'get_default_llm',
    'get_default_llm_config',
    'set_default_llm_config',
    'create_llm',
    'get_available_providers',
    'get_mock_llm',
]
