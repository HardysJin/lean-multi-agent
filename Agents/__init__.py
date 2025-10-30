"""
Agents Module - MCP-based Multi-Agent System

包含：
- core: 核心业务逻辑层（纯Python，无协议依赖）
- base_mcp_agent: MCP Agent基类
- technical_agent: 技术分析Agent (MCP Server)
- news_agent: 新闻分析Agent (MCP Server, 可选)
- sector_agent: 行业分析Agent
- meta_agent: Meta Agent (MCP Client)
- decision_makers: 决策制定器
- layered_scheduler: 分层调度器
- utils: 工具模块（LLM配置等）

Legacy (Old Implementation):
- base_agent: Agent基类 (旧版)
- multi_agent_system: 现有的Multi-Agent系统 (旧版)
"""

from .base_mcp_agent import BaseMCPAgent, ExampleAgent
from .meta_agent import MetaAgent, MetaDecision, ToolCall, create_meta_agent_with_technical
from .decision_makers import (
    StrategicDecisionMaker,
    CampaignDecisionMaker,
    TacticalDecisionMaker,
    DecisionMakerFactory,
    Decision
)
from .layered_scheduler import (
    LayeredScheduler,
    MultiSymbolScheduler,
    DecisionLevel,
    EscalationReason,
    SchedulerState
)
from .utils.llm_config import (
    LLMConfig, LLMProvider,
    get_default_llm, get_default_llm_config, set_default_llm_config,
    create_llm, get_available_providers, get_mock_llm, MockLLM
)

# 重新导出 core 模块（新架构 - 纯业务逻辑）
from .core import (
    MacroAgent, MacroContext,
    SectorAgent, SectorContext, SECTOR_MAPPING,
    NewsAgent, NewsArticle, NewsSentimentReport,
    TechnicalAnalysisAgent
)

__all__ = [
    # MCP-based Agents
    'BaseMCPAgent',
    'ExampleAgent',
    'MetaAgent',
    'MetaDecision',
    'ToolCall',
    'create_meta_agent_with_technical',
    
    # Core Business Logic (New Architecture)
    'MacroAgent',
    'MacroContext',
    'SectorAgent',
    'SectorContext',
    'SECTOR_MAPPING',
    'NewsAgent',
    'NewsArticle',
    'NewsSentimentReport',
    'TechnicalAnalysisAgent',
    
    # DecisionMakers
    'StrategicDecisionMaker',
    'CampaignDecisionMaker',
    'TacticalDecisionMaker',
    'DecisionMakerFactory',
    'Decision',
    
    # Scheduler
    'LayeredScheduler',
    'MultiSymbolScheduler',
    'DecisionLevel',
    'EscalationReason',
    'SchedulerState',
    
    # LLM Configuration
    'LLMConfig',
    'LLMProvider',
    'get_default_llm',
    'get_default_llm_config',
    'set_default_llm_config',
    'create_llm',
    'get_available_providers',
    'get_mock_llm',
    'MockLLM',  # 测试工具
]
