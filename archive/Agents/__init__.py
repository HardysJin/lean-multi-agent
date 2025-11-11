"""
Agents Module - MCP-based Multi-Agent System

包含：
- core: 核心业务逻辑层（纯Python，无协议依赖）
- orchestration: 编排层（MetaAgent, DecisionMakers, Schedulers）
- base_mcp_agent: MCP Agent基类
- utils: 工具模块（LLM配置等）

Legacy (Old Implementation):
- base_agent: Agent基类 (旧版)
- multi_agent_system: 现有的Multi-Agent系统 (旧版)
"""

from .base_mcp_agent import BaseMCPAgent, ExampleAgent

# 重新导出 orchestration 模块（编排层）
from .orchestration import (
    MetaAgent, MetaDecision,
    StrategicDecisionMaker,
    CampaignDecisionMaker,
    TacticalDecisionMaker,
    DecisionMakerFactory,
    Decision,
    LayeredScheduler,
    MultiSymbolScheduler,
    DecisionLevel,
    EscalationReason,
    SchedulerState
)

# 重新导出 core 模块（新架构 - 纯业务逻辑）
from .core import (
    MacroAgent, MacroContext,
    SectorAgent, SectorContext, SECTOR_MAPPING,
    NewsAgent, NewsArticle, NewsSentimentReport,
    TechnicalAnalysisAgent
)

# 重新导出 utils（工具类）
from .utils.llm_config import (
    LLMConfig, LLMProvider,
    get_default_llm, get_default_llm_config, set_default_llm_config,
    create_llm, get_available_providers, get_mock_llm, MockLLM
)

__all__ = [
    # MCP-based Agents
    'BaseMCPAgent',
    'ExampleAgent',
    
    # Orchestration Layer (编排层)
    'MetaAgent',
    'MetaDecision',
    'StrategicDecisionMaker',
    'CampaignDecisionMaker',
    'TacticalDecisionMaker',
    'DecisionMakerFactory',
    'Decision',
    'LayeredScheduler',
    'MultiSymbolScheduler',
    'DecisionLevel',
    'EscalationReason',
    'SchedulerState',
    
    # Core Business Logic (核心业务逻辑层)
    'MacroAgent',
    'MacroContext',
    'SectorAgent',
    'SectorContext',
    'SECTOR_MAPPING',
    'NewsAgent',
    'NewsArticle',
    'NewsSentimentReport',
    'TechnicalAnalysisAgent',
    
    # LLM Configuration (工具类)
    'LLMConfig',
    'LLMProvider',
    'get_default_llm',
    'get_default_llm_config',
    'set_default_llm_config',
    'create_llm',
    'get_available_providers',
    'get_mock_llm',
    'MockLLM',
]
