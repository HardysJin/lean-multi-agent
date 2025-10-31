"""
Orchestration Package - 编排层

包含多Agent协调和决策编排的核心组件：
- MetaAgent: 多Agent协调器
- DecisionMakers: 三层决策制定者（Strategic, Campaign, Tactical）
- LayeredScheduler: 分层调度器
"""

from Agents.orchestration.meta_agent import (
    MetaAgent, 
    MetaDecision, 
    AgentConnection, 
    ToolCall,
    create_meta_agent_with_technical
)
from Agents.orchestration.decision_makers import (
    Decision,
    StrategicDecisionMaker,
    CampaignDecisionMaker,
    TacticalDecisionMaker,
    DecisionMakerFactory
)
from Agents.orchestration.layered_scheduler import (
    LayeredScheduler,
    MultiSymbolScheduler,
    DecisionLevel,
    EscalationReason,
    SchedulerState
)

__all__ = [
    # MetaAgent
    'MetaAgent',
    'MetaDecision',
    'AgentConnection',
    'ToolCall',
    'create_meta_agent_with_technical',
    
    # Decision Makers
    'Decision',
    'StrategicDecisionMaker',
    'CampaignDecisionMaker',
    'TacticalDecisionMaker',
    'DecisionMakerFactory',
    
    # Schedulers
    'LayeredScheduler',
    'MultiSymbolScheduler',
    'DecisionLevel',
    'EscalationReason',
    'SchedulerState',
]
