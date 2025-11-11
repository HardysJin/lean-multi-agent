"""
Memory Module - 多时间尺度分层记忆系统

包含：
- schemas: 数据结构定义
- vector_store: 向量数据库wrapper
- sql_store: SQL数据库wrapper
- state_manager: 多时间尺度状态管理器
"""

from .schemas import (
    Timeframe,
    DecisionRecord,
    MemoryDocument,
    HierarchicalConstraints,
    create_decision_id,
    create_memory_id,
)

from .vector_store import (
    VectorStore,
    create_vector_store,
)

from .sql_store import (
    SQLStore,
    create_sql_store,
)

from .state_manager import (
    MultiTimeframeStateManager,
    create_state_manager,
)

from .escalation import (
    EscalationDetector,
    EscalationTrigger,
    EscalationTriggerType,
    should_trigger_escalation,
)

__all__ = [
    'Timeframe',
    'DecisionRecord',
    'MemoryDocument',
    'HierarchicalConstraints',
    'create_decision_id',
    'create_memory_id',
    'VectorStore',
    'create_vector_store',
    'SQLStore',
    'create_sql_store',
    'MultiTimeframeStateManager',
    'create_state_manager',
    'EscalationDetector',
    'EscalationTrigger',
    'EscalationTriggerType',
    'should_trigger_escalation',
]
