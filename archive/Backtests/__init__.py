"""
Backtests 模块 - VectorBT 集成

提供基于 VectorBT 的高性能回测功能，支持：
- Multi-Agent 策略回测
- 批量信号预计算
- 详细的性能分析
- 可视化报告生成
"""

from .vectorbt_engine import VectorBTBacktest
from .strategies.multi_agent_strategy import MultiAgentStrategy

__all__ = [
    'VectorBTBacktest',
    'MultiAgentStrategy'
]
