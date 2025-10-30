"""
Core Agents - 纯业务逻辑层

这一层包含所有专家 Agent 的核心业务逻辑，不依赖任何协议（如 MCP）。
设计原则：
- 纯 Python 类
- 易于测试（支持依赖注入）
- 不涉及网络通信或协议
- 可以被任何上层协议包装（MCP, REST, gRPC 等）
"""

from .base_agent import BaseAgent
from .macro_agent import MacroAgent, MacroContext
from .sector_agent import SectorAgent, SectorContext, SECTOR_MAPPING
from .news_agent import NewsAgent, NewsArticle, NewsSentimentReport
from .technical_agent import TechnicalAnalysisAgent

# 重新导出测试工具
from Agents.utils.llm_config import MockLLM

__all__ = [
    'BaseAgent',
    'MacroAgent',
    'MacroContext',
    'SectorAgent',
    'SectorContext',
    'SECTOR_MAPPING',
    'NewsAgent',
    'NewsArticle',
    'NewsSentimentReport',
    'TechnicalAnalysisAgent',
    'MockLLM',  # 测试工具
]
