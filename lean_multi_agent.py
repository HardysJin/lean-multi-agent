"""
Lean Multi-Agent Trading System

这个模块提供了一个统一的命名空间入口点，避免与其他包的命名冲突。

推荐使用方式:
    >>> import lean_multi_agent as lma
    >>> meta = lma.agents.MetaAgent()
    >>> record = lma.memory.DecisionRecord(...)

向后兼容方式（不推荐，可能有命名冲突）:
    >>> from Agents.meta_agent import MetaAgent
    >>> from Memory.schemas import DecisionRecord

为什么需要命名空间?
    如果其他包也有 "Agents" 或 "Memory" 模块，直接导入会冲突。
    使用 lean_multi_agent 前缀可以明确指定来源，避免歧义。
"""

__version__ = "0.1.0"
__author__ = "HardysJin"
__email__ = ""

# 导入所有子模块，提供命名空间访问
# 这样可以使用 lean_multi_agent.agents.MetaAgent 的方式
try:
    # 尝试使用相对导入（如果在包内）
    from . import agents
    from . import memory  
    from . import algorithm
except ImportError:
    # 如果相对导入失败，使用绝对导入（向后兼容）
    import sys
    import os
    
    # 添加项目根目录到路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # 导入模块并重命名
    import Agents as agents
    import Memory as memory
    import Algorithm as algorithm

# 便捷访问常用类
from Agents.orchestration import MetaAgent, MetaDecision
from Agents.core import TechnicalAnalysisAgent, NewsAgent
from Agents.base_mcp_agent import BaseMCPAgent
from Agents.utils.llm_config import LLMConfig, LLMProvider, get_default_llm

from Memory.state_manager import MultiTimeframeStateManager, create_state_manager
from Memory.schemas import DecisionRecord, Timeframe, MemoryDocument
from Memory.sql_store import SQLStore
from Memory.vector_store import VectorStore

__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    
    # 子模块（命名空间访问）
    'agents',
    'memory',
    'algorithm',
    
    # 常用类（直接访问）
    'MetaAgent',
    'MetaDecision',
    'TechnicalAnalysisAgent',
    'NewsAgent',
    'BaseMCPAgent',
    'LLMConfig',
    'LLMProvider',
    'get_default_llm',
    'MultiTimeframeStateManager',
    'create_state_manager',
    'DecisionRecord',
    'Timeframe',
    'MemoryDocument',
    'SQLStore',
    'VectorStore',
]


def get_version():
    """获取版本号"""
    return __version__


def list_agents():
    """列出所有可用的 Agent 类型"""
    return [
        'MetaAgent',
        'TechnicalAnalysisAgent', 
        'NewsAgent',
        'BaseMCPAgent',
    ]


def list_memory_components():
    """列出所有 Memory System 组件"""
    return [
        'MultiTimeframeStateManager',
        'SQLStore',
        'VectorStore',
        'DecisionRecord',
        'Timeframe',
        'MemoryDocument',
    ]


# 提供便捷的创建函数
def create_meta_agent(**kwargs):
    """
    便捷函数：创建 MetaAgent
    
    Args:
        **kwargs: 传递给 MetaAgent 的参数
        
    Returns:
        MetaAgent 实例
        
    Example:
        >>> import lean_multi_agent as lma
        >>> meta = lma.create_meta_agent(enable_memory=True)
    """
    return MetaAgent(**kwargs)


def create_technical_agent(**kwargs):
    """
    便捷函数：创建 TechnicalAnalysisAgent
    
    Args:
        **kwargs: 传递给 TechnicalAnalysisAgent 的参数
        
    Returns:
        TechnicalAnalysisAgent 实例
        
    Example:
        >>> import lean_multi_agent as lma
        >>> tech = lma.create_technical_agent()
    """
    return TechnicalAnalysisAgent(**kwargs)


def create_news_agent(**kwargs):
    """
    便捷函数：创建 NewsAgent
    
    Args:
        **kwargs: 传递给 NewsAgent 的参数
        
    Returns:
        NewsAgent 实例
        
    Example:
        >>> import lean_multi_agent as lma
        >>> news = lma.create_news_agent(news_api_key='your-key')
    """
    return NewsAgent(**kwargs)


# 打印包信息
def info():
    """打印包信息"""
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║     Lean Multi-Agent Trading System v{__version__}           ║
    ╚══════════════════════════════════════════════════════════╝
    
    📦 安装位置: {__file__}
    
    🤖 可用 Agents:
       - MetaAgent (协调者)
       - TechnicalAnalysisAgent (技术分析)
       - NewsAgent (新闻情绪分析)
    
    💾 Memory System:
       - MultiTimeframeStateManager
       - SQLStore (关系数据库)
       - VectorStore (向量数据库)
    
    📚 推荐使用方式:
       import lean_multi_agent as lma
       meta = lma.MetaAgent()
       
    📖 文档: https://github.com/HardysJin/lean-multi-agent
    """)


if __name__ == "__main__":
    # 如果直接运行此模块，打印包信息
    info()
