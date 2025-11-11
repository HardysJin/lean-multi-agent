"""
Base Agent interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime


class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化Agent
        
        Args:
            name: Agent名称
            config: 配置字典
        """
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
    
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析数据
        
        Args:
            data: 输入数据
        
        Returns:
            Dict: 分析结果
        """
        pass
    
    def is_enabled(self) -> bool:
        """检查Agent是否启用"""
        return self.enabled
    
    def get_name(self) -> str:
        """获取Agent名称"""
        return self.name
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据
        
        Returns:
            bool: 是否有效
        """
        return data is not None and len(data) > 0
