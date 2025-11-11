"""
Base data collector interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


class BaseCollector(ABC):
    """数据收集器基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化收集器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
    
    @abstractmethod
    def collect(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Dict[str, Any]:
        """
        收集数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
        
        Returns:
            Dict: 收集到的数据
        """
        pass
    
    def collect_last_n_days(
        self, 
        n_days: int = 7, 
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        收集过去N天数据
        
        Args:
            n_days: 收集的天数
            end_date: 结束日期，默认为今天
        
        Returns:
            Dict: 过去N天的数据
        """
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=n_days)
        return self.collect(start_date, end_date)
    
    def collect_last_week(self, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        收集上周数据（便捷方法，等同于collect_last_n_days(7)）
        
        Args:
            end_date: 结束日期，默认为今天
        
        Returns:
            Dict: 上周数据
        """
        return self.collect_last_n_days(7, end_date)
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        验证数据完整性
        
        Args:
            data: 待验证的数据
        
        Returns:
            bool: 是否有效
        """
        return data is not None and len(data) > 0
