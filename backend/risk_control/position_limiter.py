"""
Position Limiter - enforces position size constraints
仓位限制器
"""

from typing import Dict, Any
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class PositionLimiter:
    """
    仓位限制器
    
    强制执行仓位约束规则
    """
    
    def __init__(
        self,
        max_position: float = 0.3,
        min_position: float = 0.0,
        position_step: float = 0.05
    ):
        """
        初始化仓位限制器
        
        Args:
            max_position: 单只最大仓位
            min_position: 单只最小仓位（低于此值视为0）
            position_step: 仓位调整步长
        """
        self.max_position = max_position
        self.min_position = min_position
        self.position_step = position_step
    
    def limit_position(self, position: float) -> float:
        """
        限制单个仓位
        
        Args:
            position: 原始仓位
        
        Returns:
            float: 限制后的仓位
        """
        # 低于最小仓位视为0
        if position < self.min_position:
            return 0.0
        
        # 超过最大仓位限制
        if position > self.max_position:
            logger.warning(f"Position {position:.2%} exceeds max {self.max_position:.2%}, capping")
            return self.max_position
        
        # 按步长取整
        if self.position_step > 0:
            position = round(position / self.position_step) * self.position_step
        
        return position
    
    def limit_portfolio(self, positions: Dict[str, float]) -> Dict[str, float]:
        """
        限制整个组合的仓位
        
        Args:
            positions: 原始仓位字典
        
        Returns:
            Dict: 限制后的仓位字典
        """
        limited = {}
        
        for symbol, weight in positions.items():
            if symbol == 'cash':
                # 现金单独处理
                limited['cash'] = weight
            else:
                limited_weight = self.limit_position(weight)
                if limited_weight > 0:
                    limited[symbol] = limited_weight
        
        # 重新归一化
        total = sum(limited.values())
        if total > 0 and abs(total - 1.0) > 0.01:
            for symbol in limited:
                limited[symbol] /= total
        
        return limited
    
    def round_to_lots(
        self,
        shares: int,
        lot_size: int = 100
    ) -> int:
        """
        按手数取整
        
        Args:
            shares: 股票数量
            lot_size: 一手的股数（默认100股）
        
        Returns:
            int: 取整后的股数
        """
        lots = shares // lot_size
        return lots * lot_size
