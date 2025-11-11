"""
Risk Manager - validates and adjusts trading decisions
风险管理器 - 验证和调整交易决策
"""

from typing import Dict, Any, Optional, List, Tuple
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class RiskManager:
    """
    风险管理器
    
    职责：
    1. 验证LLM决策的合规性
    2. 自动调整不合规的仓位
    3. 触发熔断机制
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化风险管理器
        
        Args:
            config: 风控配置
        """
        self.config = config or {}
        
        # 风控参数
        self.max_single_position = self.config.get('max_single_position', 0.3)
        self.min_cash_reserve = self.config.get('min_cash_reserve', 0.2)
        self.max_weekly_turnover = self.config.get('max_weekly_turnover', 0.5)
        self.max_drawdown = self.config.get('max_drawdown', 0.15)
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.05)
        self.validate_before_execute = self.config.get('validate_before_execute', True)
    
    def validate_decision(
        self,
        decision: Dict[str, Any],
        current_portfolio: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, List[str]]:
        """
        验证决策是否合规
        
        Args:
            decision: LLM生成的决策
            current_portfolio: 当前持仓（{symbol: weight}）
        
        Returns:
            Tuple[bool, List[str]]: (是否合规, 违规原因列表)
        """
        violations = []
        
        suggested_positions = decision.get('suggested_positions')
        if not suggested_positions:
            # 没有仓位建议，跳过验证
            return True, []
        
        # 1. 检查单只股票仓位上限
        for symbol, weight in suggested_positions.items():
            if symbol != 'cash' and weight > self.max_single_position:
                violations.append(
                    f"{symbol} position ({weight:.1%}) exceeds max ({self.max_single_position:.1%})"
                )
        
        # 2. 检查现金储备下限
        cash_reserve = suggested_positions.get('cash', 0)
        if cash_reserve < self.min_cash_reserve:
            violations.append(
                f"Cash reserve ({cash_reserve:.1%}) below minimum ({self.min_cash_reserve:.1%})"
            )
        
        # 3. 检查总仓位是否为100%
        total_weight = sum(suggested_positions.values())
        if abs(total_weight - 1.0) > 0.01:  # 允许1%的误差
            violations.append(
                f"Total position ({total_weight:.1%}) does not equal 100%"
            )
        
        # 4. 检查换手率（如果有当前持仓）
        if current_portfolio:
            turnover = self._calculate_turnover(suggested_positions, current_portfolio)
            if turnover > self.max_weekly_turnover:
                violations.append(
                    f"Turnover ({turnover:.1%}) exceeds max ({self.max_weekly_turnover:.1%})"
                )
        
        is_valid = len(violations) == 0
        
        if not is_valid:
            logger.warning(f"Decision validation failed: {'; '.join(violations)}")
        
        return is_valid, violations
    
    def adjust_positions(
        self,
        suggested_positions: Dict[str, float],
        current_portfolio: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        调整仓位使其合规
        
        Args:
            suggested_positions: 建议仓位
            current_portfolio: 当前持仓
        
        Returns:
            Dict: 调整后的合规仓位
        """
        logger.info("Adjusting positions to comply with risk rules")
        
        adjusted = suggested_positions.copy()
        
        # 1. 限制单只股票仓位
        for symbol in list(adjusted.keys()):
            if symbol != 'cash' and adjusted[symbol] > self.max_single_position:
                excess = adjusted[symbol] - self.max_single_position
                adjusted[symbol] = self.max_single_position
                # 多余的分配到现金
                adjusted['cash'] = adjusted.get('cash', 0) + excess
                logger.info(f"Reduced {symbol} from {adjusted[symbol] + excess:.1%} to {self.max_single_position:.1%}")
        
        # 2. 确保最低现金储备
        current_cash = adjusted.get('cash', 0)
        if current_cash < self.min_cash_reserve:
            deficit = self.min_cash_reserve - current_cash
            # 按比例减少其他持仓
            non_cash_total = sum(w for s, w in adjusted.items() if s != 'cash')
            
            if non_cash_total > 0:
                for symbol in list(adjusted.keys()):
                    if symbol != 'cash':
                        reduction_ratio = deficit / non_cash_total
                        reduction = adjusted[symbol] * reduction_ratio
                        adjusted[symbol] -= reduction
                
                adjusted['cash'] = self.min_cash_reserve
                logger.info(f"Increased cash reserve to {self.min_cash_reserve:.1%}")
        
        # 3. 归一化到100%
        total = sum(adjusted.values())
        if abs(total - 1.0) > 0.01:
            for symbol in adjusted:
                adjusted[symbol] /= total
            logger.info("Normalized positions to 100%")
        
        # 4. 检查换手率限制
        if current_portfolio:
            turnover = self._calculate_turnover(adjusted, current_portfolio)
            if turnover > self.max_weekly_turnover:
                # 限制换手：保留更多当前持仓
                adjusted = self._limit_turnover(adjusted, current_portfolio)
        
        return adjusted
    
    def check_circuit_breaker(
        self,
        current_drawdown: float,
        portfolio_value: float
    ) -> bool:
        """
        检查是否触发熔断
        
        Args:
            current_drawdown: 当前回撤（负值）
            portfolio_value: 当前组合价值
        
        Returns:
            bool: 是否应该熔断（停止交易）
        """
        if abs(current_drawdown) >= self.max_drawdown:
            logger.warning(
                f"Circuit breaker triggered! Drawdown {current_drawdown:.1%} "
                f"exceeds limit {self.max_drawdown:.1%}"
            )
            return True
        
        return False
    
    def apply_stop_loss(
        self,
        positions: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        应用止损规则
        
        Args:
            positions: 持仓详情 {symbol: {"entry_price": x, "current_price": y, ...}}
        
        Returns:
            List[str]: 需要止损的股票列表
        """
        stop_loss_list = []
        
        for symbol, info in positions.items():
            if symbol == 'cash':
                continue
            
            entry_price = info.get('entry_price', 0)
            current_price = info.get('current_price', 0)
            
            if entry_price == 0:
                continue
            
            pnl = (current_price - entry_price) / entry_price
            
            if pnl <= -self.stop_loss_pct:
                stop_loss_list.append(symbol)
                logger.warning(
                    f"Stop loss triggered for {symbol}: {pnl:.1%} loss"
                )
        
        return stop_loss_list
    
    def _calculate_turnover(
        self,
        new_positions: Dict[str, float],
        old_positions: Dict[str, float]
    ) -> float:
        """
        计算换手率
        
        Args:
            new_positions: 新仓位
            old_positions: 旧仓位
        
        Returns:
            float: 换手率（0-1）
        """
        # 换手率 = sum(abs(new_weight - old_weight)) / 2
        all_symbols = set(new_positions.keys()) | set(old_positions.keys())
        
        turnover = 0
        for symbol in all_symbols:
            new_weight = new_positions.get(symbol, 0)
            old_weight = old_positions.get(symbol, 0)
            turnover += abs(new_weight - old_weight)
        
        return turnover / 2
    
    def _limit_turnover(
        self,
        suggested_positions: Dict[str, float],
        current_portfolio: Dict[str, float]
    ) -> Dict[str, float]:
        """
        限制换手率
        
        Args:
            suggested_positions: 建议仓位
            current_portfolio: 当前持仓
        
        Returns:
            Dict: 限制换手后的仓位
        """
        # 简化实现：按比例混合当前持仓和建议仓位
        # 使得换手率不超过上限
        
        turnover = self._calculate_turnover(suggested_positions, current_portfolio)
        if turnover <= self.max_weekly_turnover:
            return suggested_positions
        
        # 计算混合比例
        alpha = self.max_weekly_turnover / turnover  # alpha in [0, 1]
        
        # 混合仓位
        adjusted = {}
        all_symbols = set(suggested_positions.keys()) | set(current_portfolio.keys())
        
        for symbol in all_symbols:
            old_weight = current_portfolio.get(symbol, 0)
            new_weight = suggested_positions.get(symbol, 0)
            # 加权平均
            adjusted[symbol] = old_weight * (1 - alpha) + new_weight * alpha
        
        # 归一化
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        logger.info(f"Limited turnover from {turnover:.1%} to {self.max_weekly_turnover:.1%}")
        
        return adjusted
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        获取风控规则摘要
        
        Returns:
            Dict: 风控配置摘要
        """
        return {
            "max_single_position": self.max_single_position,
            "min_cash_reserve": self.min_cash_reserve,
            "max_weekly_turnover": self.max_weekly_turnover,
            "max_drawdown": self.max_drawdown,
            "stop_loss_pct": self.stop_loss_pct
        }
