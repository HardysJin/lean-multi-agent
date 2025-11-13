"""
Decision Validator - validates trading decisions before execution
决策验证器
"""

from typing import Dict, Any, List, Tuple, Optional
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class DecisionValidator:
    """
    决策验证器
    
    验证决策的完整性和合理性
    """
    
    def __init__(self, valid_strategies: Optional[List[str]] = None):
        """
        初始化验证器
        
        Args:
            valid_strategies: 有效策略列表，如果为None则使用默认列表
        """
        if valid_strategies is None:
            valid_strategies = ['grid_trading', 'momentum', 'mean_reversion', 'double_ema_channel', 'buy_and_hold', 'hold']
        self.valid_strategies = valid_strategies
    
    def validate(self, decision: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证决策
        
        Args:
            decision: 决策字典
        
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误列表)
        """
        errors = []
        
        # 1. 检查必需字段
        required_fields = [
            'market_state',
            'reasoning',
            'recommended_strategy',
            'confidence',
            'risk_assessment'
        ]
        
        for field in required_fields:
            if field not in decision:
                errors.append(f"Missing required field: {field}")
        
        # 2. 验证策略
        strategy = decision.get('recommended_strategy')
        if strategy and strategy not in self.valid_strategies:
            errors.append(f"Invalid strategy: {strategy}, valid strategies: {self.valid_strategies}")
        
        # 3. 验证置信度
        confidence = decision.get('confidence')
        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                errors.append(f"Confidence must be numeric, got {type(confidence)}")
            elif not (0 <= confidence <= 1):
                errors.append(f"Confidence must be in [0, 1], got {confidence}")
        
        # 4. 验证市场状态
        valid_market_states = ['trending', 'ranging', 'volatile', 'event_driven', 'uncertain']
        market_state = decision.get('market_state')
        if market_state and market_state not in valid_market_states:
            errors.append(f"Invalid market_state: {market_state}")
        
        # 5. 验证仓位建议（如果存在）
        suggested_positions = decision.get('suggested_positions')
        if suggested_positions:
            position_errors = self._validate_positions(suggested_positions)
            errors.extend(position_errors)
        
        # 6. 验证reasoning长度
        reasoning = decision.get('reasoning', '')
        if len(reasoning) < 20:
            errors.append("Reasoning too short (minimum 20 characters)")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"Decision validation failed: {'; '.join(errors)}")
        
        return is_valid, errors
    
    def _validate_positions(self, positions: Dict[str, float]) -> List[str]:
        """
        验证仓位建议
        
        Args:
            positions: 仓位字典
        
        Returns:
            List[str]: 错误列表
        """
        errors = []
        
        # 检查类型
        if not isinstance(positions, dict):
            errors.append("Positions must be a dictionary")
            return errors
        
        # 检查每个仓位
        for symbol, weight in positions.items():
            if not isinstance(weight, (int, float)):
                errors.append(f"Position weight for {symbol} must be numeric")
                continue
            
            if weight < 0:
                errors.append(f"Negative position for {symbol}: {weight}")
            
            if weight > 1:
                errors.append(f"Position for {symbol} exceeds 100%: {weight}")
        
        # 检查总仓位
        total = sum(positions.values())
        if abs(total - 1.0) > 0.05:  # 允许5%误差
            errors.append(f"Total position ({total:.1%}) far from 100%")
        
        # 检查是否包含现金
        if 'cash' not in positions:
            errors.append("Missing 'cash' position")
        
        return errors
    
    def validate_against_history(
        self,
        decision: Dict[str, Any],
        history: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """
        根据历史验证决策的合理性
        
        Args:
            decision: 当前决策
            history: 历史决策列表
        
        Returns:
            Tuple[bool, List[str]]: (是否合理, 警告列表)
        """
        warnings = []
        
        if not history:
            return True, []
        
        # 检查策略切换频率
        recent_strategies = [h.get('strategy') for h in history[-4:] if h.get('strategy')]
        current_strategy = decision.get('recommended_strategy')
        
        if recent_strategies and all(s == recent_strategies[0] for s in recent_strategies):
            # 过去4周策略相同
            if current_strategy != recent_strategies[0]:
                warnings.append(
                    f"Strategy changed from consistent {recent_strategies[0]} to {current_strategy}"
                )
        
        # 检查最近表现
        recent_pnl = [h.get('pnl', 0) for h in history[-2:]]
        if len(recent_pnl) >= 2:
            avg_pnl = sum(recent_pnl) / len(recent_pnl)
            if avg_pnl < -0.05:  # 最近平均亏损超过5%
                confidence = decision.get('confidence', 0)
                if confidence > 0.7:
                    warnings.append(
                        f"High confidence ({confidence:.2f}) despite recent losses ({avg_pnl:.1%})"
                    )
        
        # 这些是警告不是错误，所以总是返回True
        if warnings:
            logger.info(f"Decision warnings: {'; '.join(warnings)}")
        
        return True, warnings
