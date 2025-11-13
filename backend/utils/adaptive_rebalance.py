"""
动态Rebalance频率计算器 - 指数衰减公式

根据市场波动率自适应调整决策频率
使用公式: days = max_days * exp(-decay_rate * max(0, vix - vix_baseline))
"""

import math
from typing import Dict, Any, Optional
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class AdaptiveRebalanceCalculator:
    """动态决策频率计算器 - 指数衰减模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化
        
        Args:
            config: adaptive_rebalance配置
        """
        self.enabled = config.get('enabled', False)
        self.min_days = config.get('min_days', 1)
        self.max_days = config.get('max_days', 14)
        
        # 指数衰减参数
        self.vix_baseline = config.get('vix_baseline', 15.0)
        self.decay_rate = config.get('decay_rate', 0.15)
        
        # 早期触发配置
        self.early_triggers = config.get('early_triggers', {})
        self.early_triggers_enabled = self.early_triggers.get('enabled', True)
        
        logger.info(f"动态Rebalance计算器初始化完成 (enabled={self.enabled})")
        if self.enabled:
            logger.info(f"  指数衰减公式: days = {self.max_days} * exp(-{self.decay_rate} * max(0, vix - {self.vix_baseline}))")
            logger.info(f"  示例: VIX={self.vix_baseline}→{self.max_days}天, VIX=20→{self._calculate_days_from_vix(20)}天, VIX=30→{self._calculate_days_from_vix(30)}天, VIX=40→{self._calculate_days_from_vix(40)}天")
    
    def _calculate_days_from_vix(self, vix: float) -> int:
        """
        指数衰减公式: days = max_days * exp(-decay_rate * max(0, vix - vix_baseline))
        
        示例 (baseline=15, decay=0.15, max_days=14):
          VIX=15 → 14 * exp(0) = 14天
          VIX=20 → 14 * exp(-0.15*5) = 14 * 0.472 = 6.6天 → 7天
          VIX=25 → 14 * exp(-0.15*10) = 14 * 0.223 = 3.1天 → 3天
          VIX=30 → 14 * exp(-0.15*15) = 14 * 0.105 = 1.5天 → 2天
          VIX=35 → 14 * exp(-0.15*20) = 14 * 0.050 = 0.7天 → 1天
          VIX≥40 → 14 * exp(-0.15*25) = 14 * 0.024 = 0.3天 → 1天
        """
        if vix <= self.vix_baseline:
            return self.max_days
        
        exponent = -self.decay_rate * (vix - self.vix_baseline)
        days = self.max_days * math.exp(exponent)
        
        # 四舍五入并确保在范围内
        return max(self.min_days, min(self.max_days, int(round(days))))
    
    def calculate_next_rebalance_days(
        self,
        market_state: Dict[str, Any],
        current_regime: Optional[str] = None,
        default_days: int = 14
    ) -> Dict[str, Any]:
        """
        计算下次决策间隔天数（使用指数衰减公式）
        
        Args:
            market_state: 市场状态
                - vix: 当前VIX值
                - realized_volatility: 实际波动率（年化）
                - price_change_7d: 最近7天价格变化（小数）
                - price_change_1d: 最近1天价格变化（小数）
            current_regime: 当前市场regime（可选）
            default_days: 默认天数
        
        Returns:
            {
                'days': int,  # 下次决策间隔天数
                'regime': str,  # 波动率regime
                'reasoning': str  # 推理
            }
        """
        if not self.enabled:
            return {
                'days': default_days,
                'regime': 'default',
                'reasoning': f'动态频率未启用，使用默认{default_days}天'
            }
        
        vix = market_state.get('vix', 15)
        rv = market_state.get('realized_volatility', 0.15)
        price_7d = abs(market_state.get('price_change_7d', 0))
        
        # 使用指数衰减公式
        days = self._calculate_days_from_vix(vix)
        regime = self._classify_regime_by_vix(vix)
        reasoning = f"VIX={vix:.1f}→{days}天 (指数衰减: {self.max_days}*exp(-{self.decay_rate}*max(0,vix-{self.vix_baseline})))"
        
        # 价格剧烈波动修正（可选）
        if price_7d > 0.15:  # 7天>15%
            original_days = days
            days = min(days, 3)  # 最多3天
            if days < original_days:
                reasoning += f" | 7天变化{price_7d:.1%}，缩短至{days}天"
        
        logger.info(f"动态频率: regime={regime}, days={days}, {reasoning}")
        
        return {
            'days': days,
            'regime': regime,
            'reasoning': reasoning
        }
    
    def _classify_regime_by_vix(self, vix: float) -> str:
        """根据VIX分类regime"""
        if vix >= 40:
            return 'extreme_panic'
        elif vix >= 30:
            return 'high_volatility'
        elif vix >= 25:
            return 'elevated'
        elif vix >= 20:
            return 'moderate'
        else:
            return 'low'
    
    def should_trigger_early_rebalance(
        self,
        current_market_state: Dict[str, Any],
        last_market_state: Dict[str, Any],
        days_since_last_decision: int
    ) -> Dict[str, Any]:
        """
        检查是否应该提前触发决策
        
        Args:
            current_market_state: 当前市场状态
            last_market_state: 上次决策时的市场状态
            days_since_last_decision: 距上次决策天数
        
        Returns:
            {
                'should_trigger': bool,  # 是否触发
                'reason': str  # 触发原因
            }
        """
        if not self.enabled or not self.early_triggers_enabled:
            return {'should_trigger': False, 'reason': 'Early triggers disabled'}
        
        vix_current = current_market_state.get('vix', 15)
        vix_last = last_market_state.get('vix', 15) if last_market_state else vix_current
        price_change_1d = abs(current_market_state.get('price_change_1d', 0))
        
        # 触发条件1: VIX突破阈值
        vix_spike_threshold = self.early_triggers.get('vix_spike', 40)
        if vix_current >= vix_spike_threshold:
            return {
                'should_trigger': True,
                'reason': f'VIX={vix_current:.1f}突破{vix_spike_threshold}→立即决策'
            }
        
        # 触发条件2: 单日价格剧烈变化
        price_move_threshold = self.early_triggers.get('daily_price_move', 0.10)
        if price_change_1d >= price_move_threshold:
            return {
                'should_trigger': True,
                'reason': f'单日价格变化{price_change_1d:.1%}≥{price_move_threshold:.1%}→立即决策'
            }
        
        # 触发条件3: VIX单日剧烈变化
        vix_change_threshold = self.early_triggers.get('vix_change_pct', 0.25)
        vix_change = abs(vix_current - vix_last) / max(vix_last, 1)
        if vix_change >= vix_change_threshold and days_since_last_decision >= 2:
            return {
                'should_trigger': True,
                'reason': f'VIX变化{vix_change:.1%}≥{vix_change_threshold:.1%} ({vix_last:.1f}→{vix_current:.1f})→立即决策'
            }
        
        return {'should_trigger': False, 'reason': 'No early trigger conditions met'}
    
    def _extract_market_state(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从市场数据中提取状态指标
        
        Args:
            market_data: 市场数据（包含各ticker的价格、VIX等）
        
        Returns:
            市场状态字典
        """
        # 提取VIX
        vix_data = market_data.get('^VIX', {})
        vix = vix_data.get('Close', 15) if isinstance(vix_data, dict) else 15
        
        # 提取主标的价格变化（用于计算波动）
        # 默认使用SPY作为市场代表
        spy_data = market_data.get('SPY', market_data.get('QQQ', {}))
        
        # 计算实际波动率（简化版）
        # 实际应该用historical volatility，这里用最近价格标准差近似
        realized_vol = 0.15  # 默认15%
        
        # 提取价格变化
        price_change_1d = 0.0
        price_change_7d = 0.0
        
        if isinstance(spy_data, dict) and 'Close' in spy_data:
            # 假设market_data包含时间序列
            pass  # 简化处理
        
        return {
            'vix': vix,
            'realized_volatility': realized_vol,
            'price_change_1d': price_change_1d,
            'price_change_7d': price_change_7d
        }
