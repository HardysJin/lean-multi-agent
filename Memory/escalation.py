"""
Escalation System - 反向传导系统

负责检测关键事件并触发从低层时间尺度向高层时间尺度的反向传导。
这是分层决策系统的核心机制之一。

核心概念：
- 战术层(Tactical)发现重大信号 → 触发战役层(Campaign)重新评估
- 战役层发现系统性风险 → 触发战略层(Strategic)重新评估
- 黑天鹅事件（COVID-19, 闪崩等）→ 直接触发战略层

触发条件分类：
1. 市场冲击 (market_shock): 单日大幅波动
2. 新闻冲击 (news_impact): 重大新闻事件
3. 技术突破 (technical_breakout): 关键技术位突破
4. 战略冲突 (strategic_conflict): 决策与上层约束冲突
5. 黑天鹅 (black_swan): 极端事件（直达战略层）
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime

from Memory.schemas import Timeframe


class EscalationTriggerType(Enum):
    """反向传导触发类型"""
    MARKET_SHOCK = "market_shock"              # 市场冲击
    NEWS_IMPACT = "news_impact"                # 新闻冲击
    TECHNICAL_BREAKOUT = "technical_breakout"  # 技术突破
    STRATEGIC_CONFLICT = "strategic_conflict"  # 战略冲突
    BLACK_SWAN = "black_swan"                  # 黑天鹅事件
    VOLATILITY_SPIKE = "volatility_spike"      # 波动率飙升
    CORRELATION_BREAK = "correlation_break"    # 相关性破裂


@dataclass
class EscalationTrigger:
    """
    反向传导触发器
    
    封装触发反向传导所需的所有信息
    """
    trigger_type: EscalationTriggerType  # 触发类型
    from_timeframe: Timeframe            # 来源时间尺度
    to_timeframe: Timeframe              # 目标时间尺度
    score: float                         # 触发评分（0-10）
    symbol: str                          # 股票代码
    timestamp: datetime                  # 触发时间
    reason: str                          # 触发原因描述
    details: Dict[str, Any]              # 详细信息
    
    def should_escalate(self, threshold: float = 7.0) -> bool:
        """判断是否应该触发反向传导"""
        return self.score >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'trigger_type': self.trigger_type.value,
            'from_timeframe': self.from_timeframe.display_name,
            'to_timeframe': self.to_timeframe.display_name,
            'score': self.score,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'reason': self.reason,
            'details': self.details,
        }


class EscalationDetector:
    """
    反向传导检测器
    
    分析市场数据、新闻、技术指标等，判断是否需要触发反向传导
    """
    
    # === 默认阈值 ===
    DEFAULT_THRESHOLDS = {
        # 市场冲击阈值
        'market_shock_1day': 0.05,      # 单日涨跌幅 > 5%
        'market_shock_3day': 0.10,      # 3日涨跌幅 > 10%
        'market_shock_extreme': 0.15,   # 极端冲击 > 15%（直达战略层）
        
        # 新闻冲击阈值
        'news_impact_high': 8.0,        # 新闻重要性 > 8/10
        'news_impact_critical': 9.5,    # 关键新闻 > 9.5/10（直达战略层）
        
        # 技术突破阈值
        'technical_breakout_confidence': 0.90,  # 突破置信度 > 90%
        
        # 波动率阈值
        'volatility_spike_multiplier': 3.0,  # 波动率 > 历史均值的3倍
        
        # 战略冲突阈值
        'strategic_conflict_conviction': 7.0,  # 信心度 > 7/10
    }
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        初始化检测器
        
        Args:
            thresholds: 自定义阈值字典（可选）
        """
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)
    
    def detect_market_shock(
        self,
        symbol: str,
        price_change_1d: float,
        price_change_3d: Optional[float] = None,
        current_timeframe: Timeframe = Timeframe.TACTICAL,
    ) -> Optional[EscalationTrigger]:
        """
        检测市场冲击
        
        Args:
            symbol: 股票代码
            price_change_1d: 单日涨跌幅（如 -0.08 表示下跌8%）
            price_change_3d: 3日涨跌幅（可选）
            current_timeframe: 当前时间尺度
        
        Returns:
            EscalationTrigger if shock detected, None otherwise
        """
        abs_change_1d = abs(price_change_1d)
        
        # 极端冲击：直达战略层
        if abs_change_1d >= self.thresholds['market_shock_extreme']:
            return EscalationTrigger(
                trigger_type=EscalationTriggerType.BLACK_SWAN,
                from_timeframe=current_timeframe,
                to_timeframe=Timeframe.STRATEGIC,
                score=10.0,
                symbol=symbol,
                timestamp=datetime.now(),
                reason=f"Extreme market shock: {price_change_1d:.2%} in 1 day",
                details={
                    'price_change_1d': price_change_1d,
                    'threshold': self.thresholds['market_shock_extreme'],
                }
            )
        
        # 严重冲击：升至战役层
        if abs_change_1d >= self.thresholds['market_shock_1day']:
            score = min(10.0, 5.0 + (abs_change_1d / self.thresholds['market_shock_1day']) * 3.0)
            return EscalationTrigger(
                trigger_type=EscalationTriggerType.MARKET_SHOCK,
                from_timeframe=current_timeframe,
                to_timeframe=Timeframe.CAMPAIGN,
                score=score,
                symbol=symbol,
                timestamp=datetime.now(),
                reason=f"Market shock: {price_change_1d:.2%} in 1 day",
                details={
                    'price_change_1d': price_change_1d,
                    'price_change_3d': price_change_3d,
                }
            )
        
        # 3日累计冲击
        if price_change_3d and abs(price_change_3d) >= self.thresholds['market_shock_3day']:
            score = min(10.0, 6.0 + (abs(price_change_3d) / self.thresholds['market_shock_3day']) * 2.5)
            return EscalationTrigger(
                trigger_type=EscalationTriggerType.MARKET_SHOCK,
                from_timeframe=current_timeframe,
                to_timeframe=Timeframe.CAMPAIGN,
                score=score,
                symbol=symbol,
                timestamp=datetime.now(),
                reason=f"Cumulative market shock: {price_change_3d:.2%} in 3 days",
                details={
                    'price_change_1d': price_change_1d,
                    'price_change_3d': price_change_3d,
                }
            )
        
        return None
    
    def detect_news_impact(
        self,
        symbol: str,
        news_importance: float,
        news_sentiment: float,
        news_title: str,
        current_timeframe: Timeframe = Timeframe.TACTICAL,
    ) -> Optional[EscalationTrigger]:
        """
        检测新闻冲击
        
        Args:
            symbol: 股票代码
            news_importance: 新闻重要性（0-10）
            news_sentiment: 新闻情绪（-1到1）
            news_title: 新闻标题
            current_timeframe: 当前时间尺度
        
        Returns:
            EscalationTrigger if significant news detected, None otherwise
        """
        # 极端重要新闻：直达战略层
        if news_importance >= self.thresholds['news_impact_critical']:
            return EscalationTrigger(
                trigger_type=EscalationTriggerType.BLACK_SWAN,
                from_timeframe=current_timeframe,
                to_timeframe=Timeframe.STRATEGIC,
                score=10.0,
                symbol=symbol,
                timestamp=datetime.now(),
                reason=f"Critical news impact: {news_title}",
                details={
                    'news_importance': news_importance,
                    'news_sentiment': news_sentiment,
                    'news_title': news_title,
                }
            )
        
        # 重要新闻：升至战役层
        if news_importance >= self.thresholds['news_impact_high']:
            # 考虑情绪极端程度
            sentiment_multiplier = abs(news_sentiment)
            score = min(10.0, news_importance * (0.7 + 0.3 * sentiment_multiplier))
            
            return EscalationTrigger(
                trigger_type=EscalationTriggerType.NEWS_IMPACT,
                from_timeframe=current_timeframe,
                to_timeframe=Timeframe.CAMPAIGN,
                score=score,
                symbol=symbol,
                timestamp=datetime.now(),
                reason=f"High impact news: {news_title}",
                details={
                    'news_importance': news_importance,
                    'news_sentiment': news_sentiment,
                    'news_title': news_title,
                }
            )
        
        return None
    
    def detect_technical_breakout(
        self,
        symbol: str,
        breakout_type: str,
        confidence: float,
        price: float,
        key_level: float,
        current_timeframe: Timeframe = Timeframe.TACTICAL,
    ) -> Optional[EscalationTrigger]:
        """
        检测技术突破
        
        Args:
            symbol: 股票代码
            breakout_type: 突破类型（'resistance', 'support', 'trendline'）
            confidence: 突破置信度（0-1）
            price: 当前价格
            key_level: 关键位价格
            current_timeframe: 当前时间尺度
        
        Returns:
            EscalationTrigger if breakout detected, None otherwise
        """
        if confidence >= self.thresholds['technical_breakout_confidence']:
            score = min(10.0, 6.0 + confidence * 4.0)
            
            return EscalationTrigger(
                trigger_type=EscalationTriggerType.TECHNICAL_BREAKOUT,
                from_timeframe=current_timeframe,
                to_timeframe=Timeframe.CAMPAIGN,
                score=score,
                symbol=symbol,
                timestamp=datetime.now(),
                reason=f"Technical breakout: {breakout_type} at {key_level:.2f}",
                details={
                    'breakout_type': breakout_type,
                    'confidence': confidence,
                    'price': price,
                    'key_level': key_level,
                }
            )
        
        return None
    
    def detect_strategic_conflict(
        self,
        symbol: str,
        tactical_action: str,
        tactical_conviction: float,
        strategic_constraint: str,
        conflict_reason: str,
        current_timeframe: Timeframe = Timeframe.TACTICAL,
    ) -> Optional[EscalationTrigger]:
        """
        检测战略冲突
        
        Args:
            symbol: 股票代码
            tactical_action: 战术层建议动作
            tactical_conviction: 战术层信心度
            strategic_constraint: 战略层约束
            conflict_reason: 冲突原因
            current_timeframe: 当前时间尺度
        
        Returns:
            EscalationTrigger if conflict detected, None otherwise
        """
        if tactical_conviction >= self.thresholds['strategic_conflict_conviction']:
            score = min(10.0, 5.0 + tactical_conviction * 0.6)
            
            return EscalationTrigger(
                trigger_type=EscalationTriggerType.STRATEGIC_CONFLICT,
                from_timeframe=current_timeframe,
                to_timeframe=Timeframe.CAMPAIGN,  # 先升至战役层协调
                score=score,
                symbol=symbol,
                timestamp=datetime.now(),
                reason=f"Strategic conflict: {conflict_reason}",
                details={
                    'tactical_action': tactical_action,
                    'tactical_conviction': tactical_conviction,
                    'strategic_constraint': strategic_constraint,
                    'conflict_reason': conflict_reason,
                }
            )
        
        return None
    
    def detect_volatility_spike(
        self,
        symbol: str,
        current_volatility: float,
        historical_volatility: float,
        current_timeframe: Timeframe = Timeframe.TACTICAL,
    ) -> Optional[EscalationTrigger]:
        """
        检测波动率飙升
        
        Args:
            symbol: 股票代码
            current_volatility: 当前波动率
            historical_volatility: 历史平均波动率
            current_timeframe: 当前时间尺度
        
        Returns:
            EscalationTrigger if spike detected, None otherwise
        """
        if historical_volatility > 0:
            multiplier = current_volatility / historical_volatility
            
            if multiplier >= self.thresholds['volatility_spike_multiplier']:
                score = min(10.0, 5.0 + (multiplier / self.thresholds['volatility_spike_multiplier']) * 3.0)
                
                return EscalationTrigger(
                    trigger_type=EscalationTriggerType.VOLATILITY_SPIKE,
                    from_timeframe=current_timeframe,
                    to_timeframe=Timeframe.CAMPAIGN,
                    score=score,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    reason=f"Volatility spike: {multiplier:.1f}x historical average",
                    details={
                        'current_volatility': current_volatility,
                        'historical_volatility': historical_volatility,
                        'multiplier': multiplier,
                    }
                )
        
        return None
    
    def detect_all(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        news_data: Optional[Dict[str, Any]] = None,
        technical_data: Optional[Dict[str, Any]] = None,
        strategic_data: Optional[Dict[str, Any]] = None,
        current_timeframe: Timeframe = Timeframe.TACTICAL,
    ) -> List[EscalationTrigger]:
        """
        综合检测所有可能的反向传导触发条件
        
        Args:
            symbol: 股票代码
            market_data: 市场数据
            news_data: 新闻数据（可选）
            technical_data: 技术数据（可选）
            strategic_data: 战略数据（可选）
            current_timeframe: 当前时间尺度
        
        Returns:
            List of EscalationTriggers (sorted by score descending)
        """
        triggers = []
        
        # 检测市场冲击
        if 'price_change_1d' in market_data:
            trigger = self.detect_market_shock(
                symbol=symbol,
                price_change_1d=market_data['price_change_1d'],
                price_change_3d=market_data.get('price_change_3d'),
                current_timeframe=current_timeframe,
            )
            if trigger:
                triggers.append(trigger)
        
        # 检测新闻冲击
        if news_data:
            trigger = self.detect_news_impact(
                symbol=symbol,
                news_importance=news_data.get('importance', 0),
                news_sentiment=news_data.get('sentiment', 0),
                news_title=news_data.get('title', 'Unknown'),
                current_timeframe=current_timeframe,
            )
            if trigger:
                triggers.append(trigger)
        
        # 检测技术突破
        if technical_data and technical_data.get('breakout_detected'):
            trigger = self.detect_technical_breakout(
                symbol=symbol,
                breakout_type=technical_data['breakout_type'],
                confidence=technical_data['confidence'],
                price=technical_data['price'],
                key_level=technical_data['key_level'],
                current_timeframe=current_timeframe,
            )
            if trigger:
                triggers.append(trigger)
        
        # 检测波动率飙升
        if market_data.get('current_volatility') and market_data.get('historical_volatility'):
            trigger = self.detect_volatility_spike(
                symbol=symbol,
                current_volatility=market_data['current_volatility'],
                historical_volatility=market_data['historical_volatility'],
                current_timeframe=current_timeframe,
            )
            if trigger:
                triggers.append(trigger)
        
        # 检测战略冲突
        if strategic_data and strategic_data.get('conflict_detected'):
            trigger = self.detect_strategic_conflict(
                symbol=symbol,
                tactical_action=strategic_data['tactical_action'],
                tactical_conviction=strategic_data['tactical_conviction'],
                strategic_constraint=strategic_data['strategic_constraint'],
                conflict_reason=strategic_data['conflict_reason'],
                current_timeframe=current_timeframe,
            )
            if trigger:
                triggers.append(trigger)
        
        # 按评分降序排序
        triggers.sort(key=lambda t: t.score, reverse=True)
        
        return triggers


def should_trigger_escalation(
    triggers: List[EscalationTrigger],
    threshold: float = 7.0,
) -> Optional[EscalationTrigger]:
    """
    判断是否应该触发反向传导
    
    Args:
        triggers: 检测到的触发器列表
        threshold: 触发阈值
    
    Returns:
        The highest scoring trigger if threshold is met, None otherwise
    """
    if not triggers:
        return None
    
    # 返回评分最高的触发器（已排序）
    top_trigger = triggers[0]
    if top_trigger.should_escalate(threshold):
        return top_trigger
    
    return None
