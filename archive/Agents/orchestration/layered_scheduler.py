"""
Layered Scheduler - 分层调度器

负责协调三层决策制定者（Strategic, Campaign, Tactical）的运行时机和约束传播。

三层决策周期：
- Strategic: 30天（或触发escalation）
- Campaign: 7天（或触发escalation）
- Tactical: 每天

Escalation触发条件：
- 突破关键技术位
- 宏观regime变化
- 行业轮动信号
- 重大新闻事件
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from Agents.orchestration.decision_makers import (
    StrategicDecisionMaker,
    CampaignDecisionMaker,
    TacticalDecisionMaker,
    Decision
)


class DecisionLevel(Enum):
    """决策层级"""
    STRATEGIC = "strategic"  # 30天
    CAMPAIGN = "campaign"    # 7天
    TACTICAL = "tactical"    # 每天


class EscalationReason(Enum):
    """Escalation触发原因"""
    SCHEDULED = "scheduled"  # 定期调度
    REGIME_CHANGE = "regime_change"  # 宏观regime变化
    SECTOR_ROTATION = "sector_rotation"  # 行业轮动
    TECHNICAL_BREAKOUT = "technical_breakout"  # 技术突破
    VOLATILITY_SPIKE = "volatility_spike"  # 波动率飙升
    NEWS_EVENT = "news_event"  # 重大新闻


@dataclass
class SchedulerState:
    """
    调度器状态
    
    记录各层级最后运行时间和结果
    """
    # 最后运行时间
    last_strategic_time: Optional[datetime] = None
    last_campaign_time: Optional[datetime] = None
    last_tactical_time: Optional[datetime] = None
    
    # 最后决策结果（缓存）
    last_strategic_decision: Optional[Decision] = None
    last_campaign_decision: Optional[Decision] = None
    last_tactical_decision: Optional[Decision] = None
    
    # Escalation计数
    strategic_escalations: int = 0
    campaign_escalations: int = 0
    
    # 当前有效约束
    active_constraints: Dict[str, Any] = field(default_factory=dict)
    active_macro_context: Optional[Dict[str, Any]] = None
    active_sector_context: Optional[Dict[str, Any]] = None


class LayeredScheduler:
    """
    分层调度器
    
    职责：
    1. 决定何时运行哪个 DecisionMaker
    2. 处理 escalation 触发
    3. 传播约束和上下文
    4. 缓存决策结果
    """
    
    def __init__(
        self,
        strategic_maker: StrategicDecisionMaker,
        campaign_maker: CampaignDecisionMaker,
        tactical_maker: TacticalDecisionMaker,
        strategic_interval_days: int = 30,
        campaign_interval_days: int = 7,
        tactical_interval_days: int = 1
    ):
        """
        初始化调度器
        
        Args:
            strategic_maker: 战略决策制定者
            campaign_maker: 战役决策制定者
            tactical_maker: 战术决策制定者
            strategic_interval_days: 战略层运行间隔（天）
            campaign_interval_days: 战役层运行间隔（天）
            tactical_interval_days: 战术层运行间隔（天）
        """
        self.strategic_maker = strategic_maker
        self.campaign_maker = campaign_maker
        self.tactical_maker = tactical_maker
        
        self.strategic_interval = timedelta(days=strategic_interval_days)
        self.campaign_interval = timedelta(days=campaign_interval_days)
        self.tactical_interval = timedelta(days=tactical_interval_days)
        
        self.state = SchedulerState()
    
    def should_run_strategic(self, current_time: datetime) -> bool:
        """
        判断是否应该运行战略层决策
        
        Args:
            current_time: 当前时间
            
        Returns:
            是否应该运行
        """
        if self.state.last_strategic_time is None:
            return True  # 首次运行
        
        elapsed = current_time - self.state.last_strategic_time
        return elapsed >= self.strategic_interval
    
    def should_run_campaign(self, current_time: datetime) -> bool:
        """判断是否应该运行战役层决策"""
        if self.state.last_campaign_time is None:
            return True
        
        elapsed = current_time - self.state.last_campaign_time
        return elapsed >= self.campaign_interval
    
    def should_run_tactical(self, current_time: datetime) -> bool:
        """判断是否应该运行战术层决策"""
        if self.state.last_tactical_time is None:
            return True
        
        elapsed = current_time - self.state.last_tactical_time
        return elapsed >= self.tactical_interval
    
    def check_escalation_triggers(
        self,
        symbol: str,
        current_level: DecisionLevel,
        signals: Optional[Dict[str, Any]] = None
    ) -> Optional[EscalationReason]:
        """
        检查是否需要 escalation（向上升级）
        
        Args:
            symbol: 股票代码
            current_level: 当前决策层级
            signals: 各种信号指标
            
        Returns:
            Escalation原因，如果不需要escalation则返回None
        """
        if signals is None:
            return None
        
        # 检查regime变化
        if signals.get('regime_changed', False):
            return EscalationReason.REGIME_CHANGE
        
        # 检查行业轮动
        if signals.get('sector_rotation_signal') in ['rotating_in', 'rotating_out']:
            return EscalationReason.SECTOR_ROTATION
        
        # 检查技术突破
        if signals.get('technical_breakout', False):
            return EscalationReason.TECHNICAL_BREAKOUT
        
        # 检查波动率
        vix_level = signals.get('vix_level', 0)
        if vix_level > 30:  # VIX > 30 表示极端恐慌
            return EscalationReason.VOLATILITY_SPIKE
        
        # 检查新闻事件
        if signals.get('major_news_event', False):
            return EscalationReason.NEWS_EVENT
        
        return None
    
    async def decide(
        self,
        symbol: str,
        current_time: datetime,
        visible_data_end: Optional[datetime] = None,
        escalation_signals: Optional[Dict[str, Any]] = None
    ) -> Decision:
        """
        制定决策（自动选择合适的层级）
        
        流程：
        1. 检查是否需要 escalation
        2. 确定应该运行的层级
        3. 从高到低依次运行必要的层级
        4. 传播约束和上下文
        5. 返回最终决策
        
        Args:
            symbol: 股票代码
            current_time: 当前时间
            visible_data_end: 回测模式的数据截止时间
            escalation_signals: Escalation触发信号
            
        Returns:
            Decision对象
        """
        # 1. 确定需要运行的层级
        run_strategic = self.should_run_strategic(current_time)
        run_campaign = self.should_run_campaign(current_time)
        run_tactical = self.should_run_tactical(current_time)
        
        # 2. 检查 escalation
        escalation_reason = self.check_escalation_triggers(
            symbol=symbol,
            current_level=DecisionLevel.TACTICAL,
            signals=escalation_signals
        )
        
        if escalation_reason:
            # Escalation: 强制运行上层决策
            if escalation_reason in [
                EscalationReason.REGIME_CHANGE,
                EscalationReason.VOLATILITY_SPIKE
            ]:
                # 严重情况：运行战略层
                run_strategic = True
                run_campaign = True
                self.state.strategic_escalations += 1
            else:
                # 中等情况：运行战役层
                run_campaign = True
                self.state.campaign_escalations += 1
        
        # 3. 按层级运行决策
        if run_strategic:
            # 运行战略层（完整分析）
            decision = await self.strategic_maker.decide(
                symbol=symbol,
                visible_data_end=visible_data_end
            )
            
            # 更新状态
            self.state.last_strategic_time = current_time
            self.state.last_strategic_decision = decision
            self.state.active_constraints = decision.constraints or {}
            
            if decision.macro_context:
                self.state.active_macro_context = decision.macro_context.to_dict()
            if decision.sector_context:
                self.state.active_sector_context = decision.sector_context.to_dict()
            
            return decision
        
        elif run_campaign:
            # 运行战役层（继承战略层约束）
            decision = await self.campaign_maker.decide(
                symbol=symbol,
                visible_data_end=visible_data_end,
                inherited_constraints=self.state.active_constraints
            )
            
            # 更新状态
            self.state.last_campaign_time = current_time
            self.state.last_campaign_decision = decision
            
            # 更新约束（合并）
            if decision.constraints:
                self.state.active_constraints.update(decision.constraints)
            
            # 更新上下文
            if decision.macro_context:
                self.state.active_macro_context = decision.macro_context.to_dict()
            if decision.sector_context:
                self.state.active_sector_context = decision.sector_context.to_dict()
            
            return decision
        
        else:
            # 运行战术层（继承所有上下文）
            decision = await self.tactical_maker.decide(
                symbol=symbol,
                inherited_constraints=self.state.active_constraints,
                inherited_macro_context=self.state.active_macro_context,
                inherited_sector_context=self.state.active_sector_context
            )
            
            # 更新状态
            self.state.last_tactical_time = current_time
            self.state.last_tactical_decision = decision
            
            return decision
    
    def get_next_schedule(self, current_time: datetime) -> Dict[str, datetime]:
        """
        获取下次各层级调度时间
        
        Args:
            current_time: 当前时间
            
        Returns:
            {'strategic': datetime, 'campaign': datetime, 'tactical': datetime}
        """
        result = {}
        
        if self.state.last_strategic_time:
            result['strategic'] = self.state.last_strategic_time + self.strategic_interval
        else:
            result['strategic'] = current_time
        
        if self.state.last_campaign_time:
            result['campaign'] = self.state.last_campaign_time + self.campaign_interval
        else:
            result['campaign'] = current_time
        
        if self.state.last_tactical_time:
            result['tactical'] = self.state.last_tactical_time + self.tactical_interval
        else:
            result['tactical'] = current_time
        
        return result
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        获取调度器状态摘要
        
        Returns:
            状态摘要字典
        """
        return {
            'last_strategic_time': self.state.last_strategic_time.isoformat() if self.state.last_strategic_time else None,
            'last_campaign_time': self.state.last_campaign_time.isoformat() if self.state.last_campaign_time else None,
            'last_tactical_time': self.state.last_tactical_time.isoformat() if self.state.last_tactical_time else None,
            'strategic_escalations': self.state.strategic_escalations,
            'campaign_escalations': self.state.campaign_escalations,
            'active_constraints': self.state.active_constraints,
            'has_macro_context': self.state.active_macro_context is not None,
            'has_sector_context': self.state.active_sector_context is not None,
        }
    
    def reset(self):
        """重置调度器状态（用于新的回测）"""
        self.state = SchedulerState()


class MultiSymbolScheduler:
    """
    多股票调度器
    
    为多个股票管理独立的 LayeredScheduler
    """
    
    def __init__(
        self,
        strategic_maker: StrategicDecisionMaker,
        campaign_maker: CampaignDecisionMaker,
        tactical_maker: TacticalDecisionMaker,
        strategic_interval_days: int = 30,
        campaign_interval_days: int = 7,
        tactical_interval_days: int = 1
    ):
        """
        初始化多股票调度器
        
        Args:
            strategic_maker: 战略决策制定者（共享）
            campaign_maker: 战役决策制定者（共享）
            tactical_maker: 战术决策制定者（共享）
            strategic_interval_days: 战略层运行间隔
            campaign_interval_days: 战役层运行间隔
            tactical_interval_days: 战术层运行间隔
        """
        self.strategic_maker = strategic_maker
        self.campaign_maker = campaign_maker
        self.tactical_maker = tactical_maker
        
        self.strategic_interval_days = strategic_interval_days
        self.campaign_interval_days = campaign_interval_days
        self.tactical_interval_days = tactical_interval_days
        
        # 每个股票一个调度器
        self.schedulers: Dict[str, LayeredScheduler] = {}
    
    def get_scheduler(self, symbol: str) -> LayeredScheduler:
        """
        获取或创建股票的调度器
        
        Args:
            symbol: 股票代码
            
        Returns:
            LayeredScheduler实例
        """
        if symbol not in self.schedulers:
            self.schedulers[symbol] = LayeredScheduler(
                strategic_maker=self.strategic_maker,
                campaign_maker=self.campaign_maker,
                tactical_maker=self.tactical_maker,
                strategic_interval_days=self.strategic_interval_days,
                campaign_interval_days=self.campaign_interval_days,
                tactical_interval_days=self.tactical_interval_days
            )
        
        return self.schedulers[symbol]
    
    async def decide(
        self,
        symbol: str,
        current_time: datetime,
        visible_data_end: Optional[datetime] = None,
        escalation_signals: Optional[Dict[str, Any]] = None
    ) -> Decision:
        """
        为指定股票制定决策
        
        Args:
            symbol: 股票代码
            current_time: 当前时间
            visible_data_end: 回测模式的数据截止时间
            escalation_signals: Escalation触发信号
            
        Returns:
            Decision对象
        """
        scheduler = self.get_scheduler(symbol)
        return await scheduler.decide(
            symbol=symbol,
            current_time=current_time,
            visible_data_end=visible_data_end,
            escalation_signals=escalation_signals
        )
    
    async def decide_batch(
        self,
        symbols: List[str],
        current_time: datetime,
        visible_data_end: Optional[datetime] = None,
        escalation_signals: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Decision]:
        """
        批量决策（多个股票）
        
        Args:
            symbols: 股票代码列表
            current_time: 当前时间
            visible_data_end: 回测模式的数据截止时间
            escalation_signals: 每个股票的escalation信号 {symbol: signals}
            
        Returns:
            {symbol: Decision}
        """
        results = {}
        
        for symbol in symbols:
            signals = escalation_signals.get(symbol) if escalation_signals else None
            decision = await self.decide(
                symbol=symbol,
                current_time=current_time,
                visible_data_end=visible_data_end,
                escalation_signals=signals
            )
            results[symbol] = decision
        
        return results
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有股票的调度器状态
        
        Returns:
            {symbol: state_summary}
        """
        return {
            symbol: scheduler.get_state_summary()
            for symbol, scheduler in self.schedulers.items()
        }
    
    def reset(self, symbol: Optional[str] = None):
        """
        重置调度器状态
        
        Args:
            symbol: 如果指定，只重置该股票；否则重置所有
        """
        if symbol:
            if symbol in self.schedulers:
                self.schedulers[symbol].reset()
        else:
            for scheduler in self.schedulers.values():
                scheduler.reset()
