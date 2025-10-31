"""
Decision Makers - 决策制定者层

这一层组合不同的Agent（Macro, Sector, Meta）来制定不同层级的决策。

三个DecisionMaker：
1. StrategicDecisionMaker - 战略层（90天）
   - MacroAgent + SectorAgent + MetaAgent
   - 生成长期约束和策略
   
2. CampaignDecisionMaker - 战役层（7天）
   - MacroAgent(轻量) + SectorAgent + MetaAgent
   - 中期调整和战术规划
   
3. TacticalDecisionMaker - 战术层（每天）
   - MetaAgent only
   - 快速决策，使用上层约束
"""

from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from Agents.core import MacroAgent, MacroContext
from Agents.core import SectorAgent, SectorContext
from Agents.orchestration.meta_agent import MetaAgent, MetaDecision


@dataclass
class Decision:
    """
    统一的决策结果
    """
    symbol: str
    action: str  # BUY, SELL, HOLD
    conviction: float  # 1-10
    reasoning: str
    
    # 上下文
    macro_context: Optional[MacroContext] = None
    sector_context: Optional[SectorContext] = None
    
    # 约束
    constraints: Optional[Dict[str, Any]] = None
    
    # 元数据
    timestamp: datetime = None
    decision_level: str = None  # strategic, campaign, tactical
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class StrategicDecisionMaker:
    """
    战略决策制定者
    
    使用所有Agent进行全面分析：
    - MacroAgent: 宏观环境
    - SectorAgent: 行业分析
    - MetaAgent: 个股决策
    
    输出：
    - 交易决策
    - 约束条件（供下层使用）
    """
    
    def __init__(
        self,
        macro_agent: MacroAgent,
        sector_agent: SectorAgent,
        meta_agent: MetaAgent
    ):
        """
        初始化战略决策制定者
        
        Args:
            macro_agent: 宏观分析Agent
            sector_agent: 行业分析Agent
            meta_agent: 个股分析Agent（Meta Agent）
        """
        self.macro_agent = macro_agent
        self.sector_agent = sector_agent
        self.meta_agent = meta_agent
    
    async def decide(
        self,
        symbol: str,
        visible_data_end: Optional[datetime] = None
    ) -> Decision:
        """
        制定战略决策
        
        流程：
        1. 宏观分析（获取市场regime和约束）
        2. 行业分析（获取行业趋势）
        3. 个股分析（综合决策）
        
        Args:
            symbol: 股票代码
            visible_data_end: 回测模式的时间截止点（也是决策时间）
            
        Returns:
            Decision对象
        """
        # 使用visible_data_end作为决策时间（回测模式）或当前时间（实盘模式）
        decision_time = visible_data_end if visible_data_end is not None else datetime.now()
        
        # 1. 宏观分析
        macro_context = await self.macro_agent.analyze_macro_environment(
            visible_data_end=visible_data_end
        )
        
        # 2. 确定股票所属行业
        sector = self.sector_agent.get_sector_for_symbol(symbol)
        
        # 3. 行业分析
        sector_context = await self.sector_agent.analyze_sector(
            sector=sector,
            visible_data_end=visible_data_end
        )
        
        # 4. 个股决策（综合宏观和行业）
        meta_decision = await self.meta_agent.analyze_and_decide(
            symbol=symbol,
            macro_context=macro_context.to_dict(),
            sector_context=sector_context.to_dict(),
            constraints=macro_context.constraints,
            current_time=decision_time
        )
        
        # 5. 构建Decision
        decision = Decision(
            symbol=symbol,
            action=meta_decision.action,
            conviction=float(meta_decision.conviction),
            reasoning=meta_decision.reasoning,
            macro_context=macro_context,
            sector_context=sector_context,
            constraints=macro_context.constraints,
            timestamp=decision_time,
            decision_level='strategic'
        )
        
        return decision


class CampaignDecisionMaker:
    """
    战役决策制定者
    
    中期决策（7天周期）：
    - MacroAgent: 轻量级宏观检查
    - SectorAgent: 行业趋势更新
    - MetaAgent: 个股决策
    
    特点：
    - 复用宏观分析（从cache）
    - 重新分析行业
    - 应用上层约束
    """
    
    def __init__(
        self,
        macro_agent: MacroAgent,
        sector_agent: SectorAgent,
        meta_agent: MetaAgent
    ):
        self.macro_agent = macro_agent
        self.sector_agent = sector_agent
        self.meta_agent = meta_agent
    
    async def decide(
        self,
        symbol: str,
        visible_data_end: Optional[datetime] = None,
        inherited_constraints: Optional[Dict[str, Any]] = None
    ) -> Decision:
        """
        制定战役决策
        
        Args:
            symbol: 股票代码
            visible_data_end: 回测模式的时间截止点（也是决策时间）
            inherited_constraints: 从Strategic层继承的约束
            
        Returns:
            Decision对象
        """
        # 使用visible_data_end作为决策时间（回测模式）或当前时间（实盘模式）
        decision_time = visible_data_end if visible_data_end is not None else datetime.now()
        
        # 1. 宏观分析（可能从cache获取）
        macro_context = await self.macro_agent.analyze_macro_environment(
            visible_data_end=visible_data_end
        )
        
        # 2. 行业分析（重新分析）
        sector = self.sector_agent.get_sector_for_symbol(symbol)
        sector_context = await self.sector_agent.analyze_sector(
            sector=sector,
            visible_data_end=visible_data_end
        )
        
        # 3. 合并约束（继承的 + 当前宏观的）
        constraints = inherited_constraints or {}
        constraints.update(macro_context.constraints)
        
        # 4. 个股决策
        meta_decision = await self.meta_agent.analyze_and_decide(
            symbol=symbol,
            macro_context=macro_context.to_dict(),
            sector_context=sector_context.to_dict(),
            constraints=constraints,
            current_time=decision_time
        )
        
        # 5. 构建Decision
        decision = Decision(
            symbol=symbol,
            action=meta_decision.action,
            conviction=float(meta_decision.conviction),
            reasoning=meta_decision.reasoning,
            macro_context=macro_context,
            sector_context=sector_context,
            constraints=constraints,
            timestamp=decision_time,
            decision_level='campaign'
        )
        
        return decision


class TacticalDecisionMaker:
    """
    战术决策制定者
    
    日常决策（每天）：
    - MetaAgent only
    - 快速决策
    - 严格遵守上层约束
    
    特点：
    - 不调用Macro/Sector（节省时间）
    - 使用inherited约束
    - 支持fast模式
    """
    
    def __init__(
        self,
        meta_agent: MetaAgent
    ):
        self.meta_agent = meta_agent
    
    async def decide(
        self,
        symbol: str,
        inherited_constraints: Optional[Dict[str, Any]] = None,
        inherited_macro_context: Optional[Dict[str, Any]] = None,
        inherited_sector_context: Optional[Dict[str, Any]] = None,
        current_time: Optional[datetime] = None
    ) -> Decision:
        """
        制定战术决策
        
        Args:
            symbol: 股票代码
            inherited_constraints: 从上层继承的约束
            inherited_macro_context: 从上层继承的宏观背景
            inherited_sector_context: 从上层继承的行业背景
            current_time: 当前时间（回测模式下使用模拟日期，实盘模式下为None则使用当前时间）
            
        Returns:
            Decision对象
        """
        # 使用提供的时间或当前时间
        decision_time = current_time if current_time is not None else datetime.now()
        
        # 快速决策：直接使用继承的上下文
        meta_decision = await self.meta_agent.analyze_and_decide(
            symbol=symbol,
            macro_context=inherited_macro_context,
            sector_context=inherited_sector_context,
            constraints=inherited_constraints,
            current_time=decision_time
        )
        
        # 构建Decision
        decision = Decision(
            symbol=symbol,
            action=meta_decision.action,
            conviction=float(meta_decision.conviction),
            reasoning=meta_decision.reasoning,
            macro_context=None,  # 不重新分析宏观
            sector_context=None,  # 不重新分析行业
            constraints=inherited_constraints,
            timestamp=decision_time,
            decision_level='tactical'
        )
        
        return decision


class DecisionMakerFactory:
    """
    DecisionMaker工厂类
    
    简化创建过程，提供便捷方法
    """
    
    def __init__(
        self,
        macro_agent: Optional[MacroAgent] = None,
        sector_agent: Optional[SectorAgent] = None,
        meta_agent: Optional[MetaAgent] = None
    ):
        """
        初始化工厂
        
        Args:
            macro_agent: 可选的MacroAgent实例
            sector_agent: 可选的SectorAgent实例
            meta_agent: 可选的MetaAgent实例
        """
        # 如果没有提供，创建默认实例
        self.macro_agent = macro_agent or MacroAgent()
        self.sector_agent = sector_agent or SectorAgent()
        self.meta_agent = meta_agent or MetaAgent()
    
    def create_strategic(self) -> StrategicDecisionMaker:
        """创建战略决策制定者"""
        return StrategicDecisionMaker(
            macro_agent=self.macro_agent,
            sector_agent=self.sector_agent,
            meta_agent=self.meta_agent
        )
    
    def create_campaign(self) -> CampaignDecisionMaker:
        """创建战役决策制定者"""
        return CampaignDecisionMaker(
            macro_agent=self.macro_agent,
            sector_agent=self.sector_agent,
            meta_agent=self.meta_agent
        )
    
    def create_tactical(self) -> TacticalDecisionMaker:
        """创建战术决策制定者"""
        return TacticalDecisionMaker(
            meta_agent=self.meta_agent
        )
