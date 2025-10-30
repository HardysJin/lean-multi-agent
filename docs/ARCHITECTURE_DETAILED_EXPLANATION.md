# 方案B架构详解：Agent分层与Timeframe调度

## 核心问题解答

### Q1: Timeframe尺度如何产生？

**答案**: Timeframe不是由Agent产生，而是由**Scheduler（调度器）**根据时间决定。

```python
# ❌ 错误理解：Agent决定timeframe
decision = await agent.decide()  # agent自己决定是tactical还是strategic

# ✅ 正确理解：Scheduler决定timeframe
scheduler = LayeredScheduler()
if scheduler.should_run_strategic():
    decision = await strategic_maker.decide()  # 明确是strategic
elif scheduler.should_run_campaign():
    decision = await campaign_maker.decide()   # 明确是campaign
else:
    decision = await tactical_maker.decide()   # 明确是tactical
```

### Q2: 多Agent会不会让系统变慢？

**答案**: 不会！反而更快！

**原因**：
1. **Agent是逻辑分离，不是物理分离** - MacroAgent和MetaAgent可以共存于同一进程
2. **宏观分析复用** - 1次宏观分析可供10只股票使用
3. **按需调用** - Tactical层不调用MacroAgent，只有Strategic层才调用

**性能对比**：
```
方案A（Timeframe绑定）:
- 每只股票都要获取宏观新闻
- 10只股票 = 10次宏观分析

方案B（MacroAgent分离）:
- 宏观分析1次，结果缓存
- 10只股票 = 1次宏观分析 + 10次个股分析
- 速度提升: ~9倍（宏观部分）
```

### Q3: LangChain Tool Calling如何理解Agent区别？

**答案**: 通过**工具描述（tool description）**和**上下文（context）**

```python
# MetaAgent的工具
tools = [
    {
        "name": "technical_analysis",
        "description": "Analyze technical indicators for a specific stock symbol",
        "parameters": {"symbol": "string", "period": "string"}
    },
    {
        "name": "stock_news",
        "description": "Get recent news for a specific stock symbol",
        "parameters": {"symbol": "string", "days": "integer"}
    }
]

# MacroAgent的工具
tools = [
    {
        "name": "get_fed_policy",
        "description": "Get current Federal Reserve policy and interest rate decisions",
        "parameters": {}  # 注意：不需要symbol
    },
    {
        "name": "get_economic_indicators",
        "description": "Get macroeconomic indicators like GDP, unemployment, inflation",
        "parameters": {"indicators": "list"}
    }
]

# LLM会根据任务选择合适的工具
# "分析苹果公司" → 调用technical_analysis(symbol="AAPL")
# "判断市场regime" → 调用get_fed_policy()
```

---

## 完整架构图

```
┌───────────────────────────────────────────────────────────────────────┐
│                    BacktestEngine / LiveTradingEngine                 │
│  职责：推进时间、管理回测循环                                           │
└─────────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ↓
┌───────────────────────────────────────────────────────────────────────┐
│                      LayeredDecisionScheduler                          │
│  职责：根据时间决定调用哪个层级                                         │
│                                                                         │
│  • last_strategic_time: 2025-01-01                                     │
│  • last_campaign_time: 2025-10-20                                      │
│  • current_time: 2025-10-30                                            │
│                                                                         │
│  决策逻辑：                                                             │
│  if (current_time - last_strategic_time) >= 90天:                     │
│      → 调用 StrategicDecisionMaker                                     │
│  elif (current_time - last_campaign_time) >= 7天:                     │
│      → 调用 CampaignDecisionMaker                                      │
│  else:                                                                  │
│      → 调用 TacticalDecisionMaker                                      │
└─────────────────────────────────┬─────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ↓             ↓             ↓
    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
    │ Strategic        │ │ Campaign         │ │ Tactical         │
    │ DecisionMaker    │ │ DecisionMaker    │ │ DecisionMaker    │
    │                  │ │                  │ │                  │
    │ 频率: 90天       │ │ 频率: 7天        │ │ 频率: 每天        │
    │ 计算: Full LLM   │ │ 计算: Hybrid     │ │ 计算: Fast/Full  │
    └──────────────────┘ └──────────────────┘ └──────────────────┘
            │                     │                     │
            │                     │                     │
    ┌───────┴─────────┐   ┌───────┴─────────┐   ┌─────┴───────────┐
    │                 │   │                 │   │                  │
    ↓                 ↓   ↓                 ↓   ↓                  │
┌─────────┐    ┌──────────┐ ┌──────────┐ ┌──────────┐      ┌──────────┐
│ Macro   │    │  Meta    │ │ Sector   │ │  Meta    │      │  Meta    │
│ Agent   │    │  Agent   │ │ Agent    │ │  Agent   │      │  Agent   │
│         │    │          │ │          │ │          │      │          │
│ 宏观分析 │    │ 个股分析  │ │ 行业分析  │ │ 个股分析  │      │ 个股分析  │
└─────────┘    └──────────┘ └──────────┘ └──────────┘      └──────────┘
     │              │             │             │                 │
     │              │             │             │                 │
     ↓              ↓             ↓             ↓                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        Specialist Agents (MCP Servers)              │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ Technical    │  │ News         │  │ Economic     │             │
│  │ Agent        │  │ Agent        │  │ Data Agent   │             │
│  │              │  │              │  │              │             │
│  │ • RSI        │  │ • 个股新闻    │  │ • GDP        │             │
│  │ • MACD       │  │ • 财报       │  │ • 失业率      │             │
│  │ • 布林带     │  │ • 分析师     │  │ • 通胀率      │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ Fed Policy   │  │ Geopolitical │  │ Sector       │             │
│  │ Agent        │  │ Agent        │  │ Rotation     │             │
│  │              │  │              │  │ Agent        │             │
│  │ • 利率决策   │  │ • 战争       │  │ • 行业轮动    │             │
│  │ • 货币政策   │  │ • 贸易       │  │ • 相对强度    │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 详细流程图

### 流程1: 正常调度流程

```
Day 1 (2025-10-01)
    ↓
LayeredScheduler.run_daily()
    ↓
检查: 距离上次Strategic = 0天 < 90天? YES
    ↓
检查: 距离上次Campaign = 0天 < 7天? YES
    ↓
运行: TacticalDecisionMaker ✓
    ↓
TacticalDecisionMaker.decide("AAPL")
    ↓
MetaAgent.analyze_and_decide(symbol="AAPL")
    ├─→ TechnicalAgent.analyze("AAPL")    [个股技术分析]
    ├─→ NewsAgent.get_stock_news("AAPL")  [个股新闻]
    └─→ LLM.decide(technical + news)      [个股决策]
    ↓
返回: DecisionRecord(timeframe=TACTICAL, action=BUY, ...)

═════════════════════════════════════════════════════════

Day 8 (2025-10-08)
    ↓
LayeredScheduler.run_daily()
    ↓
检查: 距离上次Strategic = 7天 < 90天? YES
    ↓
检查: 距离上次Campaign = 7天 >= 7天? YES  ✓
    ↓
运行: CampaignDecisionMaker ✓
    ↓
CampaignDecisionMaker.decide("AAPL")
    ├─→ MacroAgent.analyze_macro()  [1次，结果缓存]
    │   ├─→ FedPolicyAgent.get_policy()
    │   ├─→ EconomicDataAgent.get_indicators()
    │   └─→ LLM.analyze_macro_regime()
    │
    ├─→ SectorAgent.analyze_sector("Technology")  [行业分析]
    │   ├─→ SectorRotationAgent.get_rotation_signal()
    │   └─→ LLM.analyze_sector()
    │
    └─→ MetaAgent.analyze_and_decide(
            symbol="AAPL",
            macro_context=macro_result,      ← 传入宏观背景
            sector_context=sector_result      ← 传入行业背景
        )
        ├─→ TechnicalAgent.analyze("AAPL")
        ├─→ NewsAgent.get_stock_news("AAPL")
        └─→ LLM.decide(technical + news + macro + sector)
    ↓
返回: DecisionRecord(timeframe=CAMPAIGN, action=HOLD, ...)

═════════════════════════════════════════════════════════

Day 91 (2025-12-30)
    ↓
LayeredScheduler.run_daily()
    ↓
检查: 距离上次Strategic = 90天 >= 90天? YES  ✓
    ↓
运行: StrategicDecisionMaker ✓
    ↓
StrategicDecisionMaker.decide("AAPL")
    ├─→ MacroAgent.analyze_macro()  [深度宏观分析]
    │   ├─→ FedPolicyAgent.get_policy()
    │   ├─→ EconomicDataAgent.get_indicators()
    │   ├─→ GeopoliticalAgent.get_risks()
    │   └─→ LLM.comprehensive_macro_analysis()
    │
    ├─→ SectorAgent.analyze_sector("Technology")
    │
    └─→ MetaAgent.analyze_and_decide(
            symbol="AAPL",
            macro_context=macro_result,
            sector_context=sector_result,
            timeframe=STRATEGIC  ← 明确指定strategic层级
        )
    ↓
返回: DecisionRecord(timeframe=STRATEGIC, action=REDUCE, ...)
     + HierarchicalConstraints(market_regime='bear', ...)  ← 约束传递
```

### 流程2: 反向传导流程

```
Day 45 (2025-11-15) - 正常Tactical决策
    ↓
TacticalDecisionMaker.decide("AAPL")
    ↓
MetaAgent分析发现: AAPL单日暴跌-12%  ⚠️
    ↓
TacticalDecisionMaker.check_escalation()
    ↓
EscalationDetector.detect_all(...)
    ├─→ detect_market_shock(-0.12)  → Score: 10.0, Target: STRATEGIC
    ├─→ detect_news_impact(...)     → Score: 9.5, Target: STRATEGIC
    └─→ detect_volatility_spike(...) → Score: 8.8, Target: CAMPAIGN
    ↓
选择最高分: market_shock (10.0) → STRATEGIC
    ↓
立即触发 StrategicDecisionMaker.decide()  [黑天鹅，不等90天]
    ↓
StrategicDecisionMaker紧急分析
    ├─→ MacroAgent: 检测到系统性风险
    ├─→ SectorAgent: 科技股全线下跌
    └─→ MetaAgent: AAPL基本面恶化
    ↓
返回: DecisionRecord(
        timeframe=STRATEGIC,
        action=SELL,
        escalated_from="tactical",
        escalation_trigger="market_shock",
        escalation_score=10.0
      )
      + 更新约束: market_regime='bear', max_exposure=0.3
    ↓
后续Tactical决策必须遵守新约束 ✓
```

---

## 代码实现（与图一致）

### 1. Scheduler层（顶层调度）

```python
class LayeredDecisionScheduler:
    """
    分层决策调度器
    
    职责：
    1. 根据时间决定调用哪个层级
    2. 管理各层的决策频率
    3. 处理反向传导
    """
    
    def __init__(self):
        # 创建3个DecisionMaker
        self.strategic_maker = StrategicDecisionMaker()
        self.campaign_maker = CampaignDecisionMaker()
        self.tactical_maker = TacticalDecisionMaker()
        
        # 跟踪上次运行时间
        self.last_strategic = None
        self.last_campaign = None
        
        # 反向传导检测器
        self.escalation_detector = EscalationDetector()
    
    def should_run_strategic(self, current_time: datetime) -> bool:
        """是否应该运行Strategic层（90天）"""
        if self.last_strategic is None:
            return True
        return (current_time - self.last_strategic).days >= 90
    
    def should_run_campaign(self, current_time: datetime) -> bool:
        """是否应该运行Campaign层（7天）"""
        if self.last_campaign is None:
            return True
        return (current_time - self.last_campaign).days >= 7
    
    async def run_daily(
        self,
        symbol: str,
        current_time: datetime,
        market_data: Dict[str, Any],
    ) -> List[DecisionRecord]:
        """
        每日运行（回测或实盘都调用此方法）
        
        返回: 当天做出的所有决策（可能包括escalation触发的决策）
        """
        decisions = []
        
        # === Step 1: 按频率调度 ===
        
        # 1.1 Strategic层（90天1次）
        if self.should_run_strategic(current_time):
            strategic_decision = await self.strategic_maker.decide(
                symbol=symbol,
                current_time=current_time,
                market_data=market_data,
            )
            decisions.append(strategic_decision)
            self.last_strategic = current_time
            
            # Strategic决策会更新全局约束
            self._update_global_constraints(strategic_decision)
        
        # 1.2 Campaign层（7天1次）
        elif self.should_run_campaign(current_time):
            campaign_decision = await self.campaign_maker.decide(
                symbol=symbol,
                current_time=current_time,
                market_data=market_data,
            )
            decisions.append(campaign_decision)
            self.last_campaign = current_time
        
        # 1.3 Tactical层（每天）
        tactical_decision = await self.tactical_maker.decide(
            symbol=symbol,
            current_time=current_time,
            market_data=market_data,
        )
        decisions.append(tactical_decision)
        
        # === Step 2: 检查反向传导 ===
        escalation_decisions = await self._check_escalation(
            tactical_decision=tactical_decision,
            market_data=market_data,
            current_time=current_time,
        )
        decisions.extend(escalation_decisions)
        
        return decisions
    
    async def _check_escalation(
        self,
        tactical_decision: DecisionRecord,
        market_data: Dict[str, Any],
        current_time: datetime,
    ) -> List[DecisionRecord]:
        """检查是否需要反向传导"""
        # 检测触发条件
        triggers = self.escalation_detector.detect_all(
            symbol=tactical_decision.symbol,
            market_data=market_data,
            current_timeframe=Timeframe.TACTICAL,
        )
        
        if not triggers:
            return []
        
        # 获取最高分触发器
        top_trigger = triggers[0]
        
        # 评分<7不触发
        if top_trigger.score < 7.0:
            return []
        
        # 根据目标timeframe触发相应层级
        escalation_decisions = []
        
        if top_trigger.to_timeframe == Timeframe.STRATEGIC:
            # 触发Strategic层
            strategic_decision = await self.strategic_maker.decide(
                symbol=tactical_decision.symbol,
                current_time=current_time,
                market_data=market_data,
                escalated_from=tactical_decision,
                escalation_trigger=top_trigger,
            )
            escalation_decisions.append(strategic_decision)
            self.last_strategic = current_time  # 更新时间
            self._update_global_constraints(strategic_decision)
        
        elif top_trigger.to_timeframe == Timeframe.CAMPAIGN:
            # 触发Campaign层
            campaign_decision = await self.campaign_maker.decide(
                symbol=tactical_decision.symbol,
                current_time=current_time,
                market_data=market_data,
                escalated_from=tactical_decision,
                escalation_trigger=top_trigger,
            )
            escalation_decisions.append(campaign_decision)
            self.last_campaign = current_time
        
        return escalation_decisions
    
    def _update_global_constraints(self, strategic_decision: DecisionRecord):
        """从Strategic决策中提取约束，传递给下层"""
        constraints = HierarchicalConstraints(
            strategic={
                'market_regime': strategic_decision.metadata.get('market_regime'),
                'risk_budget': strategic_decision.metadata.get('risk_budget'),
                'max_exposure': strategic_decision.metadata.get('max_exposure'),
            }
        )
        
        # 更新下层的约束
        self.campaign_maker.update_constraints(constraints)
        self.tactical_maker.update_constraints(constraints)
```

### 2. DecisionMaker层（中间层，组合Agent）

```python
class StrategicDecisionMaker:
    """
    Strategic层决策制定者
    
    组合: MacroAgent + SectorAgent + MetaAgent
    频率: 90天1次
    """
    
    def __init__(self):
        self.macro_agent = MacroAgent()       # 宏观分析
        self.sector_agent = SectorAgent()     # 行业分析
        self.meta_agent = MetaAgent()         # 个股分析
        self.timeframe = Timeframe.STRATEGIC
    
    async def decide(
        self,
        symbol: str,
        current_time: datetime,
        market_data: Dict[str, Any],
        escalated_from: Optional[DecisionRecord] = None,
        escalation_trigger: Optional[EscalationTrigger] = None,
    ) -> DecisionRecord:
        """制定Strategic层决策"""
        
        # === 1. 宏观分析（不依赖个股） ===
        macro_analysis = await self.macro_agent.analyze_macro_environment()
        # 返回: {market_regime, interest_rate_trend, risk_level, ...}
        
        # === 2. 行业分析 ===
        sector = self._get_sector(symbol)
        sector_analysis = await self.sector_agent.analyze_sector(sector)
        # 返回: {sector_trend, rotation_signal, relative_strength, ...}
        
        # === 3. 个股分析（结合宏观+行业背景） ===
        decision = await self.meta_agent.analyze_and_decide(
            symbol=symbol,
            timeframe=self.timeframe,
            macro_context=macro_analysis,      # ← 传入宏观背景
            sector_context=sector_analysis,    # ← 传入行业背景
            current_time=current_time,
            market_data=market_data,
        )
        
        # === 4. 标记escalation信息 ===
        if escalated_from:
            decision.mark_as_escalated(
                from_timeframe=escalated_from.timeframe.display_name,
                trigger=escalation_trigger.trigger_type.value,
                score=escalation_trigger.score,
            )
        
        # === 5. 输出约束供下层使用 ===
        decision.metadata['market_regime'] = macro_analysis['market_regime']
        decision.metadata['risk_budget'] = self._calculate_risk_budget(macro_analysis)
        decision.metadata['max_exposure'] = self._calculate_max_exposure(macro_analysis)
        
        return decision


class CampaignDecisionMaker:
    """
    Campaign层决策制定者
    
    组合: MacroAgent(轻量级) + SectorAgent + MetaAgent
    频率: 7天1次
    """
    
    def __init__(self):
        self.macro_agent = MacroAgent()
        self.sector_agent = SectorAgent()
        self.meta_agent = MetaAgent()
        self.timeframe = Timeframe.CAMPAIGN
        self.constraints = None  # 从Strategic层传入
    
    def update_constraints(self, constraints: HierarchicalConstraints):
        """接收Strategic层的约束"""
        self.constraints = constraints
    
    async def decide(
        self,
        symbol: str,
        current_time: datetime,
        market_data: Dict[str, Any],
        **kwargs
    ) -> DecisionRecord:
        """制定Campaign层决策"""
        
        # 1. 轻量级宏观分析（或直接使用缓存）
        macro_analysis = await self.macro_agent.get_quick_update()
        
        # 2. 行业分析
        sector = self._get_sector(symbol)
        sector_analysis = await self.sector_agent.analyze_sector(sector)
        
        # 3. 个股分析
        decision = await self.meta_agent.analyze_and_decide(
            symbol=symbol,
            timeframe=self.timeframe,
            macro_context=macro_analysis,
            sector_context=sector_analysis,
            constraints=self.constraints,  # ← 遵守Strategic约束
            current_time=current_time,
            market_data=market_data,
        )
        
        return decision


class TacticalDecisionMaker:
    """
    Tactical层决策制定者
    
    组合: MetaAgent (不需要MacroAgent)
    频率: 每天
    """
    
    def __init__(self):
        self.meta_agent = MetaAgent()
        self.timeframe = Timeframe.TACTICAL
        self.constraints = None
    
    def update_constraints(self, constraints: HierarchicalConstraints):
        """接收上层约束"""
        self.constraints = constraints
    
    async def decide(
        self,
        symbol: str,
        current_time: datetime,
        market_data: Dict[str, Any],
        **kwargs
    ) -> DecisionRecord:
        """制定Tactical层决策"""
        
        # 检查是否是关键时刻（决定用fast还是full模式）
        is_critical = self._is_critical_moment(market_data)
        computation_mode = 'full' if is_critical else 'fast'
        
        # 个股分析（不需要宏观背景）
        decision = await self.meta_agent.analyze_and_decide(
            symbol=symbol,
            timeframe=self.timeframe,
            constraints=self.constraints,  # ← 遵守上层约束
            computation_mode=computation_mode,
            current_time=current_time,
            market_data=market_data,
        )
        
        return decision
```

### 3. Agent层（底层，具体分析）

```python
class MacroAgent(BaseMCPAgent):
    """
    宏观分析Agent
    
    职责: 分析宏观经济环境，不依赖个股
    工具: FedPolicy, EconomicData, Geopolitical
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 连接宏观相关的MCP servers
        self.connect_agent("fed_policy", FedPolicyAgent)
        self.connect_agent("economic_data", EconomicDataAgent)
        self.connect_agent("geopolitical", GeopoliticalAgent)
    
    async def analyze_macro_environment(self) -> Dict[str, Any]:
        """
        分析宏观环境
        
        注意: 不需要symbol参数！
        """
        # 获取美联储政策
        fed_policy = await self.call_tool(
            agent_name="fed_policy",
            tool_name="get_current_policy",
            arguments={}
        )
        
        # 获取经济指标
        economic_data = await self.call_tool(
            agent_name="economic_data",
            tool_name="get_indicators",
            arguments={"indicators": ["GDP", "unemployment", "inflation"]}
        )
        
        # 获取地缘政治风险
        geopolitical_risk = await self.call_tool(
            agent_name="geopolitical",
            tool_name="assess_risk",
            arguments={}
        )
        
        # LLM综合分析
        prompt = f"""
        作为宏观经济分析师，综合以下信息判断当前市场环境：
        
        美联储政策: {fed_policy}
        经济指标: {economic_data}
        地缘政治: {geopolitical_risk}
        
        请判断:
        1. 市场regime (bull/bear/neutral)
        2. 利率趋势 (rising/falling/stable)
        3. 经济周期 (expansion/peak/contraction/trough)
        4. 风险等级 (0-10)
        """
        
        analysis = await self.llm_client.chat.completions.create(
            model=self.llm_config.model,
            messages=[{"role": "user", "content": prompt}],
        )
        
        # 解析LLM输出
        result = self._parse_llm_response(analysis)
        
        return {
            'market_regime': result['regime'],
            'interest_rate_trend': result['interest_rate_trend'],
            'economic_cycle': result['economic_cycle'],
            'risk_level': result['risk_level'],
            'reasoning': result['reasoning'],
            'raw_data': {
                'fed_policy': fed_policy,
                'economic_data': economic_data,
                'geopolitical_risk': geopolitical_risk,
            }
        }


class MetaAgent(BaseMCPAgent):
    """
    个股分析Agent（原有的MetaAgent，轻微修改）
    
    职责: 分析特定股票
    工具: Technical, News
    """
    
    async def analyze_and_decide(
        self,
        symbol: str,
        timeframe: Timeframe,
        current_time: datetime,
        market_data: Dict[str, Any],
        macro_context: Optional[Dict] = None,      # 新增
        sector_context: Optional[Dict] = None,     # 新增
        constraints: Optional[HierarchicalConstraints] = None,  # 新增
        computation_mode: str = 'full',
        **kwargs
    ) -> DecisionRecord:
        """
        分析个股并做出决策
        
        参数:
            macro_context: 来自MacroAgent的宏观分析（Strategic/Campaign层）
            sector_context: 来自SectorAgent的行业分析
            constraints: 来自上层的约束
        """
        
        # === 1. 获取个股数据 ===
        technical = await self.technical_agent.analyze(symbol)
        news = await self.news_agent.get_stock_news(symbol)
        
        # === 2. 构建LLM prompt（包含宏观背景） ===
        prompt = f"""
        分析股票: {symbol}
        当前时间: {current_time}
        决策层级: {timeframe.display_name}
        
        技术分析: {technical}
        新闻情绪: {news}
        """
        
        # 如果有宏观背景，加入prompt
        if macro_context:
            prompt += f"""
            
            宏观环境:
            - 市场regime: {macro_context['market_regime']}
            - 利率趋势: {macro_context['interest_rate_trend']}
            - 风险等级: {macro_context['risk_level']}
            """
        
        # 如果有行业背景，加入prompt
        if sector_context:
            prompt += f"""
            
            行业分析:
            - 行业趋势: {sector_context['trend']}
            - 轮动信号: {sector_context['rotation_signal']}
            """
        
        # 如果有约束，加入prompt
        if constraints:
            regime = constraints.get_market_regime()
            if regime == 'bear':
                prompt += "\n⚠️ 约束: 当前战略层判断为熊市，禁止做多"
        
        # === 3. LLM决策 ===
        decision = await self._llm_decide(prompt)
        
        # === 4. 检查约束 ===
        if constraints:
            decision = self._apply_constraints(decision, constraints)
        
        return decision
```

---

## 性能分析

### 场景: 回测10只股票，250天

#### 方案A（Timeframe绑定）

```
每只股票每天都获取宏观新闻:
- Strategic层: 每90天 × 10只股票 × 3次 = 30次宏观分析
- Campaign层: 每7天 × 10只股票 × 36次 = 360次宏观分析
- Tactical层: 每天 × 10只股票 × 250次 = 0次（不需要）

总宏观分析: 390次
```

#### 方案B（MacroAgent分离）

```
宏观分析结果可以复用:
- Strategic层: 每90天 × 1次（所有股票共享） × 3次 = 3次
- Campaign层: 每7天 × 1次 × 36次 = 36次
- Tactical层: 0次

总宏观分析: 39次
```

**性能提升**: 390次 → 39次 = **快10倍**

### Agent数量不会导致变慢的原因

```python
# Agent是轻量级的逻辑封装
class MacroAgent:
    def __init__(self):
        # 只是连接到MCP servers，没有重复的资源
        pass
    
    async def analyze(self):
        # 调用底层工具，不是重复计算
        result = await self.call_tool(...)
        return result

# 3个DecisionMaker共享同一个MacroAgent实例
macro_agent = MacroAgent()  # 只创建1次

strategic_maker = StrategicDecisionMaker(macro_agent=macro_agent)
campaign_maker = CampaignDecisionMaker(macro_agent=macro_agent)
# 共享，不是复制
```

---

## 总结

### Timeframe如何产生？
- **Scheduler决定**，不是Agent决定
- 根据时间间隔自动调度

### 多Agent会不会变慢？
- **不会！反而更快！**
- 宏观分析可以复用
- 按需调用，不浪费

### LangChain如何理解？
- 通过**工具描述**区分
- MacroAgent: 工具不需要symbol
- MetaAgent: 工具需要symbol

### 代码与图的一致性
- ✅ 图中的每个框都对应一个类
- ✅ 箭头对应方法调用关系
- ✅ 数据流向清晰可追踪

这个设计清楚了吗？需要我详细解释某个部分吗？
