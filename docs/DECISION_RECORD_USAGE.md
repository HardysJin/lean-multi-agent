# DecisionRecord 扩展功能使用指南

本文档介绍如何使用扩展后的 `DecisionRecord` 和反向传导系统。

## 目录

1. [新增字段说明](#新增字段说明)
2. [防止Look-Ahead Bias](#防止look-ahead-bias)
3. [计算模式](#计算模式)
4. [反向传导机制](#反向传导机制)
5. [信号缓存](#信号缓存)
6. [完整示例](#完整示例)

---

## 新增字段说明

### 时间控制字段（防止Look-Ahead）

```python
visible_data_end: Optional[datetime] = None
```

**用途**：回测模式下限制可见数据的截止时间。

- 实盘模式：`None`（无限制，使用实时数据）
- 回测模式：设置为决策时间点之前的某个时间（如延迟5分钟）

**示例**：
```python
from datetime import datetime, timedelta
from Memory import DecisionRecord, Timeframe

# 回测场景：决策时间是 10:30，但只能看到 10:25 之前的数据
decision_time = datetime(2025, 10, 29, 10, 30)
visible_end = datetime(2025, 10, 29, 10, 25)

decision = DecisionRecord(
    id="backtest_001",
    timestamp=decision_time,
    timeframe=Timeframe.TACTICAL,
    symbol="AAPL",
    action="BUY",
    quantity=100,
    price=150.0,
    reasoning="Technical signal",
    agent_name="meta_agent",
    conviction=8.0,
    visible_data_end=visible_end,  # 关键：限制可见数据
)

# 验证数据是否可见
data_time_ok = datetime(2025, 10, 29, 10, 20)  # 可见
data_time_future = datetime(2025, 10, 29, 10, 27)  # 不可见（Look-Ahead）

print(decision.validate_data_timestamp(data_time_ok))  # True
print(decision.validate_data_timestamp(data_time_future))  # False
```

### 计算模式字段

```python
computation_mode: str = 'full'  # 'full', 'hybrid', 'fast'
```

**用途**：标识决策的计算方式，用于性能优化。

- `full`：完整Multi-Agent（所有Agent + LLM）
- `hybrid`：混合模式（部分Agent + LLM）
- `fast`：快速模式（仅规则引擎，无LLM）

**使用建议**：
- **Strategic层（90天）**：始终使用 `full`
- **Campaign层（7天）**：使用 `hybrid`
- **Tactical层（每天）**：普通日用 `fast`，关键时刻用 `full`

### 缓存字段

```python
cache_key: Optional[str] = None
```

**用途**：标识可复用的计算结果，避免重复计算。

**格式**：`{symbol}_{timeframe}_{strategy_version}_{data_hash}`

### 反向传导字段

```python
escalated_from: Optional[str] = None
escalation_trigger: Optional[str] = None
escalation_score: Optional[float] = None
```

**用途**：记录反向传导的来源和原因。

---

## 防止Look-Ahead Bias

### 问题背景

回测系统的常见陷阱：Agent 可能"偷看"未来数据，导致虚假高收益。

### 解决方案

使用 `visible_data_end` 字段强制限制可见数据范围。

### 实践步骤

```python
from datetime import datetime, timedelta
from Memory import DecisionRecord, Timeframe

class BacktestEngine:
    def __init__(self, start_date, end_date):
        self.current_time = start_date
        self.end_date = end_date
        self.data_delay = timedelta(minutes=5)  # 数据延迟
    
    def advance_time(self, delta):
        """推进回测时间"""
        self.current_time += delta
    
    def get_visible_data_end(self):
        """获取可见数据截止时间"""
        return self.current_time - self.data_delay
    
    def create_decision(self, symbol, action, **kwargs):
        """创建决策时自动设置时间限制"""
        return DecisionRecord(
            timestamp=self.current_time,
            visible_data_end=self.get_visible_data_end(),  # 自动限制
            symbol=symbol,
            action=action,
            **kwargs
        )

# 使用示例
engine = BacktestEngine(
    start_date=datetime(2025, 1, 1, 9, 0),
    end_date=datetime(2025, 12, 31, 16, 0)
)

# 推进到第一个决策点
engine.advance_time(timedelta(hours=1))

# 创建决策（自动带时间限制）
decision = engine.create_decision(
    symbol="AAPL",
    action="BUY",
    quantity=100,
    price=150.0,
    reasoning="Strong signal",
    agent_name="meta_agent",
    conviction=8.0,
    timeframe=Timeframe.TACTICAL,
    id="decision_001",
)

# 验证时间控制
assert decision.is_backtest_mode()
assert decision.validate_data_timestamp(engine.get_visible_data_end())
```

---

## 计算模式

### 分层计算策略

根据时间尺度选择合适的计算模式：

| 时间尺度 | 频率 | 计算模式 | 原因 |
|---------|-----|---------|-----|
| Strategic (90天) | 3次/年 | `full` | 需要深度分析，次数少 |
| Campaign (7天) | 52次/年 | `hybrid` | 平衡速度和深度 |
| Tactical (每天) | 250次/年 | `fast` + `full`（关键时刻） | 大部分日子用规则，特殊日子深度分析 |

### 实现示例

```python
from Memory import DecisionRecord, Timeframe, EscalationDetector

class LayeredDecisionMaker:
    def __init__(self):
        self.detector = EscalationDetector()
    
    def make_tactical_decision(self, symbol, market_data, **kwargs):
        """Tactical层决策：根据市场状况选择计算模式"""
        
        # 检测是否有重大事件
        triggers = self.detector.detect_all(
            symbol=symbol,
            market_data=market_data,
            current_timeframe=Timeframe.TACTICAL,
        )
        
        # 关键时刻：使用完整模式
        if triggers and triggers[0].score >= 7.0:
            return DecisionRecord(
                timeframe=Timeframe.TACTICAL,
                symbol=symbol,
                computation_mode='full',  # 深度分析
                reasoning="Critical moment detected, full analysis required",
                **kwargs
            )
        
        # 普通日子：使用快速模式
        else:
            return DecisionRecord(
                timeframe=Timeframe.TACTICAL,
                symbol=symbol,
                computation_mode='fast',  # 规则引擎
                reasoning="Normal day, fast rule-based decision",
                **kwargs
            )
    
    def make_campaign_decision(self, symbol, **kwargs):
        """Campaign层决策：混合模式"""
        return DecisionRecord(
            timeframe=Timeframe.CAMPAIGN,
            symbol=symbol,
            computation_mode='hybrid',
            **kwargs
        )
    
    def make_strategic_decision(self, symbol, **kwargs):
        """Strategic层决策：完整模式"""
        return DecisionRecord(
            timeframe=Timeframe.STRATEGIC,
            symbol=symbol,
            computation_mode='full',
            **kwargs
        )

# 使用示例
maker = LayeredDecisionMaker()

# 普通日子（快速）
normal_decision = maker.make_tactical_decision(
    symbol="AAPL",
    market_data={'price_change_1d': -0.01},  # 小幅波动
    id="tactical_001",
    timestamp=datetime.now(),
    action="HOLD",
    quantity=0,
    price=150.0,
    agent_name="rule_engine",
    conviction=5.0,
)
print(normal_decision.computation_mode)  # 'fast'

# 关键时刻（完整）
critical_decision = maker.make_tactical_decision(
    symbol="AAPL",
    market_data={'price_change_1d': -0.08},  # 8%暴跌
    id="tactical_002",
    timestamp=datetime.now(),
    action="SELL",
    quantity=100,
    price=138.0,
    agent_name="meta_agent",
    conviction=9.0,
)
print(critical_decision.computation_mode)  # 'full'
```

---

## 反向传导机制

### 核心概念

当低层时间尺度（如Tactical）检测到重大事件时，触发高层时间尺度（如Campaign或Strategic）重新评估。

### 触发类型

1. **市场冲击** (`market_shock`)：单日大幅波动
2. **新闻冲击** (`news_impact`)：重大新闻事件
3. **技术突破** (`technical_breakout`)：关键技术位突破
4. **战略冲突** (`strategic_conflict`)：决策与上层约束冲突
5. **黑天鹅** (`black_swan`)：极端事件（直达Strategic层）

### 使用示例

```python
from datetime import datetime
from Memory import (
    DecisionRecord,
    Timeframe,
    EscalationDetector,
    should_trigger_escalation,
)

# 初始化检测器
detector = EscalationDetector()

# === Step 1: Tactical层正常运行 ===
tactical_time = datetime(2025, 10, 29, 10, 30)
tactical_decision = DecisionRecord(
    id="tactical_001",
    timestamp=tactical_time,
    timeframe=Timeframe.TACTICAL,
    symbol="AAPL",
    action="HOLD",
    quantity=0,
    price=150.0,
    reasoning="No strong signals",
    agent_name="rule_engine",
    conviction=5.0,
    computation_mode="fast",
)

# === Step 2: 检测到重大事件 ===
market_data = {
    'price_change_1d': -0.08,  # 8%下跌
    'current_volatility': 0.45,
    'historical_volatility': 0.12,
}

triggers = detector.detect_all(
    symbol="AAPL",
    market_data=market_data,
    current_timeframe=Timeframe.TACTICAL,
)

# === Step 3: 判断是否触发反向传导 ===
top_trigger = should_trigger_escalation(triggers, threshold=7.0)

if top_trigger:
    print(f"🚨 Escalation triggered: {top_trigger.trigger_type.value}")
    print(f"   Score: {top_trigger.score:.1f}")
    print(f"   From: {top_trigger.from_timeframe.display_name}")
    print(f"   To: {top_trigger.to_timeframe.display_name}")
    
    # === Step 4: 高层响应 ===
    campaign_decision = DecisionRecord(
        id="campaign_001",
        timestamp=tactical_time,
        timeframe=Timeframe.CAMPAIGN,
        symbol="AAPL",
        action="REDUCE",
        quantity=150,
        price=138.0,
        reasoning=f"Emergency response: {top_trigger.reason}",
        agent_name="meta_agent",
        conviction=9.0,
        computation_mode="full",
    )
    
    # 标记为反向传导
    campaign_decision.mark_as_escalated(
        from_timeframe=top_trigger.from_timeframe.display_name,
        trigger=top_trigger.trigger_type.value,
        score=top_trigger.score,
    )
    
    print(f"✅ Campaign layer responded: {campaign_decision.action}")
```

### 自定义阈值

```python
# 自定义阈值（更敏感）
custom_thresholds = {
    'market_shock_1day': 0.03,  # 3%就触发（默认5%）
    'news_impact_high': 7.0,    # 7分就触发（默认8分）
}

detector = EscalationDetector(thresholds=custom_thresholds)
```

---

## 信号缓存

### 使用场景

在回测中，如果数据和策略版本相同，可以复用之前的计算结果。

### 实现示例

```python
import hashlib
from datetime import datetime
from Memory import DecisionRecord, Timeframe

class SignalCache:
    def __init__(self):
        self.cache = {}
    
    def generate_data_hash(self, price_data, indicators):
        """生成数据哈希"""
        data_str = f"{price_data}_{indicators}"
        return hashlib.md5(data_str.encode()).hexdigest()[:12]
    
    def get_cache_key(self, symbol, timeframe, strategy_version, data_hash):
        """生成缓存键"""
        return f"{symbol}_{timeframe.display_name}_{strategy_version}_{data_hash}"
    
    def get_or_compute(self, symbol, timeframe, strategy_version, 
                       price_data, indicators, compute_fn):
        """获取缓存或计算"""
        # 生成键
        data_hash = self.generate_data_hash(price_data, indicators)
        cache_key = self.get_cache_key(symbol, timeframe, strategy_version, data_hash)
        
        # 尝试从缓存读取
        if cache_key in self.cache:
            print(f"✅ Cache hit: {cache_key}")
            cached_decision = self.cache[cache_key]
            # 创建新决策，但使用缓存的信号
            return DecisionRecord(
                id=f"cached_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                timeframe=timeframe,
                symbol=symbol,
                action=cached_decision.action,
                quantity=cached_decision.quantity,
                price=price_data['close'],  # 使用当前价格
                reasoning=f"[CACHED] {cached_decision.reasoning}",
                agent_name="cache_engine",
                conviction=cached_decision.conviction,
                computation_mode="fast",
                cache_key=cache_key,
            )
        
        # 缓存未命中，执行计算
        print(f"❌ Cache miss: {cache_key}")
        decision = compute_fn()
        decision.set_cache_key(strategy_version, data_hash)
        
        # 存入缓存
        self.cache[cache_key] = decision
        
        return decision

# 使用示例
cache = SignalCache()

def expensive_computation():
    """模拟耗时的LLM计算"""
    print("  🔄 Running expensive LLM computation...")
    return DecisionRecord(
        id="computed_001",
        timestamp=datetime.now(),
        timeframe=Timeframe.TACTICAL,
        symbol="AAPL",
        action="BUY",
        quantity=100,
        price=150.0,
        reasoning="Strong technical signals: RSI oversold, MACD crossover",
        agent_name="meta_agent",
        conviction=8.0,
        computation_mode="full",
    )

# 第一次：计算
decision1 = cache.get_or_compute(
    symbol="AAPL",
    timeframe=Timeframe.TACTICAL,
    strategy_version="v1.0.0",
    price_data={'close': 150.0},
    indicators={'RSI': 30, 'MACD': 0.5},
    compute_fn=expensive_computation,
)

# 第二次：相同条件，使用缓存
decision2 = cache.get_or_compute(
    symbol="AAPL",
    timeframe=Timeframe.TACTICAL,
    strategy_version="v1.0.0",
    price_data={'close': 150.0},  # 相同数据
    indicators={'RSI': 30, 'MACD': 0.5},  # 相同指标
    compute_fn=expensive_computation,  # 不会被调用
)

print(f"\nDecision 1 mode: {decision1.computation_mode}")  # 'full'
print(f"Decision 2 mode: {decision2.computation_mode}")    # 'fast' (cached)
```

---

## 完整示例

综合使用所有功能的回测引擎：

```python
from datetime import datetime, timedelta
from Memory import (
    DecisionRecord,
    Timeframe,
    EscalationDetector,
    should_trigger_escalation,
    create_decision_id,
)

class SmartBacktestEngine:
    """
    智能回测引擎
    
    特性：
    - 防止Look-Ahead Bias
    - 分层决策（Strategic/Campaign/Tactical）
    - 反向传导机制
    - 信号缓存
    """
    
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.current_time = start_date
        self.end_date = end_date
        
        self.data_delay = timedelta(minutes=5)  # 数据延迟
        self.detector = EscalationDetector()
        self.decisions = []
        
        # 分层决策调度
        self.strategic_interval = timedelta(days=90)
        self.campaign_interval = timedelta(days=7)
        self.tactical_interval = timedelta(days=1)
        
        self.last_strategic = start_date
        self.last_campaign = start_date
    
    def get_visible_data_end(self):
        """获取可见数据截止时间"""
        return self.current_time - self.data_delay
    
    def should_run_strategic(self):
        """判断是否应该运行Strategic层"""
        return (self.current_time - self.last_strategic) >= self.strategic_interval
    
    def should_run_campaign(self):
        """判断是否应该运行Campaign层"""
        return (self.current_time - self.last_campaign) >= self.campaign_interval
    
    def run_strategic_decision(self, symbol, market_data):
        """Strategic层决策"""
        decision = DecisionRecord(
            id=create_decision_id(symbol, self.current_time, Timeframe.STRATEGIC),
            timestamp=self.current_time,
            timeframe=Timeframe.STRATEGIC,
            symbol=symbol,
            action="HOLD",  # 简化示例
            quantity=0,
            price=market_data['price'],
            reasoning="Strategic review every 90 days",
            agent_name="meta_agent",
            conviction=7.0,
            visible_data_end=self.get_visible_data_end(),
            computation_mode="full",
        )
        
        self.last_strategic = self.current_time
        self.decisions.append(decision)
        print(f"📊 STRATEGIC decision at {self.current_time.date()}")
        return decision
    
    def run_campaign_decision(self, symbol, market_data, escalated=False):
        """Campaign层决策"""
        decision = DecisionRecord(
            id=create_decision_id(symbol, self.current_time, Timeframe.CAMPAIGN),
            timestamp=self.current_time,
            timeframe=Timeframe.CAMPAIGN,
            symbol=symbol,
            action="HOLD",
            quantity=0,
            price=market_data['price'],
            reasoning="Campaign review" + (" (escalated)" if escalated else ""),
            agent_name="meta_agent",
            conviction=7.5,
            visible_data_end=self.get_visible_data_end(),
            computation_mode="hybrid",
        )
        
        self.last_campaign = self.current_time
        self.decisions.append(decision)
        print(f"📈 CAMPAIGN decision at {self.current_time.date()}" + 
              (" [ESCALATED]" if escalated else ""))
        return decision
    
    def run_tactical_decision(self, symbol, market_data):
        """Tactical层决策"""
        # 检测触发条件
        triggers = self.detector.detect_all(
            symbol=symbol,
            market_data=market_data,
            current_timeframe=Timeframe.TACTICAL,
        )
        
        # 检查是否需要反向传导
        top_trigger = should_trigger_escalation(triggers, threshold=7.0)
        
        if top_trigger:
            # 触发反向传导
            print(f"🚨 ESCALATION: {top_trigger.trigger_type.value} (score: {top_trigger.score:.1f})")
            
            # 根据目标层级执行相应决策
            if top_trigger.to_timeframe == Timeframe.STRATEGIC:
                escalated_decision = self.run_strategic_decision(symbol, market_data)
            else:  # Campaign
                escalated_decision = self.run_campaign_decision(symbol, market_data, escalated=True)
            
            escalated_decision.mark_as_escalated(
                from_timeframe=top_trigger.from_timeframe.display_name,
                trigger=top_trigger.trigger_type.value,
                score=top_trigger.score,
            )
            
            # Tactical层也做决策
            computation_mode = "full"
        else:
            # 普通日子，使用快速模式
            computation_mode = "fast"
        
        decision = DecisionRecord(
            id=create_decision_id(symbol, self.current_time, Timeframe.TACTICAL),
            timestamp=self.current_time,
            timeframe=Timeframe.TACTICAL,
            symbol=symbol,
            action="HOLD",
            quantity=0,
            price=market_data['price'],
            reasoning="Daily tactical decision",
            agent_name="rule_engine" if computation_mode == "fast" else "meta_agent",
            conviction=6.0,
            visible_data_end=self.get_visible_data_end(),
            computation_mode=computation_mode,
        )
        
        self.decisions.append(decision)
        print(f"📉 TACTICAL decision at {self.current_time.date()} [{computation_mode}]")
        return decision
    
    def run_backtest(self, symbol):
        """运行完整回测"""
        print(f"Starting backtest: {self.start_date.date()} to {self.end_date.date()}")
        print("=" * 60)
        
        day_count = 0
        
        while self.current_time <= self.end_date:
            # 模拟市场数据
            market_data = {
                'price': 150.0 + day_count * 0.1,  # 简化：每天涨0.1
                'price_change_1d': 0.001 * day_count,  # 简化波动
            }
            
            # 分层决策调度
            if self.should_run_strategic():
                self.run_strategic_decision(symbol, market_data)
            
            if self.should_run_campaign():
                self.run_campaign_decision(symbol, market_data)
            
            # Tactical层每天运行
            self.run_tactical_decision(symbol, market_data)
            
            # 推进时间
            self.current_time += self.tactical_interval
            day_count += 1
        
        print("=" * 60)
        print(f"Backtest completed: {len(self.decisions)} decisions made")
        
        # 统计
        strategic_count = sum(1 for d in self.decisions if d.timeframe == Timeframe.STRATEGIC)
        campaign_count = sum(1 for d in self.decisions if d.timeframe == Timeframe.CAMPAIGN)
        tactical_count = sum(1 for d in self.decisions if d.timeframe == Timeframe.TACTICAL)
        
        full_mode = sum(1 for d in self.decisions if d.computation_mode == "full")
        fast_mode = sum(1 for d in self.decisions if d.computation_mode == "fast")
        
        escalated = sum(1 for d in self.decisions if d.escalated_from is not None)
        
        print(f"\n📊 Statistics:")
        print(f"  Strategic: {strategic_count}")
        print(f"  Campaign: {campaign_count}")
        print(f"  Tactical: {tactical_count}")
        print(f"  Full mode: {full_mode}")
        print(f"  Fast mode: {fast_mode}")
        print(f"  Escalated: {escalated}")

# 运行示例
if __name__ == "__main__":
    engine = SmartBacktestEngine(
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 3, 31),  # 3个月
    )
    
    engine.run_backtest("AAPL")
```

---

## 总结

### 关键改进

1. **Look-Ahead防护**：`visible_data_end` 字段确保回测公平性
2. **性能优化**：分层计算模式减少LLM调用（74%）
3. **反向传导**：黑天鹅事件能快速触达高层决策
4. **信号缓存**：避免重复计算，进一步提速

### 下一步

- Step 2: 实现 `BacktestClock` 和 `TimeSliceManager`
- Step 3: 实现 `LayeredDecisionScheduler`
- Step 4: 实现 `SignalCache` 持久化
- Step 5: 集成到 VectorBT 引擎
- Step 6: 全面测试和优化

---

**文档版本**: v1.0  
**最后更新**: 2025-10-29
