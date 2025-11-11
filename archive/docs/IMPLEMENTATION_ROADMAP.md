# 方案B实施路线图

## 📋 总览

基于MacroAgent分离的设计方案，完整实施路线图。

**总体目标**：
- 创建分层决策系统
- 防止Look-Ahead Bias
- 提升回测速度（20-30分钟 → 5-10分钟）
- 支持反向传导机制

**预计总时间**：约 32-40 小时

---

## 🎯 已完成

### ✅ Step 0: 基础设施 (已完成)
- [x] DecisionRecord扩展
- [x] EscalationDetector实现
- [x] 反向传导机制
- [x] 完整测试覆盖 (28/28)

**成果**：
- Memory/schemas.py (扩展)
- Memory/escalation.py (新建)
- Tests/unit/test_decision_record_extensions.py
- Tests/integration/test_decision_workflow.py

---

## 📝 待实施步骤

### Step 1: MacroAgent 创建 【核心基础】

**目标**：创建宏观分析Agent，独立于个股分析

**工作内容**：

#### 1.1 创建MacroAgent基础框架 (2小时)
- [ ] 创建 `Agents/macro_agent.py`
- [ ] 继承 `BaseMCPAgent`
- [ ] 实现 `analyze_macro_environment()` 方法
- [ ] 定义返回数据结构

**输出**：
```python
class MacroAgent(BaseMCPAgent):
    async def analyze_macro_environment(self) -> Dict[str, Any]:
        # 返回: {market_regime, interest_rate_trend, risk_level, ...}
        pass
```

#### 1.2 实现宏观数据获取 (3小时)
- [ ] 创建 `Agents/economic_data_agent.py` (MCP Server)
  - GDP、失业率、通胀率
- [ ] 创建 `Agents/fed_policy_agent.py` (MCP Server)
  - 利率决策、货币政策
- [ ] 创建 `Agents/geopolitical_agent.py` (MCP Server)
  - 地缘政治风险评估

**输出**：3个新的MCP Server Agent

#### 1.3 集成LLM分析 (2小时)
- [ ] 实现 `_llm_analyze_macro()` 方法
- [ ] 设计宏观分析prompt
- [ ] 解析LLM输出为结构化数据

#### 1.4 测试MacroAgent (2小时)
- [ ] 单元测试：测试每个数据源
- [ ] 集成测试：测试完整流程
- [ ] Mock测试：不依赖真实API

**成果验证**：
- [ ] 能独立运行宏观分析（不需要symbol）
- [ ] 返回market_regime判断
- [ ] 测试通过率 100%

**预计时间**：9小时

---

### Step 2: SectorAgent 创建 【行业分析】

**目标**：创建行业分析Agent，填补宏观与个股之间的层级

**工作内容**：

#### 2.1 创建SectorAgent框架 (2小时)
- [ ] 创建 `Agents/sector_agent.py`
- [ ] 实现 `analyze_sector(sector: str)` 方法
- [ ] 定义行业分类映射

#### 2.2 实现行业数据获取 (2小时)
- [ ] 创建 `Agents/sector_rotation_agent.py`
  - 行业轮动信号
  - 行业相对强度
- [ ] 获取行业新闻汇总

#### 2.3 测试SectorAgent (1小时)
- [ ] 测试行业分析功能
- [ ] 验证与个股的关联

**成果验证**：
- [ ] 能分析特定行业（如"Technology"）
- [ ] 返回行业趋势和轮动信号
- [ ] 测试通过

**预计时间**：5小时

---

### Step 3: 修改MetaAgent 【关键集成】

**目标**：修改现有MetaAgent，支持接收宏观和行业背景

**工作内容**：

#### 3.1 添加上下文参数 (1小时)
- [ ] 修改 `analyze_and_decide()` 签名
  - 添加 `macro_context: Optional[Dict]`
  - 添加 `sector_context: Optional[Dict]`
  - 添加 `constraints: Optional[HierarchicalConstraints]`

#### 3.2 修改LLM Prompt (2小时)
- [ ] 扩展prompt包含宏观信息
- [ ] 扩展prompt包含行业信息
- [ ] 扩展prompt包含约束条件

#### 3.3 实现约束检查 (2小时)
- [ ] 实现 `_apply_constraints()` 方法
- [ ] 熊市禁止做多
- [ ] 检查风险预算
- [ ] 检查最大敞口

#### 3.4 测试修改后的MetaAgent (2小时)
- [ ] 测试无上下文的情况（向后兼容）
- [ ] 测试有宏观背景的情况
- [ ] 测试约束生效

**成果验证**：
- [ ] 向后兼容（不传macro_context也能用）
- [ ] 能正确使用宏观背景
- [ ] 约束能正确生效
- [ ] 测试通过

**预计时间**：7小时

---

### Step 4: DecisionMaker层创建 【组合逻辑】

**目标**：创建3个DecisionMaker，组合不同的Agent

**工作内容**：

#### 4.1 创建StrategicDecisionMaker (2小时)
- [ ] 创建 `Agents/decision_makers.py`
- [ ] 实现 `StrategicDecisionMaker` 类
  - 组合 MacroAgent + SectorAgent + MetaAgent
  - 实现 `decide()` 方法
  - 输出约束供下层使用

#### 4.2 创建CampaignDecisionMaker (1小时)
- [ ] 实现 `CampaignDecisionMaker` 类
  - 轻量级宏观分析
  - 行业分析 + 个股分析
  - 接收上层约束

#### 4.3 创建TacticalDecisionMaker (1小时)
- [ ] 实现 `TacticalDecisionMaker` 类
  - 仅个股分析
  - 支持fast/full模式切换
  - 接收上层约束

#### 4.4 测试DecisionMaker (2小时)
- [ ] 测试每个Maker独立工作
- [ ] 测试约束传递
- [ ] 测试不同timeframe的输出

**成果验证**：
- [ ] 3个DecisionMaker都能独立运行
- [ ] Strategic输出的约束能传递给下层
- [ ] 测试通过

**预计时间**：6小时

---

### Step 5: LayeredScheduler创建 【调度核心】

**目标**：创建调度器，根据时间自动选择层级

**工作内容**：

#### 5.1 创建Scheduler框架 (2小时)
- [ ] 创建 `Backtests/layered_scheduler.py`
- [ ] 实现时间跟踪逻辑
- [ ] 实现 `should_run_strategic()` 等判断方法

#### 5.2 实现调度逻辑 (3小时)
- [ ] 实现 `run_daily()` 方法
- [ ] 按频率调度（30天/7天/每天）
- [ ] 更新上次运行时间

#### 5.3 集成反向传导 (3小时)
- [ ] 实现 `_check_escalation()` 方法
- [ ] 集成 `EscalationDetector`
- [ ] 实现同步/异步触发逻辑
  - 评分>=9.0：立即执行
  - 评分7.0-9.0：标记待执行

#### 5.4 实现约束更新 (1小时)
- [ ] 实现 `_update_global_constraints()` 方法
- [ ] Strategic决策 → 提取约束 → 传递给下层

#### 5.5 测试Scheduler (3小时)
- [ ] 测试正常调度（30天/7天/每天）
- [ ] 测试反向传导触发
- [ ] 测试约束传递
- [ ] 模拟250天回测

**成果验证**：
- [ ] 能自动根据时间调度层级
- [ ] 反向传导正确触发
- [ ] 约束正确传递
- [ ] 测试通过

**预计时间**：12小时

---

### Step 6: BacktestClock + TimeSliceManager 【时间控制】

**目标**：统一时间管理，防止Look-Ahead Bias

**工作内容**：

#### 6.1 创建BacktestClock (2小时)
- [ ] 创建 `Backtests/backtest_clock.py`
- [ ] 实现时间推进逻辑（只能向前）
- [ ] 实现 `get_current_time()`
- [ ] 实现 `advance_to(new_time)`

#### 6.2 创建TimeSliceManager (3小时)
- [ ] 创建 `Backtests/time_slice_manager.py`
- [ ] 实现 `get_data_slice(symbol, end_time)`
- [ ] 缓存切片数据
- [ ] 集成到TechnicalAgent和NewsAgent

#### 6.3 集成到Scheduler (2小时)
- [ ] Scheduler使用BacktestClock
- [ ] 所有Agent接收visible_data_end
- [ ] DecisionRecord自动设置visible_data_end

#### 6.4 测试时间控制 (2小时)
- [ ] 测试时间不能后退
- [ ] 测试数据切片正确
- [ ] 测试Look-Ahead防护

**成果验证**：
- [ ] 时间只能向前推进
- [ ] Agent只能访问历史数据
- [ ] DecisionRecord.validate_data_timestamp() 正确工作
- [ ] 测试通过

**预计时间**：9小时

---

### Step 7: 集成到VectorBT引擎 【最终集成】

**目标**：将新系统集成到现有回测引擎

**工作内容**：

#### 7.1 重构vectorbt_engine.py (4小时)
- [ ] 移除原有的 `precompute_signals()` 方法
- [ ] 集成 `LayeredScheduler`
- [ ] 使用 `BacktestClock` 管理时间
- [ ] 使用 `TimeSliceManager` 提供数据

#### 7.2 实现信号缓存 (3小时)
- [ ] 创建 `Backtests/signal_cache.py`
- [ ] SQLite持久化
- [ ] 缓存键生成和查找
- [ ] 版本管理

#### 7.3 端到端测试 (3小时)
- [ ] 运行完整回测（AAPL, 250天）
- [ ] 验证时间控制
- [ ] 验证分层调度
- [ ] 验证反向传导

#### 7.4 性能优化 (2小时)
- [ ] 测量回测时间
- [ ] 优化瓶颈
- [ ] 验证5-10分钟目标

**成果验证**：
- [ ] 完整回测能运行
- [ ] 无Look-Ahead Bias
- [ ] 回测时间 <= 10分钟
- [ ] 结果可重现

**预计时间**：12小时

---

## 📊 总体时间规划

| Step | 内容 | 预计时间 | 累计时间 |
|------|------|---------|----------|
| 0 | ✅ 基础设施 | 已完成 | - |
| 1 | MacroAgent创建 | 9小时 | 9小时 |
| 2 | SectorAgent创建 | 5小时 | 14小时 |
| 3 | MetaAgent修改 | 7小时 | 21小时 |
| 4 | DecisionMaker层 | 6小时 | 27小时 |
| 5 | LayeredScheduler | 12小时 | 39小时 |
| 6 | 时间控制 | 9小时 | 48小时 |
| 7 | VectorBT集成 | 12小时 | 60小时 |

**总计**：约 60 小时（8个工作日）

---

## 🎯 里程碑

### 里程碑1: Agent层完成 (Step 1-3)
- [ ] MacroAgent可用
- [ ] SectorAgent可用
- [ ] MetaAgent支持上下文
- **预计**：3个工作日

### 里程碑2: 决策层完成 (Step 4-5)
- [ ] 3个DecisionMaker可用
- [ ] Scheduler能正确调度
- [ ] 反向传导机制工作
- **预计**：2.5个工作日

### 里程碑3: 时间控制完成 (Step 6)
- [ ] BacktestClock实现
- [ ] TimeSliceManager实现
- [ ] Look-Ahead防护生效
- **预计**：1.5个工作日

### 里程碑4: 完整系统 (Step 7)
- [ ] 集成到VectorBT
- [ ] 端到端测试通过
- [ ] 性能达标
- **预计**：1.5个工作日

---

## 🔀 灵活实施选项

### 选项A: 渐进式（推荐）✅
按Step 1→2→3→4→5→6→7顺序实施
- 优点：风险低，每步都可测试
- 缺点：时间较长

### 选项B: 并行式
Step 1-3 并行 → Step 4-5 并行 → Step 6-7
- 优点：快速
- 缺点：风险高，可能需要大量返工

### 选项C: MVP优先
先做 Step 1 + Step 4 + Step 5 的简化版
- 优点：快速看到效果
- 缺点：功能不完整

---

## ✅ 每步完成标准

每个Step完成后必须满足：

1. **代码完成**
   - [ ] 所有文件创建/修改完成
   - [ ] 代码符合项目规范
   - [ ] 有清晰的文档字符串

2. **测试通过**
   - [ ] 单元测试通过（100%）
   - [ ] 集成测试通过
   - [ ] 边界情况测试

3. **文档更新**
   - [ ] 更新README（如需要）
   - [ ] 添加使用示例
   - [ ] 注释清晰

4. **验证成功**
   - [ ] 功能验证通过
   - [ ] 性能符合预期
   - [ ] 无回归问题

---

## 🚀 开始前准备

### 环境检查
- [ ] Python环境配置正确
- [ ] 所有依赖已安装
- [ ] 测试框架可用

### 代码分支
建议为每个Step创建分支：
```bash
git checkout -b feature/step1-macro-agent
# 完成后
git checkout main
git merge feature/step1-macro-agent
```

---

## 📝 下一步行动

**请选择你想开始的Step**：

1. **Step 1: MacroAgent创建** (推荐从这里开始)
   - 最核心的基础组件
   - 独立性强，可以单独开发和测试

2. **Step 2: SectorAgent创建**
   - 可以在Step 1完成后立即开始

3. **Step 3: MetaAgent修改**
   - 需要Step 1完成后才能充分测试

4. **其他选项**
   - 先做简化版MVP？
   - 先实现时间控制（Step 6）？
   - 你的其他想法？

**请告诉我你想从哪个Step开始，我会提供详细的实施指导！** 🎯

---

**文档版本**: v1.0  
**创建日期**: 2025-10-30  
**最后更新**: 2025-10-30
