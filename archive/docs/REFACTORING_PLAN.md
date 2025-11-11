# Agent 重构计划 - 方案 A

## 目标

将当前的 MCP 混合架构重构为清晰的分层架构：
- **Core Layer**: 纯业务逻辑（易测试）
- **Orchestration Layer**: 协调层
- **MCP Layer**: 协议包装层（可选）
- **Utils Layer**: 工具类

## 重构原则

1. **向后兼容**：现有代码继续工作
2. **渐进式**：分步骤重构，每步都可测试
3. **测试驱动**：先修改测试，再修改实现
4. **保持功能**：不改变业务逻辑

---

## Phase 1: 准备和基础重构

### ✅ Step 1.1: 创建新文件结构

```
Agents/
  ├─ core/              # ✅ 已创建
  ├─ orchestration/     # ✅ 已创建
  ├─ mcp/               # ✅ 已创建
  └─ utils/             # ✅ 已创建
```

### ✅ Step 1.2: 创建 BaseAgent（纯业务逻辑基类）

- ✅ `core/base_agent.py` - 已创建
- 功能：LLM 管理、缓存、日志
- 特点：支持 LLM Mock

### Step 1.3: 移动 llm_config.py 到 utils/

```bash
mv Agents/llm_config.py Agents/utils/llm_config.py
# 更新 imports
```

### Step 1.4: 创建 core/macro_agent.py（新版）

- 继承 `BaseAgent`
- 纯业务逻辑
- 支持 LLM Mock
- 保持 API 兼容

### Step 1.5: 创建向后兼容的 Adapter

在 `Agents/macro_agent.py` 中：
```python
# 导入新的实现
from Agents.core.macro_agent import MacroAgent as CoreMacroAgent

# 创建兼容层（继承 BaseMCPAgent）
class MacroAgent(BaseMCPAgent):
    def __init__(self, ...):
        super().__init__(...)
        self._core = CoreMacroAgent(...)  # 组合，不是继承
```

### Step 1.6: 更新测试

- 修改测试使用 MockLLM
- 确保所有测试通过

---

## Phase 2: 迁移所有 Core Agents

### Step 2.1: 迁移 SectorAgent

- 创建 `core/sector_agent.py`
- 更新 `Agents/sector_agent.py` 为 Adapter

### Step 2.2: 迁移 TechnicalAgent 和 NewsAgent（可选）

根据实际需求决定

---

## Phase 3: 重组协调层

### Step 3.1: 移动 MetaAgent

```
mv Agents/meta_agent.py Agents/orchestration/meta_agent.py
```

### Step 3.2: 移动 DecisionMakers

```
mv Agents/decision_makers.py Agents/orchestration/decision_makers.py
```

### Step 3.3: 移动 LayeredScheduler

```
mv Agents/layered_scheduler.py Agents/orchestration/layered_scheduler.py
```

---

## Phase 4: 创建 MCP Facade（可选）

如果需要真正的 MCP Server：

```
Agents/mcp/
  ├─ base_server.py       # 从 base_mcp_agent.py 重命名
  ├─ macro_server.py      # MacroAgent 的 MCP 包装
  └─ sector_server.py     # SectorAgent 的 MCP 包装
```

---

## Phase 5: 清理和文档

### Step 5.1: 更新所有 imports

### Step 5.2: 更新 README

### Step 5.3: 添加迁移指南

---

## 当前进度

- ✅ Phase 1, Step 1.1: 创建文件结构
- ✅ Phase 1, Step 1.2: 创建 BaseAgent
- 🔄 Phase 1, Step 1.3: 移动 llm_config.py（下一步）

---

## 测试策略

每个步骤后运行：
```bash
# 运行新增的测试
pytest Tests/unit/test_core_agents.py -v

# 运行所有测试（确保向后兼容）
pytest Tests/unit/ -v
```

---

## 回滚策略

每个 Phase 完成后提交 git：
```bash
git add -A
git commit -m "Phase X: <description>"
```

如果出现问题：
```bash
git revert HEAD
```
