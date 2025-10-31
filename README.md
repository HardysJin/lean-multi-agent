# LEAN Multi-Agent Trading System

基于 LEAN 引擎的多智能体量化交易系统，集成多时间尺度记忆系统、LangChain工具调用和实时新闻情绪分析。

## 🌟 核心特性

### ✅ 已完成功能

#### 1. Memory System - 多时间尺度分层记忆系统
- **SQL存储**: SQLite数据库存储所有交易决策（`Data/sql/trading_memory.db`）
- **Vector存储**: ChromaDB向量数据库支持语义搜索（`Data/vector_db/chroma/`）
- **5个时间尺度**: REALTIME(5分钟), EXECUTION(1小时), TACTICAL(1天), CAMPAIGN(1周), STRATEGIC(30天)
- **跨会话持久化**: 系统重启后自动恢复历史数据
- **自动存储**: 所有决策自动存入Memory，无需手动调用
- **测试覆盖**: 88个测试全部通过 ✅

#### 2. MCP (Model Context Protocol) Agent架构
- **BaseMCPAgent**: 统一的Agent基类，支持工具和资源管理
- **TechnicalAnalysisAgent**: 技术分析专家
  - 计算技术指标 (RSI, MACD, Bollinger Bands等)
  - 生成交易信号
  - 检测图表形态
  - 识别支撑/阻力位
- **NewsAgent**: 新闻情绪分析专家
  - NewsAPI集成获取实时新闻
  - LLM驱动的情绪分析（正面/负面/中性）
  - 生成综合情绪报告
  - 5分钟内存缓存避免重复API调用
- **MetaAgent**: 协调者，作为MCP Client
  - LangChain Tool Calling自动选择和调用工具
  - Memory System默认启用
  - 综合多个专家意见形成最终决策
- **测试覆盖**: 116个测试全部通过 ✅

#### 3. 统一LLM配置系统
- **多提供商支持**: OpenAI, Anthropic Claude, DeepSeek, Ollama (本地)
- **LangChain集成**: 统一的接口，方便切换模型
- **环境变量配置**: 自动从`.env`读取API密钥
- **优先级检测**: OPENAI > CLAUDE > DEEPSEEK > OLLAMA
- **默认配置**: 
  - Provider: OpenAI
  - Model: gpt-4o-mini (快速且经济)
  - Temperature: 0.0 (确定性决策)
- **测试覆盖**: 31个测试全部通过 ✅

#### 4. 真实数据集成
- **NewsAPI**: 获取实时新闻（需要API key）
- **Yahoo Finance**: 获取股票价格和技术指标
- **LEAN引擎**: 完整的回测框架支持

#### 5. 综合测试系统
- **test_comprehensive_system.py**: 统一测试文件
  - LangChain Tool Calling验证
  - Memory持久化验证
  - Multi-Agent协作验证
  - 跨会话数据恢复验证
- **总测试数**: 233个测试全部通过 ✅

### 🔄 进行中功能

#### 1. Memory自动集成到所有Agents
- ✅ MetaAgent: 决策自动存储
- ⏳ NewsAgent: 需要在获取新闻时自动存入Memory
- ⏳ TechnicalAgent: 需要在计算指标时自动存入Memory

#### 2. LEAN引擎集成
- ✅ SmartAlgorithm基类（自动数据管理）
- ⏳ 将Multi-Agent系统集成到LEAN回测
- ⏳ 实时交易支持

### 📋 待完成功能

#### 短期任务
1. **NewsAgent Memory集成**
   - 在`_fetch_news()`中自动存储新闻到Memory
   - 从Memory查询历史新闻，避免重复API调用
   - 实现新闻去重机制

2. **TechnicalAgent Memory集成**
   - 存储技术指标计算结果
   - 缓存历史计算避免重复

3. **Memory维护工具**
   - 数据清理：删除过期决策
   - 数据统计：决策质量分析
   - 数据导出：CSV/JSON格式

4. **Dashboard可视化**
   - 实时决策监控
   - 历史回测结果可视化
   - Memory数据统计图表

#### 中期任务
1. **基于Memory的智能推荐**
   - 从历史相似情况推荐决策
   - 学习成功/失败案例
   - 动态调整策略参数

2. **决策回测与评分**
   - 跟踪每个决策的执行结果
   - 计算盈亏和胜率
   - Agent质量评估

3. **多时间尺度聚合**
   - 实现从低时间尺度向高时间尺度的信息聚合
   - 战略层决策基于战术层历史
   - 自动触发不同尺度的决策

4. **完整LEAN集成**
   - Multi-Agent作为LEAN Algorithm
   - 实时市场数据接入
   - 订单执行和管理

#### 长期任务
1. **自适应Memory管理**
   - 根据数据重要性自动清理
   - 压缩历史数据
   - 增量学习

2. **更多Specialist Agents**
   - FundamentalAgent: 基本面分析
   - SentimentAgent: 社交媒体情绪
   - RiskAgent: 风险管理
   - PositionAgent: 仓位管理

3. **多策略支持**
   - Memory隔离（不同策略独立存储）
   - 策略组合与切换
   - 策略回测比较

4. **生产环境部署**
   - Docker容器化
   - API服务
   - 监控和告警
   - 高可用架构

## 🚀 快速开始

### 环境配置

```bash
# 1. 克隆项目
git clone https://github.com/HardysJin/lean-multi-agent.git
cd lean-multi-agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，添加你的API密钥：
# OPENAI_API_KEY=sk-xxx...
# NEWS_API_KEY=xxx...
```

### 运行综合测试

```bash
# 运行完整系统测试（包含LLM调用，需要API key）
python Tests/test_comprehensive_system.py

# 输出示例：
# ✅ 所有测试通过！
# - LangChain Tool Calling ✅
# - Memory System ✅
# - Multi-Agent协作 ✅
# - 跨会话恢复 ✅
```

### 快速验证Memory System

```bash
# 验证Memory自动启用
python -c "
from Agents.meta_agent import MetaAgent
meta = MetaAgent()  # Memory自动启用
print('Memory启用:', meta.state_manager is not None)
"

# 查看Memory数据库
sqlite3 Data/sql/trading_memory.db "SELECT COUNT(*) FROM decisions;"
```

### 使用Multi-Agent系统

```python
import asyncio
from Agents.meta_agent import MetaAgent
from Agents.technical_agent import TechnicalAnalysisAgent
from Agents.news_agent import NewsAgent

async def main():
    # 1. 创建MetaAgent（Memory自动启用）
    meta = MetaAgent()
    
    # 2. 连接specialist agents
    technical = TechnicalAnalysisAgent()
    await meta.connect_to_agent(
        agent_name="technical",
        agent_instance=technical,
        description="Technical analysis specialist"
    )
    
    news = NewsAgent()
    await meta.connect_to_agent(
        agent_name="news",
        agent_instance=news,
        description="News sentiment specialist"
    )
    
    # 3. 分析并决策（自动调用工具、自动存储到Memory）
    decision = await meta.analyze_and_decide(
        symbol="AAPL",
        query="综合技术分析和新闻情绪，给出交易建议"
    )
    
    # 4. 查看决策
    print(f"决策: {decision.action}")
    print(f"信心: {decision.conviction}/10")
    print(f"推理: {decision.reasoning}")
    print(f"工具调用: {len(decision.tool_calls)} 次")
    
    # 决策已自动存入Memory！

asyncio.run(main())
```

## 📁 项目结构

```
lean-multi-agent/
├── Algorithm/              # LEAN交易策略
│   ├── SmartAlgorithm.py  # 智能算法基类
│   └── MultiAgent/        # 多智能体策略（待完成）
├── Agents/                 # MCP Agent实现
│   ├── base_mcp_agent.py  # Agent基类
│   ├── meta_agent.py      # Meta Agent (协调者)
│   ├── technical_agent.py # 技术分析专家
│   ├── news_agent.py      # 新闻情绪专家
│   └── llm_config.py      # 统一LLM配置
├── Memory/                 # Memory System
│   ├── state_manager.py   # 状态管理器
│   ├── sql_store.py       # SQL存储
│   ├── vector_store.py    # Vector存储
│   └── schemas.py         # 数据结构
├── Tests/                  # 测试文件
│   ├── test_comprehensive_system.py  # 综合测试
│   ├── test_memory/       # Memory测试 (88个)
│   ├── test_agents/       # Agent测试 (116个)
│   └── test_llm_config/   # LLM配置测试 (31个)
├── Data/                   # 数据存储
│   ├── sql/               # SQL数据库
│   │   └── trading_memory.db
│   ├── vector_db/         # Vector数据库
│   │   └── chroma/
│   └── cache/             # 缓存数据
├── Utils/                  # 工具脚本
│   └── download_data.py   # 数据下载工具
├── Results/                # 回测结果
├── Configs/                # 配置文件
└── Lean/                   # LEAN引擎（子模块）
```

## 🔧 配置说明

### LLM配置

**方法1: 环境变量（推荐）**

编辑 `.env` 文件：
```bash
# 使用 OpenAI (默认)
OPENAI_API_KEY=sk-xxx...

# 或使用 Claude
ANTHROPIC_API_KEY=sk-ant-xxx...

# 或使用 DeepSeek
DEEPSEEK_API_KEY=sk-xxx...
```

**方法2: 代码中指定**

```python
from Agents.llm_config import LLMConfig, LLMProvider

# 使用Claude
llm_config = LLMConfig(
    provider=LLMProvider.CLAUDE,
    model="claude-3-5-sonnet-20241022",
    temperature=0.7
)

meta = MetaAgent(llm_config=llm_config)
```

**方法3: 修改默认配置**

编辑 `Agents/llm_config.py` 第59-64行：
```python
DEFAULT_MODELS = {
    LLMProvider.OPENAI: "gpt-4o-mini",  # 改这里
    LLMProvider.CLAUDE: "claude-3-5-sonnet-20241022",
    LLMProvider.DEEPSEEK: "deepseek-chat",
    LLMProvider.OLLAMA: "llama3.1:8b",
}
```

### Memory配置

Memory System默认启用，数据存储在：
- SQL: `Data/sql/trading_memory.db`
- Vector: `Data/vector_db/chroma/`

如需禁用Memory：
```python
meta = MetaAgent(enable_memory=False)
```

## 🧪 测试

### 运行所有测试

```bash
# Memory System测试
pytest Tests/test_memory/ -v

# Agent测试
pytest Tests/test_agents/ -v

# LLM配置测试
pytest Tests/test_llm_config/ -v

# 综合系统测试
python Tests/test_comprehensive_system.py
```

### 测试统计

| 模块 | 测试数 | 状态 |
|------|--------|------|
| Memory System | 88 | ✅ 全部通过 |
| MCP Agents | 116 | ✅ 全部通过 |
| LLM Config | 31 | ✅ 全部通过 |
| NewsAgent | 29 | ✅ 全部通过 |
| **总计** | **233** | **✅ 全部通过** |

## 📊 Memory System使用

### 查询决策历史

```python
from Memory.state_manager import MultiTimeframeStateManager

# 初始化
state_manager = MultiTimeframeStateManager(
    sql_db_path="Data/sql/trading_memory.db",
    vector_db_path="Data/vector_db/chroma"
)

# 查询决策
decisions = state_manager.sql_store.query_decisions(
    symbol="AAPL",
    start_time=datetime.now() - timedelta(days=7)
)

# 语义搜索
results = state_manager.vector_store.query_by_timeframe(
    timeframe=Timeframe.TACTICAL,
    query_text="positive news about AAPL",
    n_results=5
)
```

### 使用SQL直接查询

```bash
# 进入数据库
sqlite3 Data/sql/trading_memory.db

# 查看所有决策
SELECT * FROM decisions LIMIT 10;

# 统计决策类型
SELECT action, COUNT(*) as count 
FROM decisions 
GROUP BY action;

# 按symbol查询
SELECT * FROM decisions 
WHERE symbol = 'AAPL' 
ORDER BY timestamp DESC 
LIMIT 10;
```

## 📖 文档

- [项目结构说明](PROJECT_STRUCTURE.md)
- [SmartAlgorithm使用指南](SMART_ALGORITHM_GUIDE.md)
- [测试整合说明](TESTING_CONSOLIDATED.md)
- [Memory集成成功报告](MEMORY_INTEGRATION_SUCCESS.md)

## 🤝 贡献

欢迎贡献！请查看待完成功能列表，选择感兴趣的任务。

## 📝 许可证

MIT License

## 🔗 相关链接

- [QuantConnect LEAN](https://github.com/QuantConnect/Lean)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Model Context Protocol](https://modelcontextprotocol.io/)

---

**最后更新**: 2025-10-28  
**版本**: v0.2.0 - Multi-Agent + Memory System集成完成
