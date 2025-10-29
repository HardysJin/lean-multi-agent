# 安装与快速开始指南# 安装指南



本指南涵盖 Lean Multi-Agent Trading System 的安装、配置和快速使用方法。## 快速安装



## 📦 快速安装### 1. 使用 pip 安装（推荐）



### 开发模式安装（推荐）#### 开发模式安装（推荐用于开发）

```bash

```bash# 克隆仓库

# 1. 克隆仓库git clone https://github.com/HardysJin/lean-multi-agent.git

git clone https://github.com/HardysJin/lean-multi-agent.gitcd lean-multi-agent

cd lean-multi-agent

# 创建虚拟环境（推荐）

# 2. 创建虚拟环境（推荐）python -m venv venv

python -m venv venvsource venv/bin/activate  # Linux/Mac

source venv/bin/activate  # Linux/Mac# 或 Windows: venv\Scripts\activate

# Windows: venv\Scripts\activate

# 开发模式安装（可编辑）

# 3. 开发模式安装（代码修改立即生效）pip install -e .

pip install -e .```



# 4. 验证安装#### 标准安装

python test_installation.py```bash

```pip install git+https://github.com/HardysJin/lean-multi-agent.git

```

### 其他安装方式

#### 安装额外依赖

#### 标准安装```bash

```bash# 安装开发工具

# 从 GitHub 直接安装pip install -e ".[dev]"

pip install git+https://github.com/HardysJin/lean-multi-agent.git

# 安装文档工具

# 从本地源码安装pip install -e ".[docs]"

pip install .

```# 安装所有可选依赖

pip install -e ".[dev,docs]"

#### 安装可选依赖```

```bash

# 开发工具（black, flake8, mypy等）### 2. 使用 setup.py 安装（传统方式）

pip install -e ".[dev]"

```bash

# 文档工具# 标准安装

pip install -e ".[docs]"python setup.py install



# 所有可选依赖# 开发模式安装

pip install -e ".[dev,docs]"python setup.py develop

``````



#### 使用 setup.py（传统方式）### 3. 从源码构建

```bash

python setup.py install      # 标准安装```bash

python setup.py develop      # 开发模式# 构建分发包

```python -m build



#### 构建分发包# 安装构建的包

```bashpip install dist/lean_multi_agent-0.1.0-py3-none-any.whl

# 安装构建工具```

pip install build

## 依赖说明

# 构建

python -m build### 核心依赖

- **Python**: >= 3.10

# 安装构建的包- **MCP**: Model Context Protocol SDK

pip install dist/lean_multi_agent-0.1.0-py3-none-any.whl- **ChromaDB**: 向量数据库

```- **LangChain**: LLM框架

- **Pandas/NumPy**: 数据处理

## ⚙️ 环境配置

### LLM提供商（至少需要一个）

### 1. 配置 API Keys- **OpenAI**: gpt-4o-mini（推荐，默认）

- **Anthropic**: Claude系列

```bash- **DeepSeek**: DeepSeek系列

# 复制模板- **Ollama**: 本地模型（免费）

cp .env.template .env

### 可选依赖

# 编辑 .env 文件，添加你的 API keys- **NewsAPI**: 实时新闻获取（需要API key）

```- **pytest**: 单元测试

- **black**: 代码格式化

`.env` 文件内容：

```bash## 环境配置

# OpenAI (推荐，默认)

OPENAI_API_KEY=sk-your-openai-key### 1. 创建 .env 文件



# Anthropic Claude (可选)```bash

ANTHROPIC_API_KEY=sk-ant-your-anthropic-keycp .env.template .env

```

# DeepSeek (可选)

DEEPSEEK_API_KEY=sk-your-deepseek-key### 2. 配置 API Keys



# NewsAPI (可选，用于新闻情绪分析)编辑 `.env` 文件：

NEWS_API_KEY=your-news-api-key

``````bash

# OpenAI (推荐)

### 2. 数据目录OPENAI_API_KEY=sk-your-openai-key



系统会自动创建以下目录：# Anthropic Claude (可选)

```ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

Data/

├── sql/              # SQLite数据库# DeepSeek (可选)

│   └── trading_memory.dbDEEPSEEK_API_KEY=sk-your-deepseek-key

├── vector_db/        # ChromaDB向量数据库

│   └── chroma/# NewsAPI (可选，用于新闻情绪分析)

└── cache/           # 缓存文件NEWS_API_KEY=your-news-api-key

``````



## 🚀 快速开始### 3. 数据目录



### 推荐：使用命名空间导入系统会自动创建以下目录：

```

```pythonData/

import lean_multi_agent as lma├── sql/              # SQLite数据库

│   └── trading_memory.db

# 查看包信息├── vector_db/        # ChromaDB向量数据库

lma.info()│   └── chroma/

└── cache/           # 缓存文件

# 创建 agents```

meta = lma.MetaAgent()

tech = lma.TechnicalAnalysisAgent()## 验证安装



# 使用 Memory### 1. 运行测试

record = lma.DecisionRecord(...)

``````bash

# 运行所有单元测试

**为什么推荐？**pytest Tests/unit/ -v

- ✅ 避免与其他包的命名冲突

- ✅ 代码更清晰、可读性好# 运行特定模块测试

- ✅ 符合 Python 社区最佳实践pytest Tests/unit/test_meta_agent.py -v



### 示例 1: 创建简单的交易机器人# 运行测试并生成覆盖率报告

pytest Tests/unit/ --cov=Agents --cov=Memory --cov-report=html

```python```

import lean_multi_agent as lma

import asyncio### 2. 快速测试脚本



async def main():创建 `test_installation.py`：

    # 创建 Meta Agent（自动启用 Memory）

    meta = lma.MetaAgent()```python

    #!/usr/bin/env python

    # 创建专家 agents"""测试安装是否成功"""

    technical = lma.TechnicalAnalysisAgent()

    def test_imports():

    # 连接 agents    """测试核心模块导入"""

    await meta.connect_to_agent("technical", technical, "技术分析专家")    try:

            from Agents.meta_agent import MetaAgent

    # 执行分析        from Agents.technical_agent import TechnicalAnalysisAgent

    result = await meta.execute_tool(        from Agents.news_agent import NewsAgent

        "technical",        from Memory.state_manager import MultiTimeframeStateManager

        "calculate_indicators",        from Memory.schemas import DecisionRecord, Timeframe

        {"symbol": "AAPL"}        print("✓ 所有核心模块导入成功")

    )        return True

        except ImportError as e:

    print(f"技术指标: {result}")        print(f"✗ 导入失败: {e}")

        return False

if __name__ == "__main__":

    asyncio.run(main())def test_memory_system():

```    """测试Memory System"""

    try:

### 示例 2: 使用便捷函数        from Memory.state_manager import create_state_manager

        state_manager = create_state_manager(

```python            sql_db_path="test_memory.db",

import lean_multi_agent as lma            vector_db_path="test_vector_db"

        )

# 使用便捷创建函数        print("✓ Memory System初始化成功")

meta = lma.create_meta_agent(enable_memory=True)        

tech = lma.create_technical_agent()        # 清理测试文件

news = lma.create_news_agent(news_api_key="your-key")        import os

        import shutil

# 查看可用组件        if os.path.exists("test_memory.db"):

print(f"可用 Agents: {lma.list_agents()}")            os.remove("test_memory.db")

print(f"Memory 组件: {lma.list_memory_components()}")        if os.path.exists("test_vector_db"):

```            shutil.rmtree("test_vector_db")

        

### 示例 3: 直接导入（向后兼容）        return True

    except Exception as e:

```python        print(f"✗ Memory System测试失败: {e}")

# 方式 1: 通过命名空间（推荐）        return False

import lean_multi_agent as lma

meta = lma.MetaAgent()def test_agents():

    """测试Agents创建"""

# 方式 2: 直接导入（在项目内部使用）    try:

from Agents.meta_agent import MetaAgent        from Agents.meta_agent import MetaAgent

from Memory.schemas import DecisionRecord        from Agents.technical_agent import TechnicalAnalysisAgent

meta = MetaAgent()        

        meta = MetaAgent(enable_memory=False)

# 两种方式都支持！        technical = TechnicalAnalysisAgent()

```        

        print("✓ Agents创建成功")

### 示例 4: 完整的多 Agent 工作流        return True

    except Exception as e:

```python        print(f"✗ Agents测试失败: {e}")

import lean_multi_agent as lma        return False

import asyncio

import osif __name__ == "__main__":

    print("=" * 50)

async def analyze_stock(symbol: str):    print("LEAN Multi-Agent 安装验证")

    """分析股票并生成决策"""    print("=" * 50)

        

    # 1. 创建 Meta Agent    results = []

    meta = lma.MetaAgent()    results.append(("模块导入", test_imports()))

        results.append(("Memory System", test_memory_system()))

    # 2. 创建专家 agents    results.append(("Agents创建", test_agents()))

    technical = lma.TechnicalAnalysisAgent()    

    news = lma.NewsAgent(news_api_key=os.getenv("NEWS_API_KEY"))    print("\n" + "=" * 50)

        print("测试结果汇总:")

    # 3. 连接 agents    print("=" * 50)

    await meta.connect_to_agent("technical", technical, "技术分析专家")    

    await meta.connect_to_agent("news", news, "新闻情绪分析专家")    for name, result in results:

            status = "✓ 通过" if result else "✗ 失败"

    # 4. 收集技术分析        print(f"{name}: {status}")

    indicators = await meta.execute_tool(    

        "technical",    all_passed = all(r[1] for r in results)

        "calculate_indicators",    print("\n" + "=" * 50)

        {"symbol": symbol}    if all_passed:

    )        print("🎉 所有测试通过！安装成功！")

        else:

    signals = await meta.execute_tool(        print("❌ 部分测试失败，请检查错误信息")

        "technical",    print("=" * 50)

        "generate_signals",```

        {"symbol": symbol}

    )运行验证：

    ```bash

    # 5. 收集新闻分析python test_installation.py

    news_data = await meta.execute_tool(```

        "news",

        "get_latest_news",## 常见问题

        {"symbol": symbol, "query": symbol}

    )### Q1: ImportError: No module named 'Agents'

    

    # 6. 生成综合决策**解决方案**：

    print(f"\n{'='*60}")```bash

    print(f"{symbol} 综合分析")# 确保使用开发模式安装

    print(f"{'='*60}")pip install -e .

    print(f"\n📊 技术指标:")

    print(f"  RSI: {indicators['indicators']['rsi']['value']:.2f}")# 或者添加项目路径到PYTHONPATH

    print(f"  MACD: {indicators['indicators']['macd']['histogram']:.2f}")export PYTHONPATH="${PYTHONPATH}:/path/to/lean-multi-agent"

    ```

    print(f"\n📈 交易信号:")

    print(f"  动作: {signals['action']}")### Q2: pytest找不到模块

    print(f"  信心度: {signals['conviction']}/10")

    **解决方案**：

    print(f"\n📰 新闻情绪:")确保 `pytest.ini` 包含：

    print(f"  新闻数量: {len(news_data.get('articles', []))}")```ini

    [pytest]

    # 7. 查看工具调用历史pythonpath = .

    print(f"\n📝 工具调用历史: {len(meta.get_tool_call_history())} 次")```

    

    return {### Q3: ChromaDB初始化失败

        "symbol": symbol,

        "technical": signals,**解决方案**：

        "news_count": len(news_data.get('articles', []))```bash

    }# 安装sqlite3开发库 (Ubuntu/Debian)

sudo apt-get install libsqlite3-dev

if __name__ == "__main__":

    result = asyncio.run(analyze_stock("AAPL"))# 或者 (macOS)

    print(f"\n✅ 分析完成: {result}")brew install sqlite3

```

# 重新安装chromadb

## ✅ 验证安装pip uninstall chromadb

pip install chromadb

### 方法 1: 运行测试脚本```



```bash### Q4: LangChain版本冲突

# 项目提供的安装测试

python test_installation.py**解决方案**：

``````bash

# 升级到最新版本

预期输出：pip install --upgrade langchain langchain-core langchain-openai langchain-anthropic

``````

==================================================

LEAN Multi-Agent 安装验证## 卸载

==================================================

✓ 所有核心模块导入成功```bash

✓ Memory System初始化成功# 使用pip卸载

✓ Agents创建成功pip uninstall lean-multi-agent



==================================================# 手动清理数据（可选）

测试结果汇总:rm -rf Data/sql/trading_memory.db

==================================================rm -rf Data/vector_db/chroma/

模块导入: ✓ 通过```

Memory System: ✓ 通过

Agents创建: ✓ 通过## 下一步



==================================================- 查看 [README.md](README.md) 了解系统架构和功能

🎉 所有测试通过！安装成功！- 查看 [examples/](examples/) 目录的示例代码

==================================================- 运行测试了解系统能力：`pytest Tests/unit/ -v`

```- 阅读各个Agent的文档：

  - [Meta Agent](Agents/meta_agent.py)

### 方法 2: 运行单元测试  - [Technical Agent](Agents/technical_agent.py)

  - [News Agent](Agents/news_agent.py)

```bash
# 运行所有单元测试（233个测试）
pytest Tests/unit/ -v

# 简要输出
pytest Tests/unit/ --tb=short

# 运行特定模块
pytest Tests/unit/test_meta_agent.py -v

# 生成覆盖率报告
pytest Tests/unit/ --cov=Agents --cov=Memory --cov-report=html
```

### 方法 3: 快速验证

```bash
# 验证导入
python -c "import lean_multi_agent as lma; lma.info()"

# 验证命名空间
python -c "import lean_multi_agent as lma; print(lma.list_agents())"

# 测试创建 agent
python -c "import lean_multi_agent as lma; m=lma.MetaAgent(enable_memory=False); print('OK')"
```

## 📚 依赖说明

### 核心依赖
- **Python**: >= 3.10
- **MCP**: Model Context Protocol SDK
- **ChromaDB**: 向量数据库（用于语义搜索）
- **LangChain**: 统一 LLM 接口
- **Pandas/NumPy**: 数据处理

### LLM 提供商（至少需要一个）
- **OpenAI**: gpt-4o-mini（推荐，默认）
- **Anthropic**: Claude 系列
- **DeepSeek**: DeepSeek 系列
- **Ollama**: 本地模型（免费）

### 可选依赖
- **NewsAPI**: 实时新闻获取（需要 API key）
- **pytest**: 单元测试框架
- **black**: 代码格式化工具

## 🔧 常见问题

### Q1: ImportError: No module named 'Agents'

**原因**: 包未正确安装或 PYTHONPATH 未设置

**解决方案**:
```bash
# 方案 1: 使用开发模式安装（推荐）
cd /path/to/lean-multi-agent
pip install -e .

# 方案 2: 手动设置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/lean-multi-agent"

# 验证
python -c "from Agents.meta_agent import MetaAgent; print('OK')"
```

### Q2: pytest 找不到模块

**原因**: pytest 配置中缺少 pythonpath

**解决方案**:
确保 `pytest.ini` 包含：
```ini
[pytest]
pythonpath = .
```

### Q3: ChromaDB 初始化失败

**错误信息**: `sqlite3.OperationalError` 或 ChromaDB 相关错误

**解决方案**:
```bash
# Ubuntu/Debian
sudo apt-get install libsqlite3-dev

# macOS
brew install sqlite3

# 重新安装 chromadb
pip uninstall chromadb
pip install chromadb
```

### Q4: LangChain 版本冲突

**错误信息**: 版本不兼容警告

**解决方案**:
```bash
# 升级相关包到最新版本
pip install --upgrade langchain langchain-core langchain-openai langchain-anthropic
```

### Q5: 命名空间导入失败

**错误信息**: `ModuleNotFoundError: No module named 'lean_multi_agent'`

**解决方案**:
```bash
# 重新安装包
pip uninstall lean-multi-agent
pip install -e .

# 验证
python -c "import lean_multi_agent as lma; print(lma.__version__)"
```

## 🗑️ 卸载

```bash
# 卸载包
pip uninstall lean-multi-agent

# 清理数据（可选）
rm -rf Data/sql/trading_memory.db
rm -rf Data/vector_db/chroma/
rm -rf Data/cache/

# 清理构建文件（可选）
rm -rf build/ dist/ *.egg-info
rm -rf **/__pycache__
```

## 📖 更多资源

### 文档
- [README.md](../README.md) - 项目介绍和功能概述
- [SETUP_GUIDE.md](../SETUP_GUIDE.md) - 详细的 setup.py 使用指南

### 示例代码
- `examples/external_usage/simple_import_test.py` - 导入测试
- `examples/external_usage/namespace_comparison.py` - 命名空间对比
- `examples/llm_config_usage.py` - LLM 配置示例

### API 文档
- [Agents/meta_agent.py](../Agents/meta_agent.py) - Meta Agent API
- [Agents/technical_agent.py](../Agents/technical_agent.py) - Technical Agent API
- [Agents/news_agent.py](../Agents/news_agent.py) - News Agent API
- [Memory/state_manager.py](../Memory/state_manager.py) - Memory System API

## 🎯 下一步

1. ✅ **配置环境**: 设置 API keys 在 `.env` 文件
2. ✅ **运行测试**: `pytest Tests/unit/ -v` 确保所有功能正常
3. ✅ **查看示例**: 浏览 `examples/` 目录学习使用方法
4. ✅ **创建你的第一个 Agent**: 参考上面的示例代码
5. ✅ **探索 Memory System**: 了解如何存储和检索交易决策

---

**当前状态**: 233/233 单元测试通过 ✅

**支持**: 如有问题，请在 [GitHub Issues](https://github.com/HardysJin/lean-multi-agent/issues) 提交
