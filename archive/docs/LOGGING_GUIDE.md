# 日志系统使用指南

## 🚀 快速开始

### 1. 最简单的用法（全局logger）

```python
from Utils.execution_logger import configure_logging, get_logger, LogLevel

# 配置全局logger（程序启动时调用一次）
configure_logging(
    level=LogLevel.INFO,
    enable_console=True,
    enable_file=True
)

# 在任何地方使用
logger = get_logger()
logger.log_decision(
    agent_name="meta_agent",
    symbol="AAPL",
    action="BUY",
    conviction=8.0,
    reasoning="Strong signals"
)
```

### 2. 推荐用法（依赖注入）

```python
from Utils.execution_logger import ExecutionLogger, LogLevel

# 创建logger
logger = ExecutionLogger(
    level=LogLevel.INFO,
    enable_console=True,
    enable_file=True
)

# 传递给Agent
agent = MetaAgent(execution_logger=logger)
```

### 3. 从配置文件加载（推荐生产环境）

```python
from Configs.logging_config import LoggerConfig

# 自动检测文件类型（.yaml/.json）
logger = LoggerConfig.load('Configs/logging.yaml')

# 或使用预设配置
from Configs.logging_config import get_preset_logger
logger = get_preset_logger('production')  # development/production/backtest/performance/silent
```

## 📊 日志级别

| 级别 | 用途 | 建议场景 |
|-----|------|---------|
| `DEBUG` | 详细调试信息（所有参数、返回值） | 开发、问题排查 |
| `INFO` | 重要业务信息（决策、工具调用）⭐ | 生产、回测（推荐默认） |
| `WARNING` | 警告信息（非致命问题） | 生产环境 |
| `ERROR` | 错误信息（需要关注） | 所有环境 |
| `CRITICAL` | 严重错误（系统崩溃） | 所有环境 |

## 🎯 常用日志方法

### 决策日志
```python
logger.log_decision(
    agent_name="meta_agent",
    symbol="AAPL",
    action="BUY",               # BUY/SELL/HOLD
    conviction=8.0,             # 1-10
    reasoning="技术分析显示强烈买入信号",
    timeframe="tactical",       # 可选
    backtest_date=datetime(...) # 可选：回测日期
)
```

### 工具调用日志
```python
logger.log_tool_call(
    agent_name="technical",
    tool_name="calculate_indicators",
    arguments={"symbol": "AAPL"},
    result={"rsi": 65.2},
    execution_time_ms=123.45,
    symbol="AAPL"
)
```

### 缓存命中日志
```python
logger.log_cache_hit(
    cache_type="signal_cache",
    key="AAPL_tactical_20241015",
    symbol="AAPL",
    saved_time_ms=5000.0  # 节省的时间
)
```

### 反向传导日志
```python
logger.log_escalation(
    from_timeframe="tactical",
    to_timeframe="strategic",
    trigger="market_shock",      # 触发原因
    impact_score=9.5,            # 影响分数
    symbol="AAPL",
    details={"price_drop": -5.2}
)
```

### 错误日志
```python
try:
    result = risky_operation()
except Exception as e:
    logger.log_error(
        agent_name="my_agent",
        error_message="操作失败",
        exception=e,
        details={"context": "..."}
    )
```

### 通用日志
```python
logger.debug(agent_name="agent", message="调试信息", details={...})
logger.info(agent_name="agent", message="重要信息", details={...})
logger.warning(agent_name="agent", message="警告", details={...})
```

## 🔍 查询和分析

### 查看执行轨迹
```python
# 查询特定股票的执行轨迹
trace = logger.get_execution_trace(
    symbol="AAPL",
    backtest_date=datetime(2024, 10, 15)
)

# 可视化显示
logger.visualize_trace(symbol="AAPL", backtest_date=datetime(2024, 10, 15))
```

### 性能统计
```python
# 打印性能摘要
logger.print_performance_summary()

# 获取性能数据
stats = logger.get_performance_summary()
# 返回：
# {
#   'technical.calculate_indicators': {
#     'count': 250,
#     'total_ms': 30000,
#     'avg_ms': 120,
#     'min_ms': 80,
#     'max_ms': 200
#   },
#   ...
# }
```

### 保存摘要报告
```python
logger.save_summary()  # 自动保存到 logs/summary_{session_id}.txt
logger.save_summary("custom_path.txt")  # 自定义路径
```

## ⚙️ 配置选项

### 编程方式配置
```python
logger = ExecutionLogger(
    level=LogLevel.INFO,         # 日志级别
    enable_console=True,         # 控制台输出
    enable_file=True,            # 文件输出
    enable_database=False,       # 数据库存储
    log_dir="Data/logs",         # 日志目录
    db_path="Data/logs/exec.db", # 数据库路径
    colored_console=True,        # 彩色输出
    session_id="my_session"      # 会话ID（可选）
)
```

### 配置文件方式（推荐）

创建 `Configs/logging.yaml`:
```yaml
level: INFO
outputs:
  console:
    enabled: true
    colored: true
  file:
    enabled: true
  database:
    enabled: false
```

加载：
```python
from Configs.logging_config import LoggerConfig
logger = LoggerConfig.load('Configs/logging.yaml')
```

### 预设配置（最快捷）
```python
from Configs.logging_config import get_preset_logger

# 开发环境：DEBUG级别，彩色输出
logger = get_preset_logger('development')

# 生产环境：INFO级别，无控制台，持久化
logger = get_preset_logger('production')

# 回测环境：完整日志，自动摘要
logger = get_preset_logger('backtest')

# 性能测试：只记录错误
logger = get_preset_logger('performance')

# 静默模式：只记录严重错误
logger = get_preset_logger('silent')
```

## 🔧 动态调整

### 运行时修改日志级别
```python
# 创建logger
logger = ExecutionLogger(level=LogLevel.INFO)

# 运行时调整为DEBUG
logger.set_level(LogLevel.DEBUG)

# 调整回INFO
logger.set_level(LogLevel.INFO)
```

## 🎨 输出样式

### 控制台输出（彩色）
```
10:23:45.123 | INFO     | meta_agent           | [decision]     | Decision: BUY AAPL (conviction=8)
10:23:45.234 | INFO     | technical            | [tool_call]    | Tool call: calculate_indicators (123.45ms)
10:23:45.345 | DEBUG    | cache                | [cache]        | Cache hit: signal_cache
10:23:45.456 | WARNING  | escalation           | [escalation]   | Escalation: tactical → strategic (trigger: market_shock)
10:23:45.567 | ERROR    | my_agent             | [error]        | Tool call failed: fetch_news
```

### 文件输出（JSON格式）
每行一个JSON对象，便于日志分析工具处理：
```json
{"timestamp": "2024-10-15T10:23:45.123456", "level": "INFO", "category": "decision", "agent_name": "meta_agent", ...}
```

## 📁 文件结构

```
Data/logs/
├── execution_20241015_102345.log      # 主日志文件
├── summary_20241015_102345.txt        # 摘要报告
└── execution.db                       # SQLite数据库（可选）
```

## 💡 最佳实践

### 1. 在MetaAgent中集成
```python
class MetaAgent:
    def __init__(self, ..., execution_logger=None):
        self.logger = execution_logger or get_logger()
    
    async def execute_tool(self, agent_name, tool_name, arguments):
        start_time = time.time()
        
        try:
            result = await actual_execution()
            execution_time = (time.time() - start_time) * 1000
            
            self.logger.log_tool_call(
                agent_name=agent_name,
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                execution_time_ms=execution_time
            )
            return result
        except Exception as e:
            self.logger.log_error(
                agent_name=agent_name,
                error_message=f"Tool call failed: {tool_name}",
                exception=e
            )
            raise
```

### 2. 回测场景
```python
# 回测开始前
logger = get_preset_logger('backtest')

# 回测循环
for date in trading_days:
    decision = await agent.analyze_and_decide(
        symbol=symbol,
        backtest_date=date  # ⭐ 传递回测日期
    )
    
    # 日志自动记录backtest_date

# 回测结束后
logger.visualize_trace(symbol="AAPL")
logger.print_performance_summary()
logger.save_summary()
```

### 3. 生产环境
```python
# 启动时
logger = get_preset_logger('production')

# 定期检查错误
error_logs = logger.get_execution_trace(category=LogCategory.ERROR)
if len(error_logs) > 10:
    send_alert("系统错误过多")
```

## 🐛 调试技巧

### 临时启用DEBUG
```python
# 保存原始级别
original_level = logger.level

# 临时启用DEBUG
logger.set_level(LogLevel.DEBUG)

# 执行需要调试的代码
problematic_function()

# 恢复
logger.set_level(original_level)
```

### 查看特定Agent的日志
```python
# 过滤特定agent
agent_logs = [log for log in logger.logs if log.agent_name == "technical"]
for log in agent_logs:
    print(log.to_console_string())
```

## 📝 环境变量配置

```bash
# 设置环境变量
export LOG_LEVEL=DEBUG
export LOG_CONSOLE=true
export LOG_FILE=true
export LOG_DATABASE=false

# Python中读取
import os
from Configs.logging_config import configure_from_env

logger = configure_from_env()
```

## 🎯 不同场景的推荐配置

| 场景 | 日志级别 | 控制台 | 文件 | 数据库 | 配置 |
|-----|---------|--------|-----|--------|-----|
| 开发调试 | DEBUG | ✅ 彩色 | ✅ | ❌ | `get_preset_logger('development')` |
| 本地回测 | INFO | ✅ 彩色 | ✅ | ✅ | `get_preset_logger('backtest')` |
| 生产交易 | INFO | ❌ | ✅ | ✅ | `get_preset_logger('production')` |
| 性能测试 | ERROR | ✅ | ❌ | ❌ | `get_preset_logger('performance')` |
| CI/CD | WARNING | ✅ | ✅ | ❌ | 自定义 |

## 📞 常见问题

**Q: 如何关闭日志？**
```python
logger.set_level(LogLevel.CRITICAL)  # 只记录严重错误
# 或
logger = get_preset_logger('silent')
```

**Q: 日志文件太大怎么办？**
```python
# 方案1: 提高日志级别
logger.set_level(LogLevel.INFO)  # 从DEBUG改为INFO

# 方案2: 手动清理旧日志
import os
from pathlib import Path
log_dir = Path("Data/logs")
for log_file in log_dir.glob("*.log"):
    if log_file.stat().st_size > 100 * 1024 * 1024:  # >100MB
        log_file.unlink()
```

**Q: 如何在多进程环境下使用？**
```python
# 每个进程使用独立的session_id
import os
logger = ExecutionLogger(
    session_id=f"process_{os.getpid()}"
)
```

**Q: 如何集成到现有代码？**
```python
# 最小侵入性：使用全局logger
from Utils.execution_logger import configure_logging, get_logger

# 在main()开头
configure_logging(level=LogLevel.INFO)

# 在需要记录的地方
logger = get_logger()
logger.log_decision(...)
```
