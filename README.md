# LEAN Multi-Agent Trading System

基于 LEAN 引擎的智能交易系统，支持多智能体策略和自动数据管理。

## 快速开始

```bash
# 1. 下载数据
python3 Utils/download_data.py

# 2. 运行回测
bash test_daily_template.sh
```

## 项目结构

详见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

```
lean-multi-agent/
├── Algorithm/         # 交易策略
├── Utils/             # 工具脚本
├── Data/              # 市场数据
├── Results/           # 回测结果
└── tmp/               # 临时文件
```

## 核心功能

### 1. SmartAlgorithm 基类

自动管理数据下载，无需手动准备数据：

```python
from AlgorithmImports import *
from SmartAlgorithm import SmartAlgorithm

class MyStrategy(SmartAlgorithm):
    def initialize(self):
        # 自动下载数据
        self.spy = self.add_equity_smart("SPY", Resolution.DAILY)
```

### 2. 数据管理工具

```bash
# 下载数据
python3 Utils/download_data.py

# 或在代码中
from download_data import ensure_data_for_backtest
ensure_data_for_backtest(['SPY'], '2024-01-01', '2025-03-31')
```

## 开发指南

- **添加新策略**: 在 `Algorithm/` 目录创建，继承 `SmartAlgorithm`
- **工具脚本**: 放在 `Utils/` 目录
- **测试文件**: 放在 `tmp/` 目录（自动忽略）

## 文档

- [项目结构说明](PROJECT_STRUCTURE.md)
- [SmartAlgorithm 使用指南](SMART_ALGORITHM_GUIDE.md)
- [最终方案说明](FINAL_SOLUTION.md)
