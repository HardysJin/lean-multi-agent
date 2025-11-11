# Utils 工具目录

存放项目通用工具脚本。

## 文件说明

- **`download_data.py`** - 数据下载和管理工具
  - 检查本地数据
  - 智能下载和合并数据
  - 支持命令行和代码调用

## 使用方法

### 在算法中使用

```python
# Algorithm/SmartAlgorithm.py 已自动处理路径
from download_data import check_existing_data, download_and_convert
```

### 命令行使用

```bash
# 在项目根目录
python3 Utils/download_data.py

# 在 Docker 容器内
cd /workspace && python3 Utils/download_data.py
```
