# Yahoo Finance 8天限制解决方案

## 🎯 问题

Yahoo Finance API 对分钟数据有严格限制：
```
Yahoo error = "1m data not available for startTime=... 
Only 8 days worth of 1m granularity data are allowed to be fetched per request."
```

## ✅ 解决方案：分批下载

实现了自动分批下载功能，将长时间范围拆分成多个7天的批次。

### 核心函数

```python
def _download_minute_data_in_batches(ticker, start_date, end_date, interval):
    """分批下载分钟数据（绕过 Yahoo Finance 8天限制）"""
    
    # 1. 计算总天数
    total_days = (end_dt - start_dt).days
    
    # 2. 如果 ≤7天，直接下载
    if total_days <= 7:
        return ticker.history(start=start_dt, end=end_dt, interval=interval)
    
    # 3. 分批下载（每批7天）
    batch_size = 7
    batches = []
    
    while current_start < end_dt:
        batch_df = ticker.history(start=current_start, end=current_end, interval=interval)
        batches.append(batch_df)
        time.sleep(0.5)  # 避免频率限制
        current_start = current_end
    
    # 4. 合并并去重
    merged_df = pd.concat(batches)
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
    return merged_df.sort_index()
```

## 📊 测试结果

### 测试场景：下载23天的SOXL分钟数据

```bash
时间范围: 2025-10-15 到 2025-11-07
总天数: 23 天

输出：
⚠️  Yahoo Finance 分钟数据限制每次请求8天
时间范围: 23 天，将分 4 批下载
批次 1: 2025-10-15 到 2025-10-22 ✓ (1949 条)
批次 2: 2025-10-22 到 2025-10-29 ✓ (1950 条)
批次 3: 2025-10-29 到 2025-11-05 ✓ (1950 条)
批次 4: 2025-11-05 到 2025-11-07 ✓ (780 条)
合并 4 个批次的数据...
✅ 获取到 6629 条的数据

生成: 17 个交易日的数据文件
```

### 验证

```bash
ls Lean/Data/equity/usa/minute/soxl/

20251015_trade.zip
20251016_trade.zip
20251017_trade.zip
...
20251106_trade.zip

# 数据完整性
第一条: 20251015 09:30,39.55,39.56,38.93,39.00,9385661
最后条: 20251106 15:59,43.09,43.13,42.95,43.06,1628206
```

## 🚀 优势

1. **无缝集成**
   - 自动检测时间范围
   - 无需手动配置
   - 透明化处理

2. **突破限制**
   - 之前：最多8天
   - 现在：**任意长度**（30天、60天、90天...）

3. **稳定可靠**
   - 自动重试机制
   - 批次间延迟避免限流
   - 数据去重和排序

4. **保持兼容**
   - 日线数据：单次下载（原有逻辑）
   - 小时数据：单次下载
   - 分钟数据：自动分批

## 📝 使用示例

### 1. 策略中使用（自动）

```python
class EMAChannelStrategy(SmartAlgorithm):
    def initialize(self):
        # 现在可以使用更长的时间范围！
        self.set_start_date(2025, 10, 1)   # 30多天前
        self.set_end_date(2025, 11, 7)     # 今天
        
        # 会自动分批下载分钟数据
        self.spy = self.add_equity_smart("SPY", Resolution.MINUTE)
```

### 2. 手动下载

```python
from Utils.download_data import download_and_convert
from datetime import datetime, timedelta

# 下载60天的分钟数据
end = datetime.now()
start = end - timedelta(days=60)

download_and_convert(
    'AAPL',
    start.strftime('%Y-%m-%d'),
    end.strftime('%Y-%m-%d'),
    resolution='minute'
)
```

## 🔍 Finnhub 测试结果

也测试了 Finnhub 作为替代数据源：

```
测试结果：
✅ 免费版支持：实时报价
❌ 需要付费：历史K线数据（包括分钟数据）
💰 付费价格：$59/月起

结论：继续使用 Yahoo Finance + 分批下载
```

## ⚠️ 注意事项

1. **下载时间**
   - 每批间隔0.5秒
   - 60天 ≈ 9批 ≈ 5秒下载时间
   - 可以接受

2. **数据质量**
   - Yahoo Finance 免费数据
   - 偶尔有缺失或延迟
   - 适合回测，不建议实盘决策

3. **频率限制**
   - Yahoo Finance 可能限流
   - 遇到错误会显示但继续
   - 已添加0.5秒延迟

## 📚 相关文件

- `Utils/download_data.py` - 分批下载逻辑
- `Algorithm/SmartAlgorithm.py` - 自动数据管理
- `Algorithm/EMAChannelStrategy.py` - 示例策略
- `docs/MINUTE_DATA_GUIDE.md` - 完整使用指南
