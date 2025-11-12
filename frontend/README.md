# LLM Multi-Agent Trading Dashboard

类似 [BettaFish](https://github.com/666ghj/BettaFish) 的可视化监控界面，打开LLM多Agent交易系统的"黑盒"。

## 🎯 核心功能

### 1. 实时监控页 🎯
- **Agent实时分析**：4个Agent的实时分析结果
  - 📊 Technical Agent: 技术指标分析
  - 😊 Sentiment Agent: 市场情绪分析  
  - 📰 News Agent: 新闻事件分析
  - 🎯 Coordinator: LLM综合决策

- **决策过程透明化**：
  - Agent贡献度权重
  - LLM推理过程展示
  - 决策时间线可视化

### 2. 历史回测页 📈 (真实LLM)
**✨ 核心特性：完整集成LLM回测引擎**

- **真实LLM决策**：
  - 每次决策调用真实LLM API（GPT-4o/Claude等）
  - Technical + Sentiment + News → Coordinator (LLM)
  - 完全模拟实际交易决策流程

- **详细结果展示**：
  - 完整性能指标（收益、夏普、回撤、胜率、Alpha）
  - 净值曲线 vs 基准
  - 每次LLM决策的完整推理过程
  - 可下载完整回测报告（JSON）

### 3. Agent交互页 🤖
深入了解Agent协作过程：

- **数据流可视化**：
  - Sankey图展示数据流向
  - Agent → Coordinator → Decision

- **完整输入输出**：
  - 每个Agent的JSON输出
  - 完整的LLM Prompt
  - 完整的LLM Response

### 4. 策略对比页 📊
- 多策略性能雷达图
- 关键指标对比表

### 5. 系统设置页 ⚙️
- Agent参数配置
- 风控参数调整

## 🚀 快速启动

```bash
# 方法1: 直接启动
cd /home/hardys/git/lean-multi-agent
conda activate tradingagents
streamlit run frontend/dashboard.py --server.port 8502

# 方法2: 使用测试脚本
python tmp/test_dashboard.py
```

访问: http://localhost:8502

## 📊 使用示例

### 运行LLM回测

1. 切换到「📈 历史回测」页面
2. 配置参数：
   - 选择标的（SPY、QQQ等）
   - 设置时间范围（建议先用1-2周测试）
   - 设置初始资金
3. 点击「🚀 运行LLM回测」
4. 查看结果：
   - 关键指标卡片
   - 净值曲线图
   - LLM决策详情表
   - 展开查看每次决策的完整LLM推理

### 查看Agent交互

1. 切换到「🤖 Agent交互」页面
2. 选择决策时间点
3. 查看：
   - 数据流Sankey图
   - 各Agent的JSON输出
   - 完整LLM Prompt/Response

## ⚠️ 重要提示

### 关于LLM API消耗

**历史回测使用真实LLM API**，会产生实际费用：

- 每周一次决策 = 1次LLM API调用
- 2周回测 ≈ 2-3次LLM调用
- 1个月回测 ≈ 4-5次LLM调用
- 3个月回测 ≈ 12-13次LLM调用

**建议**：
- 首次测试使用1-2周周期
- 确认系统正常后再运行长周期回测
- 注意监控API额度消耗

### LLM决策质量

系统使用完整的多Agent协作流程：
- Technical Agent 分析技术指标
- Sentiment Agent 分析市场情绪
- News Agent 分析新闻事件（Finnhub实时数据）
- Coordinator 使用LLM综合决策

每次决策都会：
1. 收集最新市场数据
2. 调用3个专业Agent分析
3. 构建详细的LLM Prompt
4. 调用LLM生成决策
5. 记录完整推理过程

## 🔧 技术栈

- **前端框架**: Streamlit
- **可视化**: Plotly (交互式图表)
- **后端引擎**: LLMBacktestEngine
- **LLM集成**: OpenAI/Anthropic/DeepSeek
- **数据源**: yfinance + Finnhub
- **样式**: 自定义CSS + 响应式布局

## 📝 与BettaFish的对比

| 特性 | BettaFish | 本系统 |
|------|-----------|--------|
| 领域 | 舆情分析 | 量化交易 |
| Agent数量 | 4 (Query/Media/Insight/Report) | 4 (Technical/Sentiment/News/Coordinator) |
| 数据源 | 社交媒体 | 金融市场 |
| 可视化 | Flask + Streamlit | Streamlit |
| 实时监控 | ✅ | ✅ |
| Agent交互可视化 | ✅ (ForumEngine) | ✅ (Sankey图) |
| LLM决策透明化 | ✅ | ✅ |
| 历史回测 | ❌ | ✅ (真实LLM) |

## 🛠️ 未来增强

- [ ] WebSocket实时推送（当前使用轮询）
- [ ] 实时交易模式监控
- [ ] Agent间的"辩论"机制可视化
- [ ] 更丰富的性能分析图表
- [ ] 自定义Dashboard布局

## 📚 相关文档

- [项目总结](../tmp/project_summary.py)
- [LLM回测引擎](../backend/backtest_engine/llm_backtest.py)
- [Agent实现](../backend/agents/)

---

**Made with ❤️ for transparent AI trading**
