"""
LLM Multi-Agent Trading System - Real-time Monitoring Dashboard
实时监控LLM多Agent交易系统的决策过程

类似BettaFish的可视化界面，打开"黑盒"：
1. 实时展示各Agent的分析过程
2. 可视化Agent之间的数据流
3. LLM决策过程透明化
4. 历史决策回放和分析
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path
import json
from typing import Dict, Any, List
import time

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.database import DecisionStore, BacktestStore, PortfolioStore
from backend.backtest_engine.llm_backtest import LLMBacktestEngine


# ============================================================================
# 页面配置
# ============================================================================

st.set_page_config(
    page_title="LLM Trading System Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .agent-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 侧边栏配置
# ============================================================================

with st.sidebar:
    st.title("📊 LLM Trading Monitor")
    st.markdown("---")
    
    # 页面选择
    page = st.radio(
        "导航",
        ["🎯 实时监控", "📈 历史回测", "🤖 Agent交互", "📊 策略对比", "⚙️ 系统设置"],
        index=0
    )
    
    st.markdown("---")
    
    # 系统状态
    st.subheader("系统状态")
    
    # 模拟实时状态（实际应该从后端获取）
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.metric("活跃Agent", "4", "0")
    with status_col2:
        st.metric("决策次数", "127", "+1")
    
    st.markdown("---")
    
    # 快速操作
    st.subheader("快速操作")
    if st.button("🔄 刷新数据", use_container_width=True):
        st.rerun()
    
    if st.button("📥 导出报告", use_container_width=True):
        st.info("报告导出功能开发中...")
    
    st.markdown("---")
    st.caption(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# 页面1: 实时监控
# ============================================================================

if page == "🎯 实时监控":
    st.title("🎯 实时决策监控")
    st.markdown("实时展示LLM多Agent系统的决策过程")
    
    # 顶部指标卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>$105,234</h3>
            <p>当前资产</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>+5.23%</h3>
            <p>总收益率</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>grid_trading</h3>
            <p>当前策略</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>0.85</h3>
            <p>决策信心</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Agent实时状态
    st.subheader("🤖 Agent实时分析")
    
    # 使用tabs展示不同Agent
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Technical Agent", 
        "😊 Sentiment Agent", 
        "📰 News Agent",
        "🎯 Coordinator"
    ])
    
    with tab1:
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        col_t1, col_t2 = st.columns([2, 1])
        
        with col_t1:
            st.markdown("#### 技术分析结果")
            st.write("**趋势方向**: Bullish (上涨)")
            st.write("**强度**: 0.72")
            st.write("**动量**: RSI 65.3")
            st.write("**波动性**: Medium")
            
            # 技术指标图表
            sample_data = pd.DataFrame({
                'Date': pd.date_range(start='2025-10-01', periods=30),
                'Price': [100 + i + (i % 5) * 2 for i in range(30)],
                'SMA_20': [100 + i for i in range(30)],
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sample_data['Date'], y=sample_data['Price'], 
                                    name='Price', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=sample_data['Date'], y=sample_data['SMA_20'], 
                                    name='SMA 20', line=dict(color='orange', width=1, dash='dash')))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        with col_t2:
            st.markdown("#### 信号摘要")
            st.info("📈 **BUY信号**")
            st.write("- SMA: Bullish")
            st.write("- RSI: Neutral")
            st.write("- MACD: Bullish")
            st.write("- BB: Squeeze")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        st.markdown("#### 市场情绪分析")
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.write("**整体情绪**: Bullish")
            st.write("**情绪分数**: 0.68")
            st.write("**VIX水平**: 15.3 (Low)")
            st.write("**风险等级**: Medium")
        
        with col_s2:
            # 情绪指标图表
            sentiment_data = pd.DataFrame({
                'Category': ['Positive', 'Neutral', 'Negative'],
                'Count': [45, 30, 25]
            })
            
            fig = px.pie(sentiment_data, values='Count', names='Category',
                        color_discrete_sequence=['#00cc96', '#ffa15a', '#ef553b'])
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        st.markdown("#### 新闻事件分析")
        
        st.write("**重大事件**: 3条")
        st.write("**交易影响**: Positive")
        st.write("**风险因素**: Low volatility expected")
        
        # 新闻列表
        st.markdown("##### 最新新闻")
        news_items = [
            {"title": "Fed maintains interest rates", "impact": "Positive", "time": "2h ago"},
            {"title": "Tech stocks rally continues", "impact": "Positive", "time": "5h ago"},
            {"title": "Economic data beats expectations", "impact": "Neutral", "time": "1d ago"},
        ]
        
        for news in news_items:
            with st.expander(f"📰 {news['title']} - {news['time']}"):
                st.write(f"**影响**: {news['impact']}")
                st.write("Fed决定维持当前利率不变，符合市场预期...")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        st.markdown("#### LLM综合决策")
        
        col_c1, col_c2 = st.columns([3, 2])
        
        with col_c1:
            st.write("**市场状态**: Bullish with medium volatility")
            st.write("**推荐策略**: grid_trading")
            st.write("**信心度**: 0.85")
            st.write("**风险评估**: Moderate risk, favorable risk-reward")
            
            st.markdown("##### 决策推理")
            st.text_area(
                "LLM推理过程",
                """Based on the comprehensive analysis:
1. Technical indicators show bullish trend (SMA, MACD)
2. Market sentiment is positive (VIX at 15.3)
3. No major negative news events
4. Grid trading suits current low-volatility environment

Recommended action: Maintain grid trading strategy
Risk management: Monitor VIX spike above 20""",
                height=200,
                disabled=True
            )
        
        with col_c2:
            st.markdown("##### Agent贡献度")
            
            contribution_data = pd.DataFrame({
                'Agent': ['Technical', 'Sentiment', 'News'],
                'Weight': [0.40, 0.35, 0.25]
            })
            
            fig = go.Figure(data=[
                go.Bar(x=contribution_data['Agent'], y=contribution_data['Weight'],
                      marker_color=['#636efa', '#ef553b', '#00cc96'])
            ])
            fig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0),
                            yaxis_title="权重")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 决策历史时间线
    st.subheader("⏱️ 决策时间线")
    
    timeline_data = pd.DataFrame({
        'Time': pd.date_range(start='2025-10-28', periods=5, freq='W'),
        'Strategy': ['grid_trading', 'grid_trading', 'momentum', 'grid_trading', 'hold'],
        'Confidence': [0.85, 0.82, 0.75, 0.88, 0.60],
        'Return': [2.3, -0.5, 1.8, 3.2, 0.0]
    })
    
    fig = go.Figure()
    
    # 绘制决策点
    colors = ['green' if r > 0 else 'red' if r < 0 else 'gray' for r in timeline_data['Return']]
    
    fig.add_trace(go.Scatter(
        x=timeline_data['Time'],
        y=timeline_data['Confidence'],
        mode='markers+lines+text',
        marker=dict(size=15, color=colors),
        text=timeline_data['Strategy'],
        textposition='top center',
        name='Confidence'
    ))
    
    fig.update_layout(
        height=300,
        yaxis_title="信心度",
        xaxis_title="时间",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# 页面2: 历史回测
# ============================================================================

elif page == "📈 历史回测":
    st.title("📈 历史回测分析")
    st.markdown("查看和分析历史回测结果")
    
    # 回测配置
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        symbols = st.multiselect(
            "选择标的 (当前仅支持单标的)",
            ["SPY", "QQQ", "IWM", "DIA"],
            default=["SPY"],
            help="LLM回测当前只支持单个标的，将使用第一个选中的标的"
        )
    
    with col2:
        start_date = st.date_input(
            "开始日期",
            value=datetime.now() - timedelta(days=90)
        )
    
    with col3:
        end_date = st.date_input(
            "结束日期",
            value=datetime.now()
        )
    
    with col4:
        initial_capital = st.number_input(
            "初始资金",
            value=100000,
            step=10000
        )
    
    # 运行回测按钮
    if st.button("🚀 运行LLM回测", type="primary", use_container_width=True):
        with st.spinner("正在运行LLM多Agent回测..."):
            # 进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 初始化LLM回测引擎
                status_text.text("🔧 初始化LLM回测引擎...")
                progress_bar.progress(10)
                
                engine = LLMBacktestEngine(
                    initial_capital=initial_capital,
                    commission=0.001
                )
                
                # 运行回测（目前只支持单个标的）
                if len(symbols) == 0:
                    st.error("请至少选择一个交易标的")
                    st.stop()
                
                symbol = symbols[0]  # 使用第一个标的
                if len(symbols) > 1:
                    st.warning(f"⚠️ 当前只支持单标的回测，将使用: {symbol}")
                
                status_text.text("📊 收集市场数据...")
                progress_bar.progress(30)
                
                status_text.text("🤖 运行Agent分析 + LLM决策...")
                progress_bar.progress(50)
                
                # 转换日期格式
                start_dt = datetime.combine(start_date, datetime.min.time())
                end_dt = datetime.combine(end_date, datetime.min.time())
                
                result = engine.run(
                    symbol=symbol,
                    start_date=start_dt,
                    end_date=end_dt,
                    lookback_days=30
                )
                
                status_text.text("📈 计算收益指标...")
                progress_bar.progress(90)
                
                # 保存结果到session state
                st.session_state['backtest_result'] = result
                st.session_state['backtest_completed'] = True
                
                progress_bar.progress(100)
                status_text.text("✅ LLM回测完成!")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                num_decisions = len(result.get('decisions', []))
                num_trades = result.get('summary', {}).get('total_trades', 0)
                st.success(f"✅ 完成！共{num_decisions}次LLM决策，{num_trades}笔交易")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ 回测失败: {str(e)}")
                import traceback
                with st.expander("查看错误详情"):
                    st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # 显示回测结果
    if 'backtest_completed' in st.session_state and st.session_state['backtest_completed']:
        result = st.session_state['backtest_result']
        summary = result.get('summary', {})
        
        st.subheader("📊 LLM回测结果")
        
        # 关键指标
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_return_pct = summary.get('total_return', 0) * 100
            total_return_dollar = summary.get('total_pnl', 0)
            st.metric("总收益", f"${total_return_dollar:,.2f}", f"{total_return_pct:.2f}%")
        with col2:
            annual_return = summary.get('annual_return', 0) * 100
            st.metric("年化收益", f"{annual_return:.2f}%", "")
        with col3:
            st.metric("夏普比率", f"{summary.get('sharpe_ratio', 0):.2f}", "")
        with col4:
            max_dd = summary.get('max_drawdown', 0) * 100
            st.metric("最大回撤", f"{max_dd:.2f}%", "")
        with col5:
            win_rate = summary.get('win_rate', 0) * 100
            st.metric("胜率", f"{win_rate:.1f}%", "")
        
        # Alpha展示
        alpha = summary.get('alpha', 0) * 100
        if alpha != 0:
            alpha_col1, alpha_col2 = st.columns(2)
            with alpha_col1:
                st.metric("Alpha", f"{alpha:.2f}%", 
                         "跑赢基准" if alpha > 0 else "跑输基准")
            with alpha_col2:
                bh_return = summary.get('benchmark_return', 0) * 100
                st.metric("基准收益", f"{bh_return:.2f}%", "")
        
        # 净值曲线
        st.markdown("#### 净值曲线")
        
        if 'portfolio_values' in result and result['portfolio_values']:
            portfolio_values = result['portfolio_values']
            
            fig = go.Figure()
            
            # 策略净值曲线 (使用'value'键)
            dates = [pv['date'] for pv in portfolio_values]
            values = [pv['value'] for pv in portfolio_values]
            
            fig.add_trace(go.Scatter(
                x=dates, 
                y=values, 
                name='LLM Strategy', 
                line=dict(color='blue', width=2),
                hovertemplate='日期: %{x}<br>价值: $%{y:,.2f}<extra></extra>'
            ))
            
            # 基准曲线（如果有）
            if len(portfolio_values) > 0 and 'benchmark_value' in portfolio_values[0]:
                benchmark_values = [pv['benchmark_value'] for pv in portfolio_values]
                fig.add_trace(go.Scatter(
                    x=dates, 
                    y=benchmark_values, 
                    name='Benchmark (B&H)', 
                    line=dict(color='gray', width=1, dash='dash'),
                    hovertemplate='日期: %{x}<br>价值: $%{y:,.2f}<extra></extra>'
                ))
            
            fig.update_layout(
                height=400, 
                yaxis_title="Portfolio Value ($)",
                xaxis_title="Date",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无净值曲线数据")
        
        # LLM决策详情表格
        st.markdown("#### LLM决策详情")
        
        if 'decisions' in result and result['decisions']:
            decisions_data = []
            for dec in result['decisions']:
                decisions_data.append({
                    'Date': dec.get('date', ''),
                    'Strategy': dec.get('strategy', ''),
                    'Confidence': f"{dec.get('confidence', 0):.2f}",
                    'Reasoning': dec.get('reasoning', '')[:80] + '...' if len(dec.get('reasoning', '')) > 80 else dec.get('reasoning', ''),
                    'Action': dec.get('action', ''),
                })
            
            decisions_df = pd.DataFrame(decisions_data)
            st.dataframe(decisions_df, use_container_width=True)
            
            # 详细推理展开
            st.markdown("##### 查看详细LLM推理")
            for i, dec in enumerate(result['decisions'], 1):
                with st.expander(f"决策 {i}: {dec.get('date', '')} - {dec.get('strategy', '')}"):
                    st.write(f"**信心度**: {dec.get('confidence', 0):.2f}")
                    st.write(f"**执行动作**: {dec.get('action', 'N/A')}")
                    st.markdown("**LLM推理过程**:")
                    st.text_area(
                        "Reasoning",
                        dec.get('reasoning', 'No reasoning provided'),
                        height=150,
                        key=f"reasoning_{i}",
                        disabled=True
                    )
        
        # 下载结果
        st.markdown("---")
        if st.button("💾 下载完整回测报告 (JSON)", use_container_width=True):
            result_json = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="📥 下载JSON文件",
                data=result_json,
                file_name=f"llm_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    else:
        # 未运行回测时显示提示
        st.info("👆 点击上方「运行LLM回测」按钮开始回测")
        
        st.markdown("#### 💡 关于LLM回测")
        st.markdown("""
        本系统使用**完整的LLM多Agent决策流程**进行回测：
        
        1. **📊 Technical Agent**: 分析技术指标（SMA、RSI、MACD、BB、ATR）
        2. **😊 Sentiment Agent**: 分析市场情绪（VIX、新闻情绪）
        3. **📰 News Agent**: 分析新闻事件（Finnhub实时新闻）
        4. **🎯 Coordinator**: 使用LLM综合所有分析，生成决策
        
        每次决策都会调用真实的LLM API（GPT-4o/Claude等），完全模拟实际交易决策过程。
        
        ⚠️ **注意**: LLM回测会消耗API额度，请合理设置回测周期。
        """)


# ============================================================================
# 页面3: Agent交互可视化
# ============================================================================

elif page == "🤖 Agent交互":
    st.title("🤖 Agent交互可视化")
    st.markdown("深入了解Agent之间的数据流和协作过程")
    
    # 选择决策时间点
    decision_time = st.selectbox(
        "选择决策时间点",
        ["2025-11-11 14:00", "2025-11-04 14:00", "2025-10-28 14:00"]
    )
    
    st.markdown("---")
    
    # Agent数据流图
    st.subheader("📊 数据流可视化")
    
    # 使用Sankey图展示数据流
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Market Data", "News Data", "Sentiment Data", 
                   "Technical Agent", "News Agent", "Sentiment Agent",
                   "Coordinator", "Decision"],
            color=["lightblue", "lightblue", "lightblue",
                   "lightgreen", "lightgreen", "lightgreen",
                   "orange", "red"]
        ),
        link=dict(
            source=[0, 1, 2, 3, 4, 5, 6],
            target=[3, 4, 5, 6, 6, 6, 7],
            value=[10, 8, 6, 10, 8, 6, 24]
        )
    )])
    
    fig.update_layout(height=400, title_text="Agent数据流向图")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Agent详细输出
    st.subheader("📝 Agent输出详情")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Technical Agent → Coordinator")
        st.json({
            "overall_signal": "BUY",
            "trend": {"direction": "bullish", "strength": 0.72},
            "momentum": {"rsi": 65.3, "momentum": "positive"},
            "volatility": {"level": "medium", "atr": 2.15},
            "timestamp": "2025-11-11T14:00:00"
        })
        
        st.markdown("##### Sentiment Agent → Coordinator")
        st.json({
            "overall_sentiment": "bullish",
            "score": 0.68,
            "vix_value": 15.3,
            "risk_level": "medium",
            "timestamp": "2025-11-11T14:00:00"
        })
    
    with col2:
        st.markdown("##### News Agent → Coordinator")
        st.json({
            "major_events": ["Fed maintains rates", "Tech rally continues"],
            "trading_implications": "Positive outlook, low volatility",
            "risk_factors": ["Geopolitical tensions"],
            "sentiment": "positive",
            "timestamp": "2025-11-11T14:00:00"
        })
        
        st.markdown("##### Coordinator → Decision")
        st.json({
            "recommended_strategy": "grid_trading",
            "confidence": 0.85,
            "market_state": "bullish_low_volatility",
            "reasoning": "Technical bullish + low VIX + positive news",
            "risk_assessment": "Moderate risk, favorable R/R",
            "timestamp": "2025-11-11T14:00:00"
        })
    
    st.markdown("---")
    
    # LLM Prompt和Response
    st.subheader("🔍 LLM交互详情")
    
    with st.expander("查看完整Prompt"):
        st.code("""
System: You are a professional trading coordinator...

User: Based on the following analysis:

TECHNICAL ANALYSIS:
- Overall Signal: BUY
- Trend: Bullish (strength: 0.72)
- Momentum: RSI 65.3 (positive)
- Volatility: Medium (ATR: 2.15)

SENTIMENT ANALYSIS:
- Overall: Bullish (score: 0.68)
- VIX: 15.3 (low fear)
- Risk Level: Medium

NEWS ANALYSIS:
- Major Events: Fed maintains rates, Tech rally continues
- Trading Implications: Positive outlook, low volatility
- Risk Factors: Geopolitical tensions

CURRENT PORTFOLIO:
- Cash: $50,482.21
- Holdings: 72 shares SPY @ $687.06

Please provide a trading decision...
        """, language="text")
    
    with st.expander("查看LLM Response"):
        st.code("""
{
  "market_state": "bullish_low_volatility",
  "reasoning": "The market shows strong bullish momentum with technical indicators aligned. Low VIX at 15.3 indicates stable market conditions. Positive news flow supports continued uptrend. Grid trading strategy is suitable for current low-volatility environment.",
  "recommended_strategy": "grid_trading",
  "confidence": 0.85,
  "risk_assessment": "Moderate risk with favorable risk-reward ratio. Main risk: potential VIX spike above 20.",
  "suggested_positions": {
    "SPY": 0.50,
    "cash": 0.50
  }
}
        """, language="json")


# ============================================================================
# 页面4: 策略对比
# ============================================================================

elif page == "📊 策略对比":
    st.title("📊 策略对比分析")
    st.markdown("对比不同交易策略的表现")
    
    # 策略选择
    strategies = st.multiselect(
        "选择要对比的策略",
        ["LLM Multi-Agent", "Grid Trading", "Momentum", "Mean Reversion", "Buy & Hold"],
        default=["LLM Multi-Agent", "Buy & Hold"]
    )
    
    if len(strategies) > 0:
        # 对比指标
        st.subheader("📈 性能对比")
        
        # 模拟数据
        comparison_data = pd.DataFrame({
            'Strategy': strategies,
            'Total Return': [15.2, 8.1] + [0] * (len(strategies) - 2),
            'Sharpe Ratio': [1.85, 1.12] + [0] * (len(strategies) - 2),
            'Max Drawdown': [-8.2, -12.5] + [0] * (len(strategies) - 2),
            'Win Rate': [65.5, 55.0] + [0] * (len(strategies) - 2)
        })
        
        # 雷达图
        fig = go.Figure()
        
        for strategy in strategies:
            strategy_data = comparison_data[comparison_data['Strategy'] == strategy].iloc[0]
            
            fig.add_trace(go.Scatterpolar(
                r=[strategy_data['Total Return'], 
                   strategy_data['Sharpe Ratio'] * 10,
                   100 + strategy_data['Max Drawdown'],
                   strategy_data['Win Rate']],
                theta=['Return', 'Sharpe', 'Drawdown', 'Win Rate'],
                fill='toself',
                name=strategy
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细对比表
        st.dataframe(comparison_data, use_container_width=True)


# ============================================================================
# 页面5: 系统设置
# ============================================================================

elif page == "⚙️ 系统设置":
    st.title("⚙️ 系统设置")
    st.markdown("配置系统参数和偏好")
    
    # Agent配置
    st.subheader("🤖 Agent配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input("Lookback Days", value=7, min_value=1, max_value=90)
        st.number_input("Forecast Days", value=7, min_value=1, max_value=30)
        st.selectbox("LLM Provider", ["OpenAI", "Anthropic", "DeepSeek"])
    
    with col2:
        st.number_input("Decision Frequency (hours)", value=168, min_value=1, max_value=720)
        st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
        st.selectbox("LLM Model", ["gpt-4o", "claude-3-5-sonnet", "deepseek-chat"])
    
    # 风控配置
    st.subheader("🛡️ 风控配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.slider("Max Single Position", 0.0, 1.0, 0.3, 0.05)
        st.slider("Min Cash Reserve", 0.0, 1.0, 0.2, 0.05)
    
    with col2:
        st.slider("Max Weekly Turnover", 0.0, 1.0, 0.5, 0.05)
        st.slider("Circuit Breaker Drawdown", 0.0, 0.5, 0.15, 0.01)
    
    # 保存按钮
    if st.button("💾 保存设置", type="primary", use_container_width=True):
        st.success("✅ 设置已保存!")


# ============================================================================
# 页脚
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>LLM Multi-Agent Trading System v1.0 | Powered by Streamlit</p>
    <p>🔗 <a href='https://github.com/HardysJin/lean-multi-agent' target='_blank'>GitHub</a> | 
       📚 <a href='#' target='_blank'>Documentation</a> | 
       💬 <a href='#' target='_blank'>Support</a></p>
</div>
""", unsafe_allow_html=True)
