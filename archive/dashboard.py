"""
Multi-Agent 交易回测系统 Dashboard

使用 Streamlit 创建交互式回测可视化界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

from Backtests.vectorbt_engine import VectorBTBacktest
from Backtests.strategies.multi_agent_strategy import SimpleTechnicalStrategy, MultiAgentStrategy


# ════════════════════════════════════════════════════════════════
# 页面配置
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Multi-Agent 回测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1557a0;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# Session State 初始化
# ════════════════════════════════════════════════════════════════

if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

if 'backtest_engine' not in st.session_state:
    st.session_state.backtest_engine = None

if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = {}


# ════════════════════════════════════════════════════════════════
# 标题
# ════════════════════════════════════════════════════════════════

st.markdown('<h1 class="main-header">📈 Multi-Agent 交易回测系统</h1>', unsafe_allow_html=True)
st.markdown("---")


# ════════════════════════════════════════════════════════════════
# 侧边栏 - 回测配置
# ════════════════════════════════════════════════════════════════

st.sidebar.header("⚙️ 回测配置")

# 股票选择
st.sidebar.subheader("1️⃣ 股票选择")
symbol_input = st.sidebar.text_input(
    "股票代码（多个用逗号分隔）",
    value="AAPL",
    help="例如: AAPL, MSFT, GOOGL"
)
symbols = [s.strip().upper() for s in symbol_input.split(",") if s.strip()]

# 日期范围
st.sidebar.subheader("2️⃣ 回测周期")
col1, col2 = st.sidebar.columns(2)

with col1:
    start_date = st.date_input(
        "开始日期",
        value=datetime.now() - timedelta(days=180),
        max_value=datetime.now()
    )

with col2:
    end_date = st.date_input(
        "结束日期",
        value=datetime.now() - timedelta(days=1),
        max_value=datetime.now()
    )

# 资金和手续费
st.sidebar.subheader("3️⃣ 资金配置")
initial_cash = st.sidebar.number_input(
    "初始资金 ($)",
    min_value=1000,
    max_value=10000000,
    value=100000,
    step=10000
)

fees = st.sidebar.slider(
    "手续费率 (%)",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01,
    format="%.2f%%"
) / 100

# 策略选择
st.sidebar.subheader("4️⃣ 策略选择")
strategy_type = st.sidebar.selectbox(
    "选择策略",
    [
        "简单移动平均 (SMA)",
        "Multi-Agent 策略 (需要 LLM)"
    ],
    help="简单移动平均策略速度快，Multi-Agent 策略更智能但需要 API"
)

# 如果选择 SMA，显示参数设置
if strategy_type == "简单移动平均 (SMA)":
    st.sidebar.markdown("**SMA 参数**")
    short_window = st.sidebar.number_input(
        "短期均线周期",
        min_value=5,
        max_value=100,
        value=20,
        step=5
    )
    long_window = st.sidebar.number_input(
        "长期均线周期",
        min_value=20,
        max_value=200,
        value=50,
        step=10
    )

st.sidebar.markdown("---")

# 运行回测按钮
run_backtest = st.sidebar.button("🚀 运行回测", type="primary", use_container_width=True)


# ════════════════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════════════════

def format_number(num, prefix="$", suffix="", decimals=2):
    """格式化数字显示"""
    if abs(num) >= 1e6:
        return f"{prefix}{num/1e6:.{decimals}f}M{suffix}"
    elif abs(num) >= 1e3:
        return f"{prefix}{num/1e3:.{decimals}f}K{suffix}"
    else:
        return f"{prefix}{num:.{decimals}f}{suffix}"


def create_performance_chart(backtest_engine, symbol):
    """创建性能图表"""
    # 验证数据
    if symbol not in backtest_engine._portfolios:
        raise ValueError(f"No portfolio data for {symbol}")
    if symbol not in backtest_engine._price_data:
        raise ValueError(f"No price data for {symbol}")
    if symbol not in backtest_engine._signals:
        raise ValueError(f"No signals data for {symbol}")
    
    portfolio = backtest_engine._portfolios[symbol]
    price_data = backtest_engine._price_data[symbol]
    signals = backtest_engine._signals[symbol]
    
    # 验证数据不为空
    if portfolio is None or price_data is None or signals is None:
        raise ValueError(f"Invalid data for {symbol}")
    
    # 创建子图
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            f'{symbol} 价格与交易信号',
            '投资组合价值',
            '每日收益率'
        ),
        vertical_spacing=0.08
    )
    
    # 1. 价格与信号
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['Close'],
            name='价格',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>价格</b>: $%{y:.2f}<br><b>日期</b>: %{x}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 买入信号
    buy_signals = signals[signals == 1]
    if len(buy_signals) > 0:
        buy_dates = buy_signals.index
        buy_prices = price_data.loc[buy_dates, 'Close']
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                name='买入信号',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='#2ca02c',
                    line=dict(color='white', width=1)
                ),
                hovertemplate='<b>买入</b><br>价格: $%{y:.2f}<br>日期: %{x}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. 投资组合价值
    portfolio_value = portfolio.value()
    fig.add_trace(
        go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value.values,
            name='组合价值',
            line=dict(color='#ff7f0e', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 127, 14, 0.1)',
            hovertemplate='<b>价值</b>: $%{y:,.2f}<br><b>日期</b>: %{x}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 添加基准线（初始资金）
    fig.add_hline(
        y=backtest_engine.initial_cash,
        line_dash="dash",
        line_color="gray",
        row=2, col=1,
        annotation_text=f"初始资金: ${backtest_engine.initial_cash:,.0f}"
    )
    
    # 3. 每日收益率
    returns = portfolio.returns()
    colors = ['#2ca02c' if r >= 0 else '#d62728' for r in returns]
    
    fig.add_trace(
        go.Bar(
            x=returns.index,
            y=returns.values * 100,  # 转换为百分比
            name='日收益率',
            marker_color=colors,
            hovertemplate='<b>收益率</b>: %{y:.2f}%<br><b>日期</b>: %{x}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 更新布局
    fig.update_layout(
        height=900,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # 更新 y 轴标签
    fig.update_yaxes(title_text="价格 ($)", row=1, col=1)
    fig.update_yaxes(title_text="价值 ($)", row=2, col=1)
    fig.update_yaxes(title_text="收益率 (%)", row=3, col=1)
    
    # 更新 x 轴标签
    fig.update_xaxes(title_text="日期", row=3, col=1)
    
    return fig


async def run_backtest_async(symbols, start_date, end_date, initial_cash, fees, strategy_type, progress_callback=None, **kwargs):
    """异步运行回测"""
    # 创建回测引擎
    if progress_callback:
        progress_callback(None, 0, 100, "🔧 初始化回测引擎...")
    
    backtest = VectorBTBacktest(
        symbols=symbols,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        initial_cash=initial_cash,
        fees=fees
    )
    
    # 加载数据
    if progress_callback:
        progress_callback(None, 10, 100, "📊 加载股票数据...")
    backtest.load_data()
    
    # 根据策略类型生成信号
    if strategy_type == "简单移动平均 (SMA)":
        if progress_callback:
            progress_callback(None, 30, 100, "🤖 生成交易信号（SMA策略）...")
        
        # 使用简单技术策略
        strategy = SimpleTechnicalStrategy(
            short_window=kwargs.get('short_window', 20),
            long_window=kwargs.get('long_window', 50)
        )
        
        signals = {}
        for symbol in backtest.symbols:
            if symbol not in backtest._price_data:
                continue
            
            df = backtest._price_data[symbol]
            symbol_signals = []
            
            total_days = len(df)
            for idx, (date, row) in enumerate(df.iterrows()):
                if progress_callback and idx % 10 == 0:
                    progress = 30 + int(40 * idx / total_days)
                    progress_callback(symbol, idx, total_days, f"分析 {symbol}: {date.strftime('%Y-%m-%d')}")
                
                historical_data = df.loc[:date]
                signal = strategy.generate_signal(
                    symbol=symbol,
                    date=date,
                    price=row['Close'],
                    historical_data=historical_data
                )
                symbol_signals.append(1 if signal > 0 else 0)
            
            signals[symbol] = pd.Series(symbol_signals, index=df.index)
    
    else:
        if progress_callback:
            progress_callback(None, 30, 100, "🧠 初始化 Multi-Agent 策略...")
        
        # 使用 Multi-Agent 策略，传递进度回调
        def agent_progress(symbol, current, total, message):
            if progress_callback and total > 0:
                # 将 Agent 进度映射到 30-70%
                progress = 30 + int(40 * current / total)
                progress_callback(symbol, current, total, f"🤖 {message}")
        
        signals = await backtest.precompute_signals(
            use_meta_agent=True,
            progress_callback=agent_progress
        )
    
    # 保存信号到回测引擎（重要！）
    backtest._signals = signals
    
    # 运行回测
    if progress_callback:
        progress_callback(None, 80, 100, "📈 执行回测...")
    backtest.run_backtest(signals)
    
    if progress_callback:
        progress_callback(None, 100, 100, "✅ 回测完成！")
    
    return backtest


# ════════════════════════════════════════════════════════════════
# 主界面
# ════════════════════════════════════════════════════════════════

# 如果点击运行回测
if run_backtest:
    # 验证输入
    if not symbols:
        st.error("❌ 请至少输入一个股票代码！")
    elif start_date >= end_date:
        st.error("❌ 开始日期必须早于结束日期！")
    else:
        # 创建进度显示容器
        progress_container = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        detail_text = st.empty()
        
        # 进度回调函数
        progress_data = {'current': 0, 'total': 100, 'message': '准备开始...'}
        
        def update_progress(symbol, current, total, message):
            progress_data['current'] = current
            progress_data['total'] = total if total > 0 else 100
            progress_data['message'] = message
            
            # 计算总体进度
            if total > 0:
                progress_pct = min(int(100 * current / total), 100)
            else:
                progress_pct = current
            
            # 更新UI
            progress_bar.progress(progress_pct / 100)
            status_text.markdown(f"**{message}**")
            if symbol:
                detail_text.info(f"📊 当前股票: {symbol} | 进度: {current}/{total}")
        
        try:
            # 运行回测
            kwargs = {}
            if strategy_type == "简单移动平均 (SMA)":
                kwargs['short_window'] = short_window
                kwargs['long_window'] = long_window
            
            kwargs['progress_callback'] = update_progress
            
            backtest_engine = asyncio.run(
                run_backtest_async(
                    symbols, start_date, end_date, 
                    initial_cash, fees, strategy_type, 
                    **kwargs
                )
            )
            
            st.session_state.backtest_engine = backtest_engine
            st.session_state.backtest_results = backtest_engine.get_performance_stats()
            
            # 清除进度显示
            progress_container.empty()
            progress_bar.empty()
            status_text.empty()
            detail_text.empty()
            
            st.success(f"✅ 回测完成！分析了 {len(symbols)} 个股票")
            st.balloons()
            
        except Exception as e:
            # 清除进度显示
            progress_container.empty()
            progress_bar.empty()
            status_text.empty()
            detail_text.empty()
            
            st.error(f"❌ 回测失败: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# 显示结果
if st.session_state.backtest_results:
    results = st.session_state.backtest_results
    backtest_engine = st.session_state.backtest_engine
    
    # 选择要查看的股票（如果有多个）
    if len(results) > 1:
        selected_symbol = st.selectbox(
            "选择股票查看详情",
            list(results.keys()),
            format_func=lambda x: f"{x} - {results[x]['symbol']}"
        )
    else:
        selected_symbol = list(results.keys())[0]
    
    stats = results[selected_symbol]
    
    st.markdown(f"## 📊 {selected_symbol} 回测结果")
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════
    # 关键指标卡片
    # ═══════════════════════════════════════════════════════════
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return_pct = stats['total_return'] * 100
        st.metric(
            "总收益率",
            f"{total_return_pct:+.2f}%",
            delta=None,
            help="整个回测期间的总收益率"
        )
    
    with col2:
        st.metric(
            "最终价值",
            format_number(stats['final_value'], prefix="$", decimals=0),
            delta=format_number(stats['profit_loss'], prefix="$", decimals=0),
            help="回测结束时的投资组合总价值"
        )
    
    with col3:
        sharpe = stats.get('sharpe_ratio', 0)
        sharpe_color = "normal" if sharpe is None else ("inverse" if sharpe < 1 else "normal")
        st.metric(
            "夏普比率",
            f"{sharpe:.2f}" if sharpe is not None else "N/A",
            delta=None,
            delta_color=sharpe_color,
            help="风险调整后的收益率，越高越好（>1为良好）"
        )
    
    with col4:
        st.metric(
            "交易次数",
            f"{stats['total_trades']}",
            delta=None,
            help="回测期间的总交易次数"
        )
    
    # 第二行指标
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        win_rate = stats.get('win_rate', 0)
        st.metric(
            "胜率",
            f"{win_rate*100:.1f}%" if win_rate else "N/A",
            delta=None,
            help="盈利交易占总交易的比例"
        )
    
    with col6:
        max_dd = stats.get('max_drawdown', 0)
        st.metric(
            "最大回撤",
            f"{max_dd*100:.2f}%" if max_dd else "N/A",
            delta=None,
            delta_color="inverse",
            help="投资组合从峰值下跌的最大幅度"
        )
    
    with col7:
        annual_return = stats.get('annualized_return', 0)
        st.metric(
            "年化收益率",
            f"{annual_return*100:.2f}%" if annual_return else "N/A",
            delta=None,
            help="按年化计算的收益率"
        )
    
    with col8:
        days = (end_date - start_date).days
        st.metric(
            "回测天数",
            f"{days}",
            delta=None,
            help="回测时间跨度"
        )
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════
    # 性能图表
    # ═══════════════════════════════════════════════════════════
    
    st.markdown("### 📈 性能图表")
    
    try:
        chart = create_performance_chart(backtest_engine, selected_symbol)
        st.plotly_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"图表生成失败: {str(e)}")
        
        # 调试信息
        with st.expander("🔍 查看调试信息"):
            st.write("**回测引擎状态:**")
            st.write(f"- Portfolios: {list(backtest_engine._portfolios.keys()) if backtest_engine._portfolios else 'None'}")
            st.write(f"- Price Data: {list(backtest_engine._price_data.keys()) if backtest_engine._price_data else 'None'}")
            st.write(f"- Signals: {list(backtest_engine._signals.keys()) if backtest_engine._signals else 'None'}")
            st.write(f"- Selected Symbol: {selected_symbol}")
            
            st.write("**错误详情:**")
            import traceback
            st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════
    # 详细统计
    # ═══════════════════════════════════════════════════════════
    
    with st.expander("📋 查看详细统计信息"):
        st.json(stats['full_stats'])
    
    # ═══════════════════════════════════════════════════════════
    # 导出功能
    # ═══════════════════════════════════════════════════════════
    
    st.markdown("### 💾 导出结果")
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        if st.button("📄 生成完整报告", use_container_width=True):
            with st.spinner("生成报告中..."):
                reports = backtest_engine.generate_report()
                st.success("✅ 报告已生成！")
                st.info(f"报告保存在: {reports['summary']}")
    
    with col_export2:
        # 准备下载数据
        summary_df = pd.DataFrame([stats]).drop(columns=['full_stats'], errors='ignore')
        csv_data = summary_df.to_csv(index=False)
        
        st.download_button(
            label="⬇️ 下载 CSV",
            data=csv_data,
            file_name=f"{selected_symbol}_backtest_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    # 欢迎界面
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>👋 欢迎使用 Multi-Agent 交易回测系统</h2>
        <p style="font-size: 1.2rem; color: #666;">
            这是一个基于 AI 多智能体的量化交易回测平台
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 功能特点
        - 📊 多股票同时回测
        - 🤖 AI 驱动的交易策略
        - 📈 交互式可视化图表
        - 💾 完整的回测报告
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 使用步骤
        1. 在左侧选择股票和日期
        2. 配置初始资金和手续费
        3. 选择交易策略
        4. 点击"运行回测"按钮
        """)
    
    with col3:
        st.markdown("""
        ### 📚 支持的策略
        - **简单移动平均**: 快速，适合测试
        - **Multi-Agent**: AI 驱动，更智能
        - 更多策略开发中...
        """)
    
    st.markdown("---")
    
    st.info("💡 **提示**: 点击左侧边栏配置回测参数，然后点击'运行回测'开始！")


# ════════════════════════════════════════════════════════════════
# 页脚
# ════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Multi-Agent Trading Backtest System | Powered by VectorBT & Streamlit</p>
    <p>⚠️ <strong>免责声明</strong>: 本系统仅供学习研究使用，不构成投资建议</p>
</div>
""", unsafe_allow_html=True)
