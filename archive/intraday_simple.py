"""
专业级实时分时图 - 简化版

核心功能：
1. 加载今日分钟数据
2. 实时更新价格点
3. 固定时间轴（9:30-16:00 ET）
4. 动态Y轴范围
5. 多股票选择
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time as dt_time
import sys
import os
import time as time_module
import pytz

sys.path.insert(0, os.path.abspath('.'))
from Data.finnhub_client import FinnhubClient


# ════════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="实时分时图",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

ET_TIMEZONE = pytz.timezone('America/New_York')
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)

# 深色主题
st.markdown("""
<style>
    .stApp { background-color: #0a0e27; }
    .big-price { font-size: 3rem; font-weight: bold; }
    .price-up { color: #10b981; }
    .price-down { color: #ef4444; }
    .stock-header { font-size: 2rem; font-weight: bold; color: #fff; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# 初始化
# ════════════════════════════════════════════════════════════════

@st.cache_resource
def init_client():
    return FinnhubClient()

# 初始化 session state
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = 'AAPL'
if 'minute_data' not in st.session_state:
    st.session_state.minute_data = {}
if 'prev_close' not in st.session_state:
    st.session_state.prev_close = {}
if 'loaded_symbols' not in st.session_state:
    st.session_state.loaded_symbols = set()


# ════════════════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════════════════

def create_timeline():
    """创建完整交易时间轴"""
    now_et = datetime.now(ET_TIMEZONE)
    today = now_et.date()
    
    open_dt = ET_TIMEZONE.localize(datetime.combine(today, MARKET_OPEN))
    close_dt = ET_TIMEZONE.localize(datetime.combine(today, MARKET_CLOSE))
    
    return pd.date_range(start=open_dt, end=close_dt, freq='1Min')


def create_chart(minute_df, current_price, current_time, prev_close):
    """创建分时图"""
    timeline = create_timeline()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.7, 0.3]
    )
    
    # 1. 分钟K线
    if not minute_df.empty:
        # 确保时区
        if minute_df.index.tz is None:
            minute_df.index = minute_df.index.tz_localize(ET_TIMEZONE)
        else:
            minute_df.index = minute_df.index.tz_convert(ET_TIMEZONE)
        
        # 计算涨跌幅
        minute_df['change_pct'] = ((minute_df['close'] - prev_close) / prev_close * 100)
        
        # 价格线
        fig.add_trace(
            go.Scatter(
                x=minute_df.index,
                y=minute_df['close'],
                mode='lines',
                name='价格',
                line=dict(color='#3b82f6', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)',
                customdata=minute_df['change_pct'],
                hovertemplate='<b>%{x|%H:%M}</b><br>价格: $%{y:.2f}<br>涨跌: %{customdata:+.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 均价线
        minute_df['cum_pv'] = (minute_df['close'] * minute_df['volume']).cumsum()
        minute_df['cum_v'] = minute_df['volume'].cumsum()
        minute_df['avg'] = minute_df['cum_pv'] / minute_df['cum_v']
        
        fig.add_trace(
            go.Scatter(
                x=minute_df.index,
                y=minute_df['avg'],
                mode='lines',
                name='均价',
                line=dict(color='#f59e0b', width=1, dash='dot'),
                hovertemplate='%{x|%H:%M}<br>均价: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. 实时点
    if current_price and current_time:
        real_change = ((current_price - prev_close) / prev_close * 100)
        fig.add_trace(
            go.Scatter(
                x=[current_time],
                y=[current_price],
                mode='markers',
                name='实时',
                marker=dict(color='#fbbf24', size=12, line=dict(color='#fff', width=2)),
                customdata=[[real_change]],
                hovertemplate='<b>实时</b><br>价格: $%{y:.2f}<br>涨跌: %{customdata[0]:+.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 3. 昨收线
    if prev_close:
        fig.add_hline(
            y=prev_close,
            line_dash="dash",
            line_color="rgba(148, 163, 184, 0.5)",
            line_width=1,
            row=1, col=1
        )
    
    # 4. 成交量
    if not minute_df.empty:
        colors = []
        for i in range(len(minute_df)):
            if i == 0:
                color = '#10b981' if minute_df.iloc[i]['close'] >= prev_close else '#ef4444'
            else:
                color = '#10b981' if minute_df.iloc[i]['close'] >= minute_df.iloc[i-1]['close'] else '#ef4444'
            colors.append(color)
        
        fig.add_trace(
            go.Bar(
                x=minute_df.index,
                y=minute_df['volume'],
                name='成交量',
                marker_color=colors,
                hovertemplate='%{x|%H:%M}<br>成交量: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # 5. 动态Y轴
    all_prices = []
    if not minute_df.empty:
        all_prices.extend(minute_df['close'].tolist())
    if current_price:
        all_prices.append(current_price)
    if prev_close:
        all_prices.append(prev_close)
    
    if all_prices:
        price_min = min(all_prices)
        price_max = max(all_prices)
        price_range = price_max - price_min
        margin = max(price_range * 0.15, 0.5)
        y_min = price_min - margin
        y_max = price_max + margin
    else:
        y_min = None
        y_max = None
    
    # 6. 布局
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(color='#cbd5e1', size=11),
            bgcolor='rgba(15, 23, 42, 0.6)'
        ),
        hovermode='x unified',
        plot_bgcolor='#0a0e27',
        paper_bgcolor='#0a0e27',
        font=dict(color='#cbd5e1'),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    fig.update_xaxes(
        range=[timeline[0], timeline[-1]],
        gridcolor='#1e293b',
        showgrid=True,
        tickformat='%H:%M',
        dtick=1800000,
        row=1, col=1
    )
    
    fig.update_xaxes(
        range=[timeline[0], timeline[-1]],
        gridcolor='#1e293b',
        showgrid=True,
        tickformat='%H:%M',
        dtick=1800000,
        row=2, col=1
    )
    
    fig.update_yaxes(
        range=[y_min, y_max] if y_min and y_max else None,
        gridcolor='#1e293b',
        showgrid=True,
        title=dict(text='价格 ($)', font=dict(color='#94a3b8', size=11)),
        side='right',
        row=1, col=1
    )
    
    fig.update_yaxes(
        gridcolor='#1e293b',
        showgrid=True,
        title=dict(text='成交量', font=dict(color='#94a3b8', size=11)),
        side='right',
        row=2, col=1
    )
    
    return fig


# ════════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════════

def main():
    client = init_client()
    
    # ════════════════════════════════════════════════════════════
    # 侧边栏 - 股票选择
    # ════════════════════════════════════════════════════════════
    
    st.sidebar.title("📊 股票选择")
    
    # 热门股票
    popular = {
        'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Google',
        'AMZN': 'Amazon', 'TSLA': 'Tesla', 'NVDA': 'NVIDIA',
        'META': 'Meta', 'NFLX': 'Netflix'
    }
    
    # 下拉选择
    selected = st.sidebar.selectbox(
        "选择股票",
        options=list(popular.keys()),
        format_func=lambda x: f"{x} - {popular[x]}",
        index=list(popular.keys()).index(st.session_state.current_symbol) if st.session_state.current_symbol in popular else 0
    )
    
    if selected != st.session_state.current_symbol:
        st.session_state.current_symbol = selected
        st.rerun()
    
    symbol = st.session_state.current_symbol
    
    st.sidebar.markdown("---")
    
    # 快速选择按钮
    st.sidebar.subheader("🔥 快速切换")
    quick = ['AAPL', 'MSFT', 'TSLA', 'NVDA']
    cols = st.sidebar.columns(2)
    for idx, sym in enumerate(quick):
        if cols[idx % 2].button(sym, key=f"q_{sym}", use_container_width=True):
            st.session_state.current_symbol = sym
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # 设置
    refresh_sec = st.sidebar.slider("刷新间隔（秒）", 1, 10, 3)
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**当前股票**: {symbol}")
    
    # ════════════════════════════════════════════════════════════
    # 主界面
    # ════════════════════════════════════════════════════════════
    
    try:
        # 1. 加载历史数据（每个股票只加载一次）
        if symbol not in st.session_state.loaded_symbols:
            with st.spinner(f'加载 {symbol} 数据...'):
                quote = client.get_quote(symbol)
                st.session_state.prev_close[symbol] = quote['pc']
                
                minute_data = client.get_today_intraday_data(symbol)
                if not minute_data.empty:
                    minute_data = minute_data.set_index('time')
                    st.session_state.minute_data[symbol] = minute_data
                else:
                    st.session_state.minute_data[symbol] = pd.DataFrame()
                
                st.session_state.loaded_symbols.add(symbol)
        
        # 2. 获取实时报价
        quote = client.get_quote(symbol)
        current_price = quote['c']
        current_time = datetime.now(ET_TIMEZONE)
        prev_close = st.session_state.prev_close.get(symbol, quote['pc'])
        
        # 3. 获取公司信息
        try:
            profile = client.get_company_profile(symbol)
            company_name = profile.get('name', symbol)
        except:
            company_name = symbol
        
        # 4. 显示头部
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f'<div class="stock-header">{symbol} - {company_name}</div>', unsafe_allow_html=True)
        
        with col2:
            st.caption(f"更新: {current_time.strftime('%H:%M:%S ET')}")
        
        # 5. 价格显示
        change = quote['d']
        change_pct = quote['dp']
        price_class = "price-up" if change >= 0 else "price-down"
        change_symbol = "▲" if change >= 0 else "▼"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f'<div class="big-price {price_class}">${current_price:.2f}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="{price_class}" style="font-size: 1.5rem;">{change_symbol} ${abs(change):.2f} ({abs(change_pct):.2f}%)</div>',
                unsafe_allow_html=True
            )
        
        with col2:
            st.metric("开盘", f"${quote['o']:.2f}")
            st.metric("最高", f"${quote['h']:.2f}")
        
        with col3:
            st.metric("昨收", f"${prev_close:.2f}")
            st.metric("最低", f"${quote['l']:.2f}")
        
        st.markdown("---")
        
        # 6. 图表
        minute_df = st.session_state.minute_data.get(symbol, pd.DataFrame())
        
        fig = create_chart(minute_df, current_price, current_time, prev_close)
        st.plotly_chart(fig, use_container_width=True)
        
        # 7. 底部统计
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("数据点数", len(minute_df))
        
        with col2:
            if not minute_df.empty:
                st.metric("均价", f"${minute_df['close'].mean():.2f}")
            else:
                st.metric("均价", "-")
        
        with col3:
            if not minute_df.empty:
                vol = minute_df['volume'].sum()
                if vol >= 1e6:
                    st.metric("成交量", f"{vol/1e6:.2f}M")
                else:
                    st.metric("成交量", f"{vol/1e3:.2f}K")
            else:
                st.metric("成交量", "-")
        
        with col4:
            if not minute_df.empty:
                price_range = minute_df['close'].max() - minute_df['close'].min()
                st.metric("振幅", f"${price_range:.2f}")
            else:
                st.metric("振幅", "-")
        
        # 8. 自动刷新
        time_module.sleep(refresh_sec)
        st.rerun()
    
    except Exception as e:
        st.error(f"错误: {e}")
        import traceback
        st.code(traceback.format_exc())
        time_module.sleep(refresh_sec)
        st.rerun()


if __name__ == "__main__":
    main()
