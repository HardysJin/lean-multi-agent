#!/bin/bash

# Multi-Agent 回测系统 Dashboard 启动脚本

echo "🚀 启动 Multi-Agent 回测系统 Dashboard..."
echo ""

# 检查依赖
echo "📦 检查依赖..."
python -c "import streamlit" 2>/dev/null || {
    echo "❌ Streamlit 未安装，正在安装..."
    pip install streamlit plotly
}

echo "✅ 依赖检查完成"
echo ""

# 启动 Dashboard
echo "🌐 启动 Dashboard (http://localhost:8501)"
echo "按 Ctrl+C 停止服务器"
echo ""

streamlit run dashboard.py
