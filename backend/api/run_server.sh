#!/usr/bin/env bash
#
# 启动 FastAPI 服务器
#

cd "$(dirname "$0")/.."

echo "================================================================================"
echo "启动 LLM量化交易决策系统 API 服务"
echo "================================================================================"
echo ""
echo "📡 服务地址:"
echo "   - API文档 (Swagger): http://localhost:8000/docs"
echo "   - API文档 (ReDoc):  http://localhost:8000/redoc"
echo "   - 健康检查:          http://localhost:8000/health"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""
echo "================================================================================"
echo ""

conda run -n tradingagents uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
