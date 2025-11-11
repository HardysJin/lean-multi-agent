#!/usr/bin/env python3
"""
启动FastAPI服务器
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from backend.api.main import app

if __name__ == "__main__":
    print("=" * 80)
    print("启动 LLM量化交易决策系统 API 服务")
    print("=" * 80)
    print("\n📡 服务地址:")
    print("   - API文档 (Swagger): http://localhost:8000/docs")
    print("   - API文档 (ReDoc):  http://localhost:8000/redoc")
    print("   - 健康检查:          http://localhost:8000/health")
    print("\n按 Ctrl+C 停止服务\n")
    print("=" * 80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
