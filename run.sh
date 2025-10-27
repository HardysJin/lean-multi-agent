#!/bin/bash

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=============================================="
echo "   LEAN Multi-Agent 回测运行脚本"
echo "=============================================="
echo ""

# 检查环境是否已搭建
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}❌ 环境未搭建${NC}"
    echo "请先运行: ./setup.sh"
    exit 1
fi

# 检查是否配置了API Keys
echo -e "${YELLOW}[1/5]${NC} 检查配置..."
if [ -f ".env" ]; then
    source .env
    if [ -z "$CLAUDE_API_KEY" ] && [ -z "$NEWS_API_KEY" ]; then
        echo -e "${YELLOW}⚠️  未配置API Keys，将使用纯技术指标策略${NC}"
        echo "   如需使用Multi-Agent功能，请编辑.env文件"
    else
        echo -e "${GREEN}✅ API Keys已配置${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  .env文件不存在，使用默认配置${NC}"
fi
echo ""

# 清理旧的容器和结果
echo -e "${YELLOW}[2/5]${NC} 清理环境..."
docker-compose down 2>/dev/null || true
rm -rf Results/* 2>/dev/null || true
rm -rf Logs/* 2>/dev/null || true
echo -e "${GREEN}✅ 清理完成${NC}"
echo ""

# 拉取最新镜像
echo -e "${YELLOW}[3/5]${NC} 检查Docker镜像..."
if docker images | grep -q quantconnect/lean; then
    echo -e "${GREEN}✅ 镜像已存在${NC}"
else
    echo "首次运行，下载镜像（约2GB，需要几分钟）..."
    docker pull quantconnect/lean:latest
fi
echo ""

# 启动回测
echo -e "${YELLOW}[4/5]${NC} 启动回测..."
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}开始运行回测，请稍候...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 运行容器（前台运行以查看日志）
docker-compose up

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}回测完成！${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 等待容器完全停止
sleep 2

# 分析结果
echo -e "${YELLOW}[5/5]${NC} 分析结果..."
echo ""

# 检查是否有结果文件
RESULT_DIR="Results"

# 检查是否有JSON结果文件
if [ ! -f "$RESULT_DIR/ProductionMultiAgent.json" ] && [ ! -f "$RESULT_DIR/ProductionMultiAgent-log.txt" ]; then
    # 尝试查找日期子目录（旧版本格式）
    DATE_DIR=$(find Results -type d -name "2*" | sort | tail -1)
    if [ -n "$DATE_DIR" ]; then
        RESULT_DIR="$DATE_DIR"
    else
        echo -e "${RED}❌ 未找到回测结果${NC}"
        echo ""
        echo "请检查日志:"
        echo "  docker-compose logs"
        exit 1
    fi
fi

echo -e "${GREEN}✅ 发现回测结果: $RESULT_DIR${NC}"
echo ""

# 显示主要结果
echo "=============================================="
echo "   📊 回测结果摘要"
echo "=============================================="
echo ""

# 查找日志文件
LOG_FILE="$RESULT_DIR/ProductionMultiAgent-log.txt"
if [ ! -f "$LOG_FILE" ]; then
    LOG_FILE="$RESULT_DIR/log.txt"
fi

# 查找JSON结果文件
JSON_FILE="$RESULT_DIR/ProductionMultiAgent.json"
if [ ! -f "$JSON_FILE" ]; then
    JSON_FILE="$RESULT_DIR/result.json"
fi

# 从日志中提取关键信息
if [ -f "$LOG_FILE" ]; then
    echo "策略配置:"
    grep "策略初始化完成" "$LOG_FILE" 2>/dev/null | tail -1 || echo "未找到策略配置信息"
    echo ""
    
    echo "交易记录:"
    TRADES=$(grep -E "买入|卖入" "$LOG_FILE" 2>/dev/null | tail -10)
    if [ -n "$TRADES" ]; then
        echo "$TRADES"
    else
        echo "无交易记录"
    fi
    echo ""
    
    echo "最终统计:"
    STATS=$(grep -A 10 "回测结果汇总" "$LOG_FILE" 2>/dev/null | tail -10)
    if [ -n "$STATS" ]; then
        echo "$STATS"
    else
        echo "未找到回测汇总"
    fi
    echo ""
fi

# 显示简要统计
if [ -f "$RESULT_DIR/ProductionMultiAgent-summary.json" ]; then
    echo "关键指标:"
    python3 << 'PYTHON_EOF' 2>/dev/null || echo "  无法解析统计数据"
import json
with open('$RESULT_DIR/ProductionMultiAgent-summary.json', 'r') as f:
    data = json.load(f)
    stats = data.get('statistics', {})
    print(f"  总订单数: {stats.get('Total Orders', '0')}")
    print(f"  净利润: {stats.get('Net Profit', '0%')}")
    print(f"  夏普比率: {stats.get('Sharpe Ratio', '0')}")
    print(f"  最大回撤: {stats.get('Drawdown', '0%')}")
PYTHON_EOF
    echo ""
fi

echo "=============================================="
echo "   📁 完整文件位置"
echo "=============================================="
echo ""
echo "回测日志: $LOG_FILE"
if [ -f "$JSON_FILE" ]; then
    echo "结果JSON: $JSON_FILE"
fi
if [ -f "$RESULT_DIR/ProductionMultiAgent-summary.json" ]; then
    echo "汇总JSON: $RESULT_DIR/ProductionMultiAgent-summary.json"
fi
echo "图表数据: $RESULT_DIR/"
echo ""

echo "=============================================="
echo "   🔍 查看详细信息"
echo "=============================================="
echo ""
echo "查看完整日志:"
echo "  cat Results/ProductionMultiAgent-log.txt"
echo ""
echo "查看JSON结果:"
echo "  cat Results/ProductionMultiAgent.json | python3 -m json.tool"
echo ""
echo "查看汇总统计:"
echo "  cat Results/ProductionMultiAgent-summary.json | python3 -m json.tool"
echo ""
echo "重新运行回测:"
echo "  ./run.sh"
echo ""
echo "修改策略:"
echo "  vim Algorithm/MultiAgent/main.py"
echo ""

# 提供快捷命令
cat > view_results.sh << 'EOF'
#!/bin/bash
RESULT_DIR="Results"

# 检查结果文件
if [ ! -f "$RESULT_DIR/ProductionMultiAgent-log.txt" ]; then
    echo "未找到结果文件"
    exit 1
fi

echo "=== 完整日志 ==="
cat "$RESULT_DIR/ProductionMultiAgent-log.txt"
echo ""
echo "=== 汇总统计 ==="
if [ -f "$RESULT_DIR/ProductionMultiAgent-summary.json" ]; then
    cat "$RESULT_DIR/ProductionMultiAgent-summary.json" | python3 -m json.tool
fi
echo ""
echo "=== 详细JSON结果 ==="
if [ -f "$RESULT_DIR/ProductionMultiAgent.json" ]; then
    cat "$RESULT_DIR/ProductionMultiAgent.json" | python3 -m json.tool | head -100
    echo "... (更多内容请查看原始文件)"
fi
EOF
chmod +x view_results.sh

echo "💡 提示: 运行 ./view_results.sh 查看完整结果"
echo ""