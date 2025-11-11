#!/bin/bash

echo "======================================"
echo "测试复杂算法: AddAlphaModelAlgorithm"
echo "使用 Alpha 模型框架"
echo "======================================"
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1

# 检查算法文件
echo "1. 检查算法文件..."
if [ -f "Lean/Algorithm.Python/AddAlphaModelAlgorithm.py" ]; then
    echo "   ✅ 算法文件存在"
else
    echo "   ❌ 算法文件不存在"
    exit 1
fi

echo ""
echo "2. 检查配置文件..."
if [ -f "Configs/config_alpha_model.json" ]; then
    echo "   ✅ 配置文件存在"
else
    echo "   ❌ 配置文件不存在"
    exit 1
fi

echo ""
echo "3. 清理旧的结果文件..."
rm -f Results/AddAlphaModelAlgorithm-*.txt
rm -f Results/AddAlphaModelAlgorithm-*.json
echo "   ✅ 清理完成"

echo ""
echo "======================================"
echo "开始运行 LEAN 回测..."
echo "======================================"
echo ""

# 运行 Docker 容器
CONFIG_FILE=config_alpha_model.json docker-compose run --rm lean

echo ""
echo "======================================"
echo "测试完成！"
echo "======================================"
echo ""

# 显示结果
echo "查看结果文件..."

# 显示完整的回测分析报告
if [ -f "Results/AddAlphaModelAlgorithm-order-events.json" ] && [ -f "Results/AddAlphaModelAlgorithm-summary.json" ]; then
    python3 Utils/show_backtest_results.py Results/AddAlphaModelAlgorithm-order-events.json Results/AddAlphaModelAlgorithm-summary.json
fi

if [ -f "Results/AddAlphaModelAlgorithm-log.txt" ]; then
    echo ""
    echo "=== 算法日志（最后10行）==="
    tail -n 10 Results/AddAlphaModelAlgorithm-log.txt
    echo ""
fi

echo ""
echo "✅ 所有结果文件保存在 Results/ 目录"
