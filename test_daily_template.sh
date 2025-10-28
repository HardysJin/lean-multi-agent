#!/bin/bash

echo "======================================"
echo "测试修改后的 Python 算法"
echo "使用日线数据的 BasicTemplateAlgorithm"
echo "======================================"
echo ""

# 检查文件是否存在
echo "1. 检查算法文件..."
if [ -f "Algorithm/BasicTemplateAlgorithmDaily.py" ]; then
    echo "   ✅ 算法文件存在"
else
    echo "   ❌ 算法文件不存在"
    exit 1
fi

echo ""
echo "2. 检查配置文件..."
if [ -f "config_daily_template.json" ]; then
    echo "   ✅ 配置文件存在"
else
    echo "   ❌ 配置文件不存在"
    exit 1
fi

echo ""
echo "3. 清理旧的结果文件..."
rm -f Results/BasicTemplateAlgorithmDaily-*.txt
rm -f Results/BasicTemplateAlgorithmDaily-*.json
echo "   ✅ 清理完成"

echo ""
echo "======================================"
echo "开始运行 LEAN 回测..."
echo "======================================"
echo ""

# 运行 Docker 容器（使用 docker-compose）
CONFIG_FILE=config_daily_template.json docker-compose run --rm lean

echo ""
echo "======================================"
echo "测试完成！"
echo "======================================"
echo ""

# 显示结果
echo "查看结果文件..."
if [ -f "Results/BasicTemplateAlgorithmDaily-log.txt" ]; then
    echo ""
    echo "=== 最近的日志输出 ==="
    tail -n 20 Results/BasicTemplateAlgorithmDaily-log.txt
    echo ""
fi

if [ -f "Results/BasicTemplateAlgorithmDaily-summary.json" ]; then
    echo "=== 回测摘要 ==="
    cat Results/BasicTemplateAlgorithmDaily-summary.json
fi

echo ""
echo "✅ 所有结果文件保存在 Results/ 目录"
