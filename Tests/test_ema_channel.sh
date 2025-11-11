#!/bin/bash

echo "======================================"
echo "测试 EMA 通道突破策略"
echo "使用 30 分钟 K 线数据"
echo "======================================"
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1

# 检查文件是否存在
echo "1. 检查算法文件..."
if [ -f "Algorithm/EMAChannelStrategy.py" ]; then
    echo "   ✅ 算法文件存在"
else
    echo "   ❌ 算法文件不存在"
    exit 1
fi

echo ""
echo "2. 检查配置文件..."
if [ -f "Configs/config_ema_channel.json" ]; then
    echo "   ✅ 配置文件存在"
else
    echo "   ❌ 配置文件不存在"
    exit 1
fi

echo ""
echo "3. 清理旧的结果文件..."
rm -f Results/EMAChannelStrategy-*.txt
rm -f Results/EMAChannelStrategy-*.json
echo "   ✅ 清理完成"

echo ""
echo "======================================"
echo "开始运行 LEAN 回测..."
echo "======================================"
echo ""
echo "策略说明："
echo "  - 蓝色通道: UP1=EMA(H,25), LOW1=EMA(L,25)"
echo "  - 黄色通道: UP2=EMA(H,90), LOW2=EMA(L,90)"
echo "  - 买入信号: 价格突破蓝色通道上沿 (Close > UP1)"
echo "  - 卖出信号: 价格跌破蓝色通道下沿 (Close < LOW1)"
echo ""

# 运行 Docker 容器（使用 docker-compose）
CONFIG_FILE=config_ema_channel.json docker-compose run --rm lean

echo ""
echo "======================================"
echo "测试完成！"
echo "======================================"
echo ""

# 显示结果
echo "查看结果文件..."

# 显示完整的回测分析报告
if [ -f "Results/EMAChannelStrategy-order-events.json" ] && [ -f "Results/EMAChannelStrategy-summary.json" ]; then
    python3 Utils/show_backtest_results.py Results/EMAChannelStrategy-order-events.json Results/EMAChannelStrategy-summary.json
fi

if [ -f "Results/EMAChannelStrategy-log.txt" ]; then
    echo ""
    echo "=== 算法日志（最后 20 行）==="
    tail -n 20 Results/EMAChannelStrategy-log.txt
    echo ""
fi

echo ""
echo "✅ 所有结果文件保存在 Results/ 目录"
echo ""
echo "查看详细日志:"
echo "  cat Results/EMAChannelStrategy-log.txt"
echo ""
echo "查看订单记录:"
echo "  cat Results/EMAChannelStrategy-order-events.json"
echo ""
echo "查看回测总结:"
echo "  cat Results/EMAChannelStrategy-summary.json"
echo ""
