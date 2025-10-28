#!/bin/bash

echo "======================================"
echo "测试 LEAN 自带的 Python 算法"
echo "======================================"

# 使用自带的 BasicTemplateAlgorithm
docker run --rm \
  -v "$(pwd)/Data:/Lean/Data:rw" \
  -v "$(pwd)/config_builtin_python.json:/Lean/Launcher/bin/Debug/config.json:ro" \
  -v "$(pwd)/Results:/Results:rw" \
  -v "$(pwd)/Logs:/Lean/Logs:rw" \
  quantconnect/lean:latest

echo ""
echo "======================================"
echo "测试完成！请查看 Results 目录"
echo "======================================"
