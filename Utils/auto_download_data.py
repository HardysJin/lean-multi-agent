#!/usr/bin/env python3
"""
自动数据下载工具 - 在算法运行前自动下载所需数据
从配置文件中读取算法路径，分析代码提取股票列表和日期范围，自动下载数据
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

# 导入下载函数
try:
    from download_data import ensure_data_for_backtest
except ImportError:
    sys.path.insert(0, '/workspace/Utils')
    from download_data import ensure_data_for_backtest


def parse_config(config_path='/Lean/Launcher/bin/Debug/config.json'):
    """读取 LEAN 配置文件"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  无法读取配置文件: {e}")
        return None


def extract_symbols_from_code(algorithm_path):
    """从算法代码中提取股票代码"""
    try:
        with open(algorithm_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 匹配常见的股票代码模式
        patterns = [
            r'add[_-]?equity[_-]?smart\s*\(\s*["\']([A-Z]{1,5})["\']',  # add_equity_smart("SPY") 或 AddEquitySmart("SPY")
            r'add[_-]?equity\s*\(\s*["\']([A-Z]{1,5})["\']',  # add_equity("SPY") 或 AddEquity("SPY")
            r'self\.symbol\s*=\s*["\']([A-Z]{1,5})["\']',  # self.symbol = "SPY"
            r'[Ss]et[Bb]enchmark\s*\(\s*["\']([A-Z]{1,5})["\']',  # SetBenchmark("SPY")
            r'Symbol\.create\s*\(\s*["\']([A-Z]{1,5})["\']',  # Symbol.create("SPY", ...)
            r'Symbol\s*\.\s*create\s*\(\s*["\']([A-Z]{1,5})["\']',  # Symbol . create("SPY", ...)
        ]
        
        symbols = set()
        for pattern in patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            symbols.update([s.upper() for s in matches])
        
        # 过滤掉一些常见的非股票代码关键字
        exclude = {'SELF', 'TRUE', 'FALSE', 'NONE', 'DEBUG', 'INFO', 'ERROR', 'DATA', 
                   'TIME', 'DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'PRICE',
                   'BUY', 'SELL', 'ORDER', 'CASH', 'USD', 'DAILY', 'HOUR', 'MINUTE',
                   'CLASS', 'DEF', 'RETURN', 'IMPORT', 'FROM'}
        symbols = symbols - exclude
        
        return list(symbols)
    
    except Exception as e:
        print(f"⚠️  无法分析算法代码: {e}")
        return []


def extract_dates_from_code(algorithm_path):
    """从算法代码中提取日期范围"""
    try:
        with open(algorithm_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        start_date = None
        end_date = None
        
        # 匹配 SetStartDate / set_start_date
        start_pattern = r'[Ss]et[_-]?[Ss]tart[_-]?[Dd]ate\s*\(\s*(\d{4})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})\s*\)'
        match = re.search(start_pattern, code)
        if match:
            year, month, day = match.groups()
            start_date = datetime(int(year), int(month), int(day))
        
        # 匹配 SetEndDate / set_end_date
        end_pattern = r'[Ss]et[_-]?[Ee]nd[_-]?[Dd]ate\s*\(\s*(\d{4})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})\s*\)'
        match = re.search(end_pattern, code)
        if match:
            year, month, day = match.groups()
            end_date = datetime(int(year), int(month), int(day))
        
        # 如果没有找到，使用默认值
        if not start_date:
            start_date = datetime(2020, 1, 1)
        if not end_date:
            end_date = datetime.now()
        
        return start_date, end_date
    
    except Exception as e:
        print(f"⚠️  无法提取日期范围: {e}")
        return datetime(2020, 1, 1), datetime.now()


def auto_download_for_algorithm(config_path='/Lean/Launcher/bin/Debug/config.json'):
    """自动为算法下载数据"""
    print("\n" + "="*80)
    print("🔍 自动数据下载检测器")
    print("="*80)
    
    # 读取配置
    config = parse_config(config_path)
    if not config:
        print("⚠️  跳过自动下载")
        return
    
    # 获取算法路径
    algo_location = config.get('algorithm-location', '')
    if not algo_location:
        print("⚠️  未找到算法路径")
        return
    
    print(f"📁 算法路径: {algo_location}")
    
    # 检查文件是否存在
    if not Path(algo_location).exists():
        print(f"⚠️  算法文件不存在: {algo_location}")
        return
    
    # 提取股票代码
    symbols = extract_symbols_from_code(algo_location)
    if not symbols:
        print("⚠️  未检测到股票代码，跳过下载")
        return
    
    print(f"📊 检测到股票: {', '.join(symbols)}")
    
    # 提取日期范围
    start_date, end_date = extract_dates_from_code(algo_location)
    print(f"📅 日期范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    
    # 下载数据
    print("\n" + "-"*80)
    print("📥 开始检查并下载数据...")
    print("-"*80)
    
    try:
        ensure_data_for_backtest(symbols, start_date, end_date)
        print("\n" + "="*80)
        print("✅ 数据准备完成，启动算法...")
        print("="*80 + "\n")
    except Exception as e:
        print(f"\n⚠️  数据下载出错: {e}")
        print("⚠️  继续运行算法，但可能缺少数据\n")


if __name__ == "__main__":
    auto_download_for_algorithm()
