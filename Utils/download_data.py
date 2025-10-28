#!/usr/bin/env python3
"""
LEAN 市场数据下载工具 - 智能版
- 自动检查并补充缺失数据
- 支持增量更新（智能合并新旧数据）
- 只下载日线数据（更快、节省空间）
- 正确的LEAN格式（价格*10000）

使用方法：
    # 批量下载
    python3 download_data.py
    
    # 在代码中使用
    from download_data import ensure_data_for_backtest
    ensure_data_for_backtest(['SPY'], '2024-01-01', '2025-03-31')
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import zipfile
from pathlib import Path

# 默认配置
DEFAULT_SYMBOLS = ['SPY', 'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'TSLA']
DEFAULT_START = '2020-01-01'
DEFAULT_END = '2025-09-30'
DATA_DIR = 'Data/equity/usa/daily'

def check_existing_data(symbol, data_dir=DATA_DIR):
    """检查本地已有数据的日期范围"""
    zip_path = Path(data_dir) / f"{symbol.lower()}.zip"
    
    if not zip_path.exists():
        return None, None
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            csv_name = f"{symbol.lower()}.csv"
            if csv_name not in zf.namelist():
                return None, None
            
            content = zf.read(csv_name).decode('utf-8')
            lines = content.strip().split('\n')
            
            if len(lines) < 2:
                return None, None
            
            # 解析第一行和最后一行的日期
            first_date = datetime.strptime(lines[0].split(',')[0], '%Y%m%d %H:%M')
            last_date = datetime.strptime(lines[-1].split(',')[0], '%Y%m%d %H:%M')
            
            return first_date, last_date
    except Exception:
        return None, None

def download_and_convert(symbol, start_date, end_date, data_dir=DATA_DIR):
    """下载并转换单个股票的日线数据（智能增量更新）"""
    print(f"\n{'='*60}")
    print(f"下载 {symbol} 数据...")
    print(f"{'='*60}")
    
    # 检查已有数据
    existing_start, existing_end = check_existing_data(symbol, data_dir)
    
    if existing_start and existing_end:
        print(f"📁 本地数据: {existing_start.strftime('%Y-%m-%d')} 到 {existing_end.strftime('%Y-%m-%d')}")
        
        # 判断是否需要更新
        req_start = pd.to_datetime(start_date)
        req_end = pd.to_datetime(end_date)
        
        if existing_start <= req_start and existing_end >= req_end:
            print(f"✅ {symbol}: 数据充足，无需下载")
            return True
        else:
            print(f"📥 需要补充数据...")
    
    try:
        # 下载日线数据
        print(f"📥 从 Yahoo Finance 下载 {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            print(f"❌ {symbol}: 未获取到数据")
            return False
        
        print(f"✅ 获取到 {len(df)} 天的数据")
        print(f"   日期: {df.index[0].strftime('%Y-%m-%d')} 到 {df.index[-1].strftime('%Y-%m-%d')}")
        
        # 如果有已有数据，合并
        if existing_start and existing_end:
            print(f"🔄 合并新旧数据...")
            zip_path = Path(data_dir) / f"{symbol.lower()}.zip"
            with zipfile.ZipFile(zip_path, 'r') as zf:
                content = zf.read(f"{symbol.lower()}.csv").decode('utf-8')
                lines = content.strip().split('\n')
                
                # 解析旧数据到字典
                old_data = {}
                for line in lines:
                    parts = line.split(',')
                    date_str = parts[0].split()[0]  # 20240102
                    date = datetime.strptime(date_str, '%Y%m%d')
                    old_data[date] = line
                
                # 添加新数据（新数据优先）
                for date, row in df.iterrows():
                    date_only = datetime(date.year, date.month, date.day)
                    date_str = date.strftime('%Y%m%d 00:00')
                    open_price = int(row['Open'] * 10000)
                    high_price = int(row['High'] * 10000)
                    low_price = int(row['Low'] * 10000)
                    close_price = int(row['Close'] * 10000)
                    volume = int(row['Volume'])
                    old_data[date_only] = f"{date_str},{open_price},{high_price},{low_price},{close_price},{volume}"
                
                # 按日期排序
                sorted_dates = sorted(old_data.keys())
                lean_data = [old_data[d] for d in sorted_dates]
        else:
            # 转换为LEAN格式（价格*10000）
            lean_data = []
            for date, row in df.iterrows():
                date_str = date.strftime('%Y%m%d 00:00')
                open_price = int(row['Open'] * 10000)
                high_price = int(row['High'] * 10000)
                low_price = int(row['Low'] * 10000)
                close_price = int(row['Close'] * 10000)
                volume = int(row['Volume'])
                lean_data.append(f"{date_str},{open_price},{high_price},{low_price},{close_price},{volume}")
        
        # 确保目录存在并保存
        os.makedirs(data_dir, exist_ok=True)
        zip_path = Path(data_dir) / f"{symbol.lower()}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{symbol.lower()}.csv", '\n'.join(lean_data))
        
        print(f"💾 已保存到: {zip_path}")
        print(f"   总共 {len(lean_data)} 天的数据")
        
        return True
        
    except Exception as e:
        print(f"❌ {symbol}: 下载失败 - {e}")
        import traceback
        traceback.print_exc()
        return False

def ensure_data_for_backtest(symbols, start_date, end_date, data_dir=DATA_DIR):
    """
    为回测准备数据，确保所有股票数据充足
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        data_dir: 数据目录
        
    Returns:
        bool: 是否所有数据都准备完成
    """
    print("="*70)
    print("LEAN 数据自动准备")
    print("="*70)
    print(f"股票: {', '.join(symbols)}")
    print(f"日期: {start_date} 到 {end_date}")
    print("="*70)
    
    all_ready = True
    for symbol in symbols:
        if not download_and_convert(symbol, start_date, end_date, data_dir):
            all_ready = False
    
    print(f"\n{'='*70}")
    if all_ready:
        print("✅ 所有数据准备完成")
    else:
        print("❌ 部分数据准备失败")
    print("="*70)
    
    return all_ready

def main():
    """主函数 - 批量下载默认股票"""
    print("="*60)
    print("LEAN 市场数据下载工具")
    print("="*60)
    print(f"股票列表: {', '.join(DEFAULT_SYMBOLS)}")
    print(f"日期范围: {DEFAULT_START} 到 {DEFAULT_END}")
    print(f"数据目录: {DATA_DIR}")
    print(f"数据类型: 仅日线数据")
    print("="*60)
    
    # 检查yfinance是否安装
    try:
        import yfinance
        print("\n✅ yfinance 已安装")
    except ImportError:
        print("\n❌ 需要安装 yfinance")
        print("   运行: pip install yfinance pandas")
        return
    
    # 下载每个股票的数据
    success_count = 0
    for symbol in DEFAULT_SYMBOLS:
        if download_and_convert(symbol, DEFAULT_START, DEFAULT_END):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"下载完成！")
    print(f"成功: {success_count}/{len(DEFAULT_SYMBOLS)}")
    print(f"{'='*60}")
    
    if success_count == len(DEFAULT_SYMBOLS):
        print("\n✅ 所有数据下载成功")
    else:
        print(f"\n⚠️  部分股票下载失败")

if __name__ == '__main__':
    main()
