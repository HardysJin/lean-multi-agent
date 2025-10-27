#!/usr/bin/env python3
"""
下载股票历史数据并转换为LEAN格式
使用yfinance免费下载数据
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import zipfile
from pathlib import Path

# 配置
SYMBOLS = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'TSLA']
START_DATE = '2023-01-01'
END_DATE = '2024-01-31'
DATA_DIR = 'Data/equity/usa'

def download_and_convert(symbol, start_date, end_date):
    """下载并转换单个股票的数据"""
    print(f"\n{'='*60}")
    print(f"下载 {symbol} 数据...")
    print(f"{'='*60}")
    
    try:
        # 下载日线数据
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            print(f"❌ {symbol}: 未获取到数据")
            return False
        
        print(f"✅ {symbol}: 获取到 {len(df)} 天的数据")
        
        # 创建日线数据目录
        daily_dir = Path(DATA_DIR) / 'daily' / symbol.lower()
        daily_dir.mkdir(parents=True, exist_ok=True)
        
        # 转换为LEAN格式并保存
        lean_data = []
        for date, row in df.iterrows():
            # LEAN日线格式: Date,Open,High,Low,Close,Volume
            date_str = date.strftime('%Y%m%d 00:00')
            lean_data.append(f"{date_str},{row['Open']:.2f},{row['High']:.2f},{row['Low']:.2f},{row['Close']:.2f},{int(row['Volume'])}")
        
        # 写入zip文件
        zip_path = daily_dir / f"{symbol.lower()}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{symbol.lower()}.csv", '\n'.join(lean_data))
        
        print(f"✅ {symbol}: 数据已保存到 {zip_path}")
        
        # 下载分钟数据（使用1小时数据代替，因为免费API限制）
        print(f"   下载 {symbol} 分钟级数据...")
        
        # 分批下载（yfinance限制）
        minute_dir = Path(DATA_DIR) / 'minute' / symbol.lower()
        minute_dir.mkdir(parents=True, exist_ok=True)
        
        # 每次下载7天的数据
        current_date = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        downloaded_days = 0
        
        while current_date < end:
            batch_end = min(current_date + timedelta(days=7), end)
            
            try:
                df_minute = ticker.history(
                    start=current_date.strftime('%Y-%m-%d'),
                    end=batch_end.strftime('%Y-%m-%d'),
                    interval='1h'  # 使用1小时数据（免费）
                )
                
                if not df_minute.empty:
                    # 按日期分组保存
                    for date in df_minute.index.date:
                        date_data = df_minute[df_minute.index.date == date]
                        if not date_data.empty:
                            save_minute_data(symbol, date, date_data, minute_dir)
                            downloaded_days += 1
                
            except Exception as e:
                print(f"   ⚠️  下载 {current_date.strftime('%Y-%m-%d')} 批次失败: {e}")
            
            current_date = batch_end
        
        if downloaded_days > 0:
            print(f"✅ {symbol}: 已保存 {downloaded_days} 天的分钟数据")
        else:
            print(f"⚠️  {symbol}: 分钟数据下载失败，将使用日线数据")
        
        return True
        
    except Exception as e:
        print(f"❌ {symbol}: 下载失败 - {e}")
        return False

def save_minute_data(symbol, date, df, minute_dir):
    """保存分钟级数据"""
    date_str = pd.to_datetime(date).strftime('%Y%m%d')
    
    # LEAN分钟格式: DateTime,Open,High,Low,Close,Volume
    lean_data = []
    for timestamp, row in df.iterrows():
        time_str = timestamp.strftime('%Y%m%d %H:%M')
        lean_data.append(f"{time_str},{row['Open']:.2f},{row['High']:.2f},{row['Low']:.2f},{row['Close']:.2f},{int(row['Volume'])}")
    
    if lean_data:
        # 保存trade数据
        zip_path = minute_dir / f"{date_str}_trade.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{date_str}_{symbol.lower()}_minute_trade.csv", '\n'.join(lean_data))

def main():
    print("="*60)
    print("LEAN 市场数据下载工具")
    print("="*60)
    print(f"股票列表: {', '.join(SYMBOLS)}")
    print(f"日期范围: {START_DATE} 到 {END_DATE}")
    print(f"数据目录: {DATA_DIR}")
    print("="*60)
    
    # 检查yfinance是否安装
    try:
        import yfinance
        print("✅ yfinance 已安装")
    except ImportError:
        print("❌ 需要安装 yfinance")
        print("   运行: pip install yfinance")
        return
    
    # 创建数据目录
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 下载每个股票的数据
    success_count = 0
    for symbol in SYMBOLS:
        if download_and_convert(symbol, START_DATE, END_DATE):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"下载完成！")
    print(f"成功: {success_count}/{len(SYMBOLS)}")
    print(f"{'='*60}")
    
    if success_count == len(SYMBOLS):
        print("\n✅ 所有数据下载成功，可以运行回测了！")
        print("   运行: bash run.sh")
    else:
        print(f"\n⚠️  部分股票下载失败，请检查网络连接后重试")

if __name__ == '__main__':
    main()
