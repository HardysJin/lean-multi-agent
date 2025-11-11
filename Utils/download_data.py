#!/usr/bin/env python3
"""
LEAN 市场数据下载工具 - 智能版
- 自动检查并补充缺失数据
- 支持增量更新（智能合并新旧数据）
- 支持多种分辨率：daily（日线）、minute（分钟线）、hour（小时线）
- 正确的LEAN格式（价格*10000）

使用方法：
    # 批量下载
    python3 download_data.py
    
    # 在代码中使用
    from download_data import ensure_data_for_backtest
    ensure_data_for_backtest(['SPY'], '2024-01-01', '2025-03-31', resolution='daily')
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import zipfile
from pathlib import Path
import time

# 默认配置
DEFAULT_SYMBOLS = ['SPY', 'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'TSLA']
DEFAULT_START = '2020-01-01'
DEFAULT_END = '2025-09-30'
DATA_DIR = './Lean/Data/equity/usa/daily'

# 分辨率映射
RESOLUTION_MAP = {
    'daily': {
        'interval': '1d',
        'dir': 'daily',
        'time_format': '%Y%m%d 00:00',
        'split_by_day': False,  # 日线数据：所有数据在一个文件
        'price_multiplier': 10000  # 日线价格乘以10000
    },
    'hour': {
        'interval': '1h',
        'dir': 'hour',
        'time_format': '%Y%m%d %H:%M',
        'split_by_day': False,  # 小时数据：所有数据在一个文件
        'price_multiplier': 10000
    },
    'minute': {
        'interval': '1m',
        'dir': 'minute',
        'time_format': '%Y%m%d %H:%M',
        'split_by_day': True,  # 分钟数据：按天分文件
        'price_multiplier': 10000  # 分钟数据价格也乘以10000
    }
}

def get_data_dir_for_resolution(resolution='daily', symbol=None):
    """根据分辨率获取数据目录
    
    Args:
        resolution: 分辨率 (daily, hour, minute)
        symbol: 股票代码（分钟数据需要单独的文件夹）
    
    Returns:
        数据目录路径
    """
    base_dir = './Lean/Data/equity/usa'
    if resolution in RESOLUTION_MAP:
        res_dir = f"{base_dir}/{RESOLUTION_MAP[resolution]['dir']}"
        # 分钟数据需要为每个股票创建单独的文件夹
        if resolution == 'minute' and symbol:
            return f"{res_dir}/{symbol.lower()}"
        return res_dir
    return f"{base_dir}/daily"

def check_existing_data(symbol, data_dir=DATA_DIR, resolution='daily'):
    """检查本地已有数据的日期范围
    
    Args:
        symbol: 股票代码
        data_dir: 数据目录
        resolution: 分辨率
    
    Returns:
        (start_date, end_date) 或 (None, None)
    """
    # 使用正确的目录
    actual_dir = get_data_dir_for_resolution(resolution, symbol) if data_dir == DATA_DIR else data_dir
    
    # 分钟数据按天分文件，需要检查文件夹中的所有文件
    if resolution == 'minute':
        if not Path(actual_dir).exists():
            return None, None
        
        try:
            # 获取所有 YYYYMMDD_trade.zip 文件
            trade_files = sorted([f for f in os.listdir(actual_dir) if f.endswith('_trade.zip')])
            
            if not trade_files:
                return None, None
            
            # 从文件名解析日期
            first_date = datetime.strptime(trade_files[0][:8], '%Y%m%d')
            last_date = datetime.strptime(trade_files[-1][:8], '%Y%m%d')
            
            return first_date, last_date
        except Exception:
            return None, None
    
    # 日线和小时数据：单个 zip 文件
    else:
        zip_path = Path(actual_dir) / f"{symbol.lower()}.zip"
        
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

def _download_minute_data_in_batches(ticker, start_date, end_date, interval):
    """分批下载分钟数据（绕过 Yahoo Finance 8天限制）
    
    Yahoo Finance API 限制：1分钟数据每次请求最多8天
    
    Args:
        ticker: yfinance Ticker 对象
        start_date: 开始日期（字符串或datetime）
        end_date: 结束日期（字符串或datetime）
        interval: 数据间隔 (如 '1m')
    
    Returns:
        合并后的 DataFrame
    """
    from datetime import datetime, timedelta
    
    # 转换为 datetime
    if isinstance(start_date, str):
        start_dt = pd.to_datetime(start_date)
    else:
        start_dt = start_date if isinstance(start_date, datetime) else pd.to_datetime(start_date)
    
    if isinstance(end_date, str):
        end_dt = pd.to_datetime(end_date)
    else:
        end_dt = end_date if isinstance(end_date, datetime) else pd.to_datetime(end_date)
    
    # 计算总天数
    total_days = (end_dt - start_dt).days
    
    # 如果小于等于7天，直接下载
    if total_days <= 7:
        print(f"   时间范围: {total_days} 天，直接下载")
        return ticker.history(start=start_dt, end=end_dt, interval=interval)
    
    # 分批下载（每批7天）
    batch_size = 7
    batches = []
    current_start = start_dt
    batch_num = 0
    
    print(f"   时间范围: {total_days} 天，将分 {(total_days // batch_size) + 1} 批下载")
    
    while current_start < end_dt:
        batch_num += 1
        current_end = min(current_start + timedelta(days=batch_size), end_dt)
        
        print(f"   批次 {batch_num}: {current_start.strftime('%Y-%m-%d')} 到 {current_end.strftime('%Y-%m-%d')}", end=' ')
        
        try:
            batch_df = ticker.history(start=current_start, end=current_end, interval=interval)
            
            if not batch_df.empty:
                batches.append(batch_df)
                print(f"✓ ({len(batch_df)} 条)")
            else:
                print("⚠️ 无数据")
                
            # 避免触发频率限制
            time.sleep(0.5)
            
        except Exception as e:
            print(f"✗ 错误: {e}")
        
        current_start = current_end
    
    # 合并所有批次
    if not batches:
        print("❌ 所有批次都失败，返回空数据")
        return pd.DataFrame()
    
    print(f"   合并 {len(batches)} 个批次的数据...")
    merged_df = pd.concat(batches)
    
    # 去重（可能有重叠）
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
    
    # 排序
    merged_df = merged_df.sort_index()
    
    return merged_df

def _save_minute_data_by_day(df, symbol, data_dir, res_config):
    """保存分钟数据（按天分文件，LEAN 官方格式）
    
    格式：equity/usa/minute/{symbol}/YYYYMMDD_trade.zip
    文件内容：YYYYMMDD_symbol_minute_trade.csv  
    价格格式：整数（价格 × 10000）- LEAN标准格式
    
    Args:
        df: DataFrame with minute data
        symbol: Stock symbol
        data_dir: Directory to save files
        res_config: Resolution configuration
    
    Returns:
        Number of days saved
    """
    multiplier = res_config['price_multiplier']
    
    # 按日期分组
    df_grouped = df.groupby(df.index.date)
    saved_count = 0
    
    for date, day_data in df_grouped:
        date_str = date.strftime('%Y%m%d')
        
        # 文件名：YYYYMMDD_trade.zip
        zip_filename = f"{date_str}_trade.zip"
        zip_path = Path(data_dir) / zip_filename
        
        # CSV 文件名：YYYYMMDD_symbol_minute_trade.csv
        csv_filename = f"{date_str}_{symbol.lower()}_minute_trade.csv"
        
        # 转换为 LEAN 格式（时间=毫秒数，价格=整数×10000）
        lean_data = []
        for timestamp, row in day_data.iterrows():
            # LEAN分钟数据使用"从午夜开始的毫秒数"作为时间戳
            milliseconds = (timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second) * 1000
            open_price = int(row['Open'] * multiplier)
            high_price = int(row['High'] * multiplier)
            low_price = int(row['Low'] * multiplier)
            close_price = int(row['Close'] * multiplier)
            volume = int(row['Volume'])
            lean_data.append(f"{milliseconds},{open_price},{high_price},{low_price},{close_price},{volume}")
        
        # 保存到 zip 文件
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(csv_filename, '\n'.join(lean_data))
        
        saved_count += 1
    
    return saved_count

def _save_consolidated_data(df, symbol, data_dir, res_config, existing_start, existing_end):
    """保存日线/小时数据（单个文件）
    
    格式：equity/usa/{resolution}/{symbol}.zip
    价格格式：整数（乘以10000）
    
    Args:
        df: DataFrame with data
        symbol: Stock symbol
        data_dir: Directory to save file
        res_config: Resolution configuration
        existing_start: Existing data start date (for merging)
        existing_end: Existing data end date (for merging)
    """
    multiplier = res_config['price_multiplier']
    
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
                timestamp_str = parts[0]
                old_data[timestamp_str] = line
            
            # 添加新数据（新数据优先）
            for date, row in df.iterrows():
                date_str = date.strftime(res_config['time_format'])
                open_price = int(row['Open'] * multiplier)
                high_price = int(row['High'] * multiplier)
                low_price = int(row['Low'] * multiplier)
                close_price = int(row['Close'] * multiplier)
                volume = int(row['Volume'])
                old_data[date_str] = f"{date_str},{open_price},{high_price},{low_price},{close_price},{volume}"
            
            # 按时间戳排序
            sorted_timestamps = sorted(old_data.keys())
            lean_data = [old_data[ts] for ts in sorted_timestamps]
    else:
        # 转换为 LEAN 格式
        lean_data = []
        for date, row in df.iterrows():
            date_str = date.strftime(res_config['time_format'])
            open_price = int(row['Open'] * multiplier)
            high_price = int(row['High'] * multiplier)
            low_price = int(row['Low'] * multiplier)
            close_price = int(row['Close'] * multiplier)
            volume = int(row['Volume'])
            lean_data.append(f"{date_str},{open_price},{high_price},{low_price},{close_price},{volume}")
    
    # 保存到 zip 文件
    zip_path = Path(data_dir) / f"{symbol.lower()}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{symbol.lower()}.csv", '\n'.join(lean_data))

def download_and_convert(symbol, start_date, end_date, data_dir=DATA_DIR, resolution='daily'):
    """下载并转换单个股票数据（智能增量更新）
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        data_dir: 数据目录（如果是默认值，会自动根据resolution调整）
        resolution: 分辨率 ('daily', 'hour', 'minute')
    """
    # 获取分辨率配置
    res_config = RESOLUTION_MAP.get(resolution, RESOLUTION_MAP['daily'])
    
    # 使用正确的目录
    actual_dir = get_data_dir_for_resolution(resolution, symbol) if data_dir == DATA_DIR else data_dir
    
    print(f"\n{'='*60}")
    print(f"下载 {symbol} 数据 ({resolution})...")
    print(f"{'='*60}")
    
    # 检查已有数据
    existing_start, existing_end = check_existing_data(symbol, data_dir, resolution)
    
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
        # 下载数据
        print(f"📥 从 Yahoo Finance 下载 {symbol} ({resolution})...")
        print(f"   时间范围: {start_date} 到 {end_date}")
        print(f"   数据间隔: {res_config['interval']}")
        
        ticker = yf.Ticker(symbol)
        
        # 根据分辨率下载对应数据
        # 注意：Yahoo Finance 对分钟数据有时间限制（每次请求最多8天）
        if resolution == 'minute':
            print(f"⚠️  注意：Yahoo Finance 分钟数据限制每次请求8天")
            # 分批下载
            df = _download_minute_data_in_batches(ticker, start_date, end_date, res_config['interval'])
        else:
            df = ticker.history(start=start_date, end=end_date, interval=res_config['interval'])
        
        if df.empty:
            print(f"❌ {symbol}: 未获取到数据")
            if resolution == 'minute':
                print(f"   可能原因:")
                print(f"   1. Yahoo Finance 对分钟数据有时间限制")
                print(f"   2. 股票代码不存在或已退市")
                print(f"   3. 时间范围超出可用范围")
                print(f"   建议: 尝试缩短时间范围（例如最近30天）")
            return False
        
        data_type = '条' if resolution in ['minute', 'hour'] else '天'
        print(f"✅ 获取到 {len(df)} {data_type}的数据")
        print(f"   日期: {df.index[0].strftime('%Y-%m-%d %H:%M')} 到 {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
        
        # 确保目录存在
        os.makedirs(actual_dir, exist_ok=True)
        
        # 根据分辨率使用不同的保存格式
        if res_config['split_by_day']:
            # 分钟数据：按天分文件保存
            print(f"📁 按天分文件保存到: {actual_dir}/")
            saved_count = _save_minute_data_by_day(df, symbol, actual_dir, res_config)
            print(f"💾 保存完成: {saved_count} 个交易日")
        else:
            # 日线/小时数据：单个文件保存
            _save_consolidated_data(df, symbol, actual_dir, res_config, existing_start, existing_end)
            print(f"💾 已保存到: {actual_dir}/{symbol.lower()}.zip")
            print(f"   总共 {len(df)} {data_type}的数据")
        
        return True
        
    except Exception as e:
        print(f"❌ {symbol}: 下载失败")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误信息: {str(e)}")
        
        # 根据错误类型给出建议
        error_msg = str(e).lower()
        if 'no data found' in error_msg or 'empty' in error_msg:
            print(f"\n   可能原因:")
            print(f"   - 股票代码不存在或拼写错误")
            print(f"   - 该股票在指定日期范围内没有交易数据")
            if resolution == 'minute':
                print(f"   - 分钟数据的时间范围超出限制（Yahoo Finance 通常只提供最近7-60天）")
        elif 'connection' in error_msg or 'timeout' in error_msg:
            print(f"\n   可能原因:")
            print(f"   - 网络连接问题")
            print(f"   - Yahoo Finance 服务暂时不可用")
        
        print(f"\n   详细错误追踪:")
        import traceback
        traceback.print_exc()
        
        return False

def ensure_data_for_backtest(symbols, start_date, end_date, data_dir=DATA_DIR, resolution='daily'):
    """
    为回测准备数据，确保所有股票数据充足
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        data_dir: 数据目录
        resolution: 分辨率 ('daily', 'hour', 'minute')
        
    Returns:
        bool: 是否所有数据都准备完成
    """
    print("="*70)
    print(f"LEAN 数据自动准备 ({resolution})")
    print("="*70)
    print(f"股票: {', '.join(symbols)}")
    print(f"日期: {start_date} 到 {end_date}")
    print(f"分辨率: {resolution}")
    print("="*70)
    
    all_ready = True
    for symbol in symbols:
        if not download_and_convert(symbol, start_date, end_date, data_dir, resolution):
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
