# QUANTCONNECT.COM - Democratizing Finance, Empowering Individuals.
# Lean Algorithmic Trading Engine v2.0. Copyright 2014 QuantConnect Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from AlgorithmImports import *
import sys
import os

### <summary>
### SmartAlgorithm 基类 - 自动管理数据下载
### 
### 特性：
### 1. 继承此类后，add_equity_smart() 会自动检查并下载缺失数据
### 2. 无需手动运行 download_data.py
### 3. 保持代码清晰，数据管理逻辑封装在基类中
### 
### 使用方法：
###     class MyAlgorithm(SmartAlgorithm):  # 继承 SmartAlgorithm 而不是 QCAlgorithm
###         def initialize(self):
###             self.set_start_date(2024, 1, 1)
###             self.set_end_date(2025, 3, 31)
###             self.set_cash(100000)
###             
###             # 使用 add_equity_smart() 自动下载数据
###             self.spy = self.add_equity_smart("SPY", Resolution.DAILY)
### </summary>
class SmartAlgorithm(QCAlgorithm):
    '''
    智能算法基类 - 自动管理数据下载
    
    继承此类后，使用 add_equity_smart() 方法会自动：
    1. 检查本地数据是否充足
    2. 如果数据不足，自动下载
    3. 然后正常添加股票到算法中
    '''
    
    def __init__(self):
        super().__init__()
        self._data_ensured = set()  # 记录已确保数据的股票
        self._enable_auto_download = True  # 默认启用自动下载
    
    def add_equity_smart(self, ticker, resolution=Resolution.DAILY, market=Market.USA, 
                        fill_forward=True, leverage=2.0, extended_market_hours=False):
        '''
        智能添加股票 - 自动检查并下载数据
        
        Args:
            ticker: 股票代码（如 "SPY"）
            resolution: 数据分辨率（默认: Resolution.DAILY）
            其他参数与 add_equity() 相同
            
        Returns:
            Security 对象
        '''
        # 只有在启用自动下载时才检查
        if self._enable_auto_download and ticker not in self._data_ensured:
            self._ensure_data_available(ticker)
            self._data_ensured.add(ticker)
        
        # 调用原始的 add_equity 方法
        return self.add_equity(ticker, resolution, market, fill_forward, 
                              leverage, extended_market_hours)
    
    def _ensure_data_available(self, symbol):
        '''
        确保股票数据可用（内部方法）
        
        如果数据不足，会自动调用 download_data.py 下载
        '''
        try:
            # 导入 download_data 模块（从 Utils 目录）
            import sys
            sys.path.insert(0, '/workspace/Utils')
            from download_data import check_existing_data, download_and_convert
            
            # 获取回测日期范围
            start_date_str = self.start_date.strftime('%Y-%m-%d')
            end_date_str = self.end_date.strftime('%Y-%m-%d')
            
            # 检查本地数据
            existing_start, existing_end = check_existing_data(symbol)
            
            if existing_start and existing_end:
                # 判断数据是否充足
                if existing_start <= self.start_date and existing_end >= self.end_date:
                    self.debug(f"✅ {symbol}: 本地数据充足")
                    return
                else:
                    self.debug(f"⚠️ {symbol}: 数据不足，开始下载...")
            else:
                self.debug(f"📥 {symbol}: 本地无数据，开始下载...")
            
            # 下载数据
            success = download_and_convert(symbol, start_date_str, end_date_str)
            
            if success:
                self.debug(f"✅ {symbol}: 数据下载完成")
            else:
                self.debug(f"❌ {symbol}: 数据下载失败，回测可能无法正常运行")
                self.debug(f"   请手动运行: python3 download_data.py")
                
        except Exception as e:
            # 如果自动下载失败，给出清晰提示
            self.debug(f"⚠️ {symbol}: 自动下载失败 - {str(e)}")
            self.debug(f"   请手动运行: python3 download_data.py")
            self.debug(f"   或在 Docker 外运行:")
            self.debug(f"   cd /path/to/lean-multi-agent && python3 download_data.py")
    
    def disable_auto_download(self):
        '''禁用自动下载功能（如果不需要）'''
        self._enable_auto_download = False
        self.debug("已禁用自动数据下载")
    
    def enable_auto_download(self):
        '''启用自动下载功能'''
        self._enable_auto_download = True
        self.debug("已启用自动数据下载")
