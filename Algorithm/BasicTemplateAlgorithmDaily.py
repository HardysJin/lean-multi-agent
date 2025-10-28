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
from SmartAlgorithm import SmartAlgorithm

### <summary>
### 基础模板算法 - 使用 SmartAlgorithm 自动管理数据
### 
### 特性：
### - 继承 SmartAlgorithm 基类，自动检查和下载数据
### - 使用 add_equity_smart() 方法，无需手动运行 download_data.py
### - 代码简洁清晰，数据管理逻辑在基类中
### </summary>
class BasicTemplateAlgorithmDaily(SmartAlgorithm):  # 继承 SmartAlgorithm
    '''基础模板算法，使用日线数据进行回测，自动管理数据下载'''

    def initialize(self):
        '''初始化算法：设置日期范围、现金和股票代码'''
        
        # ===== 配置回测参数 =====
        self.set_start_date(2020, 1, 2)   # 开始日期
        self.set_end_date(2025, 3, 31)    # 结束日期
        self.set_cash(100000)              # 初始资金
        
        # ===== 添加股票（自动下载数据）=====
        # 使用 add_equity_smart() 会自动检查并下载缺失的数据
        self.spy = self.add_equity_smart("SPY", Resolution.DAILY)
        
        # 调试信息
        self.debug("=" * 50)
        self.debug("算法初始化成功！")
        self.debug(f"回测期间：{self.start_date} 到 {self.end_date}")
        self.debug(f"初始资金：${self.portfolio.cash:,.2f}")
        self.debug(f"数据分辨率：DAILY")
        self.debug("=" * 50)
        
        # 用于跟踪交易
        self.trade_count = 0
        self.data_points = 0

    def on_data(self, data):
        '''每个数据点到达时调用此方法
        
        Arguments:
            data: Slice 对象，包含股票数据
        '''
        self.data_points += 1
        
        # 调试：每 10 个数据点打印一次
        if self.data_points % 10 == 0:
            self.debug(f"已处理 {self.data_points} 个数据点")
        
        # 检查是否有 SPY 数据
        if "SPY" not in data or data["SPY"] is None:
            return
            
        # 获取当前价格
        spy_price = data["SPY"].close
        
        # 简单策略：如果还没有持仓，就买入
        if not self.portfolio.invested:
            # 计算可以买入的股数
            quantity = int(self.portfolio.cash / spy_price)
            
            # 买入
            self.market_order("SPY", quantity)
            self.trade_count += 1
            
            self.debug("=" * 50)
            self.debug(f"交易 #{self.trade_count}")
            self.debug(f"时间：{self.time}")
            self.debug(f"价格：${spy_price:.2f}")
            self.debug(f"买入：{quantity} 股")
            self.debug(f"投资金额：${quantity * spy_price:,.2f}")
            self.debug("=" * 50)
    
    def on_end_of_algorithm(self):
        '''算法结束时调用此方法'''
        self.debug("=" * 50)
        self.debug("算法执行完成！")
        self.debug(f"总数据点：{self.data_points}")
        self.debug(f"总交易次数：{self.trade_count}")
        self.debug(f"最终资产：${self.portfolio.total_portfolio_value:,.2f}")
        self.debug(f"收益：${self.portfolio.total_portfolio_value - 100000:,.2f}")
        self.debug(f"收益率：{((self.portfolio.total_portfolio_value / 100000) - 1) * 100:.2f}%")
        self.debug("=" * 50)
