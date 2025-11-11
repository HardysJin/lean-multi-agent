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
### EMA通道突破策略 - 基于30分钟K线
### 
### 策略逻辑：
### - 蓝色通道：UP1 = EMA(H, 25), LOW1 = EMA(L, 25)
### - 黄色通道：UP2 = EMA(H, 90), LOW2 = EMA(L, 90)
### - 买入信号：价格突破蓝色通道上沿 (收盘价 > UP1)
### - 卖出信号：价格跌破蓝色通道下沿 (收盘价 < LOW1)
### </summary>
class EMAChannelStrategy(SmartAlgorithm):
    '''EMA通道突破策略，使用30分钟K线数据'''

    def initialize(self):
        '''初始化算法：设置日期范围、现金和股票代码'''
        
        # ===== 配置回测参数 =====
        # 注意：分钟数据已支持分批下载，可以使用较长时间范围
        # Yahoo Finance 会自动分批下载（每批7天）
        self.set_start_date(2025, 10, 1)   # 开始日期（可以回测更长时间）
        self.set_end_date(2025, 11, 7)      # 结束日期
        self.set_cash(100000)                # 初始资金
        
        # ===== 添加股票（自动下载数据）=====
        # 使用分钟分辨率，然后合成30分钟K线
        self.symbol = "SOXL"
        self.equity = self.add_equity_smart(self.symbol, Resolution.MINUTE)
        
        # 设置预热期（确保有足够的数据）
        self.set_warmup(timedelta(days=5))
        
        # ===== 设置30分钟K线合成器 =====
        # 使用QuoteBarConsolidator合成30分钟K线
        self.consolidator = TradeBarConsolidator(timedelta(minutes=30))
        self.consolidator.data_consolidated += self.on_data_consolidated
        self.subscription_manager.add_consolidator(self.symbol, self.consolidator)
        
        # ===== 初始化EMA指标 =====
        # 蓝色通道 (25周期)
        self.ema_high_25 = self.ema(self.symbol, 25, Resolution.MINUTE)
        self.ema_low_25 = self.ema(self.symbol, 25, Resolution.MINUTE)
        
        # 黄色通道 (90周期) - 可用于趋势确认
        self.ema_high_90 = self.ema(self.symbol, 90, Resolution.MINUTE)
        self.ema_low_90 = self.ema(self.symbol, 90, Resolution.MINUTE)
        
        # 手动跟踪高低价的EMA，因为内置EMA只能跟踪收盘价
        # 我们需要自定义指标来跟踪高低价
        self.high_window = RollingWindow[float](25)
        self.low_window = RollingWindow[float](25)
        self.high_window_90 = RollingWindow[float](90)
        self.low_window_90 = RollingWindow[float](90)
        
        # 成交量窗口（用于计算均量）
        self.volume_window = RollingWindow[float](5)
        
        # 手动计算的EMA值
        self.up1 = 0  # EMA(H, 25)
        self.low1 = 0  # EMA(L, 25)
        self.up2 = 0  # EMA(H, 90)
        self.low2 = 0  # EMA(L, 90)
        
        # EMA平滑系数
        self.alpha_25 = 2.0 / (25 + 1)
        self.alpha_90 = 2.0 / (90 + 1)
        
        # ===== 交易参数 =====
        self.stop_loss_pct = 0.05  # 止损5%
        self.take_profit_pct = 0.10  # 止盈10%
        self.volume_multiplier = 2.0  # 成交量放大倍数（提高到2倍，减少假突破）
        self.volume_lookback = 5  # 成交量回看周期
        self.cooldown_bars = 3  # 交易后冷却3个30分钟K线（90分钟）
        
        # ===== 交易状态跟踪 =====
        self.previous_price = 0
        self.trade_count = 0
        self.data_points = 0
        self.is_ready = False
        self.entry_price = 0  # 买入价格（用于止损/止盈）
        self.last_trade_bar = -999  # 上次交易的K线编号（用于冷却）
        
        # 调试信息
        self.debug("=" * 50)
        self.debug("EMA通道突破策略初始化成功！")
        self.debug(f"回测期间：{self.start_date} 到 {self.end_date}")
        self.debug(f"初始资金：${self.portfolio.cash:,.2f}")
        self.debug(f"数据分辨率：30分钟K线")
        self.debug(f"交易品种：{self.symbol}")
        self.debug("策略逻辑（已优化 v2）：")
        self.debug("  买入条件：")
        self.debug("    1. 收盘价突破UP1（蓝色上沿）")
        self.debug("    2. 成交量 > 5周期均量 × 2.0倍")
        self.debug("    3. 价格在UP2（黄色上沿）之上")
        self.debug("    4. 距离上次交易 > 3个K线（90分钟）")
        self.debug("  卖出条件：")
        self.debug("    1. 收盘价跌破LOW1（蓝色下沿）")
        self.debug("    2. 或止损（-5%）")
        self.debug("    3. 或止盈（+10%）")
        self.debug("=" * 50)

    def on_data(self, data):
        '''每个数据点到达时更新高低价窗口并执行交易'''
        # 跳过预热期
        if self.is_warming_up:
            return
            
        if self.symbol not in data or data[self.symbol] is None:
            return
        
        bar = data[self.symbol]
        
        # 更新滚动窗口
        self.high_window.add(bar.high)
        self.low_window.add(bar.low)
        self.high_window_90.add(bar.high)
        self.low_window_90.add(bar.low)
        self.volume_window.add(bar.volume)
        
        # 计算EMA - 分别计算高价和低价的EMA
        if self.high_window.is_ready:
            if self.up1 == 0:
                # 初始化为SMA（使用高价）
                self.up1 = sum([self.high_window[i] for i in range(self.high_window.count)]) / self.high_window.count
            else:
                # 更新EMA（使用高价）
                self.up1 = bar.high * self.alpha_25 + self.up1 * (1 - self.alpha_25)
            
            if self.low1 == 0:
                # 初始化为SMA（使用低价）
                self.low1 = sum([self.low_window[i] for i in range(self.low_window.count)]) / self.low_window.count
            else:
                # 更新EMA（使用低价）
                self.low1 = bar.low * self.alpha_25 + self.low1 * (1 - self.alpha_25)
        
        if self.high_window_90.is_ready:
            if self.up2 == 0:
                # 初始化为SMA（使用高价）
                self.up2 = sum([self.high_window_90[i] for i in range(self.high_window_90.count)]) / self.high_window_90.count
            else:
                # 更新EMA（使用高价）
                self.up2 = bar.high * self.alpha_90 + self.up2 * (1 - self.alpha_90)
            
            if self.low2 == 0:
                # 初始化为SMA（使用低价）
                self.low2 = sum([self.low_window_90[i] for i in range(self.low_window_90.count)]) / self.low_window_90.count
            else:
                # 更新EMA（使用低价）
                self.low2 = bar.low * self.alpha_90 + self.low2 * (1 - self.alpha_90)
            
            if not self.is_ready:
                self.is_ready = True
                self.debug(f"指标预热完成！时间：{self.time}")
                self.debug(f"初始通道：UP1=${self.up1:.2f}, LOW1=${self.low1:.2f}")
        
        # 交易逻辑已移至 on_data_consolidated()
        # 这里只更新EMA指标

    def on_data_consolidated(self, sender, bar):
        '''30分钟K线合成后调用此方法 - 基于30分钟K线执行交易逻辑
        
        Arguments:
            sender: 发送者
            bar: TradeBar 30分钟K线数据
        '''
        if self.is_warming_up or not self.is_ready:
            return
        
        self.data_points += 1
        
        # 获取当前价格和成交量（30分钟K线的）
        current_price = bar.close
        current_volume = bar.volume
        holdings = self.portfolio[self.symbol].quantity
        invested = self.portfolio[self.symbol].invested
        
        # 更新30分钟K线的成交量窗口
        consolidated_volume_window = RollingWindow[float](5)
        if not hasattr(self, 'consolidated_volume_list'):
            self.consolidated_volume_list = []
        
        self.consolidated_volume_list.append(current_volume)
        if len(self.consolidated_volume_list) > 5:
            self.consolidated_volume_list.pop(0)
        
        # 计算30分钟K线的平均成交量
        avg_volume = sum(self.consolidated_volume_list) / len(self.consolidated_volume_list) if len(self.consolidated_volume_list) > 0 else 0
        
        # 调试信息（每10个数据点打印一次）
        if self.data_points % 10 == 0:
            self.debug(f"[{self.time}] 30分钟K线: {self.data_points}, 价格: ${current_price:.2f}, UP1: ${self.up1:.2f}, LOW1: ${self.low1:.2f}, 成交量: {current_volume:,.0f}")
        
        # ===== 买入信号（基于30分钟K线）=====
        if not invested:
            # 检查冷却期
            bars_since_last_trade = self.data_points - self.last_trade_bar
            cooldown_ok = bars_since_last_trade >= self.cooldown_bars
            
            # 条件检查
            signal_breakout = current_price > self.up1
            signal_volume = len(self.consolidated_volume_list) >= 5 and current_volume > avg_volume * self.volume_multiplier
            signal_trend = current_price > self.up2
            
            if signal_breakout and signal_volume and signal_trend and cooldown_ok:
                self.trade_count += 1
                self.last_trade_bar = self.data_points
                
                self.debug("=" * 50)
                self.debug(f"【买入信号】交易 #{self.trade_count}")
                self.debug(f"时间：{self.time}")
                self.debug(f"当前价格：${current_price:.2f}")
                self.debug(f"UP1 (蓝色上沿)：${self.up1:.2f}")
                self.debug(f"UP2 (黄色上沿)：${self.up2:.2f}")
                self.debug(f"LOW1 (蓝色下沿)：${self.low1:.2f}")
                self.debug(f"成交量：{current_volume:,.0f} (均量: {avg_volume:,.0f}, 放大倍数: {current_volume/avg_volume:.2f})")
                self.debug(f"冷却期：距上次交易 {bars_since_last_trade} 个K线")
                
                try:
                    self.set_holdings(self.symbol, 0.99)
                    self.entry_price = current_price
                    self.debug(f"订单已提交，买入价：${self.entry_price:.2f}")
                except Exception as e:
                    self.debug(f"下单失败：{e}")
                
                self.debug("=" * 50)
        
        # ===== 卖出信号（基于30分钟K线）=====
        elif invested:
            signal_breakdown = current_price < self.low1
            signal_stoploss = self.entry_price > 0 and current_price < self.entry_price * (1 - self.stop_loss_pct)
            signal_takeprofit = self.entry_price > 0 and current_price > self.entry_price * (1 + self.take_profit_pct)
            
            if signal_breakdown or signal_stoploss or signal_takeprofit:
                self.liquidate(self.symbol)
                self.trade_count += 1
                self.last_trade_bar = self.data_points
                
                if signal_takeprofit:
                    sell_reason = f"止盈（+{self.take_profit_pct*100:.0f}%）"
                elif signal_stoploss:
                    sell_reason = f"止损（-{self.stop_loss_pct*100:.0f}%）"
                else:
                    sell_reason = "跌破LOW1"
                
                profit_pct = ((current_price / self.entry_price) - 1) * 100 if self.entry_price > 0 else 0
                
                self.debug("=" * 50)
                self.debug(f"【卖出信号】交易 #{self.trade_count} - {sell_reason}")
                self.debug(f"时间：{self.time}")
                self.debug(f"买入价：${self.entry_price:.2f}")
                self.debug(f"当前价格：${current_price:.2f}")
                self.debug(f"盈亏：{profit_pct:+.2f}%")
                self.debug(f"UP1：${self.up1:.2f}, LOW1：${self.low1:.2f}")
                self.debug(f"卖出数量：{holdings} 股")
                self.debug("=" * 50)
                
                self.entry_price = 0
    
    def on_end_of_algorithm(self):
        '''算法结束时调用此方法'''
        self.debug("=" * 50)
        self.debug("EMA通道突破策略执行完成！")
        self.debug(f"总数据点（30分钟K线）：{self.data_points}")
        self.debug(f"总交易次数：{self.trade_count}")
        self.debug(f"最终资产：${self.portfolio.total_portfolio_value:,.2f}")
        profit = self.portfolio.total_portfolio_value - 100000
        profit_pct = ((self.portfolio.total_portfolio_value / 100000) - 1) * 100
        self.debug(f"收益：${profit:,.2f}")
        self.debug(f"收益率：{profit_pct:.2f}%")
        self.debug("=" * 50)
