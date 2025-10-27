from AlgorithmImports import *
import sys
import os
sys.path.append('/Lean/Algorithm/MultiAgent/Agents')
sys.path.append('/Lean/Algorithm/MultiAgent/Utils')

try:
    from multi_agent_system import MultiAgentSystem
    AGENT_AVAILABLE = True
except:
    AGENT_AVAILABLE = False

class ProductionMultiAgent(QCAlgorithm):
    """Multi-Agent量化策略"""
    
    def Initialize(self):
        """初始化"""
        
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)
        
        # 股票池
        self.symbol_names = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'TSLA']
        self.symbols = {}
        for symbol_name in self.symbol_names:
            equity = self.AddEquity(symbol_name)
            self.symbols[symbol_name] = equity.Symbol
        
        # 初始化Multi-Agent系统
        self.agent_enabled = False
        if AGENT_AVAILABLE:
            try:
                claude_key = os.environ.get('CLAUDE_API_KEY', '')
                news_key = os.environ.get('NEWS_API_KEY', '')
                
                self.agent_system = MultiAgentSystem(
                    claude_api_key=claude_key,
                    news_api_key=news_key,
                    use_local_llm=False,
                    debug_mode=True
                )
                
                self.agent_enabled = True
                self.Debug("✅ Multi-Agent系统初始化成功")
                
            except Exception as e:
                self.Error(f"⚠️ Multi-Agent初始化失败: {e}")
        else:
            self.Debug("⚠️ Multi-Agent模块未找到，使用技术指标策略")
        
        # 技术指标
        self.indicators = {}
        for symbol_name, symbol in self.symbols.items():
            self.indicators[symbol_name] = {
                'rsi': self.RSI(symbol, 14),
                'macd': self.MACD(symbol, 12, 26, 9),
                'sma50': self.SMA(symbol, 50),
                'sma200': self.SMA(symbol, 200)
            }
        
        # 定时任务
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketClose(list(self.symbols.values())[0], 30),
            self.DailyAnalysis
        )
        
        self.daily_signals = {}
        self.positions_info = {}
        self.trade_count = 0
        
        self.Debug(f"\n{'='*60}")
        self.Debug(f"策略初始化完成")
        self.Debug(f"监控股票: {', '.join(self.symbols)}")
        self.Debug(f"Multi-Agent: {'启用' if self.agent_enabled else '未启用（使用技术指标）'}")
        self.Debug(f"{'='*60}\n")
    
    def DailyAnalysis(self):
        """每日分析"""
        
        self.Debug(f"\n{'='*60}")
        self.Debug(f"📊 日期: {self.Time.strftime('%Y-%m-%d')}")
        self.Debug(f"{'='*60}")
        
        for symbol in self.symbols:
            try:
                data = self._prepare_data(symbol)
                
                if self.agent_enabled:
                    # Multi-Agent分析
                    decision = self.agent_system.analyze(symbol, data)
                else:
                    # 技术指标fallback
                    decision = self._technical_analysis(symbol)
                
                self.daily_signals[symbol] = decision
                
                # 输出分析结果
                action_emoji = {'buy': '🟢', 'sell': '🔴', 'hold': '⚪'}
                emoji = action_emoji.get(decision['action'], '⚪')
                
                self.Debug(f"\n{emoji} {symbol}:")
                self.Debug(f"  动作: {decision['action'].upper()}")
                self.Debug(f"  得分: {decision['score']:.2f}/10")
                self.Debug(f"  置信度: {decision['confidence']:.1%}")
                self.Debug(f"  理由: {decision['reasoning'][:80]}")
                
            except Exception as e:
                self.Error(f"❌ 分析{symbol}失败: {e}")
                self.daily_signals[symbol] = {
                    'action': 'hold',
                    'score': 0,
                    'confidence': 0,
                    'reasoning': f'分析失败: {str(e)}'
                }
        
        self.Debug(f"\n{'='*60}\n")
    
    def OnData(self, data):
        """执行交易"""
        
        # 每天开盘后1分钟执行
        if self.Time.hour != 9 or self.Time.minute != 31:
            return
        
        if not self.daily_signals:
            return
        
        for symbol, decision in self.daily_signals.items():
            if not data.ContainsKey(symbol):
                continue
            
            self._execute_decision(symbol, decision)
    
    def _prepare_data(self, symbol):
        """准备分析数据"""
        
        history = self.History(symbol, 60)
        security = self.Securities[symbol]
        indicators = self.indicators.get(symbol, {})
        
        return {
            'symbol': symbol,
            'history': history,
            'current_price': security.Price,
            'current_position': self.Portfolio[symbol].Invested,
            'technical': {
                'rsi': indicators['rsi'].Current.Value if indicators['rsi'].IsReady else None,
                'macd': indicators['macd'].Current.Value if indicators['macd'].IsReady else None,
                'sma50': indicators['sma50'].Current.Value if indicators['sma50'].IsReady else None,
                'sma200': indicators['sma200'].Current.Value if indicators['sma200'].IsReady else None,
            }
        }
    
    def _technical_analysis(self, symbol):
        """纯技术指标分析"""
        
        indicators = self.indicators.get(symbol, {})
        
        if not indicators['rsi'].IsReady:
            return {'action': 'hold', 'score': 0, 'confidence': 0, 'reasoning': '指标未就绪'}
        
        rsi = indicators['rsi'].Current.Value
        price = self.Securities[symbol].Price
        sma50 = indicators['sma50'].Current.Value if indicators['sma50'].IsReady else price
        sma200 = indicators['sma200'].Current.Value if indicators['sma200'].IsReady else price
        
        score = 0
        reasons = []
        
        # RSI评分
        if rsi < 30:
            score += 3
            reasons.append(f"RSI超卖({rsi:.1f})")
        elif rsi > 70:
            score -= 3
            reasons.append(f"RSI超买({rsi:.1f})")
        
        # 均线评分
        if price > sma50 > sma200:
            score += 2
            reasons.append("多头排列")
        elif price < sma50 < sma200:
            score -= 2
            reasons.append("空头排列")
        
        # 决策
        if score >= 4:
            action = 'buy'
        elif score <= -4:
            action = 'sell'
        else:
            action = 'hold'
        
        return {
            'action': action,
            'score': score,
            'confidence': min(abs(score) / 5.0, 1.0),
            'reasoning': '; '.join(reasons) if reasons else 'RSI中性'
        }
    
    def _execute_decision(self, symbol, decision):
        """执行交易"""
        
        if decision['action'] == 'buy':
            if not self.Portfolio[symbol].Invested and decision['confidence'] > 0.6:
                target_weight = min(decision['confidence'] * 0.15, 0.12)
                self.SetHoldings(symbol, target_weight)
                
                self.positions_info[symbol] = {
                    'entry_time': self.Time,
                    'entry_price': self.Securities[symbol].Price,
                    'entry_reason': decision['reasoning']
                }
                
                self.trade_count += 1
                self.Debug(f"✅ 买入 {symbol} ({target_weight:.1%}): {decision['reasoning'][:50]}")
        
        elif decision['action'] == 'sell':
            if self.Portfolio[symbol].Invested:
                pnl = self.Portfolio[symbol].UnrealizedProfit
                pnl_pct = self.Portfolio[symbol].UnrealizedProfitPercent
                
                self.Liquidate(symbol)
                
                if symbol in self.positions_info:
                    entry_info = self.positions_info[symbol]
                    hold_days = (self.Time - entry_info['entry_time']).days
                    
                    self.Debug(f"❌ 卖出 {symbol}: 持有{hold_days}天, "
                             f"盈亏${pnl:.2f} ({pnl_pct:.1%})")
                    del self.positions_info[symbol]
                
                self.trade_count += 1
    
    def OnEndOfAlgorithm(self):
        """回测结束统计"""
        
        total_return = (self.Portfolio.TotalPortfolioValue / 100000 - 1) * 100
        
        self.Debug(f"\n{'='*60}")
        self.Debug(f"📈 回测结果汇总")
        self.Debug(f"{'='*60}")
        self.Debug(f"初始资金: $100,000")
        self.Debug(f"最终权益: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Debug(f"总收益率: {total_return:.2f}%")
        self.Debug(f"总交易次数: {self.trade_count}")
        self.Debug(f"{'='*60}\n")
