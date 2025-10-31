"""
Multi-Agent 交易策略

基于 Meta Agent 的智能交易策略
"""

import pandas as pd
from typing import Dict, Optional
import asyncio

from Agents.meta_agent import MetaAgent
from Agents.core import TechnicalAnalysisAgent, NewsAgent


class MultiAgentStrategy:
    """
    Multi-Agent 交易策略
    
    结合多个 AI Agent 的决策：
    - MetaAgent: 协调和最终决策
    - TechnicalAgent: 技术指标分析
    - NewsAgent: 新闻情感分析
    """
    
    def __init__(self):
        """初始化所有 Agent"""
        self.meta_agent = MetaAgent()
        self.technical_agent = TechnicalAnalysisAgent()
        self.news_agent = NewsAgent()
    
    async def generate_signal(
        self,
        symbol: str,
        date: pd.Timestamp,
        price: float,
        historical_data: Optional[pd.DataFrame] = None
    ) -> int:
        """
        生成交易信号
        
        Args:
            symbol: 股票代码
            date: 当前日期
            price: 当前价格
            historical_data: 历史数据（可选）
        
        Returns:
            1 = BUY, 0 = SELL/HOLD, -1 = SELL
        """
        
        # 使用 MetaAgent 进行综合分析
        decision = await self.meta_agent.analyze_and_decide(
            symbol=symbol,
            query=f"""
            分析 {symbol} 在 {date.date()} 的交易机会
            
            当前价格: ${price:.2f}
            
            请综合考虑：
            1. 技术指标（RSI, MACD, 均线等）
            2. 最近的新闻情绪
            3. 市场趋势
            
            给出明确的 BUY/SELL/HOLD 建议
            """
        )
        
        # 转换为数字信号
        if decision.action == "BUY" and decision.conviction >= 7:
            return 1  # 强烈买入
        elif decision.action == "SELL":
            return -1  # 卖出
        else:
            return 0  # 持有
    
    async def batch_generate_signals(
        self,
        symbol: str,
        dates: pd.DatetimeIndex,
        prices: pd.Series,
        historical_data: pd.DataFrame
    ) -> pd.Series:
        """
        批量生成信号（优化性能）
        
        Args:
            symbol: 股票代码
            dates: 日期序列
            prices: 价格序列
            historical_data: 完整历史数据
        
        Returns:
            信号序列
        """
        signals = []
        
        for i, (date, price) in enumerate(zip(dates, prices)):
            # 获取到当前日期为止的历史数据
            hist_to_date = historical_data.loc[:date]
            
            signal = await self.generate_signal(
                symbol=symbol,
                date=date,
                price=price,
                historical_data=hist_to_date
            )
            
            signals.append(signal)
            
            # 显示进度
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(dates)} signals generated")
        
        return pd.Series(signals, index=dates)
    
    def get_strategy_description(self) -> str:
        """获取策略描述"""
        return """
        Multi-Agent Trading Strategy
        
        使用多个 AI Agent 协同工作：
        1. Technical Agent - 分析技术指标（RSI, MACD, MA等）
        2. News Agent - 分析新闻情感和市场情绪
        3. Meta Agent - 综合所有信息，做出最终决策
        
        决策逻辑：
        - 信心度 >= 7 的 BUY 信号 → 买入
        - SELL 信号 → 卖出
        - 其他情况 → 持有
        
        优势：
        - 多维度分析，降低误判
        - LLM 理解能力，适应市场变化
        - 自动学习历史决策（Memory 系统）
        """


# 简化的策略（不使用 LLM，适合快速测试）
class SimpleTechnicalStrategy:
    """
    简单技术策略（无 LLM）
    
    基于移动平均线的经典策略
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        初始化策略
        
        Args:
            short_window: 短期均线周期
            long_window: 长期均线周期
        """
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signal(
        self,
        symbol: str,
        date: pd.Timestamp,
        price: float,
        historical_data: pd.DataFrame
    ) -> int:
        """
        生成交易信号
        
        策略：金叉买入，死叉卖出
        """
        if len(historical_data) < self.long_window:
            return 0  # 数据不足，持有
        
        # 计算均线
        close = historical_data['Close']
        sma_short = close.rolling(self.short_window).mean().iloc[-1]
        sma_long = close.rolling(self.long_window).mean().iloc[-1]
        
        # 前一天的均线
        if len(historical_data) > 1:
            sma_short_prev = close.rolling(self.short_window).mean().iloc[-2]
            sma_long_prev = close.rolling(self.long_window).mean().iloc[-2]
            
            # 金叉（短期上穿长期）
            if sma_short_prev <= sma_long_prev and sma_short > sma_long:
                return 1  # 买入
            
            # 死叉（短期下穿长期）
            elif sma_short_prev >= sma_long_prev and sma_short < sma_long:
                return -1  # 卖出
        
        return 0  # 持有
    
    def get_strategy_description(self) -> str:
        """获取策略描述"""
        return f"""
        Simple Moving Average Crossover Strategy
        
        参数：
        - 短期均线: {self.short_window} 天
        - 长期均线: {self.long_window} 天
        
        规则：
        - 金叉（短期上穿长期）→ 买入
        - 死叉（短期下穿长期）→ 卖出
        - 其他情况 → 持有
        """
