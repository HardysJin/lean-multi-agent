"""
Sentiment analyzer for market sentiment
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import yfinance as yf

from .base_collector import BaseCollector
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAnalyzer(BaseCollector):
    """市场情绪分析器"""
    
    def __init__(self, **kwargs):
        """初始化情绪分析器"""
        super().__init__(kwargs)
    
    def collect(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Dict[str, Any]:
        """
        收集情绪指标
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            Dict: 情绪数据
        """
        logger.info(f"Collecting sentiment data from {start_date} to {end_date}")
        
        sentiment_data = {}
        
        # 1. VIX（恐慌指数）
        vix_data = self._get_vix(start_date, end_date)
        sentiment_data['vix'] = vix_data
        
        # 2. Put/Call Ratio（如果可用）
        # 注：这个数据在yfinance中不直接可用，需要其他数据源
        # sentiment_data['put_call_ratio'] = self._get_put_call_ratio()
        
        # 3. 综合情绪评分
        sentiment_score = self._calculate_sentiment_score(sentiment_data)
        sentiment_data['overall'] = sentiment_score
        
        return sentiment_data
    
    def _get_vix(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        获取VIX数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            Dict: VIX数据
        """
        try:
            vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            
            if vix.empty:
                return {"current": None, "change": None, "level": "unknown"}
            
            current_vix = float(vix['Close'].iloc[-1])
            start_vix = float(vix['Close'].iloc[0])
            change = current_vix - start_vix
            
            # VIX分级
            if current_vix < 15:
                level = "low"  # 低恐慌
            elif current_vix < 25:
                level = "medium"  # 中等
            else:
                level = "high"  # 高恐慌
            
            return {
                "current": current_vix,
                "start": start_vix,
                "change": change,
                "change_pct": (change / start_vix) * 100 if start_vix != 0 else 0,
                "level": level
            }
            
        except Exception as e:
            logger.error(f"Error getting VIX data: {e}")
            return {"current": None, "change": None, "level": "unknown"}
    
    def _calculate_sentiment_score(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算综合情绪评分
        
        Args:
            sentiment_data: 情绪数据
        
        Returns:
            Dict: 情绪评分
        """
        # 基于VIX计算情绪
        vix_info = sentiment_data.get('vix', {})
        vix_value = vix_info.get('current')
        
        if vix_value is None:
            return {
                "sentiment": "neutral",
                "score": 0.5,
                "confidence": 0.0
            }
        
        # VIX越低，情绪越乐观
        # VIX范围大约10-80，我们映射到0-1
        # 分数越高表示越乐观
        if vix_value < 15:
            sentiment = "greedy"
            score = 0.8
        elif vix_value < 20:
            sentiment = "optimistic"
            score = 0.65
        elif vix_value < 25:
            sentiment = "neutral"
            score = 0.5
        elif vix_value < 35:
            sentiment = "fearful"
            score = 0.35
        else:
            sentiment = "panic"
            score = 0.2
        
        return {
            "sentiment": sentiment,
            "score": score,
            "vix_level": vix_info.get('level', 'unknown'),
            "confidence": 0.7  # 基于单一指标，置信度中等
        }
    
    def get_fear_greed_proxy(self, vix: float) -> int:
        """
        计算Fear & Greed指数的代理值（0-100）
        
        Args:
            vix: VIX值
        
        Returns:
            int: Fear & Greed分数（0=极度恐惧，100=极度贪婪）
        """
        # VIX越低，贪婪指数越高
        # 简单映射：VIX 10 -> 90, VIX 40 -> 10
        if vix <= 10:
            return 90
        elif vix >= 40:
            return 10
        else:
            # 线性映射
            return int(90 - (vix - 10) * (80 / 30))
    
    def analyze_market_breadth(self, tickers: List[str]) -> Dict[str, Any]:
        """
        分析市场广度（多少股票上涨vs下跌）
        
        Args:
            tickers: 股票代码列表
        
        Returns:
            Dict: 市场广度分析
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            up_count = 0
            down_count = 0
            
            for ticker in tickers:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty and len(data) >= 2:
                    start_price = data['Close'].iloc[0]
                    end_price = data['Close'].iloc[-1]
                    if end_price > start_price:
                        up_count += 1
                    else:
                        down_count += 1
            
            total = up_count + down_count
            breadth_ratio = up_count / total if total > 0 else 0.5
            
            if breadth_ratio > 0.6:
                breadth = "strong"
            elif breadth_ratio > 0.4:
                breadth = "neutral"
            else:
                breadth = "weak"
            
            return {
                "up_count": up_count,
                "down_count": down_count,
                "breadth_ratio": breadth_ratio,
                "breadth": breadth
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market breadth: {e}")
            return {"breadth": "unknown"}
