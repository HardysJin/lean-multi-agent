"""
Sentiment Analysis Agent
市场情绪分析Agent
"""

from typing import Dict, Any, Optional
from datetime import datetime

from .base_agent import BaseAgent
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAgent(BaseAgent):
    """市场情绪分析Agent"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化情绪分析Agent"""
        super().__init__("SentimentAgent", config)
        self.sources = self.config.get('sources', ['news', 'twitter', 'reddit'])
    
    def analyze(self, data: Dict[str, Any], as_of_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        情绪分析
        
        Args:
            data: 包含news_data和sentiment_data
            as_of_date: 决策时间点（用于回测，默认None=当前时间）
        
        Returns:
            Dict: 情绪分析结果
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        if not self.validate_input(data):
            logger.warning("Invalid input data for SentimentAgent")
            return self._get_empty_analysis(as_of_date)
        
        logger.info("Running sentiment analysis")
        
        sentiment_data = data.get('sentiment_data', {})
        news_data = data.get('news_data', {})
        
        # VIX情绪分析
        vix_sentiment = self._analyze_vix(sentiment_data.get('vix', {}))
        
        # 新闻情绪分析
        news_sentiment = self._analyze_news_sentiment(news_data)
        
        # 综合情绪评估
        overall_sentiment = self._calculate_overall_sentiment(
            vix_sentiment,
            news_sentiment
        )
        
        return {
            "vix_sentiment": vix_sentiment,
            "news_sentiment": news_sentiment,
            "overall_sentiment": overall_sentiment,
            "timestamp": as_of_date.isoformat()
        }
    
    def _analyze_vix(self, vix_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析VIX情绪
        
        Args:
            vix_data: VIX数据
        
        Returns:
            Dict: VIX情绪分析
        """
        current_vix = vix_data.get('current')
        change = vix_data.get('change', 0)
        level = vix_data.get('level', 'unknown')
        
        if current_vix is None:
            return {
                "sentiment": "neutral",
                "score": 0.5,
                "interpretation": "VIX data not available"
            }
        
        # VIX越低，情绪越乐观
        if level == "low":
            sentiment = "optimistic"
            score = 0.75
            interpretation = "Low VIX suggests market complacency and confidence"
        elif level == "medium":
            sentiment = "neutral"
            score = 0.5
            interpretation = "Moderate VIX indicates balanced market sentiment"
        else:  # high
            sentiment = "fearful"
            score = 0.25
            interpretation = "High VIX signals elevated fear and uncertainty"
        
        # VIX变化趋势
        if change > 3:
            trend = "rising_fear"
        elif change < -3:
            trend = "decreasing_fear"
        else:
            trend = "stable"
        
        return {
            "sentiment": sentiment,
            "score": score,
            "vix_value": current_vix,
            "vix_change": change,
            "trend": trend,
            "interpretation": interpretation
        }
    
    def _analyze_news_sentiment(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析新闻情绪
        
        Args:
            news_data: 新闻数据
        
        Returns:
            Dict: 新闻情绪分析
        """
        headlines = news_data.get('headlines', [])
        
        if not headlines:
            return {
                "sentiment": "neutral",
                "score": 0.5,
                "key_themes": [],
                "interpretation": "No news data available"
            }
        
        # 简单关键词情绪分析
        positive_keywords = [
            "surge", "rally", "gain", "up", "bullish", "optimistic",
            "beat", "exceed", "strong", "growth", "recovery", "boom"
        ]
        negative_keywords = [
            "crash", "plunge", "drop", "down", "bearish", "pessimistic",
            "miss", "weak", "decline", "recession", "crisis", "fear"
        ]
        
        positive_count = 0
        negative_count = 0
        key_themes = []
        
        for headline in headlines[:10]:  # 分析前10条
            title = headline.get('title', '').lower()
            
            for keyword in positive_keywords:
                if keyword in title:
                    positive_count += 1
            
            for keyword in negative_keywords:
                if keyword in title:
                    negative_count += 1
            
            # 提取关键主题
            if any(word in title for word in ['tech', 'technology', 'ai']):
                key_themes.append('tech')
            if any(word in title for word in ['fed', 'rate', 'inflation']):
                key_themes.append('monetary_policy')
            if any(word in title for word in ['earning', 'profit', 'revenue']):
                key_themes.append('earnings')
        
        # 去重
        key_themes = list(set(key_themes))
        
        # 计算情绪分数
        total = positive_count + negative_count
        if total == 0:
            sentiment = "neutral"
            score = 0.5
        else:
            score = positive_count / total
            if score > 0.6:
                sentiment = "positive"
            elif score < 0.4:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "score": score,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "key_themes": key_themes,
            "interpretation": f"News sentiment is {sentiment} based on {len(headlines)} headlines"
        }
    
    def _calculate_overall_sentiment(
        self,
        vix_sentiment: Dict[str, Any],
        news_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        计算综合情绪
        
        Args:
            vix_sentiment: VIX情绪
            news_sentiment: 新闻情绪
        
        Returns:
            Dict: 综合情绪
        """
        # 加权平均（VIX权重更高）
        vix_score = vix_sentiment.get('score', 0.5)
        news_score = news_sentiment.get('score', 0.5)
        
        overall_score = vix_score * 0.6 + news_score * 0.4
        
        if overall_score > 0.65:
            sentiment = "bullish"
            risk_level = "low"
        elif overall_score > 0.55:
            sentiment = "cautiously_bullish"
            risk_level = "medium"
        elif overall_score > 0.45:
            sentiment = "neutral"
            risk_level = "medium"
        elif overall_score > 0.35:
            sentiment = "cautiously_bearish"
            risk_level = "medium"
        else:
            sentiment = "bearish"
            risk_level = "high"
        
        return {
            "sentiment": sentiment,
            "score": overall_score,
            "risk_level": risk_level,
            "summary": f"Overall market sentiment is {sentiment} with {risk_level} risk"
        }
    
    def _get_empty_analysis(self, as_of_date: Optional[datetime] = None) -> Dict[str, Any]:
        """返回空分析结果"""
        if as_of_date is None:
            as_of_date = datetime.now()
        return {
            "vix_sentiment": {"sentiment": "neutral", "score": 0.5},
            "news_sentiment": {"sentiment": "neutral", "score": 0.5},
            "overall_sentiment": {"sentiment": "neutral", "score": 0.5, "risk_level": "unknown"},
            "timestamp": as_of_date.isoformat()
        }
