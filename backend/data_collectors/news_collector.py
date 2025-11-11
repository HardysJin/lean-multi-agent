"""
News collector using NewsAPI
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from .base_collector import BaseCollector
from backend.utils.logger import get_logger

# 自动加载.env文件
load_dotenv()

logger = get_logger(__name__)


class NewsCollector(BaseCollector):
    """新闻收集器（使用NewsAPI）"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "en",
        top_headlines_count: int = 10,
        **kwargs
    ):
        """
        初始化新闻收集器
        
        Args:
            api_key: NewsAPI密钥
            language: 语言
            top_headlines_count: 获取头条数量
        """
        super().__init__(kwargs)
        # 如果未提供api_key，从环境变量读取
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        self.language = language
        self.top_headlines_count = top_headlines_count
        
        if self.api_key:
            try:
                from newsapi import NewsApiClient
                self.client = NewsApiClient(api_key=self.api_key)
            except ImportError:
                logger.warning("newsapi-python not installed. News collection will be disabled.")
                self.client = None
        else:
            logger.warning("NewsAPI key not provided. News collection will be disabled.")
            self.client = None
    
    def collect(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Dict[str, Any]:
        """
        收集新闻数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 可选参数（query, domains等）
        
        Returns:
            Dict: 新闻数据
        """
        if not self.client:
            logger.warning("News client not initialized. Returning empty data.")
            return {"headlines": [], "summary": "News collection disabled"}
        
        logger.info(f"Collecting news from {start_date} to {end_date}")
        
        try:
            # 获取财经类头条
            query = kwargs.get("query", "stock market OR finance OR trading")
            
            response = self.client.get_everything(
                q=query,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language=self.language,
                sort_by='relevancy',
                page_size=self.top_headlines_count
            )
            
            articles = response.get('articles', [])
            
            headlines = []
            for article in articles:
                headlines.append({
                    "title": article.get('title', ''),
                    "description": article.get('description', ''),
                    "source": article.get('source', {}).get('name', ''),
                    "published_at": article.get('publishedAt', ''),
                    "url": article.get('url', '')
                })
            
            # 生成摘要
            summary = self._generate_summary(headlines)
            
            return {
                "headlines": headlines,
                "summary": summary,
                "count": len(headlines)
            }
            
        except Exception as e:
            logger.error(f"Error collecting news: {e}")
            return {"headlines": [], "summary": f"Error: {str(e)}"}
    
    def _generate_summary(self, headlines: List[Dict[str, Any]]) -> str:
        """
        生成新闻摘要
        
        Args:
            headlines: 新闻列表
        
        Returns:
            str: 摘要文本
        """
        if not headlines:
            return "No major news events"
        
        # 简单摘要：列出前5条标题
        summary_lines = ["Top news headlines:"]
        for i, headline in enumerate(headlines[:5], 1):
            summary_lines.append(f"{i}. {headline['title']}")
        
        return "\n".join(summary_lines)
    
    def get_sentiment_keywords(self) -> Dict[str, List[str]]:
        """
        获取情绪关键词（用于简单情绪分析）
        
        Returns:
            Dict: 正面/负面关键词列表
        """
        return {
            "positive": [
                "surge", "rally", "gain", "up", "bullish", "optimistic",
                "beat", "exceed", "strong", "growth", "recovery"
            ],
            "negative": [
                "crash", "plunge", "drop", "down", "bearish", "pessimistic",
                "miss", "weak", "decline", "recession", "crisis"
            ]
        }
    
    def analyze_sentiment(self, headlines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        简单的情绪分析（基于关键词）
        
        Args:
            headlines: 新闻列表
        
        Returns:
            Dict: 情绪分析结果
        """
        keywords = self.get_sentiment_keywords()
        positive_count = 0
        negative_count = 0
        
        for headline in headlines:
            title_lower = headline['title'].lower()
            
            for word in keywords['positive']:
                if word in title_lower:
                    positive_count += 1
            
            for word in keywords['negative']:
                if word in title_lower:
                    negative_count += 1
        
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
            "negative_count": negative_count
        }
