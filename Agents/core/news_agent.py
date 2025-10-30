"""
News Agent - Core Business Logic (Pure Python)

新闻情绪分析核心逻辑，无MCP协议依赖

核心功能：
1. 获取特定股票/标的的最新新闻
2. 使用LLM分析新闻情绪（正面/负面/中性）
3. 生成情绪分数和总结
4. 识别关键事件和影响因素
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import Counter
import re

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    NewsApiClient = None

from langchain_core.messages import HumanMessage, SystemMessage

from Agents.core.base_agent import BaseAgent


@dataclass
class NewsArticle:
    """新闻文章数据结构"""
    title: str
    description: str
    source: str
    url: str
    published_at: datetime
    content: Optional[str] = None
    sentiment: Optional[str] = None  # positive/negative/neutral
    sentiment_score: Optional[float] = None  # -1.0 to 1.0
    sentiment_reasoning: Optional[str] = None


@dataclass
class NewsSentimentReport:
    """新闻情绪报告"""
    symbol: str
    overall_sentiment: str  # positive/negative/neutral/mixed
    sentiment_score: float  # -1.0 to 1.0
    articles_analyzed: int
    positive_count: int
    negative_count: int
    neutral_count: int
    key_themes: List[str]
    summary: str
    timestamp: datetime


class NewsAgent(BaseAgent):
    """
    News Agent - 新闻情绪分析专家（纯业务逻辑）
    
    使用NEWS_API获取新闻，LLM分析情绪
    无MCP协议依赖，便于单元测试
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_client=None,
        cache_ttl: int = 300,  # 5分钟缓存
        backtest_mode: bool = False,
        backtest_date: Optional[datetime] = None
    ):
        """
        初始化News Agent
        
        Args:
            api_key: News API key (如果为None，从环境变量NEWS_API_KEY读取)
            llm_client: LLM客户端（用于情绪分析）
            cache_ttl: 缓存过期时间（秒）
            backtest_mode: 是否为回测模式（避免使用未来信息）
            backtest_date: 回测日期（用于获取历史新闻）
        """
        super().__init__(
            name="news-agent",
            llm_client=llm_client,
            enable_cache=True,
            cache_ttl=cache_ttl
        )
        
        # 回测模式设置
        self.backtest_mode = backtest_mode
        self.backtest_date = backtest_date
        if backtest_mode and not backtest_date:
            self.logger.warning("Backtest mode enabled but no backtest_date provided. Using datetime.now()")
        
        # 获取API key
        self.api_key = api_key or os.getenv("NEWS_API_KEY")
        if not self.api_key:
            self.logger.warning("NEWS_API_KEY not provided. News fetching will be limited.")
        
        # 初始化News API客户端
        self.news_client = None
        if NEWSAPI_AVAILABLE and self.api_key:
            try:
                self.news_client = NewsApiClient(api_key=self.api_key)
                self.logger.info("NewsAPI client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize NewsAPI client: {e}")
        elif not NEWSAPI_AVAILABLE:
            self.logger.warning("newsapi-python package not installed. Install with: pip install newsapi-python")
    
    def _get_current_time(self) -> datetime:
        """
        获取"当前"时间
        
        在回测模式下返回回测日期，避免未来信息泄露
        在实时模式下返回真实当前时间
        """
        if self.backtest_mode and self.backtest_date:
            return self.backtest_date
        return datetime.now()
    
    # ═══════════════════════════════════════════════════
    # Public APIs - Business Logic
    # ═══════════════════════════════════════════════════
    
    async def fetch_news(
        self,
        symbol: str,
        limit: int = 10,
        days_back: int = 7
    ) -> List[NewsArticle]:
        """
        从News API获取新闻
        
        Args:
            symbol: 股票代码
            limit: 最大文章数
            days_back: 回溯天数
            
        Returns:
            新闻文章列表
        """
        # 检查缓存
        cache_key = f"news_{symbol}_{limit}_{days_back}"
        cached = self._get_from_cache(cache_key)
        if cached:
            self.logger.info(f"Using cached news for {symbol}")
            return cached
        
        if not self.news_client:
            # 如果没有API，返回模拟数据
            self.logger.warning("NewsAPI client not available, returning mock data")
            return self._get_mock_news(symbol, limit)
        
        try:
            # 计算日期范围（使用回测日期或当前日期）
            to_date = self._get_current_time()
            from_date = to_date - timedelta(days=days_back)
            
            # 构建查询（股票代码 + 常见财经关键词）
            query = f"{symbol} OR {self._get_company_name(symbol)}"
            
            # 调用News API
            response = self.news_client.get_everything(
                q=query,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='popularity',  # relevancy, popularity, publishedAt
                page_size=limit
            )
            
            # 解析文章
            articles = []
            if response['status'] == 'ok':
                for article_data in response.get('articles', [])[:limit]:
                    article = NewsArticle(
                        title=article_data.get('title', ''),
                        description=article_data.get('description', ''),
                        source=article_data.get('source', {}).get('name', 'Unknown'),
                        url=article_data.get('url', ''),
                        published_at=datetime.strptime(
                            article_data['publishedAt'],
                            '%Y-%m-%dT%H:%M:%SZ'
                        ) if article_data.get('publishedAt') else self._get_current_time(),
                        content=article_data.get('content')
                    )
                    articles.append(article)
            
            # 缓存结果
            self._put_to_cache(cache_key, articles)
            
            self.logger.info(f"Fetched {len(articles)} articles for {symbol}")
            return articles
        
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}", exc_info=True)
            return self._get_mock_news(symbol, limit)
    
    async def analyze_sentiment(
        self,
        articles: List[NewsArticle]
    ) -> List[NewsArticle]:
        """
        使用LLM分析新闻情绪
        
        Args:
            articles: 新闻文章列表
            
        Returns:
            带情绪分析的文章列表
        """
        if not self.llm:
            self.logger.warning("LLM not available, using neutral sentiment")
            # 返回中性情绪
            for article in articles:
                article.sentiment = "neutral"
                article.sentiment_score = 0.0
                article.sentiment_reasoning = "LLM not available"
            return articles
        
        # 批量分析（提高效率）
        for article in articles:
            try:
                sentiment_data = await self._analyze_single_article(article)
                article.sentiment = sentiment_data['sentiment']
                article.sentiment_score = sentiment_data['score']
                article.sentiment_reasoning = sentiment_data['reasoning']
            except Exception as e:
                self.logger.error(f"Error analyzing article sentiment: {e}")
                article.sentiment = "neutral"
                article.sentiment_score = 0.0
                article.sentiment_reasoning = f"Analysis failed: {str(e)}"
        
        return articles
    
    async def get_latest_news_with_sentiment(
        self,
        symbol: str,
        limit: int = 10,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        获取最新新闻（带情绪分析）
        
        Args:
            symbol: 股票代码
            limit: 最大文章数
            days_back: 回溯天数
            
        Returns:
            包含文章和元数据的字典
        """
        articles = await self.fetch_news(symbol, limit, days_back)
        articles = await self.analyze_sentiment(articles)
        
        return {
            "symbol": symbol,
            "articles": [asdict(a) for a in articles],
            "count": len(articles),
            "date_range": f"last {days_back} days"
        }
    
    async def generate_sentiment_report(
        self,
        symbol: str,
        days_back: int = 7
    ) -> NewsSentimentReport:
        """
        生成新闻情绪报告
        
        Args:
            symbol: 股票代码
            days_back: 回溯天数
            
        Returns:
            情绪报告
        """
        # 检查缓存
        cache_key = f"sentiment_report_{symbol}_{days_back}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        # 获取新闻
        articles = await self.fetch_news(symbol, limit=20, days_back=days_back)
        
        # 分析情绪
        articles = await self.analyze_sentiment(articles)
        
        # 统计情绪
        positive_count = sum(1 for a in articles if a.sentiment == "positive")
        negative_count = sum(1 for a in articles if a.sentiment == "negative")
        neutral_count = sum(1 for a in articles if a.sentiment == "neutral")
        
        # 计算平均分数
        scores = [a.sentiment_score for a in articles if a.sentiment_score is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # 判断整体情绪
        if positive_count > negative_count * 1.5:
            overall = "positive"
        elif negative_count > positive_count * 1.5:
            overall = "negative"
        elif positive_count > 0 and negative_count > 0:
            overall = "mixed"
        else:
            overall = "neutral"
        
        # 使用LLM生成总结
        summary = await self._generate_summary_with_llm(symbol, articles) if self.llm else "LLM not available for summary"
        
        # 提取关键主题
        key_themes = self._extract_key_themes(articles)
        
        report = NewsSentimentReport(
            symbol=symbol,
            overall_sentiment=overall,
            sentiment_score=avg_score,
            articles_analyzed=len(articles),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            key_themes=key_themes,
            summary=summary,
            timestamp=self._get_current_time()
        )
        
        # 缓存报告
        self._put_to_cache(cache_key, report)
        
        return report
    
    async def search_news_by_keyword(
        self,
        keyword: str,
        limit: int = 10,
        with_sentiment: bool = True
    ) -> Dict[str, Any]:
        """
        按关键词搜索新闻
        
        Args:
            keyword: 搜索关键词
            limit: 最大文章数
            with_sentiment: 是否进行情绪分析
            
        Returns:
            搜索结果字典
        """
        if not self.news_client:
            return {
                "keyword": keyword,
                "articles": [],
                "message": "NewsAPI client not available"
            }
        
        try:
            response = self.news_client.get_everything(
                q=keyword,
                language='en',
                sort_by='popularity',
                page_size=limit
            )
            
            articles = []
            if response['status'] == 'ok':
                for article_data in response.get('articles', [])[:limit]:
                    article = NewsArticle(
                        title=article_data.get('title', ''),
                        description=article_data.get('description', ''),
                        source=article_data.get('source', {}).get('name', 'Unknown'),
                        url=article_data.get('url', ''),
                        published_at=datetime.strptime(
                            article_data['publishedAt'],
                            '%Y-%m-%dT%H:%M:%SZ'
                        ) if article_data.get('publishedAt') else self._get_current_time()
                    )
                    articles.append(article)
            
            # 分析情绪
            if with_sentiment:
                articles = await self.analyze_sentiment(articles)
            
            return {
                "keyword": keyword,
                "articles": [asdict(a) for a in articles],
                "count": len(articles)
            }
        
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return {
                "keyword": keyword,
                "articles": [],
                "error": str(e)
            }
    
    # ═══════════════════════════════════════════════════
    # Private Helper Methods
    # ═══════════════════════════════════════════════════
    
    async def _analyze_single_article(self, article: NewsArticle) -> Dict[str, Any]:
        """
        使用LLM分析单篇文章情绪
        
        Args:
            article: 新闻文章
            
        Returns:
            情绪分析结果 {sentiment, score, reasoning}
        """
        prompt = f"""Analyze the sentiment of this news article for stock market trading purposes.

Title: {article.title}
Description: {article.description}
Source: {article.source}

Provide your analysis in the following format:
SENTIMENT: [positive/negative/neutral]
SCORE: [number between -1.0 (very negative) and 1.0 (very positive)]
REASONING: [brief explanation of why this sentiment and score]

Consider:
- Market impact (earnings, product launches, regulatory issues)
- Tone and language used
- Factual vs speculative content
- Source credibility"""
        
        messages = [
            SystemMessage(content="You are a financial news sentiment analyst. Provide accurate, unbiased sentiment analysis for trading decisions."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            result = self._parse_sentiment_response(response.content)
            return result
        except Exception as e:
            self.logger.error(f"LLM sentiment analysis failed: {e}")
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'reasoning': f'Analysis failed: {str(e)}'
            }
    
    def _parse_sentiment_response(self, response: str) -> Dict[str, Any]:
        """解析LLM的情绪分析响应"""
        sentiment = "neutral"
        score = 0.0
        reasoning = ""
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("SENTIMENT:"):
                sentiment_text = line.split("SENTIMENT:", 1)[1].strip().lower()
                if "positive" in sentiment_text:
                    sentiment = "positive"
                elif "negative" in sentiment_text:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
            
            elif line.startswith("SCORE:"):
                score_text = line.split("SCORE:", 1)[1].strip()
                try:
                    score = float(score_text.split()[0])
                    score = max(-1.0, min(1.0, score))  # 限制在[-1, 1]
                except (ValueError, IndexError):
                    score = 0.0
            
            elif line.startswith("REASONING:"):
                reasoning = line.split("REASONING:", 1)[1].strip()
        
        return {
            'sentiment': sentiment,
            'score': score,
            'reasoning': reasoning or response
        }
    
    async def _generate_summary_with_llm(
        self,
        symbol: str,
        articles: List[NewsArticle]
    ) -> str:
        """使用LLM生成新闻摘要"""
        if not articles:
            return "No articles to summarize"
        
        # 构建文章列表
        articles_text = "\n\n".join([
            f"[{a.sentiment.upper() if a.sentiment else 'UNKNOWN'}] {a.title}\n{a.description[:200]}"
            for a in articles[:10]  # 最多10篇
        ])
        
        prompt = f"""Summarize the recent news sentiment for {symbol} based on these articles:

{articles_text}

Provide a concise summary (2-3 sentences) covering:
1. Overall market sentiment
2. Key events or themes
3. Potential trading implications"""
        
        messages = [
            SystemMessage(content="You are a financial analyst summarizing news for traders."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")
            return "Summary generation failed"
    
    def _extract_key_themes(self, articles: List[NewsArticle]) -> List[str]:
        """提取关键主题（简单关键词统计）"""
        words = []
        for article in articles:
            # 提取标题中的单词
            title_words = re.findall(r'\b[A-Za-z]{4,}\b', article.title.lower())
            words.extend(title_words)
        
        # 过滤常见词
        stopwords = {'that', 'this', 'with', 'from', 'have', 'more', 'been', 'will', 'about', 'after', 'says'}
        filtered_words = [w for w in words if w not in stopwords]
        
        # 统计Top 5
        counter = Counter(filtered_words)
        return [word for word, count in counter.most_common(5)]
    
    def _get_company_name(self, symbol: str) -> str:
        """获取公司名称（简单映射）"""
        mapping = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta',
            'NVDA': 'Nvidia',
        }
        return mapping.get(symbol.upper(), symbol)
    
    def _get_mock_news(self, symbol: str, limit: int) -> List[NewsArticle]:
        """生成模拟新闻数据（用于测试）"""
        current_time = self._get_current_time()
        
        mock_articles = [
            NewsArticle(
                title=f"{symbol} Reports Strong Q4 Earnings",
                description="Company beats analyst expectations with revenue growth.",
                source="Mock Financial News",
                url=f"https://example.com/{symbol}-earnings",
                published_at=current_time - timedelta(hours=2)
            ),
            NewsArticle(
                title=f"{symbol} Announces New Product Launch",
                description="New innovative product expected to drive future growth.",
                source="Mock Tech News",
                url=f"https://example.com/{symbol}-product",
                published_at=current_time - timedelta(hours=5)
            ),
            NewsArticle(
                title=f"Analyst Upgrades {symbol} Rating",
                description="Major investment bank raises price target citing strong fundamentals.",
                source="Mock Market Watch",
                url=f"https://example.com/{symbol}-upgrade",
                published_at=current_time - timedelta(hours=12)
            ),
        ]
        return mock_articles[:limit]
