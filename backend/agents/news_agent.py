"""
News Analysis Agent
新闻分析Agent
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_agent import BaseAgent
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class NewsAgent(BaseAgent):
    """新闻分析Agent"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化新闻分析Agent"""
        super().__init__("NewsAgent", config)
    
    def analyze(self, data: Dict[str, Any], as_of_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        新闻分析
        
        Args:
            data: 新闻数据
            as_of_date: 决策时间点（用于回测，默认None=当前时间）
        
        Returns:
            Dict: 新闻分析结果
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        if not self.validate_input(data):
            logger.warning("Invalid input data for NewsAgent")
            return self._get_empty_analysis(as_of_date)
        
        logger.info("Running news analysis")
        
        headlines = data.get('headlines', [])
        
        if not headlines:
            return self._get_empty_analysis(as_of_date)
        
        # 识别重大事件
        major_events = self._identify_major_events(headlines)
        
        # 分析行业影响
        sector_impacts = self._analyze_sector_impacts(headlines)
        
        # 识别风险因素
        risk_factors = self._identify_risk_factors(headlines)
        
        # 交易影响分析
        trading_implications = self._analyze_trading_implications(
            major_events,
            sector_impacts,
            risk_factors
        )
        
        return {
            "major_events": major_events,
            "sector_impacts": sector_impacts,
            "risk_factors": risk_factors,
            "trading_implications": trading_implications,
            "timestamp": as_of_date.isoformat()
        }
    
    def _identify_major_events(self, headlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        识别重大事件
        
        Args:
            headlines: 新闻标题列表
        
        Returns:
            List: 重大事件列表
        """
        major_event_keywords = {
            "fed": ["fed", "federal reserve", "interest rate", "powell"],
            "earnings": ["earnings", "profit", "revenue", "guidance"],
            "geopolitical": ["war", "conflict", "sanction", "tension"],
            "economic": ["gdp", "unemployment", "inflation", "cpi", "ppi"],
            "policy": ["regulation", "policy", "law", "congress"]
        }
        
        major_events = []
        
        for headline in headlines:
            title = headline.get('title', '').lower()
            
            for event_type, keywords in major_event_keywords.items():
                if any(keyword in title for keyword in keywords):
                    major_events.append({
                        "type": event_type,
                        "title": headline.get('title', ''),
                        "source": headline.get('source', ''),
                        "published_at": headline.get('published_at', '')
                    })
                    break  # 每条新闻只算一次
        
        return major_events[:5]  # 返回前5个重要事件
    
    def _analyze_sector_impacts(self, headlines: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        分析行业影响
        
        Args:
            headlines: 新闻标题列表
        
        Returns:
            Dict: 行业影响 {sector: impact}
        """
        sector_keywords = {
            "tech": ["tech", "technology", "software", "ai", "chip", "semiconductor"],
            "finance": ["bank", "financial", "credit", "lending"],
            "energy": ["oil", "energy", "gas", "renewable"],
            "healthcare": ["healthcare", "pharma", "drug", "medical"],
            "consumer": ["retail", "consumer", "shopping", "e-commerce"]
        }
        
        positive_words = ["surge", "rally", "gain", "beat", "strong", "growth"]
        negative_words = ["crash", "plunge", "miss", "weak", "decline", "loss"]
        
        sector_impacts = {}
        
        for sector, keywords in sector_keywords.items():
            positive_count = 0
            negative_count = 0
            
            for headline in headlines:
                title = headline.get('title', '').lower()
                
                # 检查是否提到该行业
                if any(keyword in title for keyword in keywords):
                    # 检查情绪
                    if any(word in title for word in positive_words):
                        positive_count += 1
                    if any(word in title for word in negative_words):
                        negative_count += 1
            
            # 判断影响
            if positive_count > negative_count:
                sector_impacts[sector] = "positive"
            elif negative_count > positive_count:
                sector_impacts[sector] = "negative"
            elif positive_count > 0 or negative_count > 0:
                sector_impacts[sector] = "mixed"
            # 如果都是0，不添加该行业
        
        return sector_impacts
    
    def _identify_risk_factors(self, headlines: List[Dict[str, Any]]) -> List[str]:
        """
        识别风险因素
        
        Args:
            headlines: 新闻标题列表
        
        Returns:
            List: 风险因素列表
        """
        risk_keywords = {
            "high_inflation": ["inflation", "price increase", "cpi"],
            "recession": ["recession", "economic slowdown", "contraction"],
            "geopolitical": ["war", "conflict", "tension", "sanction"],
            "rate_hike": ["rate hike", "tighten", "hawkish"],
            "earnings_miss": ["miss", "disappoint", "lower guidance"],
            "market_volatility": ["volatility", "swing", "turbulent"]
        }
        
        risk_factors = []
        
        for headline in headlines:
            title = headline.get('title', '').lower()
            
            for risk_type, keywords in risk_keywords.items():
                if any(keyword in title for keyword in keywords):
                    if risk_type not in risk_factors:
                        risk_factors.append(risk_type)
        
        return risk_factors
    
    def _analyze_trading_implications(
        self,
        major_events: List[Dict[str, Any]],
        sector_impacts: Dict[str, str],
        risk_factors: List[str]
    ) -> str:
        """
        分析交易影响
        
        Args:
            major_events: 重大事件
            sector_impacts: 行业影响
            risk_factors: 风险因素
        
        Returns:
            str: 交易影响分析
        """
        implications = []
        
        # 分析重大事件
        if major_events:
            event_types = [e['type'] for e in major_events]
            if 'fed' in event_types:
                implications.append("Fed policy may impact market direction")
            if 'earnings' in event_types:
                implications.append("Earnings season driving individual stock moves")
            if 'geopolitical' in event_types:
                implications.append("Geopolitical risks creating uncertainty")
        
        # 分析行业影响
        positive_sectors = [s for s, impact in sector_impacts.items() if impact == "positive"]
        negative_sectors = [s for s, impact in sector_impacts.items() if impact == "negative"]
        
        if positive_sectors:
            implications.append(f"Positive momentum in: {', '.join(positive_sectors)}")
        if negative_sectors:
            implications.append(f"Weakness in: {', '.join(negative_sectors)}")
        
        # 分析风险
        if len(risk_factors) >= 3:
            implications.append("Multiple risk factors present - caution advised")
        elif risk_factors:
            implications.append(f"Monitor: {', '.join(risk_factors)}")
        
        if not implications:
            return "No major market-moving events detected"
        
        return "; ".join(implications)
    
    def _get_empty_analysis(self, as_of_date: Optional[datetime] = None) -> Dict[str, Any]:
        """返回空分析结果"""
        if as_of_date is None:
            as_of_date = datetime.now()
        return {
            "major_events": [],
            "sector_impacts": {},
            "risk_factors": [],
            "trading_implications": "No news data available for analysis",
            "timestamp": as_of_date.isoformat()
        }
