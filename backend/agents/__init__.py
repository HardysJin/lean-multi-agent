"""Agents module initialization"""

from .base_agent import BaseAgent
from .technical_agent import TechnicalAgent
from .sentiment_agent import SentimentAgent
from .news_agent import NewsAgent
from .coordinator import WeeklyCoordinator

__all__ = [
    'BaseAgent',
    'TechnicalAgent',
    'SentimentAgent',
    'NewsAgent',
    'WeeklyCoordinator'
]
