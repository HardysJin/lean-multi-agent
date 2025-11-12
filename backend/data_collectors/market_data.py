"""
Market data collector using yfinance with Polygon.io fallback
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import pickle
import requests
import os

from .base_collector import BaseCollector
from backend.utils.logger import get_logger
from backend.config.config_loader import get_config

logger = get_logger(__name__)


class MarketDataCollector(BaseCollector):
    """市场数据收集器（使用yfinance）"""
    
    def __init__(
        self,
        tickers: List[str] = None,
        cache_enabled: bool = True,
        cache_dir: str = "./Data/cache",
        **kwargs
    ):
        """
        初始化市场数据收集器
        
        Args:
            tickers: 股票代码列表，默认从config.yaml读取
            cache_enabled: 是否启用缓存
            cache_dir: 缓存目录
        """
        super().__init__(kwargs)
        
        # 如果未指定tickers，从配置文件读取
        if tickers is None:
            try:
                config = get_config()
                self.tickers = config.data_sources.market_data.tickers
                logger.info(f"从配置文件加载tickers: {self.tickers}")
            except Exception as e:
                logger.warning(f"无法从配置文件加载tickers，使用默认值: {e}")
                self.tickers = ["SPY", "QQQ", "^VIX"]
        else:
            self.tickers = tickers
        
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)
        
        # Polygon API key (optional fallback)
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '').strip()
        if self.polygon_api_key:
            logger.info("Polygon.io API key found - will use as fallback for insufficient data")
        
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def collect(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Dict[str, Any]:
        """
        收集市场数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            Dict: 包含价格数据、指标等
        """
        logger.info(f"Collecting market data from {start_date} to {end_date}")
        
        market_data = {}
        
        for ticker in self.tickers:
            try:
                # 为每个ticker单独检查缓存
                ticker_cache_key = f"{ticker}_{start_date.date()}_{end_date.date()}"
                cached_ticker_data = self._get_from_cache(ticker_cache_key)
                
                if cached_ticker_data:
                    logger.info(f"Using cached data for {ticker}")
                    market_data[ticker] = cached_ticker_data
                    continue
                
                # 先尝试 yfinance
                data = self._fetch_from_yfinance(ticker, start_date, end_date)
                
                # 检查数据是否充足
                expected_days = (end_date - start_date).days
                actual_days = len(data) if not data.empty else 0
                
                # 如果数据不足（少于期望天数的30%），且有 Polygon API key，则尝试 Polygon
                if actual_days < expected_days * 0.3 and self.polygon_api_key:
                    logger.warning(f"{ticker}: yfinance返回数据不足 ({actual_days}/{expected_days}天)，尝试Polygon.io...")
                    polygon_data = self._fetch_from_polygon(ticker, start_date, end_date)
                    
                    if not polygon_data.empty and len(polygon_data) > actual_days:
                        logger.info(f"{ticker}: Polygon.io返回 {len(polygon_data)} 条数据 (vs yfinance {actual_days}条)")
                        data = polygon_data
                    else:
                        logger.warning(f"{ticker}: Polygon.io也无法获取更多数据")
                
                if data.empty:
                    logger.warning(f"No data for {ticker}")
                    continue
                
                # 展平MultiIndex列（如果存在）
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # 重置index为Date列，便于序列化
                data_with_date = data.reset_index()
                if 'Date' in data_with_date.columns:
                    data_with_date['Date'] = data_with_date['Date'].dt.strftime('%Y-%m-%d')
                
                # 计算技术指标
                indicators = self._calculate_indicators(data)
                
                # 计算周度统计
                weekly_stats = self._calculate_weekly_stats(data)
                
                ticker_data = {
                    "ohlcv": data_with_date.to_dict('records'),
                    "latest_price": float(data['Close'].iloc[-1]) if len(data) > 0 else None,
                    "indicators": indicators,
                    "weekly_stats": weekly_stats
                }
                
                market_data[ticker] = ticker_data
                
                # 为每个ticker单独缓存
                if self.cache_enabled:
                    self._save_to_cache(ticker_cache_key, ticker_data)
                
            except Exception as e:
                logger.error(f"Error collecting data for {ticker}: {e}")
                continue
        
        return market_data
    
    def _fetch_from_yfinance(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        从 yfinance 获取数据
        
        Args:
            ticker: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: OHLCV数据
        """
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )
            
            # 展平MultiIndex列（如果存在）
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            return data
        except Exception as e:
            logger.error(f"Error fetching {ticker} from yfinance: {e}")
            return pd.DataFrame()
    
    def _fetch_from_polygon(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        从 Polygon.io 获取数据
        
        Args:
            ticker: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: OHLCV数据
        """
        if not self.polygon_api_key:
            logger.warning("Polygon API key not configured")
            return pd.DataFrame()
        
        try:
            # Polygon Aggregates API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apiKey': self.polygon_api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('results'):
                    results = data['results']
                    
                    # 转换为 DataFrame
                    df = pd.DataFrame(results)
                    
                    # 重命名列以匹配 yfinance 格式
                    df['Date'] = pd.to_datetime(df['t'], unit='ms')
                    df = df.rename(columns={
                        'o': 'Open',
                        'h': 'High',
                        'l': 'Low',
                        'c': 'Close',
                        'v': 'Volume'
                    })
                    
                    # 设置 Date 为索引
                    df.set_index('Date', inplace=True)
                    
                    # 只保留需要的列
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    
                    logger.info(f"Successfully fetched {len(df)} bars from Polygon.io for {ticker}")
                    return df
                else:
                    logger.warning(f"No results from Polygon.io for {ticker}")
                    return pd.DataFrame()
            else:
                logger.error(f"Polygon.io API error {response.status_code}: {response.text[:200]}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching {ticker} from Polygon.io: {e}")
            return pd.DataFrame()
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算技术指标
        
        Args:
            df: OHLCV数据
        
        Returns:
            Dict: 技术指标
        """
        indicators = {}
        
        try:
            # 移动平均
            df.ta.sma(length=20, append=True)
            df.ta.sma(length=50, append=True)
            df.ta.sma(length=200, append=True)
            
            # RSI
            df.ta.rsi(length=14, append=True)
            
            # MACD
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            
            # Bollinger Bands
            df.ta.bbands(length=20, std=2, append=True)
            
            # ATR
            df.ta.atr(length=14, append=True)
            
            # 提取最新值
            latest = df.iloc[-1]
            
            indicators = {
                "sma_20": float(latest.get('SMA_20', 0)),
                "sma_50": float(latest.get('SMA_50', 0)),
                "sma_200": float(latest.get('SMA_200', 0)),
                "rsi": float(latest.get('RSI_14', 0)),
                "macd": float(latest.get('MACD_12_26_9', 0)),
                "macd_signal": float(latest.get('MACDs_12_26_9', 0)),
                "macd_hist": float(latest.get('MACDh_12_26_9', 0)),
                "bb_upper": float(latest.get('BBU_20_2.0', 0)),
                "bb_middle": float(latest.get('BBM_20_2.0', 0)),
                "bb_lower": float(latest.get('BBL_20_2.0', 0)),
                "atr": float(latest.get('ATRr_14', 0))
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return indicators
    
    def _calculate_weekly_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算周度统计数据
        
        Args:
            df: OHLCV数据
        
        Returns:
            Dict: 周度统计
        """
        if df.empty:
            return {}
        
        try:
            start_price = float(df['Close'].iloc[0])
            end_price = float(df['Close'].iloc[-1])
            week_return = (end_price - start_price) / start_price
            
            stats = {
                "return": week_return,
                "high": float(df['High'].max()),
                "low": float(df['Low'].min()),
                "avg_volume": float(df['Volume'].mean()),
                "volatility": float(df['Close'].pct_change().std()),
                "days_traded": len(df)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating weekly stats: {e}")
            return {}
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """从缓存读取数据"""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]):
        """保存数据到缓存"""
        if not self.cache_enabled:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        获取当前价格
        
        Args:
            ticker: 股票代码
        
        Returns:
            float: 当前价格
        """
        try:
            tick = yf.Ticker(ticker)
            data = tick.history(period="1d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {e}")
        
        return None
