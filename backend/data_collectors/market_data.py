"""
Market data collector using yfinance
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import pickle

from .base_collector import BaseCollector
from backend.utils.logger import get_logger

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
            tickers: 股票代码列表，默认为["SPY", "QQQ", "^VIX"]
            cache_enabled: 是否启用缓存
            cache_dir: 缓存目录
        """
        super().__init__(kwargs)
        self.tickers = tickers or ["SPY", "QQQ", "^VIX"]
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)
        
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
        
        # 检查缓存
        cache_key = f"{','.join(self.tickers)}_{start_date.date()}_{end_date.date()}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.info("Using cached market data")
            return cached_data
        
        market_data = {}
        
        for ticker in self.tickers:
            try:
                # 下载数据
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
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
                
                market_data[ticker] = {
                    "ohlcv": data_with_date.to_dict('records'),
                    "latest_price": float(data['Close'].iloc[-1]) if len(data) > 0 else None,
                    "indicators": indicators,
                    "weekly_stats": weekly_stats
                }
                
            except Exception as e:
                logger.error(f"Error collecting data for {ticker}: {e}")
                continue
        
        # 缓存数据
        if self.cache_enabled and market_data:
            self._save_to_cache(cache_key, market_data)
        
        return market_data
    
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
