"""
Finnhub API 客户端封装

提供统一的接口访问 Finnhub API，包括：
- 实时报价
- WebSocket 实时数据流
- 历史K线数据
- 分钟级数据聚合
- 公司新闻
- 公司基本信息
- 技术指标（免费版自行计算）
"""

import os
import finnhub
import pandas as pd
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple, Callable
import time
from dotenv import load_dotenv
import websocket
import json
import threading
from collections import defaultdict

# 加载环境变量
load_dotenv()


class FinnhubClient:
    """Finnhub API 客户端"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 Finnhub 客户端
        
        Args:
            api_key: Finnhub API密钥，如果不提供则从环境变量读取
        """
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        if not self.api_key:
            raise ValueError("需要提供 FINNHUB_API_KEY")
        
        # 去除可能的引号
        self.api_key = self.api_key.strip().strip('"').strip("'")
        self.client = finnhub.Client(api_key=self.api_key)
        
        # 免费版API限制：60次/分钟
        self.rate_limit_delay = 1.0  # 秒
        self.last_request_time = 0
        
        # WebSocket 相关
        self.ws = None
        self.ws_thread = None
        self.ws_callbacks = {}
        self.ws_running = False
        
        # 实时数据缓存
        self.realtime_ticks = defaultdict(list)  # {symbol: [ticks]}
        self.minute_bars = defaultdict(list)  # {symbol: [minute_bars]}
    
    def _rate_limit(self):
        """控制API请求频率"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def get_quote(self, symbol: str) -> Dict:
        """
        获取实时报价
        
        Args:
            symbol: 股票代码，如 'AAPL'
            
        Returns:
            包含实时报价数据的字典：
            - c: 当前价
            - d: 涨跌额
            - dp: 涨跌幅%
            - h: 最高价
            - l: 最低价
            - o: 开盘价
            - pc: 前收盘价
            - t: 时间戳
        """
        self._rate_limit()
        quote = self.client.quote(symbol)
        return quote
    
    def get_candles(
        self, 
        symbol: str, 
        resolution: str = 'D',
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        获取历史K线数据
        
        注意：免费版Finnhub不支持历史K线，使用yfinance作为备选方案
        
        Args:
            symbol: 股票代码
            resolution: 时间间隔 ('1', '5', '15', '30', '60', 'D', 'W', 'M')
                       免费版只支持 'D' (日线)
            days_back: 回溯天数
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        try:
            # 尝试使用 Finnhub API（可能需要付费）
            self._rate_limit()
            
            end = int(datetime.now().timestamp())
            start = int((datetime.now() - timedelta(days=days_back)).timestamp())
            
            candles = self.client.stock_candles(symbol, resolution, start, end)
            
            if candles['s'] == 'no_data':
                raise ValueError("No data from Finnhub")
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(candles['t'], unit='s'),
                'open': candles['o'],
                'high': candles['h'],
                'low': candles['l'],
                'close': candles['c'],
                'volume': candles['v']
            })
            
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            # 免费版Finnhub不支持，使用yfinance作为替代
            print(f"Finnhub K线数据不可用 (免费版限制)，使用 yfinance 替代...")
            return self._get_candles_from_yfinance(symbol, days_back)
    
    def _get_candles_from_yfinance(self, symbol: str, days_back: int) -> pd.DataFrame:
        """
        使用yfinance获取历史K线数据（备用方案）
        
        Args:
            symbol: 股票代码
            days_back: 回溯天数
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        try:
            import yfinance as yf
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return pd.DataFrame()
            
            # 重命名列以匹配 Finnhub 格式
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # 只保留需要的列
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            print(f"yfinance 获取数据失败: {e}")
            return pd.DataFrame()
    
    def get_company_news(
        self, 
        symbol: str, 
        days_back: int = 7
    ) -> List[Dict]:
        """
        获取公司新闻
        
        Args:
            symbol: 股票代码
            days_back: 回溯天数
            
        Returns:
            新闻列表，每条新闻包含：
            - headline: 标题
            - summary: 摘要
            - source: 来源
            - url: 链接
            - datetime: 时间戳
        """
        self._rate_limit()
        
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        news = self.client.company_news(symbol, _from=start, to=end)
        return news
    
    def get_market_news(self, category: str = 'general', limit: int = 10) -> List[Dict]:
        """
        获取市场新闻
        
        Args:
            category: 新闻类别 ('general', 'forex', 'crypto', 'merger')
            limit: 返回数量限制
            
        Returns:
            新闻列表
        """
        self._rate_limit()
        news = self.client.general_news(category)
        return news[:limit]
    
    def get_company_profile(self, symbol: str) -> Dict:
        """
        获取公司基本信息
        
        Args:
            symbol: 股票代码
            
        Returns:
            公司信息字典：
            - name: 公司名称
            - ticker: 股票代码
            - finnhubIndustry: 行业
            - marketCapitalization: 市值
            - ipo: IPO日期
            - weburl: 网站
            - logo: Logo URL
        """
        self._rate_limit()
        profile = self.client.company_profile2(symbol=symbol)
        return profile
    
    def get_basic_financials(self, symbol: str) -> Dict:
        """
        获取基本财务指标
        
        Args:
            symbol: 股票代码
            
        Returns:
            财务指标字典
        """
        self._rate_limit()
        metrics = self.client.company_basic_financials(symbol, 'all')
        return metrics
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标（免费版需要自行计算）
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了技术指标的DataFrame
        """
        if df.empty:
            return df
        
        # 移动平均线
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Histogram'] = df['MACD'] - df['Signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Std'] = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        
        return df
    
    def get_realtime_data(self, symbol: str) -> Dict:
        """
        获取综合实时数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            包含报价、公司信息等的综合数据字典
        """
        data = {
            'symbol': symbol,
            'quote': self.get_quote(symbol),
            'profile': self.get_company_profile(symbol),
            'timestamp': datetime.now()
        }
        return data
    
    def get_today_intraday_data(self, symbol: str) -> pd.DataFrame:
        """
        获取今日分钟级数据（使用yfinance）
        
        Args:
            symbol: 股票代码
            
        Returns:
            包含今日分钟数据的DataFrame，列：time, open, high, low, close, volume
        """
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            # 获取今日1分钟数据
            df = ticker.history(period='1d', interval='1m')
            
            if df.empty:
                return pd.DataFrame()
            
            # 重命名列
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # 只保留需要的列
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            df.index.name = 'time'
            df = df.reset_index()
            
            return df
            
        except Exception as e:
            print(f"获取今日分钟数据失败: {e}")
            return pd.DataFrame()
    
    def start_websocket(self, symbols: List[str], callback: Optional[Callable] = None):
        """
        启动 WebSocket 连接订阅实时数据
        
        Args:
            symbols: 要订阅的股票代码列表
            callback: 收到数据时的回调函数
        """
        if self.ws_running:
            print("WebSocket 已在运行")
            return
        
        self.ws_running = True
        
        def on_message(ws, message):
            data = json.loads(message)
            if data['type'] == 'trade':
                for trade in data['data']:
                    symbol = trade['s']
                    tick = {
                        'symbol': symbol,
                        'price': trade['p'],
                        'volume': trade['v'],
                        'timestamp': datetime.fromtimestamp(trade['t'] / 1000),
                        'conditions': trade.get('c', [])
                    }
                    
                    # 保存到缓存
                    self.realtime_ticks[symbol].append(tick)
                    
                    # 调用回调
                    if callback:
                        callback(tick)
                    
                    # 只保留最近1000个tick
                    if len(self.realtime_ticks[symbol]) > 1000:
                        self.realtime_ticks[symbol] = self.realtime_ticks[symbol][-1000:]
        
        def on_error(ws, error):
            print(f"WebSocket 错误: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("WebSocket 连接关闭")
            self.ws_running = False
        
        def on_open(ws):
            print("WebSocket 连接成功")
            # 订阅股票
            for symbol in symbols:
                ws.send(json.dumps({'type': 'subscribe', 'symbol': symbol}))
                print(f"订阅 {symbol}")
        
        def run_websocket():
            websocket_url = f"wss://ws.finnhub.io?token={self.api_key}"
            self.ws = websocket.WebSocketApp(
                websocket_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            self.ws.run_forever()
        
        # 在单独线程中运行
        self.ws_thread = threading.Thread(target=run_websocket, daemon=True)
        self.ws_thread.start()
    
    def stop_websocket(self):
        """停止 WebSocket 连接"""
        if self.ws and self.ws_running:
            self.ws.close()
            self.ws_running = False
            print("WebSocket 已停止")
    
    def get_realtime_ticks(self, symbol: str) -> List[Dict]:
        """
        获取缓存的实时 tick 数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            tick 数据列表
        """
        return self.realtime_ticks.get(symbol, [])
    
    def aggregate_to_minute_bars(self, symbol: str) -> pd.DataFrame:
        """
        将实时 tick 数据聚合为分钟K线
        
        Args:
            symbol: 股票代码
            
        Returns:
            分钟K线 DataFrame
        """
        ticks = self.realtime_ticks.get(symbol, [])
        if not ticks:
            return pd.DataFrame()
        
        # 转换为 DataFrame
        df = pd.DataFrame(ticks)
        df['time'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('time')
        
        # 按分钟聚合
        minute_bars = df.groupby(pd.Grouper(freq='1Min')).agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        
        minute_bars.columns = ['open', 'high', 'low', 'close', 'volume']
        minute_bars = minute_bars.dropna()
        
        return minute_bars


if __name__ == "__main__":
    # 测试代码
    client = FinnhubClient()
    
    # 测试实时报价
    print("=" * 50)
    print("实时报价测试 (AAPL)")
    print("=" * 50)
    quote = client.get_quote('AAPL')
    print(f"当前价: ${quote['c']:.2f}")
    print(f"涨跌: ${quote['d']:.2f} ({quote['dp']:.2f}%)")
    print(f"最高: ${quote['h']:.2f} | 最低: ${quote['l']:.2f}")
    
    # 测试历史数据
    print("\n" + "=" * 50)
    print("历史K线数据测试 (AAPL, 最近7天)")
    print("=" * 50)
    df = client.get_candles('AAPL', 'D', 7)
    print(df.tail())
    
    # 测试技术指标
    print("\n" + "=" * 50)
    print("技术指标计算测试")
    print("=" * 50)
    df = client.calculate_indicators(df)
    print(df[['close', 'MA5', 'MA10', 'RSI']].tail())
