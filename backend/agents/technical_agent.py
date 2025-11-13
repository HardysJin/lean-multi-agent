"""
Technical Analysis Agent
技术分析Agent
"""

from typing import Dict, Any, Optional
from datetime import datetime

from .base_agent import BaseAgent
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class TechnicalAgent(BaseAgent):
    """技术分析Agent"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化技术分析Agent"""
        super().__init__("TechnicalAgent", config)
        self.indicators = self.config.get('indicators', [
            'SMA', 'RSI', 'MACD', 'BB', 'ATR'
        ])
    
    def analyze(self, data: Dict[str, Any], as_of_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        技术分析
        
        Args:
            data: 市场数据（来自MarketDataCollector）
            as_of_date: 决策时间点（用于回测，默认None=当前时间）
        
        Returns:
            Dict: 技术分析结果
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        if not self.validate_input(data):
            logger.warning("Invalid input data for TechnicalAgent")
            return self._get_empty_analysis(as_of_date)
        
        logger.info("Running technical analysis")
        
        # 分析主要市场指数（SPY）
        spy_data = data.get('SPY', {})
        if not spy_data:
            logger.warning("No SPY data available")
            return self._get_empty_analysis(as_of_date)
        
        indicators = spy_data.get('indicators', {})
        weekly_stats = spy_data.get('weekly_stats', {})
        latest_price = spy_data.get('latest_price', 0)
        
        # 趋势分析
        trend_analysis = self._analyze_trend(latest_price, indicators)
        
        # 动量分析
        momentum_analysis = self._analyze_momentum(indicators)
        
        # 波动性分析
        volatility_analysis = self._analyze_volatility(indicators, weekly_stats)
        
        # 支撑阻力分析
        support_resistance = self._analyze_support_resistance(latest_price, indicators, weekly_stats)
        
        # 综合评估
        overall_signal = self._generate_overall_signal(
            trend_analysis,
            momentum_analysis,
            volatility_analysis
        )
        
        return {
            "trend": trend_analysis,
            "momentum": momentum_analysis,
            "volatility": volatility_analysis,
            "support_resistance": support_resistance,
            "overall_signal": overall_signal,
            "timestamp": as_of_date.isoformat()
        }
    
    def _analyze_trend(self, price: float, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        趋势分析（V2增强版）
        
        Args:
            price: 当前价格
            indicators: 技术指标
        
        Returns:
            Dict: 趋势分析结果
        """
        sma_20 = indicators.get('sma_20', 0)
        sma_50 = indicators.get('sma_50', 0)
        sma_200 = indicators.get('sma_200', 0)
        adx = indicators.get('adx', 0)
        
        # 判断MA对齐情况
        ma_aligned_bullish = (price > sma_50 > sma_200)
        ma_aligned_bearish = (price < sma_50 < sma_200)
        
        # 趋势强度判断（基于ADX）
        # ADX > 25: 强趋势, 20-25: 中等, < 20: 弱趋势/盘整
        if adx > 25:
            adx_strength = "strong"
        elif adx > 20:
            adx_strength = "moderate"
        else:
            adx_strength = "weak"
        
        # 综合判断趋势
        if ma_aligned_bullish and adx > 25:
            trend = "strong_uptrend"
            strength = min(0.9, 0.5 + adx / 100)  # ADX越高，强度越大
        elif ma_aligned_bullish:
            trend = "uptrend"
            strength = 0.6 + adx / 200
        elif ma_aligned_bearish and adx > 25:
            trend = "strong_downtrend"
            strength = max(0.1, 0.5 - adx / 100)
        elif ma_aligned_bearish:
            trend = "downtrend"
            strength = 0.4 - adx / 200
        else:
            trend = "neutral"
            strength = 0.5
        
        # MA alignment字符串（用于prompt）
        if price > sma_50 > sma_200:
            ma_alignment = "Price > MA(50) > MA(200)"
        elif price < sma_50 < sma_200:
            ma_alignment = "Price < MA(50) < MA(200)"
        elif price > sma_50 and sma_50 < sma_200:
            ma_alignment = "Price > MA(50) < MA(200) (mixed)"
        else:
            ma_alignment = "MA alignment unclear"
        
        return {
            "direction": trend,
            "strength": strength,
            "adx": adx,
            "adx_strength": adx_strength,
            "ma_alignment": ma_alignment,
            "above_sma20": price > sma_20,
            "above_sma50": price > sma_50,
            "above_sma200": price > sma_200
        }
    
    def _analyze_momentum(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        动量分析
        
        Args:
            indicators: 技术指标
        
        Returns:
            Dict: 动量分析结果
        """
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_hist = indicators.get('macd_hist', 0)
        
        # RSI分析
        if rsi > 70:
            rsi_signal = "overbought"
        elif rsi < 30:
            rsi_signal = "oversold"
        else:
            rsi_signal = "neutral"
        
        # MACD分析
        if macd > macd_signal and macd_hist > 0:
            macd_signal_str = "bullish"
        elif macd < macd_signal and macd_hist < 0:
            macd_signal_str = "bearish"
        else:
            macd_signal_str = "neutral"
        
        # 综合动量
        if rsi > 60 and macd_signal_str == "bullish":
            momentum = "positive"
        elif rsi < 40 and macd_signal_str == "bearish":
            momentum = "negative"
        else:
            momentum = "neutral"
        
        return {
            "momentum": momentum,
            "rsi": rsi,
            "rsi_signal": rsi_signal,
            "macd_signal": macd_signal_str,
            "macd_histogram": macd_hist
        }
    
    def _analyze_volatility(
        self,
        indicators: Dict[str, Any],
        weekly_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        波动性分析（V2增强版）
        
        Args:
            indicators: 技术指标
            weekly_stats: 周度统计
        
        Returns:
            Dict: 波动性分析结果
        """
        atr = indicators.get('atr', 0)
        atr_pct = indicators.get('atr_pct', 0)
        volatility = weekly_stats.get('volatility', 0)
        realized_vol_annual = indicators.get('realized_vol_annual', 0)
        
        # 波动性分级（基于realized volatility年化）
        if realized_vol_annual < 15:
            vol_level = "low"
        elif realized_vol_annual < 25:
            vol_level = "medium"
        else:
            vol_level = "high"
        
        # 判断波动率是否扩张（简化判断：如果周度波动率 > 2%，认为在扩张）
        expanding = volatility > 0.02
        
        return {
            "level": vol_level,
            "atr": atr,
            "atr_pct": atr_pct,
            "historical_volatility": volatility,
            "realized_vol_annual": realized_vol_annual,
            "expanding": expanding
        }
    
    def _analyze_support_resistance(
        self,
        price: float,
        indicators: Dict[str, Any],
        weekly_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        支撑阻力分析
        
        Args:
            price: 当前价格
            indicators: 技术指标
            weekly_stats: 周度统计
        
        Returns:
            Dict: 支撑阻力位
        """
        bb_upper = indicators.get('bb_upper', 0)
        bb_lower = indicators.get('bb_lower', 0)
        week_high = weekly_stats.get('high', 0)
        week_low = weekly_stats.get('low', 0)
        
        return {
            "resistance_levels": [bb_upper, week_high],
            "support_levels": [bb_lower, week_low],
            "current_price": price
        }
    
    def _generate_overall_signal(
        self,
        trend: Dict[str, Any],
        momentum: Dict[str, Any],
        volatility: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成综合信号
        
        Args:
            trend: 趋势分析
            momentum: 动量分析
            volatility: 波动性分析
        
        Returns:
            Dict: 综合信号
        """
        trend_dir = trend['direction']
        momentum_dir = momentum['momentum']
        vol_level = volatility['level']
        
        # 综合判断
        if 'uptrend' in trend_dir and momentum_dir == 'positive':
            signal = "bullish"
            confidence = 0.8
        elif 'downtrend' in trend_dir and momentum_dir == 'negative':
            signal = "bearish"
            confidence = 0.8
        else:
            signal = "neutral"
            confidence = 0.5
        
        # 高波动降低置信度
        if vol_level == "high":
            confidence *= 0.8
        
        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": f"Trend: {trend_dir}, Momentum: {momentum_dir}, Volatility: {vol_level}"
        }
    
    def _get_empty_analysis(self, as_of_date: Optional[datetime] = None) -> Dict[str, Any]:
        """返回空分析结果"""
        if as_of_date is None:
            as_of_date = datetime.now()
        return {
            "trend": {"direction": "unknown", "strength": 0.5},
            "momentum": {"momentum": "neutral"},
            "volatility": {"level": "unknown"},
            "support_resistance": {},
            "overall_signal": {"signal": "neutral", "confidence": 0.0},
            "timestamp": as_of_date.isoformat()
        }
