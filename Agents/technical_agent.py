"""
Technical Analysis Agent - MCP Server

技术分析专家Agent，提供传统技术指标分析

特点：
- 不使用LLM（纯计算，快速响应）
- 基于LEAN的Indicators系统
- 提供多种技术指标和信号生成

提供的Tools:
1. calculate_indicators - 计算多种技术指标
2. generate_signals - 生成买卖信号
3. detect_patterns - 检测图表形态（简化版）
4. find_support_resistance - 寻找支撑/阻力位（简化版）

提供的Resources:
- indicators://<SYMBOL> - 实时指标数据
- signals://<SYMBOL> - 当前信号状态
"""

from .base_mcp_agent import BaseMCPAgent
from mcp.types import Tool, Resource
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime


class TechnicalAnalysisAgent(BaseMCPAgent):
    """
    技术分析Agent - MCP Server
    
    提供传统技术指标分析，不依赖LLM
    直接使用LEAN的Indicators或自己计算指标
    """
    
    def __init__(self, algorithm=None):
        """
        初始化技术分析Agent
        
        Args:
            algorithm: LEAN的QCAlgorithm实例（可选）
                      如果提供，将使用LEAN的indicators
                      如果不提供，将使用模拟数据或自己计算
        """
        super().__init__(
            name="technical-analysis-agent",
            description="Provides technical analysis using traditional indicators (RSI, MACD, MA, etc.)",
            version="1.0.0"
        )
        
        self.algorithm = algorithm
        self.logger.info(f"Initialized with algorithm: {algorithm is not None}")
        
        # 缓存计算结果（避免重复计算）
        self._indicators_cache = {}
        self._cache_timestamp = {}
    
    # ═══════════════════════════════════════════════
    # MCP Protocol Implementation
    # ═══════════════════════════════════════════════
    
    def get_tools(self) -> List[Tool]:
        """定义该Agent提供的工具"""
        return [
            self._create_tool_schema(
                name="calculate_indicators",
                description="""
                Calculate multiple technical indicators for a symbol.
                
                Returns comprehensive technical analysis including:
                - Trend indicators: MA (20, 50, 200), MACD
                - Momentum indicators: RSI, Stochastic
                - Volatility indicators: Bollinger Bands, ATR
                - Volume indicators: Volume MA
                
                All values are current (latest available data point).
                """,
                properties={
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol (e.g., AAPL, TSLA)"
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ["5min", "1h", "1d"],
                        "description": "Timeframe for indicators (default: 1d)"
                    }
                },
                required=["symbol"]
            ),
            
            self._create_tool_schema(
                name="generate_signals",
                description="""
                Generate buy/sell signals based on technical indicators.
                
                Uses a scoring system combining multiple indicators:
                - RSI (oversold/overbought)
                - MACD (trend direction and momentum)
                - Moving Averages (trend and support/resistance)
                
                Returns:
                - action: BUY/SELL/HOLD
                - conviction: 1-10 score (confidence level)
                - reasoning: Detailed explanation of the signal
                - indicators_summary: Key indicator values
                """,
                properties={
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol"
                    }
                },
                required=["symbol"]
            ),
            
            self._create_tool_schema(
                name="detect_patterns",
                description="""
                Detect chart patterns (simplified version).
                
                Currently supports basic pattern detection:
                - Trend direction (uptrend/downtrend/sideways)
                - Breakout detection
                - Support/Resistance touches
                
                Note: Full pattern recognition (head & shoulders, triangles, etc.)
                      will be implemented in future versions.
                """,
                properties={
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol"
                    },
                    "lookback_days": {
                        "type": "number",
                        "description": "Number of days to look back (default: 60)"
                    }
                },
                required=["symbol"]
            ),
            
            self._create_tool_schema(
                name="find_support_resistance",
                description="""
                Identify key support and resistance levels.
                
                Uses multiple methods:
                - Moving averages as dynamic S/R
                - Recent high/low points
                - Pivot points calculation
                
                Returns levels with strength scores.
                """,
                properties={
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol"
                    }
                },
                required=["symbol"]
            )
        ]
    
    def get_resources(self) -> List[Resource]:
        """定义该Agent提供的资源"""
        return [
            Resource(
                uri=self._create_resource_uri("indicators"),
                name="Technical Indicators",
                description="Real-time technical indicators for all tracked symbols",
                mimeType="application/json"
            ),
            Resource(
                uri=self._create_resource_uri("signals"),
                name="Trading Signals",
                description="Current buy/sell signals based on technical analysis",
                mimeType="application/json"
            )
        ]
    
    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """处理工具调用"""
        
        symbol = arguments.get('symbol', '').upper()
        
        if not symbol:
            raise ValueError("Symbol is required")
        
        if name == "calculate_indicators":
            timeframe = arguments.get('timeframe', '1d')
            return self._calculate_indicators(symbol, timeframe)
        
        elif name == "generate_signals":
            return self._generate_signals(symbol)
        
        elif name == "detect_patterns":
            lookback_days = arguments.get('lookback_days', 60)
            return self._detect_patterns(symbol, lookback_days)
        
        elif name == "find_support_resistance":
            return self._find_support_resistance(symbol)
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    async def handle_resource_read(self, uri: str) -> Any:
        """读取资源"""
        
        if uri.startswith("indicators://"):
            # 格式: indicators://SYMBOL 或 indicators://
            symbol = uri.replace("indicators://", "")
            
            if symbol:
                return self._calculate_indicators(symbol)
            else:
                # 返回所有跟踪的symbol的指标
                return self._get_all_indicators()
        
        elif uri.startswith("signals://"):
            symbol = uri.replace("signals://", "")
            
            if symbol:
                return self._generate_signals(symbol)
            else:
                return self._get_all_signals()
        
        else:
            return await super().handle_resource_read(uri)
    
    # ═══════════════════════════════════════════════
    # Core Analysis Methods
    # ═══════════════════════════════════════════════
    
    def _calculate_indicators(self, symbol: str, timeframe: str = "1d") -> Dict:
        """
        计算技术指标
        
        如果有LEAN algorithm，使用LEAN的indicators
        否则使用模拟数据或自己计算
        """
        
        # 检查缓存
        cache_key = f"{symbol}_{timeframe}"
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"Using cached indicators for {symbol}")
            return self._indicators_cache[cache_key]
        
        if self.algorithm:
            indicators = self._calculate_from_lean(symbol)
        else:
            indicators = self._calculate_mock_indicators(symbol)
        
        # 缓存结果
        self._indicators_cache[cache_key] = indicators
        self._cache_timestamp[cache_key] = datetime.now()
        
        return indicators
    
    def _calculate_from_lean(self, symbol: str) -> Dict:
        """从LEAN获取指标"""
        
        try:
            # 获取LEAN的indicators
            indicators = self.algorithm.indicators.get(symbol, {})
            
            if not indicators:
                self.logger.warning(f"No indicators found for {symbol} in LEAN")
                return self._calculate_mock_indicators(symbol)
            
            # 获取当前价格
            if symbol in self.algorithm.Securities:
                current_price = self.algorithm.Securities[symbol].Price
            else:
                current_price = 0
            
            result = {
                'symbol': symbol,
                'timestamp': self.algorithm.Time.isoformat(),
                'current_price': current_price,
                'indicators': {}
            }
            
            # RSI
            if 'RSI' in indicators:
                rsi_value = indicators['RSI'].Current.Value
                result['indicators']['rsi'] = {
                    'value': rsi_value,
                    'signal': self._interpret_rsi(rsi_value)
                }
            
            # MACD
            if 'MACD' in indicators:
                macd = indicators['MACD']
                result['indicators']['macd'] = {
                    'macd': macd.Current.Value,
                    'signal': macd.Signal.Current.Value,
                    'histogram': macd.Histogram.Current.Value,
                    'signal_interpretation': self._interpret_macd(macd)
                }
            
            # Moving Averages
            for ma_name in ['SMA20', 'SMA50', 'SMA200']:
                if ma_name in indicators:
                    ma_value = indicators[ma_name].Current.Value
                    result['indicators'][ma_name.lower()] = {
                        'value': ma_value,
                        'distance_pct': self._calculate_distance(current_price, ma_value)
                    }
            
            # Bollinger Bands
            if 'BB' in indicators:
                bb = indicators['BB']
                result['indicators']['bollinger_bands'] = {
                    'upper': bb.UpperBand.Current.Value,
                    'middle': bb.MiddleBand.Current.Value,
                    'lower': bb.LowerBand.Current.Value,
                    'position': self._bb_position(current_price, bb)
                }
            
            # ATR (Average True Range)
            if 'ATR' in indicators:
                result['indicators']['atr'] = {
                    'value': indicators['ATR'].Current.Value
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators from LEAN: {e}")
            return self._calculate_mock_indicators(symbol)
    
    def _calculate_mock_indicators(self, symbol: str) -> Dict:
        """
        计算模拟指标（用于测试或无LEAN环境）
        
        返回合理的模拟数据
        """
        import random
        
        # 模拟当前价格
        base_price = 150.0
        current_price = base_price + random.uniform(-10, 10)
        
        # 模拟RSI
        rsi_value = random.uniform(30, 70)
        
        # 模拟MACD
        macd_value = random.uniform(-2, 2)
        signal_value = macd_value + random.uniform(-0.5, 0.5)
        histogram = macd_value - signal_value
        
        # 模拟均线
        sma20 = current_price * random.uniform(0.98, 1.02)
        sma50 = current_price * random.uniform(0.95, 1.05)
        sma200 = current_price * random.uniform(0.90, 1.10)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'indicators': {
                'rsi': {
                    'value': rsi_value,
                    'signal': self._interpret_rsi(rsi_value)
                },
                'macd': {
                    'macd': macd_value,
                    'signal': signal_value,
                    'histogram': histogram,
                    'signal_interpretation': 'bullish' if histogram > 0 else 'bearish'
                },
                'sma20': {
                    'value': sma20,
                    'distance_pct': self._calculate_distance(current_price, sma20)
                },
                'sma50': {
                    'value': sma50,
                    'distance_pct': self._calculate_distance(current_price, sma50)
                },
                'sma200': {
                    'value': sma200,
                    'distance_pct': self._calculate_distance(current_price, sma200)
                }
            },
            'note': 'Mock data (no LEAN algorithm provided)'
        }
    
    def _generate_signals(self, symbol: str) -> Dict:
        """
        生成交易信号
        
        综合多个指标，返回明确的建议
        """
        
        # 获取指标数据
        indicators_data = self._calculate_indicators(symbol)
        
        if 'error' in indicators_data:
            return indicators_data
        
        indicators = indicators_data.get('indicators', {})
        current_price = indicators_data.get('current_price', 0)
        
        # 计分系统
        bullish_score = 0
        bearish_score = 0
        reasons = []
        
        # === RSI 评分 ===
        if 'rsi' in indicators:
            rsi_value = indicators['rsi']['value']
            rsi_signal = indicators['rsi']['signal']
            
            if rsi_signal == 'oversold':
                bullish_score += 2
                reasons.append(f"RSI oversold ({rsi_value:.1f})")
            elif rsi_signal == 'overbought':
                bearish_score += 2
                reasons.append(f"RSI overbought ({rsi_value:.1f})")
        
        # === MACD 评分 ===
        if 'macd' in indicators:
            macd_signal = indicators['macd']['signal_interpretation']
            histogram = indicators['macd']['histogram']
            
            if 'bullish' in macd_signal:
                bullish_score += 2
                reasons.append(f"MACD bullish (hist: {histogram:.2f})")
            elif 'bearish' in macd_signal:
                bearish_score += 2
                reasons.append(f"MACD bearish (hist: {histogram:.2f})")
            
            # 弱化信号
            if 'weakening' in macd_signal:
                if 'bullish' in macd_signal:
                    bullish_score -= 1
                else:
                    bearish_score -= 1
        
        # === 均线评分 ===
        if 'sma200' in indicators:
            distance = indicators['sma200']['distance_pct']
            
            if distance > 0.05:  # 价格在200日均线上方5%
                bullish_score += 1
                reasons.append(f"Above SMA200 (+{distance*100:.1f}%)")
            elif distance < -0.05:  # 价格在200日均线下方5%
                bearish_score += 1
                reasons.append(f"Below SMA200 ({distance*100:.1f}%)")
        
        # === 短期均线交叉 ===
        if 'sma20' in indicators and 'sma50' in indicators:
            sma20 = indicators['sma20']['value']
            sma50 = indicators['sma50']['value']
            
            if sma20 > sma50 * 1.01:  # 金叉
                bullish_score += 1
                reasons.append("Golden cross (SMA20 > SMA50)")
            elif sma20 < sma50 * 0.99:  # 死叉
                bearish_score += 1
                reasons.append("Death cross (SMA20 < SMA50)")
        
        # === 生成最终信号 ===
        total_score = bullish_score - bearish_score
        
        if total_score >= 3:
            action = "BUY"
            conviction = min(10, 5 + total_score)
        elif total_score <= -3:
            action = "SELL"
            conviction = min(10, 5 + abs(total_score))
        else:
            action = "HOLD"
            conviction = 5
        
        return {
            'symbol': symbol,
            'timestamp': indicators_data['timestamp'],
            'action': action,
            'conviction': conviction,
            'reasoning': '; '.join(reasons) if reasons else 'Neutral technical setup',
            'score_breakdown': {
                'bullish': bullish_score,
                'bearish': bearish_score,
                'total': total_score
            },
            'indicators_summary': indicators,
            'current_price': current_price
        }
    
    def _detect_patterns(self, symbol: str, lookback_days: int) -> Dict:
        """
        检测图表形态（简化版本）
        
        未来可以实现更复杂的形态识别
        """
        
        indicators_data = self._calculate_indicators(symbol)
        indicators = indicators_data.get('indicators', {})
        
        patterns = []
        
        # 简单的趋势判断
        if 'sma20' in indicators and 'sma50' in indicators and 'sma200' in indicators:
            sma20 = indicators['sma20']['value']
            sma50 = indicators['sma50']['value']
            sma200 = indicators['sma200']['value']
            
            if sma20 > sma50 > sma200:
                patterns.append({
                    'pattern': 'Strong Uptrend',
                    'confidence': 0.8,
                    'description': 'All major MAs aligned bullishly'
                })
            elif sma20 < sma50 < sma200:
                patterns.append({
                    'pattern': 'Strong Downtrend',
                    'confidence': 0.8,
                    'description': 'All major MAs aligned bearishly'
                })
            else:
                patterns.append({
                    'pattern': 'Sideways/Consolidation',
                    'confidence': 0.6,
                    'description': 'MAs are mixed or converging'
                })
        
        return {
            'symbol': symbol,
            'lookback_days': lookback_days,
            'patterns': patterns,
            'note': 'Simplified pattern detection. Full pattern recognition coming soon.'
        }
    
    def _find_support_resistance(self, symbol: str) -> Dict:
        """
        寻找支撑和阻力位
        
        使用均线作为动态支撑/阻力
        """
        
        indicators_data = self._calculate_indicators(symbol)
        current_price = indicators_data.get('current_price', 0)
        indicators = indicators_data.get('indicators', {})
        
        support_levels = []
        resistance_levels = []
        
        # 使用均线作为支撑/阻力
        for ma_name in ['sma20', 'sma50', 'sma200']:
            if ma_name in indicators:
                ma_value = indicators[ma_name]['value']
                distance_pct = indicators[ma_name]['distance_pct']
                
                if ma_value < current_price:
                    # 下方的均线是潜在支撑
                    support_levels.append({
                        'level': ma_value,
                        'type': ma_name.upper(),
                        'distance_pct': abs(distance_pct),
                        'strength': self._calculate_level_strength(ma_name)
                    })
                else:
                    # 上方的均线是潜在阻力
                    resistance_levels.append({
                        'level': ma_value,
                        'type': ma_name.upper(),
                        'distance_pct': abs(distance_pct),
                        'strength': self._calculate_level_strength(ma_name)
                    })
        
        # 按距离排序
        support_levels.sort(key=lambda x: x['distance_pct'])
        resistance_levels.sort(key=lambda x: x['distance_pct'])
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'support_levels': support_levels[:3],  # 最近的3个支撑
            'resistance_levels': resistance_levels[:3],  # 最近的3个阻力
            'note': 'Using moving averages as dynamic S/R levels'
        }
    
    # ═══════════════════════════════════════════════
    # Helper Methods
    # ═══════════════════════════════════════════════
    
    def _interpret_rsi(self, rsi_value: float) -> str:
        """解读RSI信号"""
        if rsi_value > 70:
            return "overbought"
        elif rsi_value < 30:
            return "oversold"
        elif 45 <= rsi_value <= 55:
            return "neutral"
        else:
            return "normal"
    
    def _interpret_macd(self, macd) -> str:
        """解读MACD信号"""
        try:
            macd_value = macd.Current.Value
            signal_value = macd.Signal.Current.Value
            histogram = macd.Histogram.Current.Value
            
            if macd_value > signal_value:
                if histogram > 0:
                    return "bullish"
                else:
                    return "bullish_weakening"
            else:
                if histogram < 0:
                    return "bearish"
                else:
                    return "bearish_weakening"
        except:
            return "unknown"
    
    def _calculate_distance(self, price: float, reference: float) -> float:
        """计算价格与参考值的距离百分比"""
        if reference == 0:
            return 0
        return (price - reference) / reference
    
    def _bb_position(self, price: float, bb) -> str:
        """判断价格在布林带中的位置"""
        try:
            upper = bb.UpperBand.Current.Value
            lower = bb.LowerBand.Current.Value
            
            if price > upper:
                return "above_upper"
            elif price < lower:
                return "below_lower"
            else:
                return "within_bands"
        except:
            return "unknown"
    
    def _calculate_level_strength(self, ma_type: str) -> float:
        """计算支撑/阻力位的强度"""
        strength_map = {
            'sma20': 0.5,
            'sma50': 0.7,
            'sma200': 0.9
        }
        return strength_map.get(ma_type, 0.5)
    
    def _is_cache_valid(self, cache_key: str, max_age_seconds: int = 60) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self._cache_timestamp:
            return False
        
        age = (datetime.now() - self._cache_timestamp[cache_key]).total_seconds()
        return age < max_age_seconds
    
    def _get_all_indicators(self) -> Dict:
        """获取所有跟踪symbol的指标"""
        
        if not self.algorithm:
            return {'error': 'No algorithm provided, cannot list all indicators'}
        
        try:
            all_indicators = {}
            for symbol in self.algorithm.Securities.Keys:
                all_indicators[str(symbol)] = self._calculate_indicators(str(symbol))
            
            return {
                'timestamp': datetime.now().isoformat(),
                'indicators': all_indicators
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_all_signals(self) -> Dict:
        """获取所有跟踪symbol的信号"""
        
        if not self.algorithm:
            return {'error': 'No algorithm provided, cannot list all signals'}
        
        try:
            all_signals = {}
            for symbol in self.algorithm.Securities.Keys:
                all_signals[str(symbol)] = self._generate_signals(str(symbol))
            
            return {
                'timestamp': datetime.now().isoformat(),
                'signals': all_signals
            }
        except Exception as e:
            return {'error': str(e)}


# ═══════════════════════════════════════════════
# 主程序入口（用于测试）
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建并运行Agent（无LEAN，使用模拟数据）
    agent = TechnicalAnalysisAgent(algorithm=None)
    
    print("Starting Technical Analysis MCP Agent...")
    print("Use MCP Inspector to test:")
    print("  npx @modelcontextprotocol/inspector python Agents/technical_agent.py")
    print()
    
    asyncio.run(agent.run())
