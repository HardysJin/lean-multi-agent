"""LLM Prompts for Periodly Decision System"""

from typing import List, Optional, Dict


# 策略描述映射
STRATEGY_DESCRIPTIONS = {
    "grid_trading": "For ranging/sideways markets with clear support/resistance",
    "momentum": "For strong trending markets",
    "mean_reversion": "For overextended moves likely to reverse",
    "double_ema_channel": "For strong trending markets with breakout confirmation (dual EMA channels with volume)",
    "buy_and_hold": "Baseline strategy for long-term bullish outlook",
    "hold": "When uncertain or high risk"
}

# Regime定义和量化标准 (V2增强版)
REGIME_DEFINITIONS = {
    "BULL_TREND": {
        "description": "Strong sustained uptrend - ride the trend",
        "criteria": [
            "Price > MA(50) > MA(200)",
            "ADX > 25 (strong trend)",
            "RSI between 50-80",
            "Price gain > 5% over analysis period",
            "Higher highs and higher lows pattern"
        ],
        "recommended_strategies": ["momentum", "buy_and_hold"],
        "risk_level": "medium"
    },
    "BEAR_TREND": {
        "description": "Strong sustained downtrend - defensive positioning",
        "criteria": [
            "Price < MA(50) < MA(200)",
            "ADX > 25 (strong trend)",
            "RSI between 30-50",
            "Price loss > 5% over analysis period",
            "Lower highs and lower lows pattern"
        ],
        "recommended_strategies": ["hold", "cash"],
        "risk_level": "high"
    },
    "HIGH_VOL_RANGE": {
        "description": "Violent swings without clear direction - dangerous territory",
        "criteria": [
            "ADX < 20 (weak trend)",
            "Realized volatility > 25% annualized",
            "VIX > 25",
            "Wide intraday ranges (ATR > 3% of price)"
        ],
        "recommended_strategies": ["hold", "grid_trading_small_size"],
        "risk_level": "very_high"
    },
    "LOW_VOL_RANGE": {
        "description": "Tight consolidation - good for range strategies",
        "criteria": [
            "ADX < 20 (weak trend)",
            "Realized volatility < 15% annualized",
            "VIX < 20",
            "Price contained in <5% range"
        ],
        "recommended_strategies": ["grid_trading", "mean_reversion"],
        "risk_level": "low"
    },
    "TRANSITIONING": {
        "description": "Market changing regime - wait for clarity",
        "criteria": [
            "MA crossovers in progress",
            "Conflicting signals (e.g., RSI bullish but price bearish)",
            "VIX spike >20% in 5 days",
            "Technical uncertainty (0.4 < trend_strength < 0.6)"
        ],
        "recommended_strategies": ["hold", "reduce_exposure"],
        "risk_level": "high"
    },
    "EVENT_DRIVEN": {
        "description": "External shock dominates - expect high volatility",
        "criteria": [
            "VIX spike > 30",
            "Gap move > 3% on major news",
            "Major macro event (Fed decision, earnings, geopolitical)",
            "Correlation breakdown (normal relationships disrupted)"
        ],
        "recommended_strategies": ["hold", "wait_and_see"],
        "risk_level": "very_high"
    }
}


def get_regime_classification_guide() -> str:
    """生成regime分类指南文本（V2）"""
    guide_lines = []
    for regime, details in REGIME_DEFINITIONS.items():
        guide_lines.append(f"\n**{regime}:** {details['description']}")
        guide_lines.append("  Criteria:")
        for criterion in details['criteria']:
            guide_lines.append(f"    - {criterion}")
        guide_lines.append(f"  Recommended: {', '.join(details['recommended_strategies'])}")
        guide_lines.append(f"  Risk: {details['risk_level']}")
    
    return "\n".join(guide_lines)


def get_coordinator_system_prompt(available_strategies: Optional[List[str]] = None, version: str = "v1") -> str:
    """
    生成Coordinator System Prompt
    
    Args:
        available_strategies: 可用策略列表，如果为None则使用默认列表
        version: prompt版本，"v1"=原版（简单），"v2"=增强版（6步regime分类）
    
    Returns:
        str: 格式化的system prompt
    """
    if available_strategies is None:
        available_strategies = ["grid_trading", "momentum", "mean_reversion", "double_ema_channel", "buy_and_hold", "hold"]
    
    # 构建策略列表部分
    strategy_lines = []
    for strategy in available_strategies:
        description = STRATEGY_DESCRIPTIONS.get(strategy, "Custom strategy")
        strategy_lines.append(f"- {strategy}: {description}")
    
    strategies_text = "\n".join(strategy_lines)
    
    # 构建recommended_strategy的可选值
    strategy_options = "/".join(available_strategies)
    
    # 根据版本选择不同的prompt
    if version == "v2":
        return _get_coordinator_system_prompt_v2(strategies_text, strategy_options)
    else:
        return _get_coordinator_system_prompt_v1(strategies_text, strategy_options)


def _get_coordinator_system_prompt_v1(strategies_text: str, strategy_options: str) -> str:
    """原版prompt（简洁版）"""
    return f"""You are an expert quantitative trading strategist making data-driven trading decisions.

Your task is to analyze historical market data over a specified lookback period and forecast market conditions for an upcoming period, determining:
1. Current market state based on historical data (trending/ranging/volatile/event-driven)
2. Recommended trading strategy for the forecast period
3. Optional position allocation (if enabled)

**Available Strategies:**
{strategies_text}

**Analysis Framework:**
1. Technical Analysis: Price trends, volatility, support/resistance
2. Sentiment Analysis: Market fear/greed, social media sentiment
3. News Analysis: Major events, earnings, macro news
4. Historical Performance: What worked last period?

**Risk Constraints:**
- Max single position: 30%
- Min cash reserve: 20%
- Max period turnover: 50%
- Stop loss: 5% per position

**Output Format (JSON):**
{{
  "market_state": "trending/ranging/volatile/event_driven",
  "reasoning": "Detailed analysis of current market conditions...",
  "recommended_strategy": "{strategy_options}",
  "suggested_positions": {{  // Optional, only if can_suggest_positions=true
    "AAPL": 0.25,
    "MSFT": 0.20,
    "cash": 0.55
  }},
  "confidence": 0.85,  // 0-1 scale
  "risk_assessment": "Low/Medium/High risk assessment..."
}}

Be conservative. When uncertain, recommend 'hold' and maintain cash reserves.
"""


def _get_coordinator_system_prompt_v2(strategies_text: str, strategy_options: str) -> str:
    """增强版prompt（6步regime分类）"""
    regime_guide = get_regime_classification_guide()
    
    return f"""You are an expert market regime identification system for quantitative trading.

Your task is to analyze market data systematically and classify the current regime, then recommend appropriate trading actions.

**REGIME CLASSIFICATION FRAMEWORK:**
{regime_guide}

**DECISION PROCESS (Follow strictly):**

Step 1: EVENT CHECK
- Is VIX > 30 or gap > 3% or major macro event?
- If YES → regime = EVENT_DRIVEN, recommend "hold"

Step 2: TREND ASSESSMENT  
- Calculate: Price vs MA(50), MA(50) vs MA(200), ADX value
- Is ADX > 25 AND price aligned with MAs?
- If YES → regime = BULL_TREND or BEAR_TREND

Step 3: VOLATILITY REGIME
- Calculate: Realized vol, VIX level, ATR
- Is volatility elevated (VIX > 25 or RV > 25%)?
- Classify as HIGH_VOL_RANGE or LOW_VOL_RANGE

Step 4: EDGE CASE DETECTION
- Are signals conflicting? (e.g., RSI bullish but price < MA)
- Is trend_strength between 0.4-0.6? (weak/uncertain)
- If YES → regime = TRANSITIONING

Step 5: ANOMALY DETECTION
- Check for sector divergences (any ticker >20% divergence from market?)
- Check for VIX spikes (>20% change in 5 days?)
- Flag these as risk factors

Step 6: CONFIDENCE CALIBRATION
- Clear signals (all indicators agree) → confidence 0.8-0.9
- Mixed signals (some conflict) → confidence 0.5-0.7  
- High uncertainty → confidence <0.5, recommend "hold"

**STRATEGY SELECTION RULES:**
Based on the identified regime, choose the PRIMARY strategy:
- BULL_TREND → momentum or buy_and_hold or double_ema_channel (ride the trend)
- BEAR_TREND → hold (defensive, preserve capital)
- LOW_VOL_RANGE → grid_trading or mean_reversion (range-bound strategies)
- HIGH_VOL_RANGE → grid_trading
- TRANSITIONING → hold or mean_reversion (wait for clarity)
- EVENT_DRIVEN → hold (wait for dust to settle)


**AVAILABLE STRATEGIES:**
{strategies_text}

**RISK CONSTRAINTS:**
- Max single position: 30% (reduce to 20% if confidence < 0.7)
- Min cash reserve: 20% (increase to 40% if regime = HIGH_VOL_RANGE or EVENT_DRIVEN)
- Max period turnover: 50%
- Stop loss: 5% per position (tighten to 3% in TRANSITIONING or EVENT_DRIVEN regimes)

**OUTPUT FORMAT (JSON):**
{{{{
  "regime_classification": {{{{
    "primary_regime": "BULL_TREND/BEAR_TREND/HIGH_VOL_RANGE/LOW_VOL_RANGE/TRANSITIONING/EVENT_DRIVEN",
    "confidence": 0.85,
    "alternative_regime": "BULL_TREND",
    "regime_duration_estimate": "1-2 weeks / 2-4 weeks / 4+ weeks"
  }}}},
  
  "detailed_reasoning": {{{{
    "step1_event_check": "Event check results...",
    "step2_trend": "Trend assessment...",
    "step3_volatility": "Volatility analysis...",
    "step4_edge_cases": "Edge case detection...",
    "step5_anomalies": ["Anomaly 1", "Anomaly 2"],
    "step6_confidence_factors": ["+0.3: Factor 1", "-0.1: Factor 2"]
  }}}},
  
  "strategy_recommendation": {{{{
    "primary_strategy": "ONE OF: {strategy_options}",
    "rationale": "Why this strategy fits the current regime...",
    "position_sizing": {{{{
      "aggressive": false,
      "suggested_exposure": 0.70
    }}}}
  }}}},
  
  "suggested_positions": {{{{
    "SPY": 0.25,
    "cash": 0.75
  }}}},
  
  "risk_assessment": {{{{
    "overall_risk": "Low/Medium/High/Very High",
    "key_risks": ["Risk 1", "Risk 2"],
    "risk_mitigation": ["Action 1", "Action 2"]
  }}}},
  
  "execution_guidelines": {{{{
    "entry_timing": "Immediate / Wait for pullback / Wait for breakout",
    "exit_conditions": ["Condition 1", "Condition 2"],
    "rebalance_trigger": "Trigger description"
  }}}}
}}}}

**CRITICAL RULES:**
1. When in doubt, choose "hold" and preserve capital
2. Never recommend >70% equity exposure if confidence < 0.7
3. Always explain reasoning step-by-step (show your work!)
4. Flag ALL anomalies and divergences (they matter!)
5. If regime is TRANSITIONING or EVENT_DRIVEN, default to defensive positioning
6. Historical accuracy matters: if past predictions were wrong, be more conservative

Be thorough but concise. Prioritize capital preservation over profit maximization.
"""


# 保留旧的常量作为默认值（为了向后兼容）
COORDINATOR_SYSTEM_PROMPT = get_coordinator_system_prompt()

COORDINATOR_USER_PROMPT_TEMPLATE = """
**Analysis Period:** {analysis_start_date} to {analysis_end_date} ({lookback_days} days)
**Forecast Period:** {forecast_start_date} to {forecast_end_date} ({forecast_days} days)

**Market Data (Analysis Period):**
{market_data}

**Technical Analysis:**
{technical_analysis}

**Sentiment Analysis:**
{sentiment_analysis}

**News Summary:**
{news_summary}

**Current Portfolio:**
{current_portfolio}

**Last Period P&L:** {last_period_pnl}

**Decision History:**
{decision_history}

Based on the analysis of the past {lookback_days} days, provide your trading decision for the next {forecast_days} days in JSON format.
Your decision should explain what you observed during the analysis period and what you expect for the forecast period.
"""

# V2 User Prompt (增强版)
COORDINATOR_USER_PROMPT_TEMPLATE_V2 = """
**Analysis Period:** {analysis_start_date} to {analysis_end_date} ({lookback_days} days)
**Forecast Period:** {forecast_start_date} to {forecast_end_date} ({forecast_days} days)

**Market Data (Analysis Period):**
{market_data}

**Technical Analysis:**
{technical_analysis}

**Sentiment Analysis:**
{sentiment_analysis}

**News Summary:**
{news_summary}

**Current Portfolio:**
{current_portfolio}

**Last Period P&L:** {last_period_pnl}

**Historical Performance Tracking:**
{decision_history}

---

**YOUR TASK:**
Using the 6-step decision process outlined in your system prompt, classify the current market regime and provide your recommendation in the specified JSON format.

Remember to:
1. Check for events first (Step 1)
2. Assess trend systematically (Step 2)
3. Measure volatility regime (Step 3)
4. Detect edge cases (Step 4)
5. Flag anomalies (Step 5)
6. Calibrate confidence (Step 6)

Be explicit about your reasoning at each step. Show your work!
"""


def format_decision_history(historical_decisions: List[Dict]) -> str:
    """
    格式化历史决策，包含回测结果（V2增强版）
    
    Args:
        historical_decisions: 历史决策列表
        [{
            'period': '2024-02-01 to 2024-02-14',
            'predicted_regime': 'BULL_TREND',
            'confidence': 0.8,
            'strategy_used': 'momentum',
            'outcome': {
                'market_return': 0.05,
                'portfolio_return': 0.03,
                'correct_regime': True
            }
        }]
    
    Returns:
        str: 格式化的历史文本
    """
    if not historical_decisions:
        return "No historical data available (first run)"
    
    history_lines = ["Recent Performance (Last 3 periods):"]
    
    for i, decision in enumerate(historical_decisions[-3:], 1):
        outcome = decision.get('outcome', {})
        market_ret = outcome.get('market_return', 0) * 100
        portfolio_ret = outcome.get('portfolio_return', 0) * 100
        correct = outcome.get('correct_regime', None)
        
        status = "✓ Correct" if correct else "✗ Incorrect" if correct is False else "- Pending"
        outperform = "✓" if portfolio_ret > market_ret else "✗"
        
        history_lines.append(f"\nPeriod {i}: {decision.get('period', 'Unknown')}")
        history_lines.append(f"  Predicted Regime: {decision.get('predicted_regime', 'N/A')} (confidence: {decision.get('confidence', 0):.2f})")
        history_lines.append(f"  Strategy Used: {decision.get('strategy_used', 'N/A')}")
        history_lines.append(f"  Regime Accuracy: {status}")
        history_lines.append(f"  Market Return: {market_ret:+.2f}%")
        history_lines.append(f"  Portfolio Return: {portfolio_ret:+.2f}% {outperform}")
        
        if correct is False:
            history_lines.append(f"  ⚠️  Lesson: {decision.get('lesson', 'Review why regime was misidentified')}")
    
    # 计算总体准确率
    total_decisions = len([d for d in historical_decisions if d.get('outcome', {}).get('correct_regime') is not None])
    correct_decisions = len([d for d in historical_decisions if d.get('outcome', {}).get('correct_regime') is True])
    
    if total_decisions > 0:
        accuracy = correct_decisions / total_decisions
        history_lines.append(f"\n**Overall Accuracy: {accuracy:.1%} ({correct_decisions}/{total_decisions})**")
        
        if accuracy < 0.6:
            history_lines.append("⚠️  LOW ACCURACY WARNING: Be more conservative (increase confidence threshold)")
        elif accuracy > 0.8:
            history_lines.append("✓ HIGH ACCURACY: Your regime identification is reliable")
    
    return "\n".join(history_lines)


# Few-shot examples
COORDINATOR_EXAMPLES = [
    {
        "input": """
**Period Ending:** 2024-01-15
**Market Data:** SPY +3.2%, VIX: 12.5 (down from 18), Volume: Above average
**Technical:** Strong uptrend, RSI 68, MACD bullish crossover
**Sentiment:** Bullish (Fear & Greed: 72)
**News:** Strong earnings from tech sector, Fed dovish comments
**Current Portfolio:** 50% cash, 25% AAPL, 25% MSFT
**Last Period P&L:** +2.1%
""",
        "output": {
            "market_state": "trending",
            "reasoning": "Market shows strong momentum with declining VIX and positive breadth. Tech earnings beat expectations and Fed dovish stance supports risk-on. RSI not yet overbought. Trend is intact.",
            "recommended_strategy": "momentum",
            "suggested_positions": {
                "AAPL": 0.30,
                "MSFT": 0.25,
                "NVDA": 0.20,
                "cash": 0.25
            },
            "confidence": 0.82,
            "risk_assessment": "Medium. Trend is strong but monitor RSI for overbought conditions."
        }
    },
    {
        "input": """
**Period Ending:** 2024-03-10
**Market Data:** SPY -1.5%, VIX: 25 (up from 15), Volume: High
**Technical:** Range-bound between 480-500, RSI 45, choppy
**Sentiment:** Neutral to bearish (Fear & Greed: 38)
**News:** Mixed earnings, geopolitical tensions rising
**Current Portfolio:** 30% cash, 40% QQQ, 30% SPY
**Last Period P&L:** -2.5%
""",
        "output": {
            "market_state": "ranging",
            "reasoning": "Market stuck in consolidation range with elevated VIX. No clear direction. Geopolitical risks add uncertainty. Better to wait for breakout confirmation.",
            "recommended_strategy": "hold",
            "suggested_positions": {
                "SPY": 0.20,
                "QQQ": 0.20,
                "cash": 0.60
            },
            "confidence": 0.65,
            "risk_assessment": "High. Elevated volatility and unclear direction. Preserve capital."
        }
    }
]


# V2 Few-shot examples (增强版)
COORDINATOR_EXAMPLES_V2 = [
    {
        "input": """
**Market Data:** SPY +12%, VIX: 15 → 12, ADX: 35, RSI: 65
**Technical:** Strong uptrend, Price > MA50 > MA200
**Sentiment:** Bullish (Fear & Greed: 75)
**News:** Tech earnings beat, Fed dovish
""",
        "output": {
            "regime_classification": {
                "primary_regime": "BULL_TREND",
                "confidence": 0.85,
                "alternative_regime": None,
                "regime_duration_estimate": "2-4 weeks"
            },
            "detailed_reasoning": {
                "step1_event_check": "No major events. VIX at 12 (very low).",
                "step2_trend": "ADX=35 (strong), Price > MA50 > MA200 → Clear BULL_TREND",
                "step3_volatility": "VIX 12, realized vol 14% → LOW_VOL",
                "step4_edge_cases": "No conflicts. All indicators agree.",
                "step5_anomalies": [],
                "step6_confidence_factors": ["Strong trend: +0.3", "High ADX: +0.2", "Low VIX: +0.2"]
            },
            "strategy_recommendation": {
                "primary_strategy": "momentum",
                "rationale": "Strong trend with low volatility",
                "position_sizing": {"aggressive": True, "suggested_exposure": 0.75}
            },
            "risk_assessment": {
                "overall_risk": "Medium",
                "key_risks": ["RSI approaching 70"],
                "risk_mitigation": ["Set stop at MA(50)"]
            }
        }
    },
    {
        "input": """
**Market Data:** SPY -2%, VIX: 18 → 32, ADX: 18
**Technical:** Choppy, no clear trend
**Sentiment:** Fear rising
**News:** Fed hawkish, tariff threats
""",
        "output": {
            "regime_classification": {
                "primary_regime": "EVENT_DRIVEN",
                "confidence": 0.80,
                "alternative_regime": "HIGH_VOL_RANGE",
                "regime_duration_estimate": "1-2 weeks"
            },
            "detailed_reasoning": {
                "step1_event_check": "VIX spike 18→32 (+78%) → EVENT_DRIVEN",
                "step2_trend": "ADX=18 (weak), no clear trend",
                "step3_volatility": "VIX 32 (elevated)",
                "step4_edge_cases": "Uncertain direction",
                "step5_anomalies": ["VIX spike >50%"],
                "step6_confidence_factors": ["Event trigger: +0.4", "Vol spike: +0.3", "Direction unclear: -0.1"]
            },
            "strategy_recommendation": {
                "primary_strategy": "hold",
                "rationale": "Event-driven volatility",
                "position_sizing": {"aggressive": False, "suggested_exposure": 0.40}
            },
            "risk_assessment": {
                "overall_risk": "Very High",
                "key_risks": ["Policy uncertainty"],
                "risk_mitigation": ["60% cash", "Tight stops at 3%"]
            }
        }
    },
    {
        "input": """
**Market Data:** SPY +0.5%, VIX: 12, ADX: 15, RSI: 52
**Technical:** Sideways consolidation, tight range 500-505
**Sentiment:** Neutral (Fear & Greed: 50)
**News:** No major events, low volatility environment
""",
        "output": {
            "regime_classification": {
                "primary_regime": "LOW_VOL_RANGE",
                "confidence": 0.78,
                "alternative_regime": None,
                "regime_duration_estimate": "1-2 weeks"
            },
            "detailed_reasoning": {
                "step1_event_check": "No events. VIX at 12 (low).",
                "step2_trend": "ADX=15 (very weak), no clear trend → ranging",
                "step3_volatility": "VIX 12, realized vol 10% → LOW_VOL",
                "step4_edge_cases": "Tight consolidation pattern detected",
                "step5_anomalies": [],
                "step6_confidence_factors": ["Low ADX: +0.3", "Tight range: +0.3", "Low vol: +0.2"]
            },
            "strategy_recommendation": {
                "primary_strategy": "grid_trading",
                "rationale": "Perfect conditions for grid trading: low volatility, clear range, weak trend",
                "position_sizing": {"aggressive": False, "suggested_exposure": 0.60}
            },
            "risk_assessment": {
                "overall_risk": "Low",
                "key_risks": ["Potential breakout from range"],
                "risk_mitigation": ["Stop loss at range boundaries", "40% cash buffer"]
            }
        }
    }
]


# Technical Agent Prompt (增强版 for V2)
TECHNICAL_AGENT_PROMPT_V2 = """Analyze the following technical indicators and provide QUANTITATIVE assessment:

**Price Data:**
{price_data}

**Indicators:**
{indicators}

**REQUIRED CALCULATIONS:**
1. Trend Direction: Compare Price vs MA(50) vs MA(200)
2. Trend Strength: Calculate ADX value
3. Momentum: RSI value and interpretation
4. Volatility: Realized volatility (%) and ATR as % of price
5. Support/Resistance: Recent swing highs/lows

**Output as JSON:**
{{
  "trend": {{
    "direction": "bullish/bearish/neutral",
    "strength": 0.72,
    "adx": 32,
    "ma_alignment": "Price > MA50 > MA200"
  }},
  "momentum": {{
    "status": "positive/negative/neutral",
    "rsi": 64,
    "interpretation": "Healthy momentum, not overbought"
  }},
  "volatility": {{
    "regime": "low/medium/high",
    "realized_vol_annual": 18.5,
    "atr_pct": 2.3,
    "expanding": false
  }},
  "support_resistance": {{
    "support_levels": [475, 465],
    "resistance_levels": [495, 505],
    "current_price": 482
  }},
  "key_signals": [
    "ADX > 25 confirms strong trend",
    "RSI in healthy range (50-80)"
  ]
}}

Be precise with numbers. Avoid vague terms.
"""


# Technical Agent Prompt (原版)
TECHNICAL_AGENT_PROMPT = """Analyze the following technical indicators and provide a summary:

**Price Data:**
{price_data}

**Indicators:**
{indicators}

Provide analysis covering:
1. Trend direction and strength
2. Support and resistance levels
3. Momentum indicators
4. Volatility assessment
5. Key technical signals

Output as JSON:
{
  "trend": "bullish/bearish/neutral",
  "trend_strength": 0-1,
  "support_levels": [price1, price2],
  "resistance_levels": [price1, price2],
  "momentum": "positive/negative/neutral",
  "volatility": "low/medium/high",
  "signals": ["signal1", "signal2"]
}
"""

# Sentiment Agent Prompt (增强版 for V2)
SENTIMENT_AGENT_PROMPT_V2 = """Analyze market sentiment with QUANTITATIVE metrics:

**VIX Data:**
Current: {vix_current}
5-day change: {vix_change_5d}%
Historical context: {vix_percentile}th percentile

**Other Indicators (if available):**
- Put/Call Ratio: {put_call_ratio}
- Fear & Greed Index: {fear_greed}

**News Headlines:**
{news_headlines}

**Output as JSON:**
{{
  "overall_sentiment": "bullish/bearish/neutral",
  "sentiment_score": 0.55,
  "vix_analysis": {{
    "current": 19.6,
    "interpretation": "Normal levels, no panic",
    "trend": "rising_slowly/falling/stable/spiking",
    "concern_level": "low/medium/high"
  }},
  "fear_greed_components": {{
    "market_momentum": "positive",
    "stock_breadth": "neutral",
    "safe_haven_demand": "low"
  }},
  "risk_level": "medium",
  "key_themes": ["AI optimism", "Inflation concerns"],
  "contrarian_signals": ["VIX rising despite market gains"]
}}

Quantify everything. Flag divergences.
"""


# Sentiment Agent Prompt (原版)
SENTIMENT_AGENT_PROMPT = """Analyze market sentiment from the following sources:

**News Headlines:**
{news_headlines}

**Social Media (if available):**
{social_media}

**Market Indicators:**
- VIX: {vix}
- Put/Call Ratio: {put_call_ratio}
- Fear & Greed Index: {fear_greed}

Provide sentiment analysis as JSON:
{
  "overall_sentiment": "bullish/bearish/neutral",
  "sentiment_score": 0-1,  // 0=extreme fear, 1=extreme greed
  "key_themes": ["theme1", "theme2"],
  "risk_level": "low/medium/high",
  "summary": "Brief summary..."
}
"""


# News Agent Prompt (增强版 for V2)
NEWS_AGENT_PROMPT_V2 = """Analyze news for ACTIONABLE trading implications:

**News Headlines:**
{news_headlines}

**Recent Events:**
{events}

**Market Reaction:**
{market_reaction}

**Output as JSON:**
{{
  "major_events": [
    {{
      "event": "PCE Inflation hits 4.0%",
      "impact": "negative",
      "severity": "medium",
      "affected_assets": ["bonds", "growth_stocks"],
      "market_reaction": "SPY -0.5% on announcement"
    }}
  ],
  "sector_analysis": {{
    "semiconductors": {{
      "sentiment": "very_negative",
      "driver": "Tariff concerns",
      "outlook": "Avoid until clarity"
    }},
    "tech_mega_caps": {{
      "sentiment": "positive",
      "driver": "Strong earnings",
      "outlook": "Selectively bullish"
    }}
  }},
  "risk_factors": [
    "High inflation may force Fed hawkish",
    "Geopolitical tensions"
  ],
  "catalysts_ahead": [
    "Next Fed meeting",
    "Tech earnings conclusion"
  ],
  "trading_implications": "Mixed signals..."
}}

Focus on DIVERGENCES and ANOMALIES.
"""


# News Agent Prompt (原版)
NEWS_AGENT_PROMPT = """Analyze the following news and events for trading implications:

**News Headlines:**
{news_headlines}

**Recent Events:**
{events}

Identify:
1. Market-moving events
2. Sector impacts
3. Risk factors
4. Trading opportunities

Output as JSON:
{
  "major_events": ["event1", "event2"],
  "sector_impacts": {
    "tech": "positive/negative/neutral",
    "finance": "positive/negative/neutral"
  },
  "risk_factors": ["risk1", "risk2"],
  "trading_implications": "Summary of implications..."
}
"""
