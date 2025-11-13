"""LLM Prompts for Periodly Decision System"""

from typing import List, Optional


# 策略描述映射
STRATEGY_DESCRIPTIONS = {
    "grid_trading": "For ranging/sideways markets with clear support/resistance",
    "momentum": "For strong trending markets",
    "mean_reversion": "For overextended moves likely to reverse",
    "double_ema_channel": "For trending markets with breakout confirmation (dual EMA channels with volume)",
    "buy_and_hold": "Baseline strategy for long-term bullish outlook",
    "hold": "When uncertain or high risk"
}


def get_coordinator_system_prompt(available_strategies: Optional[List[str]] = None) -> str:
    """
    生成Coordinator System Prompt
    
    Args:
        available_strategies: 可用策略列表，如果为None则使用默认列表
    
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


# Technical Agent Prompt
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

# Sentiment Agent Prompt
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

# News Agent Prompt
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
