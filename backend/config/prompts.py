"""LLM Prompts for Periodly Decision System"""

from typing import List, Optional, Dict


# 策略描述映射
STRATEGY_DESCRIPTIONS = {
    "grid_trading": "For ranging/sideways markets with clear support/resistance",
    "momentum": "For strong trending markets",
    "mean_reversion": "For overextended moves likely to reverse",
    "double_ema_channel": "For strong trending markets with breakout confirmation (dual EMA channels with volume)",
    "buy_and_hold": "Baseline strategy for long-term bullish outlook (buy once and hold, no rebalancing)",
    "hold": "When uncertain or high risk - maintain current position",
    # 新增：独立仓位控制策略
    "buy": "Buy/add to position - supports percentage-based position sizing (e.g., buy to 70% position)",
    "sell": "Sell/reduce position - supports percentage-based position sizing (e.g., sell to 30% position, or 0% for full exit)"
}

# Regime定义和量化标准 (V2增强版)
REGIME_DEFINITIONS = {
    "BULL_TREND": {
        "description": "Strong sustained uptrend - ride the trend",
        "criteria": [
            "Price > MA(50) > MA(200)",
            "ADX > 15 (trend present) OR strong price momentum (+5% or more)",
            "RSI between 50-80",
            "Price gain > 5% over analysis period (PRIMARY indicator)",
            "Higher highs and higher lows pattern"
        ],
        "recommended_strategies": ["momentum", "double_ema_channel"],
        "risk_level": "medium"
    },
    "BEAR_TREND": {
        "description": "Strong sustained downtrend - defensive positioning",
        "criteria": [
            "Price < MA(50) < MA(200)",
            "ADX > 15 (trend present) OR strong price decline (-5% or more)",
            "RSI between 30-50",
            "Price loss > 5% over analysis period (PRIMARY indicator)",
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
        "description": "Tight consolidation - good for range strategies (USE SPARINGLY!)",
        "criteria": [
            "ADX < 15 (very weak trend) - NOTE: < 15, not < 20!",
            "Price change < 3% over analysis period (KEY: small price move)",
            "Realized volatility < 15% annualized",
            "VIX < 20",
            "Price contained in narrow range"
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
        "description": "External shock dominates - TWO scenarios based on panic level",
        "criteria": [
            "VIX spike > 30",
            "Gap move > 3% on major news",
            "Major macro event (Fed decision, earnings, geopolitical)",
            "Correlation breakdown (normal relationships disrupted)"
        ],
        "sub_regimes": {
            "PANIC_EXTREME": {
                "description": "🔥 EXTREME PANIC - Contrarian opportunity window",
                "criteria": [
                    "VIX > 35 (extreme fear, historically unsustainable)",
                    "Price decline > 15% in < 30 days (panic selling)",
                    "Quality assets oversold (e.g., TQQQ/QQQ -20%+ while fundamentals intact)",
                    "Fear indicators at multi-year highs",
                    "Indiscriminate selling (even strong stocks hit)"
                ],
                "panic_duration_framework": {
                    "SHORT_PANIC": {
                        "duration": "3-7天快速反弹",
                        "characteristics": [
                            "VIX > 40但无系统性风险（如个股暴跌、技术性回调）",
                            "无明确催化剂（纯情绪恐慌）",
                            "历史类似：闪崩、个股事件、技术性调整",
                            "基本面完好，仅恐慌性抛售"
                        ],
                        "entry_timing": "立即分批建仓（首批30-40%）",
                        "strategy": "激进抄底，快速反弹后获利了结"
                    },
                    "MEDIUM_PANIC": {
                        "duration": "2-4周震荡筑底",
                        "characteristics": [
                            "VIX 35-45，有明确催化剂（如加息、地缘风险、财报季）",
                            "市场需要消化负面信息",
                            "历史类似：政策收紧、中等经济数据恶化",
                            "基本面受影响但非致命"
                        ],
                        "entry_timing": "分批建仓（每周加仓20-30%，观察VIX回落）",
                        "strategy": "耐心等待VIX < 30确认底部，波段操作"
                    },
                    "LONG_PANIC": {
                        "duration": "1-3个月系统性调整",
                        "characteristics": [
                            "VIX持续 > 35，伴随经济衰退担忧或系统性风险",
                            "基本面恶化（如衰退、金融危机、持续通胀）",
                            "历史类似：2008金融危机、2020疫情初期、2022熊市",
                            "多轮恐慌抛售，反弹后再创新低"
                        ],
                        "entry_timing": "极度谨慎，等待VIX < 30并确认企稳信号（如连续3周不创新低）",
                        "strategy": "只在VIX峰值（>50）小仓位试探，主要观望"
                    }
                },
                "recommended_strategies": ["buy_dip_gradually", "momentum_reversal", "buy"],
                "reasoning": "When VIX > 35, fear is extreme and mean-reversion likely. History shows these panic levels don't last. Position for bounce.",
                "risk_level": "very_high_but_asymmetric"
            },
            "NORMAL_SHOCK": {
                "description": "⚠️ NORMAL EVENT SHOCK - Stay defensive",
                "criteria": [
                    "VIX 25-35 (elevated but not extreme)",
                    "Moderate decline or high uncertainty",
                    "Event outcome unclear"
                ],
                "recommended_strategies": ["hold", "wait_and_see", "sell"],
                "risk_level": "very_high"
            }
        },
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
        available_strategies = ["buy", "sell", "hold", "momentum", "mean_reversion", "double_ema_channel", "buy_and_hold", "grid_trading"]
    
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

Step 1: BEAR MARKET EARLY DETECTION (CRITICAL - Act before panic sets in!)
- **1A. BEAR_TREND INITIATION - Defensive positioning IMMEDIATELY:**
  * **Trigger conditions (ANY TWO triggers → High cash position):**
    1. Recent sharp decline: Last 30 days loss > -8%
    2. Breakdown below key support: Price < MA(50) AND MA(50) turning down
    3. Negative momentum acceleration: Last 7 days worse than last 30 days
    4. VIX rising trend: VIX > 22 AND rising (not yet panic, but fear building)
    5. Breadth deterioration: Multiple quality stocks declining together (>70% of watchlist red)
  * **Action if 2+ triggers met:**
    - Immediately reduce to 30-40% equity (high cash 60-70%)
    - **DO NOT wait for VIX > 35 to act!** Act early when VIX 20-25
    - Better to be early than late in bear market protection
  * **Reasoning:** Bear markets develop gradually, then accelerate. Protect capital early.

Step 2: EXTREME PANIC CHECK - Contrarian opportunity (After bear decline established)
- **2A. Check for PANIC_EXTREME (Only after significant decline has occurred):**
  * **Prerequisites (ALL must be met):**
    1. Already in drawdown: Price down > -15% from recent high (within 60 days)
    2. VIX spiked to extreme: VIX > 35 (fear at unsustainable levels)
    3. Capitulation signals: Massive volume, wide intraday swings (ATR > 4%)
    4. Quality assets oversold: TQQQ/QQQ/SPY down 20-30%+ but fundamentals not collapsed
  
  * **CRITICAL: Classify PANIC TYPE (determines entry timing):**
    
    **PANIC_TYPE_1: CAPITULATION SPIKE (1-2 weeks bounce) - Aggressive contrarian:**
    - VIX > 45 (extreme panic peak, historically unsustainable)
    - Price waterfall drop (single day -5%+ or -15%+ in 5 days)
    - Indiscriminate selling (even safe-haven assets dumped)
    - NO new systemic catalyst (just amplified existing fears)
    - **Key question:** Is this a NEW crisis or overreaction to known risks?
    - Historical analog: Oct-2023 Israel war spike, Dec-2018 Powell pivot drop
    - **Action:** Deploy 40-50% immediately, add to 70-80% within 1 week as VIX drops
    - **Target:** VIX 45 → 25 typically happens in 3-10 days, capture bounce
    
    **PANIC_TYPE_2: ROLLING PANIC (3-6 weeks) - Gradual scale-in:**
    - VIX 35-45 sustained (not a single spike, but elevated fear)
    - Multiple down legs (drops, bounces, retests lows)
    - Clear catalyst still unfolding (Fed tightening cycle, earnings recession, etc.)
    - Market digesting bad news over weeks
    - Historical analog: Mar-Apr 2025 tariff panic, Oct-Nov 2022 inflation fears
    - **Action:** Deploy 20-25% initially, add 15-20% each week for 3-4 weeks
    - **Target:** Scale in as VIX declines from 40 → 30 → 25 over 4-6 weeks
    
    **PANIC_TYPE_3: SYSTEMIC CRISIS (2-4 months) - Extreme caution:**
    - VIX > 35 persistently (elevated for 8+ weeks)
    - Systemic risk: Recession confirmed, financial system stress, prolonged policy error
    - Multiple sectors broken (tech, financials, consumer all declining)
    - Fundamentals deteriorating (unemployment rising, credit spreads widening)
    - Historical analog: 2008-09 financial crisis, Mar-May 2020 pandemic
    - **Action:** Stay 80-90% cash, only nibble 5-10% at VIX > 50 peaks
    - **Target:** Wait for VIX sustained < 25 for 4+ weeks AND economic stabilization
  
  * **Decision Framework for Panic Classification:**
    1. **Check VIX trajectory:** Single spike (Type 1) vs sustained elevation (Type 2/3)?
    2. **Assess catalyst:** Over-reaction to known risk (Type 1) vs new unfolding crisis (Type 2/3)?
    3. **Examine fundamentals:** Intact (Type 1), impaired (Type 2), or collapsing (Type 3)?
    4. **Historical comparison:** Which past episode does this resemble most closely?
    5. **Duration elapsed:** How long has panic lasted? Fresh spike (Type 1) or prolonged (Type 2/3)?
  
- **2B. Check for NORMAL VOLATILITY EVENT (Cautious, not contrarian):**
  * VIX 25-35 (elevated but not extreme)
  * Price decline modest (-5% to -15% over 30 days)
  * Could be early bear OR temporary correction (unclear)
  * If YES → regime = EVENT_DRIVEN (sub: NORMAL_SHOCK), recommend "hold" (60% cash)

- **2C. No significant event:**
  * VIX < 25 AND price change moderate → proceed to Step 3 (normal regime analysis)

Step 3: MULTI-TIMEFRAME MOMENTUM ANALYSIS (CRITICAL - Detect trend changes early!)
- **Last 7 Days (Weight: 40%) - MOST SENSITIVE for early detection:**
  * Last 7d loss > -5% → **RED FLAG: Possible bear initiation**
  * Last 7d gain > +5% → Strong recent momentum (possible reversal from panic)
  * Acceleration/deceleration: Is trend speeding up or slowing down?
  
- **Last 30 Days (Weight: 40%) - Primary trend classifier:**
  * Last 30d gain > +8% AND Price > MA(20) → Strong BULL_TREND
  * Last 30d gain +5% to +8% AND Price > MA(20) → Moderate BULL_TREND
  * Last 30d loss -5% to -8% AND Price < MA(20) → Early BEAR_TREND (defensive!)
  * Last 30d loss > -8% AND Price < MA(20) → Strong BEAR_TREND (high cash!)
  * Last 30d change < ±3% → Consolidation (possible ranging)
  
- **Last 60 Days (Weight: 15%) - Confirmation:**
  * Last 60d gain > +10% → Medium-term bull
  * Last 60d loss > -10% → Medium-term bear
  * Check for trend change: 60d bull but 30d bear? → **Reversal in progress!**
  
- **Full Period 150d (Weight: 5%) - Context only:**
  * Only used to identify major cycle turns
  * Example: 150d +15% but last 30d -10% → **Late-stage bull, bear starting**

- **MA Structural Analysis (Critical for bear detection):**
  * **Death Cross Warning:** MA(50) crossing below MA(200) → Bear market signal
  * Price < MA(20) < MA(50) → Clear downtrend structure
  * Price < MA(50) BUT above MA(200) → Intermediate correction (not yet full bear)
  * MA(20) slope: Turning down rapidly? → Momentum deteriorating

- **ADX Confirmation (relaxed for early bear detection):**
  * ADX > 25 = Strong trend (bull or bear, check price direction!)
  * ADX 15-25 = Moderate trend (accept if price action clear)
  * ADX < 15 = No trend (only if price truly flat < ±2%)
  * **Key:** ADX rising + Price falling = Strengthening bear trend!

Step 4: VOLATILITY REGIME
- Calculate: Realized vol, VIX level, ATR
- Is volatility elevated (VIX > 25 or RV > 25%)?
- Classify as HIGH_VOL_RANGE or LOW_VOL_RANGE
- **Important:** Don't confuse low ADX with LOW_VOL_RANGE if price is moving strongly!

Step 5: EDGE CASE DETECTION
- Are signals conflicting? (e.g., RSI bullish but price < MA)
- Is trend_strength between 0.4-0.6? (weak/uncertain)
- If YES → regime = TRANSITIONING

Step 6: ANOMALY DETECTION
- Check for sector divergences (any ticker >20% divergence from market?)
- Check for VIX spikes (>20% change in 5 days?)
- Flag these as risk factors

Step 7: CONFIDENCE CALIBRATION
- Clear signals (all indicators agree) → confidence 0.8-0.9
- Mixed signals (some conflict) → confidence 0.5-0.7  
- High uncertainty → confidence <0.5, recommend "hold"

**STRATEGY SELECTION RULES (Enhanced for bear/panic detection):**

**PRIMARY RULES (Check in order):**

1. **BEAR_TREND (Early detection priority!)** → hold (60-80% cash defensive)
   * **Triggers:** ANY TWO of:
     - Last 30d loss > -8% 
     - Price < MA(50) with MA(50) turning down
     - VIX rising above 22
     - Breadth deterioration (most stocks declining)
   * **Action:** Immediately reduce to 30-40% equity
   * **Reasoning:** Better early defense than late panic selling!
   * **Do NOT wait for VIX > 35 to act defensively!**

2. **PANIC_EXTREME (Contrarian opportunity - After decline established)** → buy_dip strategies
   * **Prerequisites:** Already down > -15% from recent high + VIX > 35
   * **Strategy based on panic classification:**
   
   **PANIC_TYPE_1 (Capitulation Spike)** → momentum or buy_and_hold (AGGRESSIVE)
   - VIX > 45, waterfall drop, indiscriminate selling
   - Deploy 40-50% immediately, scale to 70-80% as VIX drops
   - Stop loss: -8% (tight, expect quick bounce)
   - Historical: Dec-2018, Oct-2023 war spike
   
   **PANIC_TYPE_2 (Rolling Panic)** → buy_and_hold or momentum (GRADUAL)
   - VIX 35-45 sustained, multiple down legs
   - Week 1: Deploy 20-25%, then add 15-20% weekly for 3-4 weeks
   - Stop loss: -12% (expect retest of lows)
   - Historical: Mar-Apr 2025 tariffs, 2022 rate hikes
   
   **PANIC_TYPE_3 (Systemic Crisis)** → hold primary (EXTREME CAUTION)
   - VIX > 35 for 8+ weeks, systemic risk present
   - Stay 80-90% cash, only nibble 5-10% at VIX > 50 peaks
   - Stop loss: -15% (prolonged bear possible)
   - Historical: 2008-09, Mar-May 2020, 2022 full year

3. **BULL_TREND** → momentum, buy_and_hold, or double_ema_channel
   * Last 30d gain > +5% AND Price > MA(20)
   * ADX > 15 or strong momentum
   * Prefer momentum if RSI 50-70 and accelerating
   * Prefer buy_and_hold if slow steady rise
   * Target: 70-80% equity exposure

4. **LOW_VOL_RANGE (Use SPARINGLY!)** → grid_trading or mean_reversion
   * **Strict criteria (ALL must be met):**
     - Price change < 3% over last 30 days
     - ADX < 15 (truly weak trend)
     - VIX < 20 (low volatility)
     - No bear signals present
   * Prefer mean_reversion if at range extremes
   * Target: 50-60% equity exposure

5. **HIGH_VOL_RANGE** → hold (defensive)
   * VIX > 25, Realized vol > 25%
   * Chaotic price action without clear direction
   * Target: 30-40% equity exposure

6. **TRANSITIONING** → hold or mean_reversion
   * Mixed signals, MA crossovers in progress
   * Wait for regime clarity
   * Target: 40-50% equity exposure

7. **EVENT_DRIVEN (Normal Shock)** → hold
   * VIX 25-35, elevated uncertainty
   * Target: 40% equity exposure

**CRITICAL DECISION RULES (Prioritize capital preservation!):**

1. **BEAR DEFENSE FIRST:** Detect bear trends EARLY (don't wait for -20% drawdown!)
   - Act when VIX 20-25 + price declining (before panic sets in)
   - Last 30d loss > -8% = immediate defensive posture
   - Better to be early defensive than late panicked

2. **PANIC RECOGNITION:** After decline established, identify capitulation opportunities
   - VIX > 45 + indiscriminate selling = likely capitulation (contrarian entry)
   - VIX 35-45 sustained = rolling panic (gradual scale-in)
   - VIX > 35 for 8+ weeks = systemic (extreme caution)

3. **DON'T OVER-USE GRID_TRADING:** It should be RARE (< 10% of decisions)
   - Only use when: Price flat < ±3%, ADX < 15, VIX < 20, no bear signals
   - Most markets are trending or transitioning, not ranging!

4. **RECENT DATA PRIORITY:** Last 7-30 days matter most for 14-day forecast
   - 150-day history is context only, not primary driver
   - If last 30d shows clear trend, trust it (even if 150d differs)

5. **ASYMMETRIC RISK/REWARD:**
   - In bull: Can afford to be patient (miss 5% gains not fatal)
   - In bear: Must act fast (missing -20% decline is devastating)
   - In panic: Aggressive entry justified (VIX > 45 mean-reverts fast)


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
    "sub_regime": "PANIC_EXTREME/NORMAL_SHOCK/null",  // Only if primary_regime = EVENT_DRIVEN
    "confidence": 0.85,
    "alternative_regime": "BULL_TREND",
    "regime_duration_estimate": "1-2 weeks / 2-4 weeks / 4+ weeks",
    "panic_analysis": {{{{  // Only if sub_regime = PANIC_EXTREME - CRITICAL for timing
      "panic_type": "SHORT_PANIC/MEDIUM_PANIC/LONG_PANIC",
      "vix_level": 45.2,
      "price_decline_30d": -22.5,
      "systemic_risk": false,  // Whether systemic risk present (recession, financial crisis)
      "catalyst": "Technical correction/Policy tightening/Geopolitical/Systemic crisis",
      "fundamental_health": "Healthy/Impaired/Severely damaged",
      "historical_analog": "Dec-2018 flash crash/2022 rate hike cycle/2008 financial crisis",
      "entry_timing": "Immediate/Gradual tranches/Extreme caution",
      "initial_position_size": "30-40%/20-25%/5-10%",
      "contrarian_confidence": 0.85,  // Confidence in dip-buying (0-1)
      "reasoning": "Detailed explanation for SHORT/MEDIUM/LONG panic classification and trade rationale"
    }}}}
  }}}},
  
  "detailed_reasoning": {{{{
    "step1_bear_detection": "Bear market early warning signals...",
    "step2_panic_check": "Extreme panic / capitulation analysis...",
    "step3_multi_timeframe_trend": "7d/30d/60d/150d momentum analysis...",
    "step4_volatility": "Volatility regime classification...",
    "step5_edge_cases": "Conflicting signals detected...",
    "step6_anomalies": ["Sector divergence", "VIX spike"],
    "step7_confidence_factors": ["+0.3: Clear trend", "-0.1: Mixed signals"]
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

# V2 User Prompt (增强版 - 多时间尺度)
COORDINATOR_USER_PROMPT_TEMPLATE_V2 = """
**Analysis Period:** {analysis_start_date} to {analysis_end_date} ({lookback_days} days total)
**Forecast Period:** {forecast_start_date} to {forecast_end_date} ({forecast_days} days)

**⚠️ IMPORTANT - Multi-Timeframe Analysis:**
Focus on RECENT data (last 30 days) as it's most relevant for {forecast_days}-day forecast!
Long-term data (150 days) is for context only, not primary decision driver.

**Market Data - Multi-Timeframe View:**

📊 **Recent Trend (Last 30 Days) - WEIGHT: 70%** [MOST CRITICAL]
{market_data_recent}

� **Last 7 Days Daily Prices** [Fine-grained momentum analysis]
{recent_prices_7d}

�📈 **Medium-Term (Last 60 Days) - WEIGHT: 20%**
{market_data_medium}

📉 **Full Period ({lookback_days} Days) - WEIGHT: 10%** [Context only]
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
Using the 7-step decision process outlined in your system prompt, classify the current market regime and provide your recommendation in the specified JSON format.

**Critical Steps (Follow in order):**
1. **BEAR DETECTION FIRST:** Check for early bear market signals (Last 30d < -8%? VIX rising? MA breakdown?)
2. **PANIC CHECK:** If already declining > -15%, check for capitulation opportunity (VIX > 35?)
3. **MULTI-TIMEFRAME TREND:** Analyze 7d/30d/60d/150d momentum (weight recent data 80%!)
4. **VOLATILITY REGIME:** Measure current volatility environment
5. **EDGE CASES:** Detect conflicting signals or transitioning regimes
6. **ANOMALIES:** Flag sector divergences, VIX spikes, unusual patterns
7. **CONFIDENCE:** Calibrate confidence based on signal clarity

**Key Decision Points:**
- If Step 1 triggers (bear signals) → Defensive positioning (60-80% cash) immediately
- If Step 2 triggers (extreme panic) → Classify panic type, prepare contrarian entry
- Else → Normal regime analysis (bull/range/transitioning)

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
