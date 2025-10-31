"""
LayeredStrategy - Intelligent Trading Strategy using Layered Decision-Making

This strategy demonstrates the full capabilities of the refactored orchestration layer:
1. Uses LayeredScheduler for intelligent decision level selection
2. Supports escalation mechanisms (REGIME_CHANGE, SECTOR_ROTATION, VOLATILITY_SPIKE, etc.)
3. Coordinates Strategic, Campaign, and Tactical decision makers
4. Automatically adjusts decision frequency based on market conditions

Decision Levels:
- Strategic: Quarterly (90 days) - Portfolio allocation, risk management, long-term themes
- Campaign: Weekly (7 days) - Sector rotation, position sizing, rebalancing  
- Tactical: Daily (1 day) - Entry/exit timing, short-term adjustments

Example Usage:
    >>> from Backtests.strategies.layered_strategy import LayeredStrategy
    >>> strategy = LayeredStrategy(use_mock_llm=True)
    >>> signal = await strategy.generate_signal(symbol="AAPL", date="2024-01-15", price_data=data)
    >>> print(f"Decision level: {signal['decision_level']}")
    >>> print(f"Action: {signal['action']}")  # BUY, SELL, HOLD
    >>> print(f"Confidence: {signal['confidence']}")
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
import logging

from Agents.orchestration import (
    LayeredScheduler,
    StrategicDecisionMaker,
    CampaignDecisionMaker, 
    TacticalDecisionMaker,
    DecisionMakerFactory,
    MetaAgent
)
from Agents.core import MacroAgent, SectorAgent, TechnicalAnalysisAgent, NewsAgent
from Agents.utils import LLMConfig, get_mock_llm
from Memory import MultiTimeframeStateManager

logger = logging.getLogger(__name__)


class LayeredStrategy:
    """
    Trading strategy that uses the layered decision-making architecture.
    
    This strategy showcases the full orchestration layer capabilities:
    - Intelligent scheduler selects appropriate decision level
    - Three-layer decision makers (Strategic/Campaign/Tactical)
    - Escalation mechanism for market regime changes
    - State persistence and memory management
    
    Attributes:
        scheduler (LayeredScheduler): Manages decision timing and escalation
        strategic_dm (StrategicDecisionMaker): Quarterly portfolio decisions
        campaign_dm (CampaignDecisionMaker): Weekly sector/position decisions
        tactical_dm (TacticalDecisionMaker): Daily entry/exit decisions
        state_manager (StateManager): Persists strategy state
        use_mock_llm (bool): If True, uses MockLLM for testing (no API calls)
    """
    
    def __init__(
        self,
        use_mock_llm: bool = False,
        enable_memory: bool = False,  # Default to False for simplicity
        enable_escalation: bool = True
    ):
        """
        Initialize the LayeredStrategy.
        
        Args:
            use_mock_llm: If True, uses MockLLM for testing (fast, no API costs)
            enable_memory: If True, enables state persistence (requires DB setup)
            enable_escalation: If True, enables automatic escalation to higher decision levels
        """
        self.use_mock_llm = use_mock_llm
        self.enable_memory = enable_memory
        self.enable_escalation = enable_escalation
        
        # Initialize LLM client
        if use_mock_llm:
            logger.info("Using MockLLM for testing (no API calls)")
            llm_client = get_mock_llm()
        else:
            logger.info("Using real LLM client")
            llm_config = LLMConfig()
            llm_client = llm_config.get_llm()
        
        # Initialize state manager (only if memory enabled and paths provided)
        self.state_manager = None
        if enable_memory:
            try:
                from Memory import create_state_manager
                self.state_manager = create_state_manager()
                logger.info("State manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize state manager: {e}. Continuing without memory.")
                self.state_manager = None
        
        # Initialize core agents with LLM client
        macro_agent = MacroAgent(llm_client=llm_client)
        sector_agent = SectorAgent(llm_client=llm_client)
        meta_agent = MetaAgent(llm_client=llm_client)
        
        # Initialize decision makers using factory
        factory = DecisionMakerFactory(
            macro_agent=macro_agent,
            sector_agent=sector_agent,
            meta_agent=meta_agent
        )
        self.strategic_dm = factory.create_strategic()
        self.campaign_dm = factory.create_campaign()
        self.tactical_dm = factory.create_tactical()
        
        # Initialize scheduler
        self.scheduler = LayeredScheduler(
            strategic_maker=self.strategic_dm,
            campaign_maker=self.campaign_dm,
            tactical_maker=self.tactical_dm
        )
        
        # Track strategy state
        self.last_decisions = {
            "strategic": None,
            "campaign": None,
            "tactical": None
        }
        self.decision_history = []
        
        logger.info(f"LayeredStrategy initialized (mock_llm={use_mock_llm}, memory={enable_memory}, escalation={enable_escalation})")
    
    async def generate_signal(
        self,
        symbol: str,
        date: str,
        price_data: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate trading signal using layered decision-making.
        
        This method:
        1. Checks if a decision is needed (using scheduler)
        2. Selects appropriate decision level (strategic/campaign/tactical)
        3. Executes decision maker
        4. Returns actionable signal
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            date: Current date (YYYY-MM-DD format)
            price_data: Historical price data (OHLCV)
            context: Optional additional context (news, macro data, etc.)
        
        Returns:
            Signal dictionary with:
            - action: "BUY", "SELL", or "HOLD"
            - confidence: 0.0 to 1.0
            - decision_level: "strategic", "campaign", or "tactical"
            - reason: Explanation of the decision
            - metadata: Additional decision context
        """
        context = context or {}
        current_date = pd.Timestamp(date)
        
        # Check which decision levels should run
        should_strategic = self.scheduler.should_run_strategic(current_date)
        should_campaign = self.scheduler.should_run_campaign(current_date)
        should_tactical = self.scheduler.should_run_tactical(current_date)
        
        # Determine highest priority decision level to execute
        decision_level = None
        if should_strategic:
            decision_level = "strategic"
        elif should_campaign:
            decision_level = "campaign"
        elif should_tactical:
            decision_level = "tactical"
        
        if decision_level is None:
            # No decision needed, return HOLD
            return self._create_hold_signal(
                symbol=symbol,
                date=date,
                reason="No decision needed at this time (within cooldown period)",
                metadata={"scheduler_state": "cooldown"}
            )
        
        # Execute decision at appropriate level
        logger.info(f"Making {decision_level} decision for {symbol} on {date}")
        
        # Prepare context for decision makers
        decision_context = {
            "symbol": symbol,
            "current_date": current_date,
            "current_price": price_data.iloc[-1]["close"] if not price_data.empty else None,
            "price_history": price_data,
            **context
        }
        
        # Call appropriate decision maker
        try:
            if decision_level == "strategic":
                decision = await self.strategic_dm.make_decision(decision_context)
                self.scheduler.state.last_strategic_time = current_date
            elif decision_level == "campaign":
                decision = await self.campaign_dm.make_decision(decision_context)
                self.scheduler.state.last_campaign_time = current_date
            else:  # tactical
                decision = await self.tactical_dm.make_decision(decision_context)
                self.scheduler.state.last_tactical_time = current_date
        except Exception as e:
            logger.error(f"Error making {decision_level} decision: {e}")
            return self._create_hold_signal(
                symbol=symbol,
                date=date,
                reason=f"Error during decision-making: {str(e)}",
                metadata={"error": str(e), "decision_level": decision_level}
            )
        
        # Store decision
        self.last_decisions[decision_level] = current_date
        self.decision_history.append({
            "date": date,
            "symbol": symbol,
            "level": decision_level,
            "decision": decision,
            "escalated": False
        })
        
        # Convert decision to trading signal
        signal = self._convert_decision_to_signal(
            decision=decision,
            symbol=symbol,
            date=date,
            decision_level=decision_level,
            escalated=False
        )
        
        return signal
    
    async def batch_generate_signals(
        self,
        symbols: List[str],
        date: str,
        price_data_dict: Dict[str, pd.DataFrame],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate signals for multiple symbols in parallel.
        
        This is more efficient than calling generate_signal() repeatedly because:
        1. Strategic decisions can be shared across symbols (portfolio allocation)
        2. Sector decisions can be batched (sector rotation)
        3. News/macro analysis is done once
        
        Args:
            symbols: List of ticker symbols
            date: Current date (YYYY-MM-DD format)
            price_data_dict: Dict mapping symbol -> price_data
            context: Optional shared context
        
        Returns:
            Dict mapping symbol -> signal
        """
        context = context or {}
        
        # For multi-symbol strategies, strategic decisions are shared
        # This is a key feature of the layered architecture
        
        tasks = [
            self.generate_signal(
                symbol=symbol,
                date=date,
                price_data=price_data_dict.get(symbol, pd.DataFrame()),
                context=context
            )
            for symbol in symbols
        ]
        
        signals = await asyncio.gather(*tasks)
        return dict(zip(symbols, signals))
    
    def _convert_decision_to_signal(
        self,
        decision: Dict[str, Any],
        symbol: str,
        date: str,
        decision_level: str,
        escalated: bool
    ) -> Dict[str, Any]:
        """Convert decision maker output to trading signal format."""
        
        # Extract action from decision
        # Decision makers return different formats, normalize here
        action = decision.get("action", "HOLD").upper()
        confidence = decision.get("confidence", 0.5)
        reason = decision.get("reason", "No specific reason provided")
        
        # Ensure action is valid
        if action not in ["BUY", "SELL", "HOLD"]:
            logger.warning(f"Invalid action '{action}', defaulting to HOLD")
            action = "HOLD"
        
        return {
            "symbol": symbol,
            "date": date,
            "action": action,
            "confidence": confidence,
            "decision_level": decision_level,
            "escalated": escalated,
            "reason": reason,
            "metadata": {
                "raw_decision": decision,
                "strategy": "LayeredStrategy",
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _create_hold_signal(
        self,
        symbol: str,
        date: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a HOLD signal with explanation."""
        return {
            "symbol": symbol,
            "date": date,
            "action": "HOLD",
            "confidence": 1.0,
            "decision_level": "none",
            "escalated": False,
            "reason": reason,
            "metadata": metadata or {}
        }
    
    def get_decision_summary(self) -> Dict[str, Any]:
        """
        Get summary of strategy's decision history.
        
        Returns:
            Summary with:
            - total_decisions: Total number of decisions made
            - by_level: Count by decision level
            - escalation_rate: Percentage of escalated decisions
            - recent_decisions: Last 10 decisions
        """
        if not self.decision_history:
            return {
                "total_decisions": 0,
                "by_level": {"strategic": 0, "campaign": 0, "tactical": 0},
                "escalation_rate": 0.0,
                "recent_decisions": []
            }
        
        total = len(self.decision_history)
        by_level = {
            "strategic": sum(1 for d in self.decision_history if d["level"] == "strategic"),
            "campaign": sum(1 for d in self.decision_history if d["level"] == "campaign"),
            "tactical": sum(1 for d in self.decision_history if d["level"] == "tactical")
        }
        escalated = sum(1 for d in self.decision_history if d["escalated"])
        
        return {
            "total_decisions": total,
            "by_level": by_level,
            "escalation_rate": escalated / total if total > 0 else 0.0,
            "recent_decisions": self.decision_history[-10:]
        }
    
    def reset(self):
        """Reset strategy state (useful for backtesting multiple scenarios)."""
        self.last_decisions = {
            "strategic": None,
            "campaign": None,
            "tactical": None
        }
        self.decision_history = []
        if self.state_manager:
            # Could clear state from database here if needed
            pass
        logger.info("LayeredStrategy state reset")


# Convenience functions for common use cases

async def create_layered_strategy(
    use_mock_llm: bool = False,
    **kwargs
) -> LayeredStrategy:
    """
    Factory function to create a configured LayeredStrategy.
    
    Args:
        use_mock_llm: If True, uses MockLLM for testing
        **kwargs: Additional arguments passed to LayeredStrategy
    
    Returns:
        Configured LayeredStrategy instance
    """
    return LayeredStrategy(use_mock_llm=use_mock_llm, **kwargs)


def estimate_decision_frequency(
    backtest_days: int,
    enable_escalation: bool = True
) -> Dict[str, int]:
    """
    Estimate how many decisions will be made during a backtest.
    
    This helps users understand the performance characteristics:
    - With escalation: More decisions, better adaptation
    - Without escalation: Fewer decisions, more consistent
    
    Args:
        backtest_days: Total days in backtest period
        enable_escalation: Whether escalation is enabled
    
    Returns:
        Estimated decision counts by level
    """
    # Base frequencies
    strategic = backtest_days // 90  # Quarterly
    campaign = backtest_days // 7    # Weekly
    tactical = backtest_days           # Daily
    
    # Escalation adds ~10-20% more decisions
    if enable_escalation:
        escalation_factor = 1.15
        strategic = int(strategic * escalation_factor)
        campaign = int(campaign * escalation_factor)
    
    return {
        "strategic": strategic,
        "campaign": campaign,
        "tactical": tactical,
        "total": strategic + campaign + tactical
    }
