"""
LayeredStrategy - Intelligent Trading Strategy using Layered Decision-Making

This strategy demonstrates the full capabilities of the refactored orchestration layer:
1. Uses LayeredScheduler for intelligent decision level selection
2. Supports escalation mechanisms (REGIME_CHANGE, SECTOR_ROTATION, VOLATILITY_SPIKE, etc.)
3. Coordinates Strategic, Campaign, and Tactical decision makers
4. Automatically adjusts decision frequency based on market conditions

Decision Levels:
- Strategic: Monthly (30 days) - Portfolio allocation, risk management, long-term themes
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
        strategic_dm (StrategicDecisionMaker): Monthly portfolio decisions
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
        technical_agent = TechnicalAnalysisAgent()
        news_agent = NewsAgent(llm_client=llm_client)
        
        # Initialize meta agent
        meta_agent = MetaAgent(llm_client=llm_client, state_manager=self.state_manager)
        
        # Store agents for later use
        self.meta_agent = meta_agent
        self.technical_agent = technical_agent
        self.news_agent = news_agent
        self.macro_agent = macro_agent
        self.sector_agent = sector_agent
        
        # Note: Agent connection will be done lazily on first use
        # to avoid event loop issues during __init__
        self._agents_connected = False
        
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
        
        # Store latest contexts from upper decision levels for inheritance
        self.latest_strategic_decision = None
        self.latest_campaign_decision = None
        
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
        # Lazy connect agents on first use
        if not self._agents_connected:
            await self._connect_agents(self.meta_agent, self.technical_agent, self.news_agent)
            self._agents_connected = True
        
        context = context or {}
        current_date = pd.Timestamp(date)
        
        # Check which decision levels should run
        should_strategic = self.scheduler.should_run_strategic(current_date)
        should_campaign = self.scheduler.should_run_campaign(current_date)
        should_tactical = self.scheduler.should_run_tactical(current_date)
        
        # Execute decisions in order: Strategic -> Campaign -> Tactical
        # Higher levels set context for lower levels
        
        # No decision at all? (shouldn't happen with daily tactical)
        if not (should_strategic or should_campaign or should_tactical):
            # No decision needed, return HOLD
            return self._create_hold_signal(
                symbol=symbol,
                date=date,
                reason="No decision needed at this time (within cooldown period)",
                metadata={"scheduler_state": "cooldown"}
            )
        
        # Determine which level to execute
        # Note: We only execute ONE decision per day, but choose the highest priority one
        decision_level = None
        if should_strategic:
            decision_level = "strategic"
        elif should_campaign:
            decision_level = "campaign"
        elif should_tactical:
            decision_level = "tactical"
        
        # Execute decision at appropriate level
        logger.info(f"Making {decision_level} decision for {symbol} on {date}")
        
        # Prepare context for decision makers
        visible_data_end = current_date if context.get('backtest_mode') else None
        
        # Call appropriate decision maker
        try:
            if decision_level == "strategic":
                decision = await self.strategic_dm.decide(
                    symbol=symbol,
                    visible_data_end=visible_data_end
                )
                self.scheduler.state.last_strategic_time = current_date
                # Store strategic decision for inheritance
                self.latest_strategic_decision = decision
                logger.info(f"Strategic decision stored: constraints={decision.constraints}")
                
            elif decision_level == "campaign":
                decision = await self.campaign_dm.decide(
                    symbol=symbol,
                    visible_data_end=visible_data_end
                )
                self.scheduler.state.last_campaign_time = current_date
                # Store campaign decision for inheritance
                self.latest_campaign_decision = decision
                logger.info(f"Campaign decision stored: constraints={decision.constraints}")
                
            else:  # tactical
                # Extract inherited contexts from upper-level decisions
                inherited_constraints = None
                inherited_macro_context = None
                inherited_sector_context = None
                
                # Inherit from most recent strategic decision
                if self.latest_strategic_decision:
                    inherited_constraints = self.latest_strategic_decision.constraints
                    if self.latest_strategic_decision.macro_context:
                        inherited_macro_context = self.latest_strategic_decision.macro_context.to_dict()
                    if self.latest_strategic_decision.sector_context:
                        inherited_sector_context = self.latest_strategic_decision.sector_context.to_dict()
                    logger.info(f"Tactical inheriting from Strategic: constraints={inherited_constraints is not None}, "
                              f"macro={inherited_macro_context is not None}, sector={inherited_sector_context is not None}")
                
                # Override with campaign decision if available (more recent)
                if self.latest_campaign_decision:
                    if self.latest_campaign_decision.constraints:
                        inherited_constraints = self.latest_campaign_decision.constraints
                    if self.latest_campaign_decision.macro_context:
                        inherited_macro_context = self.latest_campaign_decision.macro_context.to_dict()
                    if self.latest_campaign_decision.sector_context:
                        inherited_sector_context = self.latest_campaign_decision.sector_context.to_dict()
                    logger.info(f"Tactical also inheriting from Campaign (overriding Strategic)")
                
                # Build portfolio state from context (if available)
                portfolio_state = self._build_portfolio_state(symbol, context)
                
                # Tactical decision maker uses current_date as decision time
                decision = await self.tactical_dm.decide(
                    symbol=symbol,
                    price_data=price_data,
                    portfolio_state=portfolio_state,
                    inherited_constraints=inherited_constraints,
                    inherited_macro_context=inherited_macro_context,
                    inherited_sector_context=inherited_sector_context,
                    current_time=current_date  # Pass backtest date as decision time
                )
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
        decision,  # Decision object from decision makers
        symbol: str,
        date: str,
        decision_level: str,
        escalated: bool
    ) -> Dict[str, Any]:
        """Convert decision maker output to trading signal format."""
        
        # Extract action from Decision object
        action = decision.action.upper() if hasattr(decision, 'action') else "HOLD"
        conviction = decision.conviction if hasattr(decision, 'conviction') else 5.0
        reasoning = decision.reasoning if hasattr(decision, 'reasoning') else "No reasoning provided"
        
        # DEBUG: Log the decision
        logger.info(f"🔍 Converting decision: action={action}, conviction={conviction}, level={decision_level}")
        
        # Convert conviction (1-10) to confidence (0.0-1.0)
        confidence = conviction / 10.0
        
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
            "reason": reasoning,
            "metadata": {
                "raw_decision": {
                    "action": action,
                    "conviction": conviction,
                    "reasoning": reasoning,
                    "timestamp": decision.timestamp.isoformat() if hasattr(decision, 'timestamp') and decision.timestamp else None
                },
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
    
    def _build_portfolio_state(
        self,
        symbol: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        构建投资组合状态信息
        
        Args:
            symbol: 当前分析的股票代码
            context: 可选的上下文（可能包含portfolio信息）
            
        Returns:
            Portfolio state字典，包含：
            - current_holdings: 当前持仓 {symbol: {quantity, value, avg_cost}}
            - available_cash: 可用现金
            - total_portfolio_value: 总资产
            - position_for_symbol: 当前symbol的持仓（如果有）
            - position_concentration: 各仓位占比
        """
        if not context or 'portfolio' not in context:
            # 如果没有提供portfolio信息，返回默认状态
            return {
                'current_holdings': {},
                'available_cash': 100000.0,  # 默认10万现金
                'total_portfolio_value': 100000.0,
                'position_for_symbol': None,
                'position_concentration': {},
                'note': 'Default portfolio state (no data provided)'
            }
        
        portfolio = context['portfolio']
        
        # 提取关键信息
        holdings = portfolio.get('holdings', {})
        cash = portfolio.get('cash', 0.0)
        total_value = portfolio.get('total_value', cash)
        
        # 计算各仓位占比
        concentration = {}
        if total_value > 0:
            for sym, position in holdings.items():
                position_value = position.get('value', 0.0)
                concentration[sym] = position_value / total_value
        
        # 获取当前symbol的持仓
        symbol_position = holdings.get(symbol)
        
        return {
            'current_holdings': holdings,
            'available_cash': cash,
            'total_portfolio_value': total_value,
            'position_for_symbol': symbol_position,
            'position_concentration': concentration,
            'holdings_count': len(holdings),
            'cash_ratio': cash / total_value if total_value > 0 else 1.0
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
    
    async def _connect_agents(self, meta_agent, technical_agent, news_agent):
        """Connect specialist agents to meta agent (async helper)"""
        await meta_agent.connect_to_agent(
            agent_name="technical",
            agent_instance=technical_agent,
            description="Technical analysis specialist - provides indicators, signals, and chart patterns"
        )
        await meta_agent.connect_to_agent(
            agent_name="news",
            agent_instance=news_agent,
            description="News sentiment specialist - analyzes news articles and market sentiment"
        )
        logger.info("✓ Connected TechnicalAgent and NewsAgent to MetaAgent")


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
    strategic = backtest_days // 30  # Monthly
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
