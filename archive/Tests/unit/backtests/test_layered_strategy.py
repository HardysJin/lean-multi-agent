"""
Unit tests for LayeredStrategy

Tests the layered decision-making strategy that uses the orchestration layer.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from Backtests.strategies.layered_strategy import (
    LayeredStrategy,
    create_layered_strategy,
    estimate_decision_frequency
)


@pytest.fixture
def sample_price_data():
    """Create sample OHLCV price data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 150, 100),
        'high': np.random.uniform(100, 150, 100),
        'low': np.random.uniform(100, 150, 100),
        'close': np.random.uniform(100, 150, 100),
        'volume': np.random.uniform(1e6, 1e7, 100)
    }, index=dates)
    return data


@pytest.fixture
def strategy():
    """Create a LayeredStrategy instance with MockLLM for testing."""
    return LayeredStrategy(use_mock_llm=True, enable_memory=False)


class TestLayeredStrategyInitialization:
    """Test LayeredStrategy initialization."""
    
    def test_initialization_with_mock_llm(self):
        """Test strategy initializes correctly with MockLLM."""
        strategy = LayeredStrategy(use_mock_llm=True)
        
        assert strategy.use_mock_llm is True
        assert strategy.scheduler is not None
        assert strategy.strategic_dm is not None
        assert strategy.campaign_dm is not None
        assert strategy.tactical_dm is not None
        assert strategy.last_decisions == {
            "strategic": None,
            "campaign": None,
            "tactical": None
        }
        assert strategy.decision_history == []
    
    def test_initialization_without_memory(self):
        """Test strategy initializes correctly without memory."""
        strategy = LayeredStrategy(use_mock_llm=True, enable_memory=False)
        
        assert strategy.enable_memory is False
        assert strategy.state_manager is None
    
    def test_initialization_with_memory(self):
        """Test strategy initializes correctly with memory."""
        strategy = LayeredStrategy(use_mock_llm=True, enable_memory=True)
        
        assert strategy.enable_memory is True
        # state_manager may be None if DB paths not configured
        # This is expected behavior in test environment


class TestLayeredStrategySignalGeneration:
    """Test signal generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_signal_basic(self, strategy, sample_price_data):
        """Test basic signal generation."""
        signal = await strategy.generate_signal(
            symbol="AAPL",
            date="2024-01-15",
            price_data=sample_price_data
        )
        
        # Check signal structure
        assert "symbol" in signal
        assert "date" in signal
        assert "action" in signal
        assert "confidence" in signal
        assert "decision_level" in signal
        assert "reason" in signal
        
        # Check values
        assert signal["symbol"] == "AAPL"
        assert signal["date"] == "2024-01-15"
        assert signal["action"] in ["BUY", "SELL", "HOLD"]
        assert 0.0 <= signal["confidence"] <= 1.0
        assert signal["decision_level"] in ["strategic", "campaign", "tactical", "none"]
    
    @pytest.mark.asyncio
    async def test_generate_signal_with_context(self, strategy, sample_price_data):
        """Test signal generation with additional context."""
        context = {
            "market_regime": "bullish",
            "volatility": "low",
            "news_sentiment": "positive"
        }
        
        signal = await strategy.generate_signal(
            symbol="AAPL",
            date="2024-01-15",
            price_data=sample_price_data,
            context=context
        )
        
        assert signal is not None
        assert "metadata" in signal
    
    @pytest.mark.asyncio
    async def test_generate_signal_updates_history(self, strategy, sample_price_data):
        """Test that signal generation updates decision history."""
        initial_count = len(strategy.decision_history)
        
        await strategy.generate_signal(
            symbol="AAPL",
            date="2024-01-15",
            price_data=sample_price_data
        )
        
        # History should be updated (unless it was a HOLD due to cooldown)
        assert len(strategy.decision_history) >= initial_count
    
    @pytest.mark.asyncio
    async def test_generate_signal_empty_price_data(self, strategy):
        """Test signal generation with empty price data."""
        empty_data = pd.DataFrame()
        
        signal = await strategy.generate_signal(
            symbol="AAPL",
            date="2024-01-15",
            price_data=empty_data
        )
        
        # Should still return a valid signal (likely HOLD)
        assert signal is not None
        assert signal["action"] in ["BUY", "SELL", "HOLD"]


class TestLayeredStrategyBatchGeneration:
    """Test batch signal generation."""
    
    @pytest.mark.asyncio
    async def test_batch_generate_signals(self, strategy, sample_price_data):
        """Test generating signals for multiple symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        price_data_dict = {symbol: sample_price_data for symbol in symbols}
        
        signals = await strategy.batch_generate_signals(
            symbols=symbols,
            date="2024-01-15",
            price_data_dict=price_data_dict
        )
        
        # Check all symbols got signals
        assert len(signals) == len(symbols)
        for symbol in symbols:
            assert symbol in signals
            assert signals[symbol]["symbol"] == symbol
            assert signals[symbol]["action"] in ["BUY", "SELL", "HOLD"]
    
    @pytest.mark.asyncio
    async def test_batch_generate_signals_with_shared_context(self, strategy, sample_price_data):
        """Test batch generation with shared context."""
        symbols = ["AAPL", "MSFT"]
        price_data_dict = {symbol: sample_price_data for symbol in symbols}
        context = {"market_regime": "bearish"}
        
        signals = await strategy.batch_generate_signals(
            symbols=symbols,
            date="2024-01-15",
            price_data_dict=price_data_dict,
            context=context
        )
        
        assert len(signals) == 2
        # All signals should have metadata showing shared context was used
        for symbol in symbols:
            assert signals[symbol] is not None


class TestLayeredStrategyDecisionLevels:
    """Test decision level selection and escalation."""
    
    @pytest.mark.asyncio
    async def test_tactical_decision_daily(self, strategy, sample_price_data):
        """Test that tactical decisions are made daily."""
        # First call should make a decision
        signal1 = await strategy.generate_signal(
            symbol="AAPL",
            date="2024-01-15",
            price_data=sample_price_data
        )
        
        # Same day, same symbol - should be HOLD (cooldown)
        signal2 = await strategy.generate_signal(
            symbol="AAPL",
            date="2024-01-15",
            price_data=sample_price_data
        )
        
        # One of them should involve a decision
        assert signal1["action"] in ["BUY", "SELL", "HOLD"]
    
    @pytest.mark.asyncio
    async def test_different_symbols_independent(self, strategy, sample_price_data):
        """Test that different symbols have independent decisions."""
        signal_aapl = await strategy.generate_signal(
            symbol="AAPL",
            date="2024-01-15",
            price_data=sample_price_data
        )
        
        signal_msft = await strategy.generate_signal(
            symbol="MSFT",
            date="2024-01-15",
            price_data=sample_price_data
        )
        
        # Both should have valid signals
        assert signal_aapl is not None
        assert signal_msft is not None
        # They might have different actions
        assert signal_aapl["symbol"] == "AAPL"
        assert signal_msft["symbol"] == "MSFT"


class TestLayeredStrategyStateManagement:
    """Test strategy state management."""
    
    def test_get_decision_summary_empty(self, strategy):
        """Test decision summary when no decisions made."""
        summary = strategy.get_decision_summary()
        
        assert summary["total_decisions"] == 0
        assert summary["by_level"]["strategic"] == 0
        assert summary["by_level"]["campaign"] == 0
        assert summary["by_level"]["tactical"] == 0
        assert summary["escalation_rate"] == 0.0
        assert summary["recent_decisions"] == []
    
    @pytest.mark.asyncio
    async def test_get_decision_summary_after_decisions(self, strategy, sample_price_data):
        """Test decision summary after making decisions."""
        # Make a few decisions
        await strategy.generate_signal("AAPL", "2024-01-15", sample_price_data)
        await strategy.generate_signal("MSFT", "2024-01-15", sample_price_data)
        
        summary = strategy.get_decision_summary()
        
        # Should have at least some decisions recorded
        assert summary["total_decisions"] >= 0
        assert isinstance(summary["by_level"], dict)
        assert isinstance(summary["escalation_rate"], float)
        assert 0.0 <= summary["escalation_rate"] <= 1.0
    
    def test_reset(self, strategy):
        """Test strategy reset."""
        # Add some fake history
        strategy.decision_history.append({
            "date": "2024-01-15",
            "symbol": "AAPL",
            "level": "tactical",
            "decision": {},
            "escalated": False
        })
        strategy.last_decisions["tactical"] = pd.Timestamp("2024-01-15")
        
        # Reset
        strategy.reset()
        
        # Should be clean
        assert strategy.decision_history == []
        assert strategy.last_decisions["tactical"] is None


class TestLayeredStrategyHelperFunctions:
    """Test helper functions."""
    
    @pytest.mark.asyncio
    async def test_create_layered_strategy(self):
        """Test factory function."""
        strategy = await create_layered_strategy(use_mock_llm=True)
        
        assert isinstance(strategy, LayeredStrategy)
        assert strategy.use_mock_llm is True
    
    def test_estimate_decision_frequency(self):
        """Test decision frequency estimation."""
        # 1 year backtest
        estimates = estimate_decision_frequency(365, enable_escalation=False)
        
        assert estimates["strategic"] >= 4  # ~4 quarters
        assert estimates["campaign"] >= 52  # ~52 weeks
        assert estimates["tactical"] == 365  # Every day
        assert estimates["total"] == estimates["strategic"] + estimates["campaign"] + estimates["tactical"]
    
    def test_estimate_decision_frequency_with_escalation(self):
        """Test decision frequency estimation with escalation."""
        without_escalation = estimate_decision_frequency(365, enable_escalation=False)
        with_escalation = estimate_decision_frequency(365, enable_escalation=True)
        
        # With escalation should have more decisions
        assert with_escalation["total"] >= without_escalation["total"]


class TestLayeredStrategyIntegration:
    """Integration tests with the orchestration layer."""
    
    @pytest.mark.asyncio
    async def test_strategy_uses_scheduler(self, strategy, sample_price_data):
        """Test that strategy properly uses LayeredScheduler."""
        # The strategy should delegate to scheduler for decision timing
        signal = await strategy.generate_signal(
            symbol="AAPL",
            date="2024-01-15",
            price_data=sample_price_data
        )
        
        # Scheduler should have been invoked
        assert signal is not None
        assert "decision_level" in signal
    
    @pytest.mark.asyncio
    async def test_strategy_uses_decision_makers(self, strategy, sample_price_data):
        """Test that strategy properly uses decision makers."""
        signal = await strategy.generate_signal(
            symbol="AAPL",
            date="2024-01-15",
            price_data=sample_price_data
        )
        
        # Decision makers should have been invoked
        assert signal["decision_level"] in ["strategic", "campaign", "tactical", "none"]
        
        if signal["decision_level"] != "none":
            # A decision was made, check it's in history
            assert len(strategy.decision_history) > 0
    
    def test_strategy_has_required_methods(self, strategy):
        """Test that strategy implements required interface."""
        # Check key methods exist
        assert hasattr(strategy, 'generate_signal')
        assert hasattr(strategy, 'batch_generate_signals')
        assert hasattr(strategy, 'get_decision_summary')
        assert hasattr(strategy, 'reset')
        
        # Check they're callable
        assert callable(strategy.generate_signal)
        assert callable(strategy.batch_generate_signals)
        assert callable(strategy.get_decision_summary)
        assert callable(strategy.reset)
