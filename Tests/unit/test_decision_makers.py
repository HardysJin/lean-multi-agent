"""
测试 DecisionMaker 层
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from Agents.decision_makers import (
    StrategicDecisionMaker,
    CampaignDecisionMaker,
    TacticalDecisionMaker,
    DecisionMakerFactory,
    Decision
)
from Agents.core import MacroAgent, MacroContext
from Agents.sector_agent import SectorAgent, SectorContext
from Agents.meta_agent import MetaAgent, MetaDecision


# ===== Fixtures =====

@pytest.fixture
def mock_macro_agent():
    """Mock MacroAgent"""
    agent = AsyncMock(spec=MacroAgent)
    
    # 模拟宏观分析结果
    agent.analyze_macro_environment.return_value = MacroContext(
        market_regime='bull',
        regime_confidence=0.8,
        interest_rate_trend='stable',
        current_rate=5.0,
        risk_level=3.0,
        volatility_level='medium',
        gdp_trend='expanding',
        inflation_level='moderate',
        market_sentiment='greed',
        vix_level=15.0,
        constraints={'allow_long': True, 'allow_short': False, 'max_position_size': 0.3},
        analysis_timestamp=datetime.now(),
        data_end_time=None,
        confidence_score=0.85,
        reasoning='Bull market confirmed.'
    )
    
    return agent


@pytest.fixture
def mock_sector_agent():
    """Mock SectorAgent"""
    agent = AsyncMock(spec=SectorAgent)
    
    # 模拟行业分析结果
    agent.analyze_sector.return_value = SectorContext(
        sector='Technology',
        trend='bullish',
        relative_strength=0.7,
        momentum='accelerating',
        sector_rotation_signal='rotating_in',
        avg_pe_ratio=25.0,
        avg_growth_rate=15.0,
        sentiment='bullish',
        confidence=0.85,
        recommendation='overweight',
        reasoning='Tech sector showing strength.',
        analysis_timestamp=datetime.now(),
        data_end_time=None
    )
    
    # 模拟符号到行业的映射
    agent.get_sector_for_symbol.return_value = 'Technology'
    
    return agent


@pytest.fixture
def mock_meta_agent():
    """Mock MetaAgent"""
    agent = AsyncMock(spec=MetaAgent)
    
    # 模拟个股决策结果
    agent.analyze_and_decide.return_value = MetaDecision(
        symbol='AAPL',
        action='BUY',
        conviction=8,
        reasoning='Strong fundamentals with bullish macro and sector backdrop.',
        evidence={'macro': 'bull', 'sector': 'bullish'},
        tool_calls=[],
        timestamp=datetime.now()
    )
    
    return agent


# ===== Test StrategicDecisionMaker =====

@pytest.mark.asyncio
async def test_strategic_decide_basic(mock_macro_agent, mock_sector_agent, mock_meta_agent):
    """测试 StrategicDecisionMaker 基本决策流程"""
    maker = StrategicDecisionMaker(
        macro_agent=mock_macro_agent,
        sector_agent=mock_sector_agent,
        meta_agent=mock_meta_agent
    )
    
    decision = await maker.decide(symbol='AAPL')
    
    # 验证调用顺序
    mock_macro_agent.analyze_macro_environment.assert_called_once()
    mock_sector_agent.get_sector_for_symbol.assert_called_once_with('AAPL')
    mock_sector_agent.analyze_sector.assert_called_once()
    mock_meta_agent.analyze_and_decide.assert_called_once()
    
    # 验证决策结果
    assert decision.symbol == 'AAPL'
    assert decision.action == 'BUY'
    assert decision.conviction == 8.0
    assert decision.decision_level == 'strategic'
    assert decision.macro_context is not None
    assert decision.sector_context is not None
    assert decision.constraints is not None


@pytest.mark.asyncio
async def test_strategic_context_propagation(mock_macro_agent, mock_sector_agent, mock_meta_agent):
    """测试上下文传播到 MetaAgent"""
    maker = StrategicDecisionMaker(
        macro_agent=mock_macro_agent,
        sector_agent=mock_sector_agent,
        meta_agent=mock_meta_agent
    )
    
    await maker.decide(symbol='AAPL')
    
    # 检查 MetaAgent 是否收到了正确的上下文
    call_args = mock_meta_agent.analyze_and_decide.call_args
    assert call_args.kwargs['symbol'] == 'AAPL'
    assert call_args.kwargs['macro_context'] is not None
    assert call_args.kwargs['sector_context'] is not None
    assert call_args.kwargs['constraints'] is not None


@pytest.mark.asyncio
async def test_strategic_with_visible_data_end(mock_macro_agent, mock_sector_agent, mock_meta_agent):
    """测试回测模式（visible_data_end）"""
    maker = StrategicDecisionMaker(
        macro_agent=mock_macro_agent,
        sector_agent=mock_sector_agent,
        meta_agent=mock_meta_agent
    )
    
    end_time = datetime(2024, 1, 1)
    await maker.decide(symbol='AAPL', visible_data_end=end_time)
    
    # 验证时间参数传递
    macro_call = mock_macro_agent.analyze_macro_environment.call_args
    assert macro_call.kwargs['visible_data_end'] == end_time
    
    sector_call = mock_sector_agent.analyze_sector.call_args
    assert sector_call.kwargs['visible_data_end'] == end_time


# ===== Test CampaignDecisionMaker =====

@pytest.mark.asyncio
async def test_campaign_decide_basic(mock_macro_agent, mock_sector_agent, mock_meta_agent):
    """测试 CampaignDecisionMaker 基本决策"""
    maker = CampaignDecisionMaker(
        macro_agent=mock_macro_agent,
        sector_agent=mock_sector_agent,
        meta_agent=mock_meta_agent
    )
    
    decision = await maker.decide(symbol='AAPL')
    
    # 验证调用
    mock_macro_agent.analyze_macro_environment.assert_called_once()
    mock_sector_agent.analyze_sector.assert_called_once()
    mock_meta_agent.analyze_and_decide.assert_called_once()
    
    # 验证决策结果
    assert decision.symbol == 'AAPL'
    assert decision.action == 'BUY'
    assert decision.decision_level == 'campaign'


@pytest.mark.asyncio
async def test_campaign_constraint_inheritance(mock_macro_agent, mock_sector_agent, mock_meta_agent):
    """测试约束继承和合并"""
    maker = CampaignDecisionMaker(
        macro_agent=mock_macro_agent,
        sector_agent=mock_sector_agent,
        meta_agent=mock_meta_agent
    )
    
    inherited_constraints = {
        'allow_long': True,
        'allow_short': False,
        'max_leverage': 1.5
    }
    
    await maker.decide(symbol='AAPL', inherited_constraints=inherited_constraints)
    
    # 检查约束合并
    call_args = mock_meta_agent.analyze_and_decide.call_args
    constraints = call_args.kwargs['constraints']
    
    # 应该包含继承的约束
    assert constraints['allow_long'] is True
    assert constraints['allow_short'] is False
    assert constraints['max_leverage'] == 1.5
    
    # 应该包含宏观的约束
    assert 'max_position_size' in constraints


@pytest.mark.asyncio
async def test_campaign_no_inherited_constraints(mock_macro_agent, mock_sector_agent, mock_meta_agent):
    """测试没有继承约束的情况"""
    maker = CampaignDecisionMaker(
        macro_agent=mock_macro_agent,
        sector_agent=mock_sector_agent,
        meta_agent=mock_meta_agent
    )
    
    decision = await maker.decide(symbol='AAPL')
    
    # 应该只使用宏观约束
    assert decision.constraints is not None
    assert 'allow_long' in decision.constraints


# ===== Test TacticalDecisionMaker =====

@pytest.mark.asyncio
async def test_tactical_decide_basic(mock_meta_agent):
    """测试 TacticalDecisionMaker 基本决策"""
    maker = TacticalDecisionMaker(meta_agent=mock_meta_agent)
    
    constraints = {'allow_long': True, 'allow_short': False}
    decision = await maker.decide(
        symbol='AAPL',
        inherited_constraints=constraints
    )
    
    # 验证只调用 MetaAgent
    mock_meta_agent.analyze_and_decide.assert_called_once()
    
    # 验证决策结果
    assert decision.symbol == 'AAPL'
    assert decision.action == 'BUY'
    assert decision.decision_level == 'tactical'
    assert decision.macro_context is None  # 不重新分析宏观
    assert decision.sector_context is None  # 不重新分析行业


@pytest.mark.asyncio
async def test_tactical_with_inherited_contexts(mock_meta_agent):
    """测试继承上下文"""
    maker = TacticalDecisionMaker(meta_agent=mock_meta_agent)
    
    macro_ctx = {'market_regime': 'bull'}
    sector_ctx = {'trend': 'bullish'}
    constraints = {'allow_long': True}
    
    await maker.decide(
        symbol='AAPL',
        inherited_constraints=constraints,
        inherited_macro_context=macro_ctx,
        inherited_sector_context=sector_ctx
    )
    
    # 检查上下文传递
    call_args = mock_meta_agent.analyze_and_decide.call_args
    assert call_args.kwargs['macro_context'] == macro_ctx
    assert call_args.kwargs['sector_context'] == sector_ctx
    assert call_args.kwargs['constraints'] == constraints


@pytest.mark.asyncio
async def test_tactical_minimal_call(mock_meta_agent):
    """测试最小调用（无继承上下文）"""
    maker = TacticalDecisionMaker(meta_agent=mock_meta_agent)
    
    decision = await maker.decide(symbol='AAPL')
    
    # 仍然能正常工作
    assert decision.symbol == 'AAPL'
    assert decision.action == 'BUY'
    
    # 检查 MetaAgent 调用
    call_args = mock_meta_agent.analyze_and_decide.call_args
    assert call_args.kwargs['macro_context'] is None
    assert call_args.kwargs['sector_context'] is None
    assert call_args.kwargs['constraints'] is None


# ===== Test DecisionMakerFactory =====

def test_factory_create_strategic():
    """测试工厂创建 StrategicDecisionMaker"""
    factory = DecisionMakerFactory()
    maker = factory.create_strategic()
    
    assert isinstance(maker, StrategicDecisionMaker)
    assert maker.macro_agent is not None
    assert maker.sector_agent is not None
    assert maker.meta_agent is not None


def test_factory_create_campaign():
    """测试工厂创建 CampaignDecisionMaker"""
    factory = DecisionMakerFactory()
    maker = factory.create_campaign()
    
    assert isinstance(maker, CampaignDecisionMaker)
    assert maker.macro_agent is not None
    assert maker.sector_agent is not None
    assert maker.meta_agent is not None


def test_factory_create_tactical():
    """测试工厂创建 TacticalDecisionMaker"""
    factory = DecisionMakerFactory()
    maker = factory.create_tactical()
    
    assert isinstance(maker, TacticalDecisionMaker)
    assert maker.meta_agent is not None


def test_factory_with_custom_agents(mock_macro_agent, mock_sector_agent, mock_meta_agent):
    """测试使用自定义Agent的工厂"""
    factory = DecisionMakerFactory(
        macro_agent=mock_macro_agent,
        sector_agent=mock_sector_agent,
        meta_agent=mock_meta_agent
    )
    
    maker = factory.create_strategic()
    
    # 应该使用提供的实例
    assert maker.macro_agent is mock_macro_agent
    assert maker.sector_agent is mock_sector_agent
    assert maker.meta_agent is mock_meta_agent


def test_factory_shares_agents():
    """测试工厂创建的多个Maker共享Agent实例"""
    factory = DecisionMakerFactory()
    
    strategic = factory.create_strategic()
    campaign = factory.create_campaign()
    tactical = factory.create_tactical()
    
    # 应该共享同一个 meta_agent 实例
    assert strategic.meta_agent is campaign.meta_agent
    assert campaign.meta_agent is tactical.meta_agent


# ===== Test Decision Dataclass =====

def test_decision_creation():
    """测试 Decision 对象创建"""
    decision = Decision(
        symbol='AAPL',
        action='BUY',
        conviction=8.0,
        reasoning='Test',
        decision_level='strategic'
    )
    
    assert decision.symbol == 'AAPL'
    assert decision.action == 'BUY'
    assert decision.conviction == 8.0
    assert decision.reasoning == 'Test'
    assert decision.decision_level == 'strategic'
    assert decision.timestamp is not None  # auto-generated


def test_decision_with_contexts():
    """测试带上下文的 Decision"""
    macro_ctx = MacroContext(
        market_regime='bull',
        regime_confidence=0.8,
        interest_rate_trend='stable',
        current_rate=5.0,
        risk_level=3.0,
        volatility_level='medium',
        gdp_trend='expanding',
        inflation_level='moderate',
        market_sentiment='greed',
        vix_level=15.0,
        constraints={},
        analysis_timestamp=datetime.now(),
        data_end_time=None,
        confidence_score=0.85,
        reasoning='Bull market confirmed.'
    )
    
    decision = Decision(
        symbol='AAPL',
        action='BUY',
        conviction=8.0,
        reasoning='Test',
        macro_context=macro_ctx,
        decision_level='strategic'
    )
    
    assert decision.macro_context is not None
    assert decision.macro_context.market_regime == 'bull'


# ===== Integration Tests =====

@pytest.mark.asyncio
async def test_three_level_hierarchy(mock_macro_agent, mock_sector_agent, mock_meta_agent):
    """测试三层决策层级的协调工作"""
    factory = DecisionMakerFactory(
        macro_agent=mock_macro_agent,
        sector_agent=mock_sector_agent,
        meta_agent=mock_meta_agent
    )
    
    # 战略层决策
    strategic_maker = factory.create_strategic()
    strategic_decision = await strategic_maker.decide(symbol='AAPL')
    
    # 战役层决策（继承战略层约束）
    campaign_maker = factory.create_campaign()
    campaign_decision = await campaign_maker.decide(
        symbol='AAPL',
        inherited_constraints=strategic_decision.constraints
    )
    
    # 战术层决策（继承战役层约束和上下文）
    tactical_maker = factory.create_tactical()
    tactical_decision = await tactical_maker.decide(
        symbol='AAPL',
        inherited_constraints=campaign_decision.constraints,
        inherited_macro_context=strategic_decision.macro_context.to_dict() if strategic_decision.macro_context else None,
        inherited_sector_context=strategic_decision.sector_context.to_dict() if strategic_decision.sector_context else None
    )
    
    # 验证所有决策都成功
    assert strategic_decision.decision_level == 'strategic'
    assert campaign_decision.decision_level == 'campaign'
    assert tactical_decision.decision_level == 'tactical'
    
    # 验证约束继承
    assert tactical_decision.constraints is not None
