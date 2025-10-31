"""
测试 LayeredScheduler
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta

from Agents.orchestration import (
    LayeredScheduler,
    MultiSymbolScheduler,
    DecisionLevel,
    EscalationReason,
    SchedulerState,
    StrategicDecisionMaker,
    CampaignDecisionMaker,
    TacticalDecisionMaker,
    Decision
)
from Agents.core import MacroContext
from Agents.core import SectorContext


# ===== Fixtures =====

@pytest.fixture
def mock_strategic_maker():
    """Mock StrategicDecisionMaker"""
    maker = AsyncMock(spec=StrategicDecisionMaker)
    
    # 模拟决策结果
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
        constraints={'allow_long': True, 'allow_short': False, 'max_position_size': 0.3},
        analysis_timestamp=datetime.now(),
        data_end_time=None,
        confidence_score=0.85,
        reasoning='Bull market'
    )
    
    sector_ctx = SectorContext(
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
        reasoning='Tech strong',
        analysis_timestamp=datetime.now(),
        data_end_time=None
    )
    
    maker.decide.return_value = Decision(
        symbol='AAPL',
        action='BUY',
        conviction=8.0,
        reasoning='Strategic: Bull market + Tech strength',
        macro_context=macro_ctx,
        sector_context=sector_ctx,
        constraints={'allow_long': True, 'allow_short': False, 'max_position_size': 0.3},
        decision_level='strategic'
    )
    
    return maker


@pytest.fixture
def mock_campaign_maker():
    """Mock CampaignDecisionMaker"""
    maker = AsyncMock(spec=CampaignDecisionMaker)
    
    maker.decide.return_value = Decision(
        symbol='AAPL',
        action='BUY',
        conviction=7.0,
        reasoning='Campaign: Maintain position',
        constraints={'allow_long': True, 'allow_short': False},
        decision_level='campaign'
    )
    
    return maker


@pytest.fixture
def mock_tactical_maker():
    """Mock TacticalDecisionMaker"""
    maker = AsyncMock(spec=TacticalDecisionMaker)
    
    maker.decide.return_value = Decision(
        symbol='AAPL',
        action='BUY',
        conviction=6.0,
        reasoning='Tactical: Continue',
        decision_level='tactical'
    )
    
    return maker


@pytest.fixture
def scheduler(mock_strategic_maker, mock_campaign_maker, mock_tactical_maker):
    """LayeredScheduler实例"""
    return LayeredScheduler(
        strategic_maker=mock_strategic_maker,
        campaign_maker=mock_campaign_maker,
        tactical_maker=mock_tactical_maker,
        strategic_interval_days=30,
        campaign_interval_days=7,
        tactical_interval_days=1
    )


# ===== Test LayeredScheduler Initialization =====

def test_scheduler_initialization(scheduler):
    """测试调度器初始化"""
    assert scheduler.strategic_maker is not None
    assert scheduler.campaign_maker is not None
    assert scheduler.tactical_maker is not None
    
    assert scheduler.strategic_interval == timedelta(days=30)
    assert scheduler.campaign_interval == timedelta(days=7)
    assert scheduler.tactical_interval == timedelta(days=1)
    
    assert scheduler.state.last_strategic_time is None
    assert scheduler.state.last_campaign_time is None
    assert scheduler.state.last_tactical_time is None


# ===== Test Scheduling Logic =====

def test_should_run_strategic_first_time(scheduler):
    """测试首次运行战略层"""
    current_time = datetime(2024, 1, 1)
    assert scheduler.should_run_strategic(current_time) is True


def test_should_run_strategic_after_interval(scheduler):
    """测试间隔后运行战略层"""
    scheduler.state.last_strategic_time = datetime(2024, 1, 1)
    
    # 30天后应该运行
    current_time = datetime(2024, 1, 31)  # 1月1日 + 30天
    assert scheduler.should_run_strategic(current_time) is True
    
    # 29天后不应该运行
    current_time = datetime(2024, 1, 30)
    assert scheduler.should_run_strategic(current_time) is False


def test_should_run_campaign_logic(scheduler):
    """测试战役层调度逻辑"""
    scheduler.state.last_campaign_time = datetime(2024, 1, 1)
    
    # 7天后应该运行
    assert scheduler.should_run_campaign(datetime(2024, 1, 8)) is True
    
    # 6天后不应该运行
    assert scheduler.should_run_campaign(datetime(2024, 1, 7)) is False


def test_should_run_tactical_logic(scheduler):
    """测试战术层调度逻辑"""
    scheduler.state.last_tactical_time = datetime(2024, 1, 1, 10, 0)
    
    # 1天后应该运行
    assert scheduler.should_run_tactical(datetime(2024, 1, 2, 10, 0)) is True
    
    # 不到1天不应该运行
    assert scheduler.should_run_tactical(datetime(2024, 1, 2, 9, 0)) is False


# ===== Test Escalation =====

def test_check_escalation_regime_change(scheduler):
    """测试regime变化触发escalation"""
    signals = {'regime_changed': True}
    reason = scheduler.check_escalation_triggers('AAPL', DecisionLevel.TACTICAL, signals)
    
    assert reason == EscalationReason.REGIME_CHANGE


def test_check_escalation_sector_rotation(scheduler):
    """测试行业轮动触发escalation"""
    signals = {'sector_rotation_signal': 'rotating_in'}
    reason = scheduler.check_escalation_triggers('AAPL', DecisionLevel.TACTICAL, signals)
    
    assert reason == EscalationReason.SECTOR_ROTATION


def test_check_escalation_volatility_spike(scheduler):
    """测试波动率飙升触发escalation"""
    signals = {'vix_level': 35}
    reason = scheduler.check_escalation_triggers('AAPL', DecisionLevel.TACTICAL, signals)
    
    assert reason == EscalationReason.VOLATILITY_SPIKE


def test_check_escalation_technical_breakout(scheduler):
    """测试技术突破触发escalation"""
    signals = {'technical_breakout': True}
    reason = scheduler.check_escalation_triggers('AAPL', DecisionLevel.TACTICAL, signals)
    
    assert reason == EscalationReason.TECHNICAL_BREAKOUT


def test_check_escalation_no_trigger(scheduler):
    """测试无escalation触发"""
    signals = {'vix_level': 15}  # 正常水平
    reason = scheduler.check_escalation_triggers('AAPL', DecisionLevel.TACTICAL, signals)
    
    assert reason is None


# ===== Test Decision Making =====

@pytest.mark.asyncio
async def test_decide_strategic_first_run(scheduler, mock_strategic_maker):
    """测试首次运行战略层决策"""
    current_time = datetime(2024, 1, 1)
    
    decision = await scheduler.decide(
        symbol='AAPL',
        current_time=current_time
    )
    
    # 应该调用战略层
    mock_strategic_maker.decide.assert_called_once()
    
    # 验证决策结果
    assert decision.action == 'BUY'
    assert decision.decision_level == 'strategic'
    
    # 验证状态更新
    assert scheduler.state.last_strategic_time == current_time
    assert scheduler.state.last_strategic_decision is not None
    assert scheduler.state.active_constraints is not None
    assert scheduler.state.active_macro_context is not None
    assert scheduler.state.active_sector_context is not None


@pytest.mark.asyncio
async def test_decide_tactical_after_strategic(scheduler, mock_tactical_maker):
    """测试战略层运行后的战术层决策"""
    # 先运行战略层
    scheduler.state.last_strategic_time = datetime(2024, 1, 1)
    scheduler.state.last_campaign_time = datetime(2024, 1, 1)  # 也设置campaign时间
    scheduler.state.last_tactical_time = datetime(2024, 1, 1)  # 设置tactical时间
    scheduler.state.active_constraints = {'allow_long': True}
    scheduler.state.active_macro_context = {'market_regime': 'bull'}
    scheduler.state.active_sector_context = {'trend': 'bullish'}
    
    # 第二天运行战术层（需要确保campaign也不到7天）
    current_time = datetime(2024, 1, 2)
    
    decision = await scheduler.decide(
        symbol='AAPL',
        current_time=current_time
    )
    
    # 应该调用战术层
    mock_tactical_maker.decide.assert_called_once()
    
    # 验证继承的上下文
    call_args = mock_tactical_maker.decide.call_args
    assert call_args.kwargs['inherited_constraints'] == {'allow_long': True}
    assert call_args.kwargs['inherited_macro_context'] == {'market_regime': 'bull'}
    assert call_args.kwargs['inherited_sector_context'] == {'trend': 'bullish'}


@pytest.mark.asyncio
async def test_decide_campaign_scheduled(scheduler, mock_campaign_maker):
    """测试定期运行战役层"""
    # 战略层上次运行（不到30天）
    scheduler.state.last_strategic_time = datetime(2024, 1, 1)
    scheduler.state.last_campaign_time = datetime(2024, 1, 1)  # Campaign上次运行
    scheduler.state.active_constraints = {'allow_long': True, 'allow_short': False}
    
    # 7天后运行战役层（但strategic还没到30天）
    current_time = datetime(2024, 1, 8)
    
    decision = await scheduler.decide(
        symbol='AAPL',
        current_time=current_time
    )
    
    # 应该调用战役层
    mock_campaign_maker.decide.assert_called_once()
    
    # 验证继承约束
    call_args = mock_campaign_maker.decide.call_args
    assert call_args.kwargs['inherited_constraints'] == {'allow_long': True, 'allow_short': False}


@pytest.mark.asyncio
async def test_decide_with_regime_escalation(scheduler, mock_strategic_maker):
    """测试regime变化触发战略层escalation"""
    # 战略层上次运行
    scheduler.state.last_strategic_time = datetime(2024, 1, 1)
    scheduler.state.last_tactical_time = datetime(2024, 1, 10)
    
    # 第11天，正常应该运行战术层，但有regime变化
    current_time = datetime(2024, 1, 11)
    escalation_signals = {'regime_changed': True}
    
    decision = await scheduler.decide(
        symbol='AAPL',
        current_time=current_time,
        escalation_signals=escalation_signals
    )
    
    # 应该触发战略层
    mock_strategic_maker.decide.assert_called_once()
    
    # escalation计数增加
    assert scheduler.state.strategic_escalations == 1


@pytest.mark.asyncio
async def test_decide_with_sector_rotation_escalation(scheduler, mock_campaign_maker):
    """测试行业轮动触发战役层escalation"""
    scheduler.state.last_strategic_time = datetime(2024, 1, 1)  # Strategic不到期
    scheduler.state.last_campaign_time = datetime(2024, 1, 1)
    scheduler.state.last_tactical_time = datetime(2024, 1, 2)
    
    current_time = datetime(2024, 1, 3)
    escalation_signals = {'sector_rotation_signal': 'rotating_out'}
    
    decision = await scheduler.decide(
        symbol='AAPL',
        current_time=current_time,
        escalation_signals=escalation_signals
    )
    
    # 应该触发战役层
    mock_campaign_maker.decide.assert_called_once()
    
    # escalation计数增加
    assert scheduler.state.campaign_escalations == 1


# ===== Test State Management =====

def test_get_next_schedule(scheduler):
    """测试获取下次调度时间"""
    scheduler.state.last_strategic_time = datetime(2024, 1, 1)
    scheduler.state.last_campaign_time = datetime(2024, 1, 1)
    scheduler.state.last_tactical_time = datetime(2024, 1, 1)
    
    current_time = datetime(2024, 1, 10)
    schedule = scheduler.get_next_schedule(current_time)
    
    assert schedule['strategic'] == datetime(2024, 1, 31)  # 1月1日 + 30天
    assert schedule['campaign'] == datetime(2024, 1, 8)   # +7 days
    assert schedule['tactical'] == datetime(2024, 1, 2)   # +1 day


def test_get_state_summary(scheduler):
    """测试获取状态摘要"""
    scheduler.state.last_strategic_time = datetime(2024, 1, 1)
    scheduler.state.strategic_escalations = 2
    scheduler.state.campaign_escalations = 5
    scheduler.state.active_constraints = {'allow_long': True}
    scheduler.state.active_macro_context = {'regime': 'bull'}
    
    summary = scheduler.get_state_summary()
    
    assert summary['last_strategic_time'] == '2024-01-01T00:00:00'
    assert summary['strategic_escalations'] == 2
    assert summary['campaign_escalations'] == 5
    assert summary['active_constraints'] == {'allow_long': True}
    assert summary['has_macro_context'] is True


def test_reset_scheduler(scheduler):
    """测试重置调度器"""
    scheduler.state.last_strategic_time = datetime(2024, 1, 1)
    scheduler.state.strategic_escalations = 10
    
    scheduler.reset()
    
    assert scheduler.state.last_strategic_time is None
    assert scheduler.state.strategic_escalations == 0


# ===== Test MultiSymbolScheduler =====

def test_multi_symbol_initialization(mock_strategic_maker, mock_campaign_maker, mock_tactical_maker):
    """测试多股票调度器初始化"""
    multi = MultiSymbolScheduler(
        strategic_maker=mock_strategic_maker,
        campaign_maker=mock_campaign_maker,
        tactical_maker=mock_tactical_maker
    )
    
    assert multi.strategic_maker is not None
    assert len(multi.schedulers) == 0  # 初始没有调度器


def test_multi_symbol_get_scheduler(mock_strategic_maker, mock_campaign_maker, mock_tactical_maker):
    """测试获取单个股票的调度器"""
    multi = MultiSymbolScheduler(
        strategic_maker=mock_strategic_maker,
        campaign_maker=mock_campaign_maker,
        tactical_maker=mock_tactical_maker
    )
    
    scheduler1 = multi.get_scheduler('AAPL')
    assert scheduler1 is not None
    assert 'AAPL' in multi.schedulers
    
    scheduler2 = multi.get_scheduler('AAPL')
    assert scheduler1 is scheduler2  # 应该返回同一个实例


@pytest.mark.asyncio
async def test_multi_symbol_decide(mock_strategic_maker, mock_campaign_maker, mock_tactical_maker):
    """测试多股票决策"""
    multi = MultiSymbolScheduler(
        strategic_maker=mock_strategic_maker,
        campaign_maker=mock_campaign_maker,
        tactical_maker=mock_tactical_maker
    )
    
    current_time = datetime(2024, 1, 1)
    decision = await multi.decide(
        symbol='AAPL',
        current_time=current_time
    )
    
    assert decision is not None
    assert decision.action == 'BUY'
    assert 'AAPL' in multi.schedulers


@pytest.mark.asyncio
async def test_multi_symbol_decide_batch(mock_strategic_maker, mock_campaign_maker, mock_tactical_maker):
    """测试批量决策"""
    multi = MultiSymbolScheduler(
        strategic_maker=mock_strategic_maker,
        campaign_maker=mock_campaign_maker,
        tactical_maker=mock_tactical_maker
    )
    
    current_time = datetime(2024, 1, 1)
    decisions = await multi.decide_batch(
        symbols=['AAPL', 'GOOGL', 'MSFT'],
        current_time=current_time
    )
    
    assert len(decisions) == 3
    assert 'AAPL' in decisions
    assert 'GOOGL' in decisions
    assert 'MSFT' in decisions
    
    # 每个股票应该有独立的调度器
    assert len(multi.schedulers) == 3


@pytest.mark.asyncio
async def test_multi_symbol_decide_batch_with_signals(mock_strategic_maker, mock_campaign_maker, mock_tactical_maker):
    """测试带escalation信号的批量决策"""
    multi = MultiSymbolScheduler(
        strategic_maker=mock_strategic_maker,
        campaign_maker=mock_campaign_maker,
        tactical_maker=mock_tactical_maker
    )
    
    current_time = datetime(2024, 1, 1)
    escalation_signals = {
        'AAPL': {'regime_changed': True},
        'GOOGL': {'sector_rotation_signal': 'rotating_out'}
    }
    
    decisions = await multi.decide_batch(
        symbols=['AAPL', 'GOOGL'],
        current_time=current_time,
        escalation_signals=escalation_signals
    )
    
    assert len(decisions) == 2


def test_multi_symbol_get_all_states(mock_strategic_maker, mock_campaign_maker, mock_tactical_maker):
    """测试获取所有状态"""
    multi = MultiSymbolScheduler(
        strategic_maker=mock_strategic_maker,
        campaign_maker=mock_campaign_maker,
        tactical_maker=mock_tactical_maker
    )
    
    # 创建两个调度器
    multi.get_scheduler('AAPL')
    multi.get_scheduler('GOOGL')
    
    states = multi.get_all_states()
    
    assert len(states) == 2
    assert 'AAPL' in states
    assert 'GOOGL' in states


def test_multi_symbol_reset_single(mock_strategic_maker, mock_campaign_maker, mock_tactical_maker):
    """测试重置单个股票"""
    multi = MultiSymbolScheduler(
        strategic_maker=mock_strategic_maker,
        campaign_maker=mock_campaign_maker,
        tactical_maker=mock_tactical_maker
    )
    
    scheduler1 = multi.get_scheduler('AAPL')
    scheduler2 = multi.get_scheduler('GOOGL')
    
    scheduler1.state.last_strategic_time = datetime(2024, 1, 1)
    scheduler2.state.last_strategic_time = datetime(2024, 1, 1)
    
    # 只重置AAPL
    multi.reset('AAPL')
    
    assert scheduler1.state.last_strategic_time is None
    assert scheduler2.state.last_strategic_time == datetime(2024, 1, 1)


def test_multi_symbol_reset_all(mock_strategic_maker, mock_campaign_maker, mock_tactical_maker):
    """测试重置所有股票"""
    multi = MultiSymbolScheduler(
        strategic_maker=mock_strategic_maker,
        campaign_maker=mock_campaign_maker,
        tactical_maker=mock_tactical_maker
    )
    
    scheduler1 = multi.get_scheduler('AAPL')
    scheduler2 = multi.get_scheduler('GOOGL')
    
    scheduler1.state.last_strategic_time = datetime(2024, 1, 1)
    scheduler2.state.last_strategic_time = datetime(2024, 1, 1)
    
    # 重置所有
    multi.reset()
    
    assert scheduler1.state.last_strategic_time is None
    assert scheduler2.state.last_strategic_time is None
