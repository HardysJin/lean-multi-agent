"""
测试扩展的 DecisionRecord 和反向传导系统

测试内容：
1. DecisionRecord 新字段的基本功能
2. 时间验证（防止Look-Ahead）
3. 反向传导标记
4. 缓存键生成
5. 序列化/反序列化
6. EscalationDetector 各种触发条件
"""

import pytest
from datetime import datetime, timedelta
from Memory.schemas import (
    DecisionRecord, 
    Timeframe, 
    create_decision_id
)
from Memory.escalation import (
    EscalationDetector,
    EscalationTrigger,
    EscalationTriggerType,
    should_trigger_escalation,
)


class TestDecisionRecordExtensions:
    """测试 DecisionRecord 的新增字段和方法"""
    
    def test_basic_decision_creation(self):
        """测试基础决策创建"""
        timestamp = datetime(2025, 10, 29, 10, 30)
        decision = DecisionRecord(
            id="test_001",
            timestamp=timestamp,
            timeframe=Timeframe.TACTICAL,
            symbol="AAPL",
            action="BUY",
            quantity=100,
            price=150.0,
            reasoning="Strong technical signals",
            agent_name="meta_agent",
            conviction=8.5,
        )
        
        assert decision.id == "test_001"
        assert decision.symbol == "AAPL"
        assert decision.action == "BUY"
        assert decision.computation_mode == "full"  # 默认值
        assert decision.visible_data_end is None  # 默认无限制
    
    def test_backtest_mode_detection(self):
        """测试回测模式检测"""
        timestamp = datetime(2025, 10, 29, 10, 30)
        
        # 实盘模式（无visible_data_end）
        live_decision = DecisionRecord(
            id="live_001",
            timestamp=timestamp,
            timeframe=Timeframe.TACTICAL,
            symbol="AAPL",
            action="BUY",
            quantity=100,
            price=150.0,
            reasoning="Test",
            agent_name="meta_agent",
            conviction=7.0,
        )
        assert not live_decision.is_backtest_mode()
        
        # 回测模式（有visible_data_end）
        backtest_decision = DecisionRecord(
            id="backtest_001",
            timestamp=timestamp,
            timeframe=Timeframe.TACTICAL,
            symbol="AAPL",
            action="BUY",
            quantity=100,
            price=150.0,
            reasoning="Test",
            agent_name="meta_agent",
            conviction=7.0,
            visible_data_end=timestamp - timedelta(minutes=5),
        )
        assert backtest_decision.is_backtest_mode()
    
    def test_data_timestamp_validation(self):
        """测试数据时间戳验证（防止Look-Ahead）"""
        decision_time = datetime(2025, 10, 29, 10, 30)
        visible_end = datetime(2025, 10, 29, 10, 25)  # 5分钟前
        
        decision = DecisionRecord(
            id="test_002",
            timestamp=decision_time,
            timeframe=Timeframe.TACTICAL,
            symbol="AAPL",
            action="BUY",
            quantity=100,
            price=150.0,
            reasoning="Test",
            agent_name="meta_agent",
            conviction=7.0,
            visible_data_end=visible_end,
        )
        
        # 测试各种时间点
        past_time = datetime(2025, 10, 29, 10, 20)  # 可见范围内
        exact_time = datetime(2025, 10, 29, 10, 25)  # 边界
        future_time = datetime(2025, 10, 29, 10, 35)  # 未来数据
        
        assert decision.validate_data_timestamp(past_time) is True
        assert decision.validate_data_timestamp(exact_time) is True
        assert decision.validate_data_timestamp(future_time) is False  # 不能看未来
    
    def test_computation_modes(self):
        """测试计算模式设置"""
        timestamp = datetime(2025, 10, 29, 10, 30)
        
        # Full模式
        full_decision = DecisionRecord(
            id="full_001",
            timestamp=timestamp,
            timeframe=Timeframe.STRATEGIC,
            symbol="AAPL",
            action="BUY",
            quantity=100,
            price=150.0,
            reasoning="Strategic allocation",
            agent_name="meta_agent",
            conviction=9.0,
            computation_mode="full",
        )
        assert full_decision.computation_mode == "full"
        
        # Fast模式
        fast_decision = DecisionRecord(
            id="fast_001",
            timestamp=timestamp,
            timeframe=Timeframe.TACTICAL,
            symbol="AAPL",
            action="HOLD",
            quantity=0,
            price=150.0,
            reasoning="No signal",
            agent_name="rule_engine",
            conviction=5.0,
            computation_mode="fast",
        )
        assert fast_decision.computation_mode == "fast"
        
        # Hybrid模式
        hybrid_decision = DecisionRecord(
            id="hybrid_001",
            timestamp=timestamp,
            timeframe=Timeframe.CAMPAIGN,
            symbol="AAPL",
            action="ADD",
            quantity=50,
            price=150.0,
            reasoning="Moderate signal",
            agent_name="meta_agent",
            conviction=7.5,
            computation_mode="hybrid",
        )
        assert hybrid_decision.computation_mode == "hybrid"
    
    def test_escalation_marking(self):
        """测试反向传导标记"""
        timestamp = datetime(2025, 10, 29, 10, 30)
        decision = DecisionRecord(
            id="escalated_001",
            timestamp=timestamp,
            timeframe=Timeframe.CAMPAIGN,
            symbol="AAPL",
            action="SELL",
            quantity=200,
            price=145.0,
            reasoning="Emergency exit due to market shock",
            agent_name="meta_agent",
            conviction=9.5,
        )
        
        # 标记为反向传导
        decision.mark_as_escalated(
            from_timeframe="tactical",
            trigger="market_shock",
            score=9.2,
        )
        
        assert decision.escalated_from == "tactical"
        assert decision.escalation_trigger == "market_shock"
        assert decision.escalation_score == 9.2
        assert decision.metadata['escalated'] is True
        assert 'escalation_details' in decision.metadata
    
    def test_cache_key_generation(self):
        """测试缓存键生成"""
        timestamp = datetime(2025, 10, 29, 10, 30)
        decision = DecisionRecord(
            id="cached_001",
            timestamp=timestamp,
            timeframe=Timeframe.TACTICAL,
            symbol="AAPL",
            action="BUY",
            quantity=100,
            price=150.0,
            reasoning="Test",
            agent_name="meta_agent",
            conviction=7.0,
        )
        
        # 设置缓存键
        decision.set_cache_key(
            strategy_version="v1.0.0",
            data_hash="abc123def456",
        )
        
        expected_key = "AAPL_tactical_v1.0.0_abc123def456"
        assert decision.cache_key == expected_key
    
    def test_serialization_with_new_fields(self):
        """测试包含新字段的序列化/反序列化"""
        timestamp = datetime(2025, 10, 29, 10, 30)
        visible_end = datetime(2025, 10, 29, 10, 25)
        
        original = DecisionRecord(
            id="serial_001",
            timestamp=timestamp,
            timeframe=Timeframe.TACTICAL,
            symbol="AAPL",
            action="BUY",
            quantity=100,
            price=150.0,
            reasoning="Test serialization",
            agent_name="meta_agent",
            conviction=8.0,
            visible_data_end=visible_end,
            computation_mode="hybrid",
            cache_key="AAPL_tactical_v1_hash123",
        )
        
        # 标记反向传导
        original.mark_as_escalated("tactical", "news_impact", 8.5)
        
        # 序列化
        data_dict = original.to_dict()
        
        # 验证字段存在
        assert 'visible_data_end' in data_dict
        assert 'computation_mode' in data_dict
        assert 'cache_key' in data_dict
        assert 'escalated_from' in data_dict
        assert 'escalation_trigger' in data_dict
        assert 'escalation_score' in data_dict
        
        # 反序列化
        restored = DecisionRecord.from_dict(data_dict)
        
        # 验证所有字段正确恢复
        assert restored.id == original.id
        assert restored.timestamp == original.timestamp
        assert restored.visible_data_end == original.visible_data_end
        assert restored.computation_mode == original.computation_mode
        assert restored.cache_key == original.cache_key
        assert restored.escalated_from == original.escalated_from
        assert restored.escalation_trigger == original.escalation_trigger
        assert restored.escalation_score == original.escalation_score


class TestEscalationDetector:
    """测试反向传导检测器"""
    
    def test_market_shock_detection_normal(self):
        """测试正常市场冲击检测（升至Campaign）"""
        detector = EscalationDetector()
        
        # 6%单日下跌
        trigger = detector.detect_market_shock(
            symbol="AAPL",
            price_change_1d=-0.06,
            current_timeframe=Timeframe.TACTICAL,
        )
        
        assert trigger is not None
        assert trigger.trigger_type == EscalationTriggerType.MARKET_SHOCK
        assert trigger.from_timeframe == Timeframe.TACTICAL
        assert trigger.to_timeframe == Timeframe.CAMPAIGN
        assert trigger.score >= 5.0
        assert trigger.symbol == "AAPL"
    
    def test_market_shock_detection_extreme(self):
        """测试极端市场冲击检测（直达Strategic）"""
        detector = EscalationDetector()
        
        # 20%单日暴跌（黑天鹅）
        trigger = detector.detect_market_shock(
            symbol="AAPL",
            price_change_1d=-0.20,
            current_timeframe=Timeframe.TACTICAL,
        )
        
        assert trigger is not None
        assert trigger.trigger_type == EscalationTriggerType.BLACK_SWAN
        assert trigger.to_timeframe == Timeframe.STRATEGIC  # 直达战略层
        assert trigger.score == 10.0  # 最高评分
    
    def test_market_shock_no_trigger(self):
        """测试小幅波动不触发"""
        detector = EscalationDetector()
        
        # 2%小幅下跌
        trigger = detector.detect_market_shock(
            symbol="AAPL",
            price_change_1d=-0.02,
            current_timeframe=Timeframe.TACTICAL,
        )
        
        assert trigger is None
    
    def test_news_impact_detection_high(self):
        """测试重要新闻检测（升至Campaign）"""
        detector = EscalationDetector()
        
        trigger = detector.detect_news_impact(
            symbol="AAPL",
            news_importance=8.5,
            news_sentiment=-0.8,
            news_title="Apple faces major antitrust lawsuit",
            current_timeframe=Timeframe.TACTICAL,
        )
        
        assert trigger is not None
        assert trigger.trigger_type == EscalationTriggerType.NEWS_IMPACT
        assert trigger.to_timeframe == Timeframe.CAMPAIGN
        assert trigger.score >= 7.0
    
    def test_news_impact_detection_critical(self):
        """测试关键新闻检测（直达Strategic）"""
        detector = EscalationDetector()
        
        trigger = detector.detect_news_impact(
            symbol="AAPL",
            news_importance=9.8,
            news_sentiment=-0.9,
            news_title="Apple CEO announces resignation",
            current_timeframe=Timeframe.TACTICAL,
        )
        
        assert trigger is not None
        assert trigger.trigger_type == EscalationTriggerType.BLACK_SWAN
        assert trigger.to_timeframe == Timeframe.STRATEGIC
        assert trigger.score == 10.0
    
    def test_technical_breakout_detection(self):
        """测试技术突破检测"""
        detector = EscalationDetector()
        
        trigger = detector.detect_technical_breakout(
            symbol="AAPL",
            breakout_type="resistance",
            confidence=0.92,
            price=155.0,
            key_level=150.0,
            current_timeframe=Timeframe.TACTICAL,
        )
        
        assert trigger is not None
        assert trigger.trigger_type == EscalationTriggerType.TECHNICAL_BREAKOUT
        assert trigger.to_timeframe == Timeframe.CAMPAIGN
        assert trigger.score >= 6.0
    
    def test_strategic_conflict_detection(self):
        """测试战略冲突检测"""
        detector = EscalationDetector()
        
        trigger = detector.detect_strategic_conflict(
            symbol="AAPL",
            tactical_action="BUY",
            tactical_conviction=8.5,
            strategic_constraint="bearish_regime",
            conflict_reason="Tactical sees strong buy signal but strategic is bearish",
            current_timeframe=Timeframe.TACTICAL,
        )
        
        assert trigger is not None
        assert trigger.trigger_type == EscalationTriggerType.STRATEGIC_CONFLICT
        assert trigger.to_timeframe == Timeframe.CAMPAIGN
        assert trigger.score >= 5.0
    
    def test_volatility_spike_detection(self):
        """测试波动率飙升检测"""
        detector = EscalationDetector()
        
        trigger = detector.detect_volatility_spike(
            symbol="AAPL",
            current_volatility=0.45,
            historical_volatility=0.12,  # 当前是历史的3.75倍
            current_timeframe=Timeframe.TACTICAL,
        )
        
        assert trigger is not None
        assert trigger.trigger_type == EscalationTriggerType.VOLATILITY_SPIKE
        assert trigger.to_timeframe == Timeframe.CAMPAIGN
        assert trigger.score >= 5.0
    
    def test_detect_all_multiple_triggers(self):
        """测试综合检测（多个触发条件）"""
        detector = EscalationDetector()
        
        market_data = {
            'price_change_1d': -0.08,  # 8%下跌
            'price_change_3d': -0.12,  # 3日12%下跌
            'current_volatility': 0.50,
            'historical_volatility': 0.15,
        }
        
        news_data = {
            'importance': 8.5,
            'sentiment': -0.7,
            'title': 'Major product recall announced',
        }
        
        triggers = detector.detect_all(
            symbol="AAPL",
            market_data=market_data,
            news_data=news_data,
            current_timeframe=Timeframe.TACTICAL,
        )
        
        # 应该检测到多个触发条件
        assert len(triggers) >= 2
        
        # 验证按评分降序排序
        for i in range(len(triggers) - 1):
            assert triggers[i].score >= triggers[i+1].score
    
    def test_custom_thresholds(self):
        """测试自定义阈值"""
        custom_thresholds = {
            'market_shock_1day': 0.03,  # 降低阈值到3%
        }
        detector = EscalationDetector(thresholds=custom_thresholds)
        
        # 4%下跌，使用自定义阈值应该触发
        trigger = detector.detect_market_shock(
            symbol="AAPL",
            price_change_1d=-0.04,
            current_timeframe=Timeframe.TACTICAL,
        )
        
        assert trigger is not None


class TestEscalationTrigger:
    """测试 EscalationTrigger 数据结构"""
    
    def test_trigger_creation(self):
        """测试触发器创建"""
        trigger = EscalationTrigger(
            trigger_type=EscalationTriggerType.MARKET_SHOCK,
            from_timeframe=Timeframe.TACTICAL,
            to_timeframe=Timeframe.CAMPAIGN,
            score=8.5,
            symbol="AAPL",
            timestamp=datetime(2025, 10, 29, 10, 30),
            reason="Sharp price decline",
            details={'price_change': -0.08},
        )
        
        assert trigger.trigger_type == EscalationTriggerType.MARKET_SHOCK
        assert trigger.from_timeframe == Timeframe.TACTICAL
        assert trigger.to_timeframe == Timeframe.CAMPAIGN
        assert trigger.score == 8.5
    
    def test_should_escalate(self):
        """测试是否应该触发判断"""
        high_score_trigger = EscalationTrigger(
            trigger_type=EscalationTriggerType.MARKET_SHOCK,
            from_timeframe=Timeframe.TACTICAL,
            to_timeframe=Timeframe.CAMPAIGN,
            score=8.5,
            symbol="AAPL",
            timestamp=datetime.now(),
            reason="Test",
            details={},
        )
        
        low_score_trigger = EscalationTrigger(
            trigger_type=EscalationTriggerType.TECHNICAL_BREAKOUT,
            from_timeframe=Timeframe.TACTICAL,
            to_timeframe=Timeframe.CAMPAIGN,
            score=6.0,
            symbol="AAPL",
            timestamp=datetime.now(),
            reason="Test",
            details={},
        )
        
        assert high_score_trigger.should_escalate(threshold=7.0) is True
        assert low_score_trigger.should_escalate(threshold=7.0) is False
    
    def test_trigger_serialization(self):
        """测试触发器序列化"""
        trigger = EscalationTrigger(
            trigger_type=EscalationTriggerType.NEWS_IMPACT,
            from_timeframe=Timeframe.TACTICAL,
            to_timeframe=Timeframe.STRATEGIC,
            score=9.5,
            symbol="AAPL",
            timestamp=datetime(2025, 10, 29, 10, 30),
            reason="Critical news",
            details={'news_id': 'news_123'},
        )
        
        data = trigger.to_dict()
        
        assert data['trigger_type'] == 'news_impact'
        assert data['from_timeframe'] == 'tactical'
        assert data['to_timeframe'] == 'strategic'
        assert data['score'] == 9.5
        assert 'timestamp' in data


class TestShouldTriggerEscalation:
    """测试全局触发判断函数"""
    
    def test_no_triggers(self):
        """测试无触发条件"""
        result = should_trigger_escalation([])
        assert result is None
    
    def test_below_threshold(self):
        """测试所有触发条件都低于阈值"""
        triggers = [
            EscalationTrigger(
                trigger_type=EscalationTriggerType.MARKET_SHOCK,
                from_timeframe=Timeframe.TACTICAL,
                to_timeframe=Timeframe.CAMPAIGN,
                score=6.0,
                symbol="AAPL",
                timestamp=datetime.now(),
                reason="Small shock",
                details={},
            )
        ]
        
        result = should_trigger_escalation(triggers, threshold=7.0)
        assert result is None
    
    def test_above_threshold(self):
        """测试返回最高评分的触发条件"""
        triggers = [
            EscalationTrigger(
                trigger_type=EscalationTriggerType.MARKET_SHOCK,
                from_timeframe=Timeframe.TACTICAL,
                to_timeframe=Timeframe.CAMPAIGN,
                score=8.5,
                symbol="AAPL",
                timestamp=datetime.now(),
                reason="Strong shock",
                details={},
            ),
            EscalationTrigger(
                trigger_type=EscalationTriggerType.NEWS_IMPACT,
                from_timeframe=Timeframe.TACTICAL,
                to_timeframe=Timeframe.CAMPAIGN,
                score=7.5,
                symbol="AAPL",
                timestamp=datetime.now(),
                reason="Important news",
                details={},
            ),
        ]
        
        result = should_trigger_escalation(triggers, threshold=7.0)
        
        assert result is not None
        assert result.score == 8.5  # 返回最高评分的
        assert result.trigger_type == EscalationTriggerType.MARKET_SHOCK


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
