"""
DecisionRecord 扩展和反向传导系统集成测试

演示完整的工作流程：
1. 创建回测模式的决策
2. 验证时间控制防止Look-Ahead
3. 检测触发条件并执行反向传导
4. 不同计算模式的决策流程
5. 信号缓存和复用
"""

import pytest
from datetime import datetime, timedelta
from Memory.schemas import (
    DecisionRecord,
    Timeframe,
    create_decision_id,
)
from Memory.escalation import (
    EscalationDetector,
    EscalationTriggerType,
)


class TestIntegratedWorkflow:
    """集成测试：完整工作流程"""
    
    def test_backtest_workflow_with_time_control(self):
        """
        测试回测工作流程中的时间控制
        
        场景：
        1. 回测从 10:00 开始
        2. 决策时间是 10:30
        3. 可见数据截止到 10:25（5分钟延迟）
        4. 验证只能使用 <= 10:25 的数据
        """
        backtest_start = datetime(2025, 10, 29, 10, 0)
        decision_time = datetime(2025, 10, 29, 10, 30)
        visible_end = datetime(2025, 10, 29, 10, 25)
        
        # 创建回测决策
        decision = DecisionRecord(
            id=create_decision_id("AAPL", decision_time, Timeframe.TACTICAL),
            timestamp=decision_time,
            timeframe=Timeframe.TACTICAL,
            symbol="AAPL",
            action="BUY",
            quantity=100,
            price=150.0,
            reasoning="Strong technical signal detected",
            agent_name="meta_agent",
            conviction=8.0,
            visible_data_end=visible_end,
            computation_mode="full",
        )
        
        # 验证回测模式
        assert decision.is_backtest_mode()
        
        # 模拟不同时间点的数据
        past_data_time = datetime(2025, 10, 29, 10, 20)  # 5分钟前
        boundary_time = datetime(2025, 10, 29, 10, 25)   # 边界
        recent_time = datetime(2025, 10, 29, 10, 27)     # 2分钟前但超过visible_end
        future_time = datetime(2025, 10, 29, 10, 35)     # 5分钟后
        
        # 验证时间控制
        assert decision.validate_data_timestamp(past_data_time) is True
        assert decision.validate_data_timestamp(boundary_time) is True
        assert decision.validate_data_timestamp(recent_time) is False  # Look-Ahead!
        assert decision.validate_data_timestamp(future_time) is False  # Look-Ahead!
        
        print(f"✓ Time control validated: Only data <= {visible_end} is visible")
    
    def test_escalation_triggered_workflow(self):
        """
        测试反向传导触发的完整流程
        
        场景：
        1. Tactical层检测到市场冲击（8%下跌）
        2. 触发反向传导至Campaign层
        3. Campaign层做出调整决策
        4. 记录完整的反向传导链路
        """
        detector = EscalationDetector()
        
        # === Step 1: Tactical层正常决策 ===
        tactical_time = datetime(2025, 10, 29, 10, 30)
        tactical_decision = DecisionRecord(
            id=create_decision_id("AAPL", tactical_time, Timeframe.TACTICAL),
            timestamp=tactical_time,
            timeframe=Timeframe.TACTICAL,
            symbol="AAPL",
            action="HOLD",
            quantity=0,
            price=150.0,
            reasoning="Normal day, no strong signals",
            agent_name="rule_engine",
            conviction=5.0,
            computation_mode="fast",  # 快速模式
        )
        
        # === Step 2: 市场突然暴跌 ===
        market_data = {
            'price_change_1d': -0.08,  # 8%暴跌
            'current_volatility': 0.45,
            'historical_volatility': 0.12,
        }
        
        # 检测触发条件
        triggers = detector.detect_all(
            symbol="AAPL",
            market_data=market_data,
            current_timeframe=Timeframe.TACTICAL,
        )
        
        assert len(triggers) > 0
        top_trigger = triggers[0]
        
        # 验证触发反向传导
        assert top_trigger.should_escalate(threshold=7.0)
        assert top_trigger.trigger_type == EscalationTriggerType.MARKET_SHOCK
        assert top_trigger.to_timeframe == Timeframe.CAMPAIGN
        
        print(f"✓ Escalation triggered: {top_trigger.trigger_type.value} (score: {top_trigger.score:.1f})")
        
        # === Step 3: Campaign层响应 ===
        campaign_time = tactical_time + timedelta(minutes=5)
        campaign_decision = DecisionRecord(
            id=create_decision_id("AAPL", campaign_time, Timeframe.CAMPAIGN),
            timestamp=campaign_time,
            timeframe=Timeframe.CAMPAIGN,
            symbol="AAPL",
            action="REDUCE",
            quantity=150,
            price=138.0,  # 价格已经下跌
            reasoning=f"Emergency response to {top_trigger.reason}",
            agent_name="meta_agent",
            conviction=9.0,
            computation_mode="full",  # Campaign层使用完整模式
        )
        
        # 标记为反向传导触发
        campaign_decision.mark_as_escalated(
            from_timeframe=top_trigger.from_timeframe.display_name,
            trigger=top_trigger.trigger_type.value,
            score=top_trigger.score,
        )
        
        # 验证反向传导标记
        assert campaign_decision.escalated_from == "tactical"
        assert campaign_decision.escalation_trigger == "market_shock"
        assert campaign_decision.escalation_score >= 7.0
        
        print(f"✓ Campaign layer responded with {campaign_decision.action} action")
    
    def test_multi_timeframe_decision_cascade(self):
        """
        测试多时间尺度决策级联
        
        场景：
        1. Strategic层设定熊市约束
        2. Campaign层制定防守策略
        3. Tactical层执行日常决策（受上层约束）
        4. 黑天鹅事件触发 → 直达Strategic层
        """
        base_time = datetime(2025, 10, 29, 9, 0)
        
        # === Strategic层：每30天决策一次 ===
        strategic_decision = DecisionRecord(
            id=create_decision_id("AAPL", base_time, Timeframe.STRATEGIC),
            timestamp=base_time,
            timeframe=Timeframe.STRATEGIC,
            symbol="AAPL",
            action="REDUCE",
            quantity=500,
            price=160.0,
            reasoning="Market regime shift to bearish, reduce exposure",
            agent_name="meta_agent",
            conviction=8.5,
            computation_mode="full",
            metadata={
                'market_regime': 'bear',
                'risk_budget': 0.5,
                'max_exposure': 0.6,
            }
        )
        
        # === Campaign层：每周决策一次 ===
        campaign_time = base_time + timedelta(days=7)
        campaign_decision = DecisionRecord(
            id=create_decision_id("AAPL", campaign_time, Timeframe.CAMPAIGN),
            timestamp=campaign_time,
            timeframe=Timeframe.CAMPAIGN,
            symbol="AAPL",
            action="HOLD",
            quantity=0,
            price=155.0,
            reasoning="Defensive stance following strategic directive",
            agent_name="meta_agent",
            conviction=7.0,
            computation_mode="hybrid",
            metadata={
                'strategic_constraint': 'bearish_regime',
                'parent_decision_id': strategic_decision.id,
            }
        )
        
        # === Tactical层：每天决策（正常日） ===
        tactical_time = campaign_time + timedelta(days=1)
        tactical_decisions = []
        
        for day in range(3):  # 3天正常交易
            daily_time = tactical_time + timedelta(days=day)
            decision = DecisionRecord(
                id=create_decision_id("AAPL", daily_time, Timeframe.TACTICAL),
                timestamp=daily_time,
                timeframe=Timeframe.TACTICAL,
                symbol="AAPL",
                action="HOLD",
                quantity=0,
                price=154.0 - day * 0.5,  # 每天小幅下跌
                reasoning=f"Day {day+1}: No strong signals, following campaign stance",
                agent_name="rule_engine",
                conviction=5.0,
                computation_mode="fast",  # 快速模式，无LLM
            )
            tactical_decisions.append(decision)
        
        # === 黑天鹅事件：第4天 ===
        crisis_day = tactical_time + timedelta(days=3)
        detector = EscalationDetector()
        
        # 极端事件：15%单日暴跌 + 重大负面新闻
        market_data = {
            'price_change_1d': -0.15,
            'current_volatility': 0.60,
            'historical_volatility': 0.12,
        }
        
        news_data = {
            'importance': 9.8,
            'sentiment': -0.95,
            'title': 'Major accounting fraud uncovered at Apple',
        }
        
        triggers = detector.detect_all(
            symbol="AAPL",
            market_data=market_data,
            news_data=news_data,
            current_timeframe=Timeframe.TACTICAL,
        )
        
        # 找到黑天鹅触发器
        black_swan_triggers = [t for t in triggers if t.trigger_type == EscalationTriggerType.BLACK_SWAN]
        assert len(black_swan_triggers) > 0
        
        top_trigger = black_swan_triggers[0]
        assert top_trigger.to_timeframe == Timeframe.STRATEGIC  # 直达战略层
        
        # Strategic层紧急响应
        emergency_strategic_decision = DecisionRecord(
            id=create_decision_id("AAPL", crisis_day, Timeframe.STRATEGIC),
            timestamp=crisis_day,
            timeframe=Timeframe.STRATEGIC,
            symbol="AAPL",
            action="SELL",
            quantity=1000,  # 全部清仓
            price=136.0,  # 价格大跌
            reasoning=f"BLACK SWAN EVENT: {top_trigger.reason}. Emergency exit.",
            agent_name="meta_agent",
            conviction=10.0,
            computation_mode="full",
        )
        
        emergency_strategic_decision.mark_as_escalated(
            from_timeframe="tactical",
            trigger=top_trigger.trigger_type.value,
            score=top_trigger.score,
        )
        
        # 验证决策链路
        assert len(tactical_decisions) == 3
        assert all(d.computation_mode == "fast" for d in tactical_decisions)
        assert strategic_decision.computation_mode == "full"
        assert campaign_decision.computation_mode == "hybrid"
        assert emergency_strategic_decision.escalated_from == "tactical"
        assert emergency_strategic_decision.escalation_score == 10.0
        
        print(f"✓ Multi-timeframe cascade validated:")
        print(f"  - Strategic (normal): {strategic_decision.action}")
        print(f"  - Campaign: {campaign_decision.action}")
        print(f"  - Tactical (3 days): {[d.action for d in tactical_decisions]}")
        print(f"  - Strategic (emergency): {emergency_strategic_decision.action}")
    
    def test_signal_caching_workflow(self):
        """
        测试信号缓存工作流程
        
        场景：
        1. 首次计算信号（full模式）
        2. 生成缓存键
        3. 后续相同条件直接使用缓存（fast模式）
        """
        base_time = datetime(2025, 10, 29, 10, 0)
        
        # === 首次计算：Full模式 ===
        first_decision = DecisionRecord(
            id=create_decision_id("AAPL", base_time, Timeframe.TACTICAL),
            timestamp=base_time,
            timeframe=Timeframe.TACTICAL,
            symbol="AAPL",
            action="BUY",
            quantity=100,
            price=150.0,
            reasoning="Technical analysis: RSI oversold, MACD bullish crossover",
            agent_name="meta_agent",
            conviction=8.0,
            computation_mode="full",
        )
        
        # 生成缓存键
        strategy_version = "v1.0.0"
        data_hash = "abc123_20251029_1000"  # 基于数据内容的哈希
        first_decision.set_cache_key(strategy_version, data_hash)
        
        expected_cache_key = f"AAPL_tactical_{strategy_version}_{data_hash}"
        assert first_decision.cache_key == expected_cache_key
        
        # === 后续相同条件：使用缓存 ===
        # 在回测中，如果数据和策略版本相同，可以直接复用信号
        cached_time = base_time + timedelta(hours=1)
        cached_decision = DecisionRecord(
            id=create_decision_id("AAPL", cached_time, Timeframe.TACTICAL),
            timestamp=cached_time,
            timeframe=Timeframe.TACTICAL,
            symbol="AAPL",
            action="BUY",  # 从缓存读取
            quantity=100,
            price=151.0,
            reasoning="[CACHED] Technical analysis: RSI oversold, MACD bullish crossover",
            agent_name="cache_engine",
            conviction=8.0,
            computation_mode="fast",  # 使用缓存，快速模式
            cache_key=expected_cache_key,
        )
        
        # 验证缓存使用
        assert cached_decision.cache_key == first_decision.cache_key
        assert cached_decision.computation_mode == "fast"
        assert "[CACHED]" in cached_decision.reasoning
        
        print(f"✓ Signal caching validated:")
        print(f"  - First computation: {first_decision.computation_mode}")
        print(f"  - Cache key: {first_decision.cache_key}")
        print(f"  - Cached retrieval: {cached_decision.computation_mode}")
    
    def test_serialization_roundtrip_with_all_features(self):
        """
        测试包含所有新功能的完整序列化/反序列化
        
        验证所有新字段在持久化后能正确恢复
        """
        original_time = datetime(2025, 10, 29, 10, 30)
        visible_end = datetime(2025, 10, 29, 10, 25)
        
        # 创建包含所有新功能的决策
        original = DecisionRecord(
            id="comprehensive_test_001",
            timestamp=original_time,
            timeframe=Timeframe.CAMPAIGN,
            symbol="AAPL",
            action="REDUCE",
            quantity=200,
            price=145.0,
            reasoning="Escalated response to market shock",
            agent_name="meta_agent",
            conviction=9.0,
            
            # 新增字段
            visible_data_end=visible_end,
            computation_mode="hybrid",
            cache_key="AAPL_campaign_v1.0.0_xyz789",
        )
        
        # 标记反向传导
        original.mark_as_escalated(
            from_timeframe="tactical",
            trigger="market_shock",
            score=8.7,
        )
        
        # 序列化
        data_dict = original.to_dict()
        
        # 验证所有字段都在字典中
        assert 'visible_data_end' in data_dict
        assert 'computation_mode' in data_dict
        assert 'cache_key' in data_dict
        assert 'escalated_from' in data_dict
        assert 'escalation_trigger' in data_dict
        assert 'escalation_score' in data_dict
        
        # 反序列化
        restored = DecisionRecord.from_dict(data_dict)
        
        # 验证所有字段完整恢复
        assert restored.id == original.id
        assert restored.timestamp == original.timestamp
        assert restored.visible_data_end == original.visible_data_end
        assert restored.computation_mode == original.computation_mode
        assert restored.cache_key == original.cache_key
        assert restored.escalated_from == original.escalated_from
        assert restored.escalation_trigger == original.escalation_trigger
        assert restored.escalation_score == original.escalation_score
        assert restored.timeframe == original.timeframe
        
        # 验证方法依然可用
        assert restored.is_backtest_mode()
        assert restored.validate_data_timestamp(visible_end)
        assert not restored.validate_data_timestamp(original_time)
        
        print(f"✓ Full serialization roundtrip validated:")
        print(f"  - All {len(data_dict)} fields preserved")
        print(f"  - Methods still functional")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
