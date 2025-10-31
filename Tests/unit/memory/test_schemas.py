"""
Unit tests for Memory/schemas.py

Tests all core data structures:
- Timeframe enum
- DecisionRecord dataclass
- MemoryDocument dataclass
- HierarchicalConstraints dataclass
"""

import pytest
from datetime import datetime, timedelta
from Memory.schemas import (
    Timeframe,
    DecisionRecord,
    MemoryDocument,
    HierarchicalConstraints,
    create_decision_id,
    create_memory_id,
)


class TestTimeframe:
    """测试 Timeframe 枚举"""
    
    def test_timeframe_ordering(self):
        """测试时间尺度排序"""
        ordered = Timeframe.get_all_ordered()
        assert len(ordered) == 5
        assert ordered[0] == Timeframe.REALTIME
        assert ordered[-1] == Timeframe.STRATEGIC
    
    def test_timeframe_comparison(self):
        """测试时间尺度比较"""
        assert Timeframe.TACTICAL < Timeframe.CAMPAIGN
        assert Timeframe.STRATEGIC > Timeframe.EXECUTION
        assert Timeframe.TACTICAL <= Timeframe.TACTICAL
        assert Timeframe.CAMPAIGN >= Timeframe.TACTICAL
    
    def test_get_higher_timeframes(self):
        """测试获取更高时间尺度"""
        higher = Timeframe.get_higher_timeframes(Timeframe.TACTICAL)
        assert len(higher) == 2
        assert Timeframe.CAMPAIGN in higher
        assert Timeframe.STRATEGIC in higher
    
    def test_get_lower_timeframes(self):
        """测试获取更低时间尺度"""
        lower = Timeframe.get_lower_timeframes(Timeframe.TACTICAL)
        assert len(lower) == 2
        assert Timeframe.REALTIME in lower
        assert Timeframe.EXECUTION in lower
    
    def test_from_string(self):
        """测试从字符串创建"""
        tf = Timeframe.from_string("tactical")
        assert tf == Timeframe.TACTICAL
        
        tf = Timeframe.from_string("STRATEGIC")
        assert tf == Timeframe.STRATEGIC
    
    def test_from_string_invalid(self):
        """测试无效字符串"""
        with pytest.raises(ValueError):
            Timeframe.from_string("invalid")
    
    def test_characteristic_periods(self):
        """测试特征周期"""
        assert Timeframe.REALTIME.seconds == 300
        assert Timeframe.EXECUTION.seconds == 3600
        assert Timeframe.TACTICAL.seconds == 86400
        assert Timeframe.CAMPAIGN.seconds == 604800
        assert Timeframe.STRATEGIC.seconds == 2592000


class TestDecisionRecord:
    """测试 DecisionRecord 数据类"""
    
    @pytest.fixture
    def sample_decision(self):
        """创建示例决策记录"""
        now = datetime.now()
        return DecisionRecord(
            id=create_decision_id("AAPL", now, Timeframe.TACTICAL),
            timestamp=now,
            timeframe=Timeframe.TACTICAL,
            symbol="AAPL",
            action="BUY",
            quantity=100,
            price=175.50,
            reasoning="RSI超卖，价格突破50日均线",
            agent_name="technical_agent",
            conviction=8.5,
            market_regime="bull",
            technical_signals={'rsi': 28.5, 'macd': 2.3},
        )
    
    def test_decision_creation(self, sample_decision):
        """测试决策记录创建"""
        assert sample_decision.symbol == "AAPL"
        assert sample_decision.action == "BUY"
        assert sample_decision.quantity == 100
        assert sample_decision.price == 175.50
        assert sample_decision.conviction == 8.5
    
    def test_decision_to_text(self, sample_decision):
        """测试转换为文本"""
        text = sample_decision.to_text()
        assert "AAPL" in text
        assert "BUY" in text
        assert "tactical" in text
        assert "bull" in text
    
    def test_decision_update_execution(self, sample_decision):
        """测试更新执行信息"""
        execution_time = datetime.now()
        sample_decision.update_execution(175.75, execution_time)
        
        assert sample_decision.executed is True
        assert sample_decision.execution_price == 175.75
        assert sample_decision.execution_time == execution_time
    
    def test_decision_update_outcome(self, sample_decision):
        """测试更新结果信息"""
        # 先更新执行
        execution_time = datetime.now()
        sample_decision.update_execution(175.75, execution_time)
        
        # 再更新结果
        exit_time = execution_time + timedelta(days=5)
        sample_decision.update_outcome(182.30, exit_time)
        
        assert sample_decision.exit_price == 182.30
        assert sample_decision.hold_duration_days == 5
        assert sample_decision.pnl > 0  # 盈利
        assert sample_decision.pnl_percent > 0.02  # >2%
        assert sample_decision.outcome == 'success'
    
    def test_decision_serialization(self, sample_decision):
        """测试序列化和反序列化"""
        # 转换为字典
        decision_dict = sample_decision.to_dict()
        assert isinstance(decision_dict, dict)
        assert decision_dict['symbol'] == "AAPL"
        assert decision_dict['timeframe'] == "tactical"
        
        # 从字典恢复
        restored = DecisionRecord.from_dict(decision_dict)
        assert restored.symbol == sample_decision.symbol
        assert restored.action == sample_decision.action
        assert restored.timestamp == sample_decision.timestamp
        assert restored.timeframe == sample_decision.timeframe


class TestMemoryDocument:
    """测试 MemoryDocument 数据类"""
    
    @pytest.fixture
    def sample_decision(self):
        """创建示例决策"""
        now = datetime.now()
        return DecisionRecord(
            id=create_decision_id("TSLA", now, Timeframe.TACTICAL),
            timestamp=now,
            timeframe=Timeframe.TACTICAL,
            symbol="TSLA",
            action="SELL",
            quantity=50,
            price=245.80,
            reasoning="技术指标超买",
            agent_name="technical_agent",
            conviction=7.5,
            market_regime="neutral",
        )
    
    def test_from_decision(self, sample_decision):
        """测试从决策创建文档"""
        doc = MemoryDocument.from_decision(sample_decision)
        
        assert doc.id == sample_decision.id
        assert "TSLA" in doc.text
        assert "SELL" in doc.text
        assert doc.metadata['symbol'] == "TSLA"
        assert doc.metadata['action'] == "SELL"
        assert doc.metadata['timeframe'] == "tactical"
    
    def test_to_chroma_format(self, sample_decision):
        """测试转换为ChromaDB格式"""
        doc = MemoryDocument.from_decision(sample_decision)
        chroma_format = doc.to_chroma_format()
        
        assert 'id' in chroma_format
        assert 'document' in chroma_format
        assert 'metadata' in chroma_format
        assert chroma_format['id'] == doc.id
        assert chroma_format['document'] == doc.text


class TestHierarchicalConstraints:
    """测试 HierarchicalConstraints 数据类"""
    
    @pytest.fixture
    def sample_constraints(self):
        """创建示例约束"""
        return HierarchicalConstraints(
            strategic={
                'market_regime': 'bear',
                'risk_budget': 0.5,
                'max_exposure': 0.6,
                'forbidden_sectors': ['crypto'],
            },
            campaign={
                'sector_allocation': {
                    'tech': 0.3,
                    'healthcare': 0.4,
                },
            },
            tactical={
                'win_rate': 0.65,
            }
        )
    
    def test_get_market_regime(self, sample_constraints):
        """测试获取市场状态"""
        assert sample_constraints.get_market_regime() == 'bear'
    
    def test_get_risk_budget(self, sample_constraints):
        """测试获取风险预算"""
        assert sample_constraints.get_risk_budget() == 0.5
    
    def test_can_open_long(self, sample_constraints):
        """测试是否允许做多"""
        assert sample_constraints.can_open_long() is False  # bear market
    
    def test_can_open_short(self, sample_constraints):
        """测试是否允许做空"""
        assert sample_constraints.can_open_short() is True  # bear market
    
    def test_is_sector_forbidden(self, sample_constraints):
        """测试行业是否被禁止"""
        assert sample_constraints.is_sector_forbidden('crypto') is True
        assert sample_constraints.is_sector_forbidden('tech') is False
    
    def test_get_sector_allocation(self, sample_constraints):
        """测试获取行业配置"""
        assert sample_constraints.get_sector_allocation('tech') == 0.3
        assert sample_constraints.get_sector_allocation('healthcare') == 0.4
        assert sample_constraints.get_sector_allocation('unknown') == 0.0
    
    def test_serialization(self, sample_constraints):
        """测试序列化"""
        # 转换为JSON
        json_str = sample_constraints.to_json()
        assert isinstance(json_str, str)
        
        # 从JSON恢复
        restored = HierarchicalConstraints.from_json(json_str)
        assert restored.get_market_regime() == 'bear'
        assert restored.get_risk_budget() == 0.5


class TestHelperFunctions:
    """测试辅助函数"""
    
    def test_create_decision_id(self):
        """测试创建决策ID"""
        now = datetime.now()
        decision_id = create_decision_id("NVDA", now, Timeframe.TACTICAL)
        
        assert "decision" in decision_id
        assert "NVDA" in decision_id
        assert "tactical" in decision_id
    
    def test_create_memory_id(self):
        """测试创建记忆ID"""
        now = datetime.now()
        memory_id = create_memory_id("news", "GOOGL", now)
        
        assert "news" in memory_id
        assert "GOOGL" in memory_id
    
    def test_decision_id_uniqueness(self):
        """测试决策ID唯一性"""
        now = datetime.now()
        id1 = create_decision_id("AAPL", now, Timeframe.TACTICAL)
        id2 = create_decision_id("AAPL", now + timedelta(seconds=1), Timeframe.TACTICAL)
        
        assert id1 != id2  # 不同时间应该产生不同ID
