"""
Unit tests for Memory/sql_store.py

Tests the SQLite database wrapper functionality:
- Decision record CRUD operations
- Constraints storage and retrieval
- Performance statistics
- Data cleanup
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta

from Memory.schemas import (
    Timeframe,
    DecisionRecord,
    HierarchicalConstraints,
    create_decision_id,
)
from Memory.sql_store import SQLStore, create_sql_store


@pytest.fixture
def temp_db_path():
    """创建临时数据库路径"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    # 清理
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def sql_store(temp_db_path):
    """创建 SQLStore 实例"""
    return SQLStore(db_path=temp_db_path)


@pytest.fixture
def sample_decision():
    """创建示例决策"""
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


@pytest.fixture
def multiple_decisions():
    """创建多个决策"""
    now = datetime.now()
    decisions = []
    
    for i, (symbol, action) in enumerate([('AAPL', 'BUY'), ('TSLA', 'SELL'), ('MSFT', 'HOLD')]):
        decision = DecisionRecord(
            id=create_decision_id(symbol, now + timedelta(hours=i), Timeframe.TACTICAL),
            timestamp=now + timedelta(hours=i),
            timeframe=Timeframe.TACTICAL,
            symbol=symbol,
            action=action,
            quantity=100,
            price=100.0 + i * 50,
            reasoning=f"Test reasoning for {symbol}",
            agent_name="test_agent",
            conviction=7.0 + i,
        )
        decisions.append(decision)
    
    return decisions


class TestSQLStoreBasics:
    """测试 SQLStore 基础功能"""
    
    def test_initialization(self, temp_db_path):
        """测试初始化"""
        store = SQLStore(db_path=temp_db_path)
        assert os.path.exists(temp_db_path)
        assert store.db_path == temp_db_path
    
    def test_create_sql_store_helper(self, temp_db_path):
        """测试便捷创建函数"""
        store = create_sql_store(temp_db_path)
        assert isinstance(store, SQLStore)
    
    def test_tables_created(self, sql_store):
        """测试表是否创建"""
        with sql_store._get_connection() as conn:
            cursor = conn.cursor()
            
            # 检查 decisions 表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='decisions'")
            assert cursor.fetchone() is not None
            
            # 检查 constraints 表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='constraints'")
            assert cursor.fetchone() is not None


class TestDecisionOperations:
    """测试决策记录操作"""
    
    def test_save_decision(self, sql_store, sample_decision):
        """测试保存决策"""
        success = sql_store.save_decision(sample_decision)
        assert success is True
        
        # 验证已保存
        retrieved = sql_store.get_decision(sample_decision.id)
        assert retrieved is not None
        assert retrieved.symbol == sample_decision.symbol
    
    def test_get_decision(self, sql_store, sample_decision):
        """测试获取决策"""
        sql_store.save_decision(sample_decision)
        
        retrieved = sql_store.get_decision(sample_decision.id)
        assert retrieved is not None
        assert retrieved.id == sample_decision.id
        assert retrieved.symbol == sample_decision.symbol
        assert retrieved.action == sample_decision.action
        assert retrieved.conviction == sample_decision.conviction
    
    def test_get_decision_not_found(self, sql_store):
        """测试获取不存在的决策"""
        retrieved = sql_store.get_decision("nonexistent_id")
        assert retrieved is None
    
    def test_update_decision(self, sql_store, sample_decision):
        """测试更新决策"""
        # 保存初始决策
        sql_store.save_decision(sample_decision)
        
        # 更新执行信息
        sample_decision.update_execution(176.00, datetime.now())
        sql_store.save_decision(sample_decision)
        
        # 验证更新
        retrieved = sql_store.get_decision(sample_decision.id)
        assert retrieved.executed is True
        assert retrieved.execution_price == 176.00
    
    def test_update_decision_outcome(self, sql_store, sample_decision):
        """测试更新决策结果"""
        # 保存并更新执行
        sample_decision.update_execution(176.00, datetime.now())
        sql_store.save_decision(sample_decision)
        
        # 更新结果
        exit_time = datetime.now() + timedelta(days=3)
        success = sql_store.update_decision_outcome(
            sample_decision.id,
            182.50,
            exit_time
        )
        assert success is True
        
        # 验证结果
        retrieved = sql_store.get_decision(sample_decision.id)
        assert retrieved.exit_price == 182.50
        assert retrieved.pnl is not None
        assert retrieved.outcome in ['success', 'failure', 'neutral']
    
    def test_delete_decision(self, sql_store, sample_decision):
        """测试删除决策"""
        sql_store.save_decision(sample_decision)
        
        success = sql_store.delete_decision(sample_decision.id)
        assert success is True
        
        # 验证已删除
        retrieved = sql_store.get_decision(sample_decision.id)
        assert retrieved is None


class TestQuerying:
    """测试查询功能"""
    
    def test_query_decisions_all(self, sql_store, multiple_decisions):
        """测试查询所有决策"""
        for decision in multiple_decisions:
            sql_store.save_decision(decision)
        
        results = sql_store.query_decisions()
        assert len(results) == len(multiple_decisions)
    
    def test_query_by_symbol(self, sql_store, multiple_decisions):
        """测试按股票代码查询"""
        for decision in multiple_decisions:
            sql_store.save_decision(decision)
        
        results = sql_store.query_decisions(symbol="AAPL")
        assert len(results) == 1
        assert all(d.symbol == "AAPL" for d in results)
    
    def test_query_by_timeframe(self, sql_store, multiple_decisions):
        """测试按时间尺度查询"""
        for decision in multiple_decisions:
            sql_store.save_decision(decision)
        
        results = sql_store.query_decisions(timeframe=Timeframe.TACTICAL)
        assert len(results) == len(multiple_decisions)
    
    def test_query_by_action(self, sql_store, multiple_decisions):
        """测试按动作查询"""
        for decision in multiple_decisions:
            sql_store.save_decision(decision)
        
        results = sql_store.query_decisions(action="BUY")
        assert len(results) == 1
        assert all(d.action == "BUY" for d in results)
    
    def test_query_by_time_range(self, sql_store, multiple_decisions):
        """测试按时间范围查询"""
        for decision in multiple_decisions:
            sql_store.save_decision(decision)
        
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now() + timedelta(hours=1)
        
        results = sql_store.query_decisions(
            start_time=start_time,
            end_time=end_time
        )
        assert len(results) >= 1
    
    def test_query_with_limit(self, sql_store, multiple_decisions):
        """测试限制查询数量"""
        for decision in multiple_decisions:
            sql_store.save_decision(decision)
        
        results = sql_store.query_decisions(limit=2)
        assert len(results) == 2
    
    def test_get_recent_decisions(self, sql_store, multiple_decisions):
        """测试获取最近决策"""
        for decision in multiple_decisions:
            sql_store.save_decision(decision)
        
        recent = sql_store.get_recent_decisions(limit=2)
        assert len(recent) == 2


class TestConstraints:
    """测试约束条件操作"""
    
    def test_save_constraints(self, sql_store):
        """测试保存约束"""
        constraints = HierarchicalConstraints(
            strategic={'market_regime': 'bull', 'risk_budget': 1.0},
            campaign={'sector_allocation': {'tech': 0.4}},
        )
        
        success = sql_store.save_constraints(Timeframe.STRATEGIC, constraints)
        assert success is True
    
    def test_get_constraints(self, sql_store):
        """测试获取约束"""
        constraints = HierarchicalConstraints(
            strategic={'market_regime': 'bear', 'risk_budget': 0.5},
        )
        
        sql_store.save_constraints(Timeframe.STRATEGIC, constraints)
        
        retrieved = sql_store.get_constraints(Timeframe.STRATEGIC)
        assert retrieved is not None
        assert retrieved.get_market_regime() == 'bear'
        assert retrieved.get_risk_budget() == 0.5
    
    def test_get_constraints_not_found(self, sql_store):
        """测试获取不存在的约束"""
        retrieved = sql_store.get_constraints(Timeframe.TACTICAL)
        assert retrieved is None
    
    def test_save_symbol_specific_constraints(self, sql_store):
        """测试保存特定股票的约束"""
        constraints = HierarchicalConstraints(
            tactical={'max_position': 1000},
        )
        
        success = sql_store.save_constraints(
            Timeframe.TACTICAL,
            constraints,
            symbol="AAPL"
        )
        assert success is True
        
        retrieved = sql_store.get_constraints(Timeframe.TACTICAL, symbol="AAPL")
        assert retrieved is not None


class TestStatistics:
    """测试统计功能"""
    
    def test_get_performance_stats(self, sql_store):
        """测试获取性能统计"""
        # 创建带结果的决策
        now = datetime.now()
        
        # 成功的交易
        decision1 = DecisionRecord(
            id=create_decision_id("AAPL", now, Timeframe.TACTICAL),
            timestamp=now,
            timeframe=Timeframe.TACTICAL,
            symbol="AAPL",
            action="BUY",
            quantity=100,
            price=100.0,
            reasoning="Test",
            agent_name="test_agent",
            conviction=8.0,
        )
        decision1.update_execution(100.0, now)
        decision1.update_outcome(110.0, now + timedelta(days=1))
        sql_store.save_decision(decision1)
        
        # 失败的交易
        decision2 = DecisionRecord(
            id=create_decision_id("TSLA", now + timedelta(hours=1), Timeframe.TACTICAL),
            timestamp=now + timedelta(hours=1),
            timeframe=Timeframe.TACTICAL,
            symbol="TSLA",
            action="BUY",
            quantity=100,
            price=200.0,
            reasoning="Test",
            agent_name="test_agent",
            conviction=7.0,
        )
        decision2.update_execution(200.0, now)
        decision2.update_outcome(190.0, now + timedelta(days=1))
        sql_store.save_decision(decision2)
        
        # 获取统计
        stats = sql_store.get_performance_stats()
        
        assert stats['total_trades'] == 2
        assert stats['wins'] == 1
        assert stats['losses'] == 1
        assert stats['win_rate'] == 0.5
        # total_pnl 应该是赢的利润减去亏的损失
        assert 'total_pnl' in stats
    
    def test_get_action_distribution(self, sql_store, multiple_decisions):
        """测试获取动作分布"""
        for decision in multiple_decisions:
            sql_store.save_decision(decision)
        
        distribution = sql_store.get_action_distribution()
        
        assert 'BUY' in distribution
        assert 'SELL' in distribution
        assert 'HOLD' in distribution
        assert distribution['BUY'] == 1
        assert distribution['SELL'] == 1
        assert distribution['HOLD'] == 1
    
    def test_get_database_stats(self, sql_store, multiple_decisions):
        """测试获取数据库统计"""
        for decision in multiple_decisions:
            sql_store.save_decision(decision)
        
        stats = sql_store.get_database_stats()
        
        assert 'db_path' in stats
        assert 'decisions_count' in stats
        assert stats['decisions_count'] == len(multiple_decisions)


class TestCleanup:
    """测试清理功能"""
    
    def test_cleanup_old_decisions(self, sql_store):
        """测试清理旧决策"""
        now = datetime.now()
        
        # 旧决策
        old_decision = DecisionRecord(
            id=create_decision_id("AAPL", now - timedelta(days=30), Timeframe.TACTICAL),
            timestamp=now - timedelta(days=30),
            timeframe=Timeframe.TACTICAL,
            symbol="AAPL",
            action="BUY",
            quantity=100,
            price=100.0,
            reasoning="Old",
            agent_name="test_agent",
            conviction=7.0,
        )
        
        # 新决策
        new_decision = DecisionRecord(
            id=create_decision_id("TSLA", now, Timeframe.TACTICAL),
            timestamp=now,
            timeframe=Timeframe.TACTICAL,
            symbol="TSLA",
            action="BUY",
            quantity=100,
            price=200.0,
            reasoning="New",
            agent_name="test_agent",
            conviction=8.0,
        )
        
        sql_store.save_decision(old_decision)
        sql_store.save_decision(new_decision)
        
        # 清理7天前的数据
        cutoff = now - timedelta(days=7)
        deleted_count = sql_store.cleanup_old_decisions(cutoff, keep_outcomes=False)
        
        assert deleted_count == 1
        
        # 验证旧决策被删除
        assert sql_store.get_decision(old_decision.id) is None
        
        # 验证新决策仍存在
        assert sql_store.get_decision(new_decision.id) is not None
    
    def test_vacuum(self, sql_store, multiple_decisions):
        """测试数据库压缩"""
        for decision in multiple_decisions:
            sql_store.save_decision(decision)
        
        # 应该不抛出异常
        sql_store.vacuum()
