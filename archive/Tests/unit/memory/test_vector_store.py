"""
Unit tests for Memory/vector_store.py

Tests the ChromaDB wrapper functionality:
- Collection creation and management
- Document storage and retrieval
- Similarity search
- Filtering and cleanup
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from Memory.schemas import (
    Timeframe,
    DecisionRecord,
    MemoryDocument,
    create_decision_id,
)
from Memory.vector_store import VectorStore, create_vector_store


@pytest.fixture
def temp_db_dir():
    """创建临时数据库目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # 清理
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def vector_store(temp_db_dir):
    """创建 VectorStore 实例"""
    return VectorStore(persist_directory=temp_db_dir)


@pytest.fixture
def sample_decisions():
    """创建示例决策记录列表"""
    now = datetime.now()
    decisions = []
    
    # 创建不同的决策
    symbols = ['AAPL', 'TSLA', 'MSFT']
    actions = ['BUY', 'SELL', 'HOLD']
    
    for i, (symbol, action) in enumerate(zip(symbols, actions)):
        decision = DecisionRecord(
            id=create_decision_id(symbol, now + timedelta(minutes=i), Timeframe.TACTICAL),
            timestamp=now + timedelta(minutes=i),
            timeframe=Timeframe.TACTICAL,
            symbol=symbol,
            action=action,
            quantity=100,
            price=100.0 + i * 10,
            reasoning=f"Test reasoning for {symbol}: {action}",
            agent_name="test_agent",
            conviction=7.0 + i,
        )
        decisions.append(decision)
    
    return decisions


class TestVectorStoreBasics:
    """测试 VectorStore 基础功能"""
    
    def test_initialization(self, temp_db_dir):
        """测试初始化"""
        store = VectorStore(persist_directory=temp_db_dir)
        assert store.persist_directory == temp_db_dir
        assert Path(temp_db_dir).exists()
    
    def test_create_vector_store_helper(self, temp_db_dir):
        """测试便捷创建函数"""
        store = create_vector_store(temp_db_dir)
        assert isinstance(store, VectorStore)
    
    def test_get_or_create_collection(self, vector_store):
        """测试创建 collection"""
        collection = vector_store.get_or_create_collection("test_collection")
        assert collection is not None
        assert collection.name == "test_collection"
        
        # 再次获取应该返回相同的 collection
        collection2 = vector_store.get_or_create_collection("test_collection")
        assert collection.name == collection2.name
    
    def test_get_collection_for_timeframe(self, vector_store):
        """测试按时间尺度获取 collection"""
        collection = vector_store.get_collection_for_timeframe(Timeframe.TACTICAL)
        assert collection is not None
        assert "tactical" in collection.name
    
    def test_list_collections(self, vector_store):
        """测试列出 collections"""
        # 创建几个 collections
        vector_store.get_or_create_collection("test1")
        vector_store.get_or_create_collection("test2")
        
        collections = vector_store.list_collections()
        assert len(collections) >= 2
        assert "test1" in collections
        assert "test2" in collections


class TestDocumentOperations:
    """测试文档操作"""
    
    def test_add_document(self, vector_store, sample_decisions):
        """测试添加单个文档"""
        decision = sample_decisions[0]
        doc = MemoryDocument.from_decision(decision)
        
        success = vector_store.add_document("test_collection", doc)
        assert success is True
        
        # 验证文档已添加
        retrieved = vector_store.get_by_id("test_collection", doc.id)
        assert retrieved is not None
        assert retrieved['id'] == doc.id
    
    def test_add_documents_batch(self, vector_store, sample_decisions):
        """测试批量添加文档"""
        docs = [MemoryDocument.from_decision(d) for d in sample_decisions]
        
        count = vector_store.add_documents_batch("test_collection", docs)
        assert count == len(docs)
        
        # 验证所有文档都已添加
        for doc in docs:
            retrieved = vector_store.get_by_id("test_collection", doc.id)
            assert retrieved is not None
    
    def test_get_by_id(self, vector_store, sample_decisions):
        """测试按 ID 获取文档"""
        decision = sample_decisions[0]
        doc = MemoryDocument.from_decision(decision)
        
        vector_store.add_document("test_collection", doc)
        
        retrieved = vector_store.get_by_id("test_collection", doc.id)
        assert retrieved is not None
        assert retrieved['id'] == doc.id
        assert decision.symbol in retrieved['document']
    
    def test_get_by_id_not_found(self, vector_store):
        """测试获取不存在的文档"""
        retrieved = vector_store.get_by_id("test_collection", "nonexistent_id")
        assert retrieved is None
    
    def test_delete_document(self, vector_store, sample_decisions):
        """测试删除文档"""
        decision = sample_decisions[0]
        doc = MemoryDocument.from_decision(decision)
        
        vector_store.add_document("test_collection", doc)
        
        # 删除
        success = vector_store.delete_document("test_collection", doc.id)
        assert success is True
        
        # 验证已删除
        retrieved = vector_store.get_by_id("test_collection", doc.id)
        assert retrieved is None


class TestQuerying:
    """测试查询功能"""
    
    def test_query(self, vector_store, sample_decisions):
        """测试语义查询"""
        # 添加文档
        docs = [MemoryDocument.from_decision(d) for d in sample_decisions]
        vector_store.add_documents_batch("test_collection", docs)
        
        # 查询
        results = vector_store.query(
            collection_name="test_collection",
            query_text="Buy AAPL",
            n_results=2
        )
        
        assert len(results['ids']) <= 2
        assert len(results['ids']) == len(results['documents'])
        assert len(results['ids']) == len(results['metadatas'])
    
    def test_query_with_filter(self, vector_store, sample_decisions):
        """测试带过滤条件的查询"""
        docs = [MemoryDocument.from_decision(d) for d in sample_decisions]
        vector_store.add_documents_batch("test_collection", docs)
        
        # 只查询 AAPL 的决策
        results = vector_store.query(
            collection_name="test_collection",
            query_text="trading decision",
            n_results=5,
            where={"symbol": "AAPL"}
        )
        
        # 验证结果
        for metadata in results['metadatas']:
            assert metadata['symbol'] == "AAPL"
    
    def test_query_by_timeframe(self, vector_store, sample_decisions):
        """测试按时间尺度查询"""
        # 添加到 tactical collection
        docs = [MemoryDocument.from_decision(d) for d in sample_decisions]
        collection_name = f"{Timeframe.TACTICAL.display_name}_memory"
        vector_store.add_documents_batch(collection_name, docs)
        
        # 查询
        results = vector_store.query_by_timeframe(
            timeframe=Timeframe.TACTICAL,
            query_text="AAPL",
            n_results=5,
            symbol="AAPL"
        )
        
        assert len(results['ids']) > 0


class TestCleanup:
    """测试清理功能"""
    
    def test_delete_by_filter(self, vector_store, sample_decisions):
        """测试按条件删除"""
        docs = [MemoryDocument.from_decision(d) for d in sample_decisions]
        vector_store.add_documents_batch("test_collection", docs)
        
        # 删除所有 AAPL 的记录
        count = vector_store.delete_by_filter(
            collection_name="test_collection",
            where={"symbol": "AAPL"}
        )
        
        assert count >= 1
        
        # 验证 AAPL 的记录已删除
        results = vector_store.query(
            collection_name="test_collection",
            query_text="AAPL",
            n_results=10,
            where={"symbol": "AAPL"}
        )
        assert len(results['ids']) == 0
    
    def test_cleanup_old_data(self, vector_store):
        """测试清理旧数据"""
        # 创建旧的和新的决策
        old_time = datetime.now() - timedelta(days=30)
        new_time = datetime.now()
        
        old_decision = DecisionRecord(
            id=create_decision_id("AAPL", old_time, Timeframe.TACTICAL),
            timestamp=old_time,
            timeframe=Timeframe.TACTICAL,
            symbol="AAPL",
            action="BUY",
            quantity=100,
            price=100.0,
            reasoning="Old decision",
            agent_name="test_agent",
            conviction=7.0,
        )
        
        new_decision = DecisionRecord(
            id=create_decision_id("TSLA", new_time, Timeframe.TACTICAL),
            timestamp=new_time,
            timeframe=Timeframe.TACTICAL,
            symbol="TSLA",
            action="BUY",
            quantity=100,
            price=200.0,
            reasoning="New decision",
            agent_name="test_agent",
            conviction=8.0,
        )
        
        # 添加文档
        old_doc = MemoryDocument.from_decision(old_decision)
        new_doc = MemoryDocument.from_decision(new_decision)
        vector_store.add_document("test_collection", old_doc)
        vector_store.add_document("test_collection", new_doc)
        
        # 清理 7 天前的数据
        cutoff_time = datetime.now() - timedelta(days=7)
        deleted_count = vector_store.cleanup_old_data(
            collection_name="test_collection",
            before_timestamp=cutoff_time,
            keep_outcomes=False
        )
        
        assert deleted_count >= 1
        
        # 验证旧数据被删除
        old_retrieved = vector_store.get_by_id("test_collection", old_doc.id)
        assert old_retrieved is None
        
        # 验证新数据仍然存在
        new_retrieved = vector_store.get_by_id("test_collection", new_doc.id)
        assert new_retrieved is not None
    
    def test_reset_collection(self, vector_store, sample_decisions):
        """测试重置 collection"""
        # 添加数据
        docs = [MemoryDocument.from_decision(d) for d in sample_decisions]
        vector_store.add_documents_batch("test_collection", docs)
        
        # 重置
        success = vector_store.reset_collection("test_collection")
        assert success is True
        
        # 验证数据已清空
        collection = vector_store.get_or_create_collection("test_collection")
        assert collection.count() == 0


class TestStatistics:
    """测试统计功能"""
    
    def test_get_collection_stats(self, vector_store, sample_decisions):
        """测试获取统计信息"""
        # 添加数据
        docs = [MemoryDocument.from_decision(d) for d in sample_decisions]
        vector_store.add_documents_batch("test_collection", docs)
        
        # 获取统计
        stats = vector_store.get_collection_stats("test_collection")
        
        assert 'name' in stats
        assert 'total_documents' in stats
        assert stats['total_documents'] == len(docs)
        assert 'action_distribution' in stats
        assert 'top_symbols' in stats
