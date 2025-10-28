"""
Vector Store - ChromaDB 向量数据库 wrapper

提供向量数据库的统一接口，用于：
1. 存储决策记忆的 embedding 向量
2. 语义相似度搜索
3. 分时间尺度的记忆管理
4. 元数据过滤和查询

使用 ChromaDB 作为底层实现，支持本地持久化。
"""

import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.api.models.Collection import Collection
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not installed. Vector store will not be available.")

from .schemas import MemoryDocument, Timeframe


class VectorStore:
    """
    向量数据库 wrapper
    
    封装 ChromaDB 的操作，提供简单的 API：
    - 创建/获取 collection
    - 添加文档（自动生成 embedding）
    - 查询相似文档
    - 删除文档
    - 清理旧数据
    
    每个时间尺度使用独立的 collection。
    """
    
    def __init__(self, 
                 persist_directory: str,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 distance_metric: str = "cosine"):
        """
        初始化向量数据库
        
        Args:
            persist_directory: 持久化目录路径
            embedding_model: embedding 模型名称（用于 sentence-transformers）
            distance_metric: 距离度量方式 ("cosine", "l2", "ip")
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Please install it with: "
                "pip install chromadb"
            )
        
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.distance_metric = distance_metric
        
        # 确保持久化目录存在
        os.makedirs(persist_directory, exist_ok=True)
        
        # 初始化 embedding function
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # 初始化 ChromaDB 客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # 缓存已创建的 collections
        self._collections: Dict[str, Collection] = {}
        
        logging.info(f"VectorStore initialized at {persist_directory}")
    
    def get_or_create_collection(self, 
                                  name: str,
                                  metadata: Optional[Dict[str, Any]] = None) -> Collection:
        """
        获取或创建一个 collection
        
        Args:
            name: collection 名称
            metadata: collection 元数据
            
        Returns:
            ChromaDB Collection 对象
        """
        # 从缓存中获取
        if name in self._collections:
            return self._collections[name]
        
        # 创建或获取
        try:
            # ChromaDB 要求 metadata 不能为空，至少要有一个字段
            if not metadata:
                metadata = {"created_at": datetime.now().isoformat()}
            
            collection = self.client.get_or_create_collection(
                name=name,
                metadata=metadata,
                embedding_function=self.embedding_function,  # 使用默认 embedding
            )
            self._collections[name] = collection
            logging.info(f"Collection '{name}' ready (size: {collection.count()})")
            return collection
            
        except Exception as e:
            logging.error(f"Failed to get/create collection '{name}': {e}")
            raise
    
    def get_collection_for_timeframe(self, timeframe: Timeframe) -> Collection:
        """
        获取指定时间尺度的 collection
        
        Args:
            timeframe: 时间尺度
            
        Returns:
            对应的 Collection
        """
        collection_name = f"{timeframe.display_name}_memory"
        metadata = {
            "timeframe": timeframe.display_name,
            "characteristic_period_seconds": timeframe.characteristic_period_seconds,
        }
        return self.get_or_create_collection(collection_name, metadata)
    
    def add_document(self,
                    collection_name: str,
                    document: MemoryDocument) -> bool:
        """
        添加单个文档到 collection
        
        Args:
            collection_name: collection 名称
            document: 要添加的文档
            
        Returns:
            是否成功添加
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # 准备数据
            chroma_format = document.to_chroma_format()
            
            # 添加到 ChromaDB
            collection.add(
                ids=[chroma_format['id']],
                documents=[chroma_format['document']],
                metadatas=[chroma_format['metadata']],
            )
            
            logging.debug(f"Added document {document.id} to {collection_name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to add document to {collection_name}: {e}")
            return False
    
    def add_documents_batch(self,
                           collection_name: str,
                           documents: List[MemoryDocument]) -> int:
        """
        批量添加文档
        
        Args:
            collection_name: collection 名称
            documents: 文档列表
            
        Returns:
            成功添加的文档数量
        """
        if not documents:
            return 0
        
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # 准备批量数据
            ids = []
            docs = []
            metadatas = []
            
            for doc in documents:
                chroma_format = doc.to_chroma_format()
                ids.append(chroma_format['id'])
                docs.append(chroma_format['document'])
                metadatas.append(chroma_format['metadata'])
            
            # 批量添加
            collection.add(
                ids=ids,
                documents=docs,
                metadatas=metadatas,
            )
            
            logging.info(f"Added {len(documents)} documents to {collection_name}")
            return len(documents)
            
        except Exception as e:
            logging.error(f"Failed to batch add documents: {e}")
            return 0
    
    def query(self,
             collection_name: str,
             query_text: str,
             n_results: int = 5,
             where: Optional[Dict[str, Any]] = None,
             where_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        查询相似文档
        
        Args:
            collection_name: collection 名称
            query_text: 查询文本
            n_results: 返回结果数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            
        Returns:
            查询结果字典，包含 ids, documents, metadatas, distances
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                where_document=where_document,
            )
            
            # 展平结果（因为 query_texts 只有一个元素）
            if results['ids']:
                return {
                    'ids': results['ids'][0],
                    'documents': results['documents'][0],
                    'metadatas': results['metadatas'][0],
                    'distances': results['distances'][0],
                }
            else:
                return {
                    'ids': [],
                    'documents': [],
                    'metadatas': [],
                    'distances': [],
                }
            
        except Exception as e:
            logging.error(f"Query failed for {collection_name}: {e}")
            return {
                'ids': [],
                'documents': [],
                'metadatas': [],
                'distances': [],
            }
    
    def query_by_timeframe(self,
                          timeframe: Timeframe,
                          query_text: str,
                          n_results: int = 5,
                          symbol: Optional[str] = None,
                          action: Optional[str] = None) -> Dict[str, Any]:
        """
        按时间尺度查询
        
        Args:
            timeframe: 时间尺度
            query_text: 查询文本
            n_results: 返回结果数量
            symbol: 可选，按股票代码过滤
            action: 可选，按动作过滤
            
        Returns:
            查询结果
        """
        collection_name = f"{timeframe.display_name}_memory"
        
        # 构建过滤条件
        where = {}
        if symbol:
            where['symbol'] = symbol
        if action:
            where['action'] = action
        
        return self.query(
            collection_name=collection_name,
            query_text=query_text,
            n_results=n_results,
            where=where if where else None,
        )
    
    def get_by_id(self,
                 collection_name: str,
                 document_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取文档
        
        Args:
            collection_name: collection 名称
            document_id: 文档 ID
            
        Returns:
            文档数据，如果不存在返回 None
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            results = collection.get(
                ids=[document_id],
                include=['documents', 'metadatas']
            )
            
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0],
                }
            return None
            
        except Exception as e:
            logging.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def delete_document(self,
                       collection_name: str,
                       document_id: str) -> bool:
        """
        删除文档
        
        Args:
            collection_name: collection 名称
            document_id: 文档 ID
            
        Returns:
            是否成功删除
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            collection.delete(ids=[document_id])
            logging.debug(f"Deleted document {document_id} from {collection_name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def delete_by_filter(self,
                        collection_name: str,
                        where: Dict[str, Any]) -> int:
        """
        按条件批量删除
        
        Args:
            collection_name: collection 名称
            where: 过滤条件
            
        Returns:
            删除的文档数量
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # 先查询满足条件的文档
            results = collection.get(where=where)
            count = len(results['ids'])
            
            if count > 0:
                collection.delete(where=where)
                logging.info(f"Deleted {count} documents from {collection_name}")
            
            return count
            
        except Exception as e:
            logging.error(f"Failed to delete by filter: {e}")
            return 0
    
    def cleanup_old_data(self,
                        collection_name: str,
                        before_timestamp: datetime,
                        keep_outcomes: bool = True) -> int:
        """
        清理旧数据
        
        Args:
            collection_name: collection 名称
            before_timestamp: 删除此时间之前的数据
            keep_outcomes: 是否保留有结果的决策（成功/失败案例）
            
        Returns:
            删除的文档数量
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # 获取所有文档
            all_docs = collection.get(include=['metadatas'])
            
            ids_to_delete = []
            for doc_id, metadata in zip(all_docs['ids'], all_docs['metadatas']):
                # 解析时间戳
                try:
                    doc_time = datetime.fromisoformat(metadata.get('timestamp', ''))
                    
                    # 如果是旧数据
                    if doc_time < before_timestamp:
                        # 如果要保留有结果的，检查 outcome
                        if keep_outcomes and metadata.get('outcome') in ['success', 'failure']:
                            continue
                        ids_to_delete.append(doc_id)
                        
                except (ValueError, TypeError):
                    # 无效的时间戳，跳过
                    continue
            
            # 批量删除
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logging.info(f"Cleaned up {len(ids_to_delete)} old documents from {collection_name}")
            
            return len(ids_to_delete)
            
        except Exception as e:
            logging.error(f"Failed to cleanup old data: {e}")
            return 0
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        获取 collection 统计信息
        
        Args:
            collection_name: collection 名称
            
        Returns:
            统计信息字典
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            count = collection.count()
            
            # 获取一些示例来分析
            sample = collection.peek(limit=min(100, count))
            
            # 统计不同 action 的数量
            action_counts = {}
            symbol_counts = {}
            
            for metadata in sample.get('metadatas', []):
                action = metadata.get('action', 'unknown')
                symbol = metadata.get('symbol', 'unknown')
                
                action_counts[action] = action_counts.get(action, 0) + 1
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            return {
                'name': collection_name,
                'total_documents': count,
                'sample_size': len(sample.get('ids', [])),
                'action_distribution': action_counts,
                'top_symbols': dict(sorted(symbol_counts.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True)[:5]),
            }
            
        except Exception as e:
            logging.error(f"Failed to get stats for {collection_name}: {e}")
            return {
                'name': collection_name,
                'error': str(e),
            }
    
    def reset_collection(self, collection_name: str) -> bool:
        """
        重置 collection（删除所有数据）
        
        Args:
            collection_name: collection 名称
            
        Returns:
            是否成功重置
        """
        try:
            self.client.delete_collection(name=collection_name)
            
            # 从缓存中移除
            if collection_name in self._collections:
                del self._collections[collection_name]
            
            logging.info(f"Reset collection {collection_name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to reset collection {collection_name}: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """
        列出所有 collections
        
        Returns:
            collection 名称列表
        """
        try:
            collections = self.client.list_collections()
            return [c.name for c in collections]
        except Exception as e:
            logging.error(f"Failed to list collections: {e}")
            return []
    
    def close(self):
        """关闭数据库连接"""
        self._collections.clear()
        logging.info("VectorStore closed")


# === 辅助函数 ===

def create_vector_store(persist_directory: str) -> VectorStore:
    """
    创建向量数据库实例的便捷函数
    
    Args:
        persist_directory: 持久化目录
        
    Returns:
        VectorStore 实例
    """
    return VectorStore(persist_directory=persist_directory)
