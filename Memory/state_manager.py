"""
State Manager - 多时间尺度状态管理器

核心模块：管理5个时间尺度的分层记忆系统

关键功能：
1. 决策存储到向量数据库和SQL数据库
2. 分层上下文检索（跨时间尺度）
3. 时间衰减权重计算
4. 分层约束检查和获取
5. 向上传播机制（重要信号自动escalate）
6. 记忆清理和维护

设计原则：
- 下层决策必须遵守上层约束（军事化指挥）
- 每层独立的记忆存储和检索
- 时间衰减权重：上层记忆衰减慢，下层快速衰减
- 向上传播：重要信号自动传递到上层
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from .schemas import (
    Timeframe,
    DecisionRecord,
    MemoryDocument,
    HierarchicalConstraints,
    create_decision_id,
)
from .vector_store import VectorStore
from .sql_store import SQLStore


class MultiTimeframeStateManager:
    """
    多时间尺度状态管理器
    
    核心职责：
    - 管理5个时间尺度的独立记忆存储
    - 计算时间衰减权重
    - 提供分层上下文检索
    - 执行约束检查
    - 协调向量数据库和SQL数据库
    """
    
    def __init__(self,
                 vector_db_path: str,
                 sql_db_path: str,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        初始化状态管理器
        
        Args:
            vector_db_path: ChromaDB持久化路径
            sql_db_path: SQLite数据库路径
            embedding_model: SentenceTransformer模型名称
        """
        # 初始化存储
        self.vector_store = VectorStore(
            persist_directory=vector_db_path,
            embedding_model=embedding_model
        )
        self.sql_store = SQLStore(db_path=sql_db_path)
        
        # 为每个时间尺度创建collection
        self.collections = {}
        for tf in Timeframe:
            self.collections[tf] = self.vector_store.get_collection_for_timeframe(tf)
        
        # 时间衰减参数（lambda值 - 越大衰减越快）
        self.decay_rates = {
            Timeframe.REALTIME: 5.0,    # 快速遗忘
            Timeframe.EXECUTION: 2.0,
            Timeframe.TACTICAL: 0.5,
            Timeframe.CAMPAIGN: 0.1,
            Timeframe.STRATEGIC: 0.02   # 长期记忆
        }
        
        # 向上传播阈值
        self.propagation_thresholds = {
            'conviction': 8.0,           # 信心度阈值
            'significance': 0.8,         # 重要性阈值
        }
        
        logging.info("MultiTimeframeStateManager initialized")
    
    # === 决策存储 ===
    
    def store_decision(self, decision: DecisionRecord) -> bool:
        """
        存储一次决策到对应时间尺度的记忆库
        
        完整流程：
        1. 构建文本描述
        2. 生成embedding向量（自动）
        3. 存入向量数据库
        4. 存入结构化数据库
        5. 检查是否需要向上传播
        
        Args:
            decision: 决策记录
            
        Returns:
            是否成功存储
        """
        try:
            # 1. 存入SQL数据库（结构化存储）
            sql_success = self.sql_store.save_decision(decision)
            if not sql_success:
                logging.error(f"Failed to save decision to SQL: {decision.id}")
                return False
            
            # 2. 创建向量文档
            doc = MemoryDocument.from_decision(decision)
            
            # 3. 存入向量数据库（语义搜索）
            collection_name = f"{decision.timeframe.display_name}_memory"
            vector_success = self.vector_store.add_document(collection_name, doc)
            if not vector_success:
                logging.warning(f"Failed to save decision to vector store: {decision.id}")
                # 不返回False，因为SQL已经保存成功
            
            # 4. 检查是否需要向上传播
            if self._should_propagate_upward(decision):
                self._propagate_to_higher_timeframe(decision)
            
            logging.info(f"Stored decision {decision.id} at {decision.timeframe.display_name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to store decision: {e}")
            return False
    
    def update_decision_outcome(self,
                               decision_id: str,
                               exit_price: float,
                               exit_time: datetime) -> bool:
        """
        更新决策结果
        
        Args:
            decision_id: 决策ID
            exit_price: 退出价格
            exit_time: 退出时间
            
        Returns:
            是否成功更新
        """
        return self.sql_store.update_decision_outcome(decision_id, exit_price, exit_time)
    
    # === 分层检索 ===
    
    def retrieve_hierarchical_context(self,
                                      query: str,
                                      symbol: str,
                                      current_timeframe: Timeframe,
                                      max_results_per_layer: int = 5,
                                      lookback_days: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        检索分层上下文（最核心的方法）
        
        从当前时间尺度及更高层级检索相关记忆，并计算时间衰减权重
        
        Args:
            query: 查询文本（用于语义搜索）
            symbol: 股票代码
            current_timeframe: 当前决策的时间尺度
            max_results_per_layer: 每层返回的最大结果数
            lookback_days: 回溯天数（None表示不限制）
            
        Returns:
            Dict[str, List[Dict]]: 每个时间尺度的加权记忆列表
            {
                'strategic': [
                    {'text': '...', 'weight': 0.95, 'metadata': {...}, 'decision_id': '...'},
                    ...
                ],
                'campaign': [...],
                ...
            }
        """
        try:
            results = {}
            current_time = datetime.now()
            
            # 获取当前及更高的时间尺度
            relevant_timeframes = [current_timeframe] + Timeframe.get_higher_timeframes(current_timeframe)
            
            for timeframe in relevant_timeframes:
                # 从向量数据库查询相似记忆
                vector_results = self.vector_store.query_by_timeframe(
                    timeframe=timeframe,
                    query_text=query,
                    n_results=max_results_per_layer * 2,  # 多查一些，然后过滤
                    symbol=symbol
                )
                
                # 处理结果并计算权重
                weighted_results = []
                for i, (doc_id, text, metadata, distance) in enumerate(zip(
                    vector_results['ids'],
                    vector_results['documents'],
                    vector_results['metadatas'],
                    vector_results['distances']
                )):
                    # 解析时间戳
                    try:
                        timestamp = datetime.fromisoformat(metadata.get('timestamp', ''))
                    except (ValueError, TypeError):
                        timestamp = current_time
                    
                    # 检查回溯期限
                    if lookback_days is not None:
                        age_days = (current_time - timestamp).days
                        if age_days > lookback_days:
                            continue
                    
                    # 计算时间衰减权重
                    time_delta = current_time - timestamp
                    decay_weight = self._calculate_decay_weight(time_delta, timeframe)
                    
                    # 计算相似度权重（距离转换为相似度）
                    # ChromaDB使用L2距离，需要转换
                    similarity_weight = 1.0 / (1.0 + distance)
                    
                    # 综合权重
                    final_weight = decay_weight * similarity_weight
                    
                    weighted_results.append({
                        'decision_id': doc_id,
                        'text': text,
                        'metadata': metadata,
                        'weight': final_weight,
                        'decay_weight': decay_weight,
                        'similarity_weight': similarity_weight,
                        'age_days': (current_time - timestamp).days,
                    })
                
                # 按权重排序并限制数量
                weighted_results.sort(key=lambda x: x['weight'], reverse=True)
                results[timeframe.display_name] = weighted_results[:max_results_per_layer]
            
            logging.debug(f"Retrieved hierarchical context for {symbol}: {len(results)} layers")
            return results
            
        except Exception as e:
            logging.error(f"Failed to retrieve hierarchical context: {e}")
            return {}
    
    def get_similar_past_decisions(self,
                                   query: str,
                                   timeframe: Timeframe,
                                   symbol: Optional[str] = None,
                                   limit: int = 10,
                                   min_weight: float = 0.1) -> List[DecisionRecord]:
        """
        获取相似的历史决策
        
        Args:
            query: 查询描述
            timeframe: 时间尺度
            symbol: 股票代码（可选）
            limit: 最大返回数量
            min_weight: 最小权重阈值
            
        Returns:
            相似决策列表（按权重排序）
        """
        try:
            # 向量搜索
            vector_results = self.vector_store.query_by_timeframe(
                timeframe=timeframe,
                query_text=query,
                n_results=limit * 2,
                symbol=symbol
            )
            
            # 从SQL获取完整决策记录
            decisions = []
            current_time = datetime.now()
            
            for doc_id, distance, metadata in zip(
                vector_results['ids'],
                vector_results['distances'],
                vector_results['metadatas']
            ):
                # 获取完整决策
                decision = self.sql_store.get_decision(doc_id)
                if decision:
                    # 计算权重
                    time_delta = current_time - decision.timestamp
                    weight = self._calculate_decay_weight(time_delta, timeframe)
                    
                    if weight >= min_weight:
                        # 将权重作为临时属性
                        decision.metadata['retrieval_weight'] = weight
                        decisions.append(decision)
            
            # 按权重排序
            decisions.sort(key=lambda d: d.metadata.get('retrieval_weight', 0), reverse=True)
            return decisions[:limit]
            
        except Exception as e:
            logging.error(f"Failed to get similar past decisions: {e}")
            return []
    
    # === 约束管理 ===
    
    def get_current_constraints(self, symbol: Optional[str] = None) -> HierarchicalConstraints:
        """
        获取当前的分层约束
        
        自动整合所有时间尺度的约束：
        - 战略层：市场regime、风险预算
        - 战役层：行业配置权重
        - 战术层：近期决策历史
        
        Args:
            symbol: 股票代码（获取特定股票的约束）
            
        Returns:
            分层约束对象
        """
        try:
            constraints = HierarchicalConstraints()
            
            # 获取战略层约束
            strategic = self.sql_store.get_constraints(Timeframe.STRATEGIC, symbol)
            if strategic:
                constraints.strategic = strategic.strategic
            else:
                # 默认约束
                constraints.strategic = {
                    'market_regime': 'neutral',
                    'risk_budget': 1.0,
                    'max_exposure': 1.0,
                    'forbidden_sectors': [],
                }
            
            # 获取战役层约束
            campaign = self.sql_store.get_constraints(Timeframe.CAMPAIGN, symbol)
            if campaign:
                constraints.campaign = campaign.campaign
            else:
                constraints.campaign = {}
            
            # 获取战术层历史（最近的决策）
            recent_decisions = self.sql_store.get_recent_decisions(limit=10, symbol=symbol)
            if recent_decisions:
                # 计算统计
                stats = self.sql_store.get_performance_stats(symbol=symbol)
                constraints.tactical = {
                    'recent_decisions_count': len(recent_decisions),
                    'win_rate': stats.get('win_rate', 0),
                    'avg_return': stats.get('avg_return', 0),
                    'last_action': recent_decisions[0].action if recent_decisions else None,
                }
            else:
                constraints.tactical = {}
            
            return constraints
            
        except Exception as e:
            logging.error(f"Failed to get current constraints: {e}")
            return HierarchicalConstraints()
    
    def save_constraints(self,
                        timeframe: Timeframe,
                        constraints: HierarchicalConstraints,
                        symbol: Optional[str] = None) -> bool:
        """
        保存约束条件
        
        Args:
            timeframe: 时间尺度
            constraints: 约束条件
            symbol: 股票代码（None表示全局约束）
            
        Returns:
            是否成功保存
        """
        return self.sql_store.save_constraints(timeframe, constraints, symbol)
    
    def check_constraints_compliance(self,
                                    decision: DecisionRecord,
                                    constraints: Optional[HierarchicalConstraints] = None) -> Tuple[bool, List[str]]:
        """
        检查决策是否符合分层约束
        
        Args:
            decision: 待检查的决策
            constraints: 约束条件（None则自动获取）
            
        Returns:
            (是否符合, 违规原因列表)
        """
        violations = []
        
        if constraints is None:
            constraints = self.get_current_constraints(decision.symbol)
        
        # 检查战略层约束
        if constraints.strategic:
            # 市场regime检查
            market_regime = constraints.get_market_regime()
            if market_regime == 'bear' and decision.action in ['BUY', 'ADD']:
                violations.append(f"Market regime is BEAR, cannot open long positions")
            elif market_regime == 'bull' and decision.action in ['SELL', 'REDUCE'] and decision.quantity > 0:
                violations.append(f"Market regime is BULL, should not reduce positions aggressively")
            
            # 风险预算检查
            risk_budget = constraints.get_risk_budget()
            if risk_budget < 0.5 and decision.conviction < 8.0:
                violations.append(f"Low risk budget ({risk_budget}), requires high conviction (>8.0)")
        
        # 检查战役层约束
        if constraints.campaign:
            # 行业配置检查（这里简化，实际需要知道股票所属行业）
            pass
        
        is_compliant = len(violations) == 0
        return is_compliant, violations
    
    # === 权重计算 ===
    
    def _calculate_decay_weight(self,
                                time_delta: timedelta,
                                timeframe: Timeframe) -> float:
        """
        计算时间衰减权重
        
        公式: w(t) = exp(-λ * Δt / T)
        其中：
        - λ: 衰减率（decay_rate）
        - Δt: 时间间隔
        - T: 特征周期（characteristic_period）
        
        Args:
            time_delta: 时间间隔
            timeframe: 时间尺度
            
        Returns:
            权重值 [0.01, 1.0]
        """
        lambda_decay = self.decay_rates[timeframe]
        T = timeframe.characteristic_period_seconds
        delta_t = time_delta.total_seconds()
        
        # 指数衰减
        weight = np.exp(-lambda_decay * delta_t / T)
        
        # 限制最小权重
        return max(weight, 0.01)
    
    # === 向上传播 ===
    
    def _should_propagate_upward(self,
                                 decision: DecisionRecord) -> bool:
        """
        判断是否需要向上传播到更高时间尺度
        
        传播条件：
        1. conviction >= 8 (高信心)
        2. 已有结果且表现极好/极差
        3. 检测到regime change
        
        Args:
            decision: 决策记录
            
        Returns:
            是否应该传播
        """
        # 条件1: 高信心决策
        if decision.conviction >= self.propagation_thresholds['conviction']:
            return True
        
        # 条件2: 极端结果
        if decision.outcome:
            if decision.outcome == 'success' and decision.pnl_percent and decision.pnl_percent > 0.1:
                return True  # >10% 收益
            if decision.outcome == 'failure' and decision.pnl_percent and decision.pnl_percent < -0.1:
                return True  # <-10% 损失
        
        # 条件3: Market regime相关（通过reasoning判断）
        if decision.reasoning and any(keyword in decision.reasoning.lower() for keyword in
                                     ['regime change', 'trend reversal', 'market shift', 'breakthrough']):
            return True
        
        return False
    
    def _propagate_to_higher_timeframe(self, decision: DecisionRecord):
        """
        将重要信号传播到更高时间尺度
        
        Args:
            decision: 决策记录
        """
        higher_timeframes = Timeframe.get_higher_timeframes(decision.timeframe)
        
        if not higher_timeframes:
            return  # 已经是最高层
        
        # 传播到直接上一层
        target_timeframe = higher_timeframes[0]
        
        logging.info(f"Propagating decision {decision.id} from {decision.timeframe.display_name} "
                    f"to {target_timeframe.display_name}")
        
        # 创建传播决策（标记为来自下层）
        propagated_decision = DecisionRecord(
            id=create_decision_id(decision.symbol, decision.timestamp, target_timeframe),
            timestamp=decision.timestamp,
            timeframe=target_timeframe,
            symbol=decision.symbol,
            action=decision.action,
            quantity=decision.quantity,
            price=decision.price,
            reasoning=f"[Propagated from {decision.timeframe.display_name}] {decision.reasoning}",
            agent_name=f"{decision.agent_name}_propagated",
            conviction=decision.conviction,
            market_regime=decision.market_regime,
            metadata={
                'propagated_from': decision.id,
                'propagated_from_timeframe': decision.timeframe.display_name,
                'original_conviction': decision.conviction,
            }
        )
        
        # 存储到更高层
        self.store_decision(propagated_decision)
    
    # === 统计和分析 ===
    
    def get_performance_summary(self,
                               symbol: Optional[str] = None,
                               timeframe: Optional[Timeframe] = None) -> Dict[str, Any]:
        """
        获取性能汇总
        
        Args:
            symbol: 股票代码过滤
            timeframe: 时间尺度过滤
            
        Returns:
            性能统计字典
        """
        return self.sql_store.get_performance_stats(symbol, timeframe)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取记忆系统统计
        
        Returns:
            统计信息
        """
        stats = {
            'vector_store': {},
            'sql_store': self.sql_store.get_database_stats(),
            'collections': {},
        }
        
        # 获取每个collection的统计
        for timeframe in Timeframe:
            collection_name = f"{timeframe.display_name}_memory"
            stats['collections'][timeframe.display_name] = \
                self.vector_store.get_collection_stats(collection_name)
        
        return stats
    
    # === 数据清理 ===
    
    def cleanup_old_memories(self,
                            days_to_keep: Dict[Timeframe, int],
                            keep_successful_trades: bool = True) -> Dict[str, int]:
        """
        清理旧记忆
        
        不同时间尺度保留不同的天数：
        - REALTIME: 7天
        - EXECUTION: 30天
        - TACTICAL: 90天
        - CAMPAIGN: 180天
        - STRATEGIC: 365天
        
        Args:
            days_to_keep: 每个时间尺度保留的天数
            keep_successful_trades: 是否保留成功的交易案例
            
        Returns:
            每个时间尺度删除的记录数
        """
        deleted_counts = {}
        current_time = datetime.now()
        
        for timeframe, days in days_to_keep.items():
            cutoff_time = current_time - timedelta(days=days)
            
            # 清理SQL
            sql_deleted = self.sql_store.cleanup_old_decisions(
                cutoff_time,
                keep_outcomes=keep_successful_trades
            )
            
            # 清理向量数据库
            collection_name = f"{timeframe.display_name}_memory"
            vector_deleted = self.vector_store.cleanup_old_data(
                collection_name,
                cutoff_time,
                keep_outcomes=keep_successful_trades
            )
            
            deleted_counts[timeframe.display_name] = {
                'sql': sql_deleted,
                'vector': vector_deleted,
            }
            
            logging.info(f"Cleaned up {timeframe.display_name}: "
                        f"SQL={sql_deleted}, Vector={vector_deleted}")
        
        return deleted_counts
    
    def reset_timeframe_memories(self, timeframe: Timeframe) -> bool:
        """
        重置指定时间尺度的所有记忆
        
        Args:
            timeframe: 要重置的时间尺度
            
        Returns:
            是否成功
        """
        try:
            collection_name = f"{timeframe.display_name}_memory"
            self.vector_store.reset_collection(collection_name)
            
            # SQL中删除该时间尺度的决策
            decisions = self.sql_store.query_decisions(timeframe=timeframe)
            for decision in decisions:
                self.sql_store.delete_decision(decision.id)
            
            logging.info(f"Reset memories for {timeframe.display_name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to reset memories: {e}")
            return False


# === 辅助函数 ===

def create_state_manager(vector_db_path: str, sql_db_path: str) -> MultiTimeframeStateManager:
    """
    创建状态管理器的便捷函数
    
    Args:
        vector_db_path: 向量数据库路径
        sql_db_path: SQL数据库路径
        
    Returns:
        MultiTimeframeStateManager实例
    """
    return MultiTimeframeStateManager(
        vector_db_path=vector_db_path,
        sql_db_path=sql_db_path
    )
