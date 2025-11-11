"""
Decision storage layer
"""

import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import desc, and_, or_
from sqlalchemy.orm import Session

from .models import DecisionRecord, TradeRecord
from .connection import get_db_manager


class DecisionStore:
    """决策存储层"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
    
    def save_decision(
        self,
        symbol: str,
        action: str,
        lookback_days: int,
        forecast_days: int,
        llm_provider: str,
        llm_model: str,
        reasoning: str,
        confidence: Optional[float] = None,
        recommended_strategy: Optional[str] = None,
        strategy_reasoning: Optional[str] = None,
        current_price: Optional[float] = None,
        position_size: Optional[float] = None,
        risk_score: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        execution_time_ms: Optional[float] = None,
        **kwargs
    ) -> DecisionRecord:
        """
        保存决策记录
        
        Args:
            symbol: 交易标的
            action: 决策动作 buy/sell/hold
            lookback_days: 回看天数
            forecast_days: 预测天数
            llm_provider: LLM提供商
            llm_model: LLM模型
            reasoning: 推理过程
            其他可选参数...
            
        Returns:
            保存的决策记录
        """
        with self.db_manager.session_scope() as session:
            decision = DecisionRecord(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                action=action,
                lookback_days=lookback_days,
                forecast_days=forecast_days,
                llm_provider=llm_provider,
                llm_model=llm_model,
                reasoning=reasoning,
                confidence=confidence,
                recommended_strategy=recommended_strategy,
                strategy_reasoning=strategy_reasoning,
                current_price=current_price,
                position_size=position_size,
                risk_score=risk_score,
                max_drawdown=max_drawdown,
                stop_loss=stop_loss,
                take_profit=take_profit,
                execution_time_ms=execution_time_ms,
            )
            
            session.add(decision)
            session.flush()  # 获取ID
            session.refresh(decision)
            
            # 从session中分离，避免延迟加载错误
            session.expunge(decision)
            
        return decision
    
    def get_decision(self, decision_id: int) -> Optional[DecisionRecord]:
        """根据ID获取决策"""
        with self.db_manager.session_scope() as session:
            decision = session.query(DecisionRecord).filter(
                DecisionRecord.id == decision_id
            ).first()
            
            if decision:
                session.expunge(decision)  # 从session中分离
            
        return decision
    
    def query_decisions(
        self,
        symbol: Optional[str] = None,
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DecisionRecord]:
        """
        查询决策记录
        
        Args:
            symbol: 筛选标的
            action: 筛选动作
            start_date: 起始时间
            end_date: 结束时间
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            决策记录列表
        """
        with self.db_manager.session_scope() as session:
            query = session.query(DecisionRecord)
            
            # 添加筛选条件
            if symbol:
                query = query.filter(DecisionRecord.symbol == symbol)
            if action:
                query = query.filter(DecisionRecord.action == action)
            if start_date:
                query = query.filter(DecisionRecord.timestamp >= start_date)
            if end_date:
                query = query.filter(DecisionRecord.timestamp <= end_date)
            
            # 排序和分页
            query = query.order_by(desc(DecisionRecord.timestamp))
            query = query.limit(limit).offset(offset)
            
            decisions = query.all()
            
            # 从session中分离
            for decision in decisions:
                session.expunge(decision)
        
        return decisions
    
    def get_latest_decision(self, symbol: str) -> Optional[DecisionRecord]:
        """获取指定标的的最新决策"""
        with self.db_manager.session_scope() as session:
            decision = session.query(DecisionRecord).filter(
                DecisionRecord.symbol == symbol
            ).order_by(desc(DecisionRecord.timestamp)).first()
            
            if decision:
                session.expunge(decision)
        
        return decision
    
    def get_decision_history(
        self,
        symbol: str,
        days: int = 30,
    ) -> List[DecisionRecord]:
        """
        获取指定标的的决策历史
        
        Args:
            symbol: 交易标的
            days: 回看天数
            
        Returns:
            决策记录列表
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        return self.query_decisions(
            symbol=symbol,
            start_date=start_date,
            limit=1000,
        )
    
    def get_decision_stats(
        self,
        symbol: Optional[str] = None,
        days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        获取决策统计信息
        
        Args:
            symbol: 可选标的筛选
            days: 可选时间范围
            
        Returns:
            统计信息字典
        """
        with self.db_manager.session_scope() as session:
            query = session.query(DecisionRecord)
            
            if symbol:
                query = query.filter(DecisionRecord.symbol == symbol)
            if days:
                start_date = datetime.utcnow() - timedelta(days=days)
                query = query.filter(DecisionRecord.timestamp >= start_date)
            
            decisions = query.all()
            
            if not decisions:
                return {
                    'total_decisions': 0,
                    'buy_count': 0,
                    'sell_count': 0,
                    'hold_count': 0,
                    'avg_confidence': None,
                    'avg_execution_time_ms': None,
                }
            
            # 统计
            buy_count = sum(1 for d in decisions if d.action == 'buy')
            sell_count = sum(1 for d in decisions if d.action == 'sell')
            hold_count = sum(1 for d in decisions if d.action == 'hold')
            
            confidences = [d.confidence for d in decisions if d.confidence is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else None
            
            exec_times = [d.execution_time_ms for d in decisions if d.execution_time_ms is not None]
            avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else None
            
            stats = {
                'total_decisions': len(decisions),
                'buy_count': buy_count,
                'sell_count': sell_count,
                'hold_count': hold_count,
                'avg_confidence': avg_confidence,
                'avg_execution_time_ms': avg_exec_time,
            }
        
        return stats
    
    def mark_decision_executed(self, decision_id: int) -> bool:
        """标记决策已执行"""
        with self.db_manager.session_scope() as session:
            decision = session.query(DecisionRecord).filter(
                DecisionRecord.id == decision_id
            ).first()
            
            if decision:
                decision.is_executed = True
                return True
            
        return False
    
    def delete_old_decisions(self, days: int = 365) -> int:
        """
        删除旧的决策记录
        
        Args:
            days: 保留天数，超过此时间的记录将被删除
            
        Returns:
            删除的记录数
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        with self.db_manager.session_scope() as session:
            deleted_count = session.query(DecisionRecord).filter(
                DecisionRecord.timestamp < cutoff_date
            ).delete()
        
        return deleted_count
