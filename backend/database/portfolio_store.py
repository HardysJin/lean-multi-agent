"""
Portfolio snapshot storage layer
"""

import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import desc
from sqlalchemy.orm import Session

from .models import PortfolioSnapshot
from .connection import get_db_manager


class PortfolioStore:
    """投资组合存储层"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
    
    def save_snapshot(
        self,
        total_equity: float,
        cash: float,
        positions_value: float,
        positions: Dict[str, Dict],
        total_return: Optional[float] = None,
        total_return_pct: Optional[float] = None,
        daily_return: Optional[float] = None,
        daily_return_pct: Optional[float] = None,
        current_drawdown: Optional[float] = None,
        current_drawdown_pct: Optional[float] = None,
        notes: Optional[str] = None,
        **kwargs
    ) -> PortfolioSnapshot:
        """
        保存投资组合快照
        
        Args:
            total_equity: 总权益
            cash: 现金
            positions_value: 持仓市值
            positions: 持仓详情 {symbol: {quantity, price, value, weight}}
            其他指标...
            
        Returns:
            保存的快照记录
        """
        with self.db_manager.session_scope() as session:
            snapshot = PortfolioSnapshot(
                timestamp=datetime.utcnow(),
                total_equity=total_equity,
                cash=cash,
                positions_value=positions_value,
                positions=json.dumps(positions),
                total_return=total_return,
                total_return_pct=total_return_pct,
                daily_return=daily_return,
                daily_return_pct=daily_return_pct,
                current_drawdown=current_drawdown,
                current_drawdown_pct=current_drawdown_pct,
                notes=notes,
            )
            
            session.add(snapshot)
            session.flush()
            session.refresh(snapshot)
            session.expunge(snapshot)
        
        return snapshot
    
    def get_snapshot(self, snapshot_id: int) -> Optional[PortfolioSnapshot]:
        """根据ID获取快照"""
        with self.db_manager.session_scope() as session:
            snapshot = session.query(PortfolioSnapshot).filter(
                PortfolioSnapshot.id == snapshot_id
            ).first()
            
            if snapshot:
                session.expunge(snapshot)
        
        return snapshot
    
    def get_latest_snapshot(self) -> Optional[PortfolioSnapshot]:
        """获取最新快照"""
        with self.db_manager.session_scope() as session:
            snapshot = session.query(PortfolioSnapshot).order_by(
                desc(PortfolioSnapshot.timestamp)
            ).first()
            
            if snapshot:
                session.expunge(snapshot)
        
        return snapshot
    
    def query_snapshots(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[PortfolioSnapshot]:
        """
        查询快照记录
        
        Args:
            start_date: 起始时间
            end_date: 结束时间
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            快照记录列表
        """
        with self.db_manager.session_scope() as session:
            query = session.query(PortfolioSnapshot)
            
            if start_date:
                query = query.filter(PortfolioSnapshot.timestamp >= start_date)
            if end_date:
                query = query.filter(PortfolioSnapshot.timestamp <= end_date)
            
            query = query.order_by(desc(PortfolioSnapshot.timestamp))
            query = query.limit(limit).offset(offset)
            
            snapshots = query.all()
            
            for snapshot in snapshots:
                session.expunge(snapshot)
        
        return snapshots
    
    def get_equity_curve(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        获取权益曲线数据
        
        Args:
            days: 回看天数
            
        Returns:
            权益曲线数据列表 [{timestamp, total_equity, cash, positions_value}]
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        snapshots = self.query_snapshots(start_date=start_date, limit=10000)
        
        curve_data = []
        for snapshot in reversed(snapshots):  # 按时间正序
            curve_data.append({
                'timestamp': snapshot.timestamp,
                'total_equity': snapshot.total_equity,
                'cash': snapshot.cash,
                'positions_value': snapshot.positions_value,
                'total_return_pct': snapshot.total_return_pct,
                'current_drawdown_pct': snapshot.current_drawdown_pct,
            })
        
        return curve_data
    
    def get_position_history(
        self,
        symbol: str,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        获取特定标的的持仓历史
        
        Args:
            symbol: 交易标的
            days: 回看天数
            
        Returns:
            持仓历史数据列表
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        snapshots = self.query_snapshots(start_date=start_date, limit=10000)
        
        position_history = []
        for snapshot in reversed(snapshots):
            positions = json.loads(snapshot.positions) if snapshot.positions else {}
            
            if symbol in positions:
                position_data = positions[symbol].copy()
                position_data['timestamp'] = snapshot.timestamp
                position_history.append(position_data)
        
        return position_history
    
    def get_performance_stats(
        self,
        days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        获取组合性能统计
        
        Args:
            days: 可选时间范围
            
        Returns:
            性能统计数据
        """
        start_date = datetime.utcnow() - timedelta(days=days) if days else None
        snapshots = self.query_snapshots(start_date=start_date, limit=10000)
        
        if not snapshots:
            return {
                'total_snapshots': 0,
                'latest_equity': None,
                'avg_equity': None,
                'max_equity': None,
                'min_equity': None,
                'current_return_pct': None,
                'max_drawdown_pct': None,
            }
        
        # 按时间正序
        snapshots = list(reversed(snapshots))
        
        equities = [s.total_equity for s in snapshots]
        returns = [s.total_return_pct for s in snapshots if s.total_return_pct is not None]
        drawdowns = [s.current_drawdown_pct for s in snapshots if s.current_drawdown_pct is not None]
        
        stats = {
            'total_snapshots': len(snapshots),
            'latest_equity': snapshots[-1].total_equity,
            'avg_equity': sum(equities) / len(equities),
            'max_equity': max(equities),
            'min_equity': min(equities),
            'current_return_pct': snapshots[-1].total_return_pct,
            'max_drawdown_pct': min(drawdowns) if drawdowns else None,
        }
        
        return stats
    
    def delete_old_snapshots(self, days: int = 365) -> int:
        """
        删除旧的快照记录
        
        Args:
            days: 保留天数
            
        Returns:
            删除的记录数
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        with self.db_manager.session_scope() as session:
            deleted_count = session.query(PortfolioSnapshot).filter(
                PortfolioSnapshot.timestamp < cutoff_date
            ).delete()
        
        return deleted_count
