"""
SQL Store - SQLite 数据库 wrapper

提供结构化数据库的统一接口，用于：
1. 存储决策记录的完整信息
2. 精确查询（按ID、时间、symbol等）
3. 结果跟踪（P&L、胜率统计）
4. 约束条件存储
5. 性能分析和报告

使用 SQLite 作为底层实现，支持本地持久化。
"""

import os
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
import json
import logging

from .schemas import (
    DecisionRecord,
    HierarchicalConstraints,
    Timeframe,
)


class SQLStore:
    """
    SQL 数据库 wrapper
    
    封装 SQLite 的操作，提供简单的 API：
    - 决策记录的 CRUD
    - 约束条件的存储和查询
    - 性能统计和分析
    - 数据清理和维护
    
    数据库表结构：
    - decisions: 决策记录
    - constraints: 分层约束
    - performance_summary: 性能汇总（可选）
    """
    
    def __init__(self, db_path: str):
        """
        初始化 SQL 数据库
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        
        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 初始化数据库表
        self._init_tables()
        
        logging.info(f"SQLStore initialized at {db_path}")
    
    @contextmanager
    def _get_connection(self):
        """
        获取数据库连接的上下文管理器
        
        Yields:
            sqlite3.Connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 支持字典访问
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_tables(self):
        """初始化数据库表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 决策记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    reasoning TEXT,
                    agent_name TEXT NOT NULL,
                    conviction REAL NOT NULL,
                    
                    -- 上下文信息
                    market_regime TEXT,
                    technical_signals TEXT,  -- JSON
                    fundamental_data TEXT,   -- JSON
                    news_sentiment TEXT,     -- JSON
                    related_news_ids TEXT,   -- JSON array
                    
                    -- 执行跟踪
                    executed INTEGER DEFAULT 0,
                    execution_price REAL,
                    execution_time TEXT,
                    
                    -- 结果跟踪
                    outcome TEXT,
                    exit_price REAL,
                    exit_time TEXT,
                    pnl REAL,
                    pnl_percent REAL,
                    hold_duration_days INTEGER,
                    
                    -- 元数据
                    metadata TEXT,  -- JSON
                    
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建索引加速查询
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_timestamp 
                ON decisions(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_symbol 
                ON decisions(symbol)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_timeframe 
                ON decisions(timeframe)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_outcome 
                ON decisions(outcome)
            """)
            
            # 约束条件表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS constraints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timeframe TEXT NOT NULL,
                    symbol TEXT,  -- NULL 表示全局约束
                    strategic TEXT,  -- JSON
                    campaign TEXT,   -- JSON
                    tactical TEXT,   -- JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timeframe, symbol)
                )
            """)
            
            logging.info("Database tables initialized")
    
    # === 决策记录操作 ===
    
    def save_decision(self, decision: DecisionRecord) -> bool:
        """
        保存决策记录
        
        Args:
            decision: 决策记录对象
            
        Returns:
            是否成功保存
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 准备数据
                data = {
                    'id': decision.id,
                    'timestamp': decision.timestamp.isoformat(),
                    'timeframe': decision.timeframe.display_name,
                    'symbol': decision.symbol,
                    'action': decision.action,
                    'quantity': decision.quantity,
                    'price': decision.price,
                    'reasoning': decision.reasoning,
                    'agent_name': decision.agent_name,
                    'conviction': decision.conviction,
                    'market_regime': decision.market_regime,
                    'technical_signals': json.dumps(decision.technical_signals) if decision.technical_signals else None,
                    'fundamental_data': json.dumps(decision.fundamental_data) if decision.fundamental_data else None,
                    'news_sentiment': json.dumps(decision.news_sentiment) if decision.news_sentiment else None,
                    'related_news_ids': json.dumps(decision.related_news_ids) if decision.related_news_ids else None,
                    'executed': 1 if decision.executed else 0,
                    'execution_price': decision.execution_price,
                    'execution_time': decision.execution_time.isoformat() if decision.execution_time else None,
                    'outcome': decision.outcome,
                    'exit_price': decision.exit_price,
                    'exit_time': decision.exit_time.isoformat() if decision.exit_time else None,
                    'pnl': decision.pnl,
                    'pnl_percent': decision.pnl_percent,
                    'hold_duration_days': decision.hold_duration_days,
                    'metadata': json.dumps(decision.metadata) if decision.metadata else None,
                }
                
                # 插入或替换
                cursor.execute("""
                    INSERT OR REPLACE INTO decisions 
                    (id, timestamp, timeframe, symbol, action, quantity, price, reasoning, 
                     agent_name, conviction, market_regime, technical_signals, fundamental_data,
                     news_sentiment, related_news_ids, executed, execution_price, execution_time,
                     outcome, exit_price, exit_time, pnl, pnl_percent, hold_duration_days, metadata)
                    VALUES 
                    (:id, :timestamp, :timeframe, :symbol, :action, :quantity, :price, :reasoning,
                     :agent_name, :conviction, :market_regime, :technical_signals, :fundamental_data,
                     :news_sentiment, :related_news_ids, :executed, :execution_price, :execution_time,
                     :outcome, :exit_price, :exit_time, :pnl, :pnl_percent, :hold_duration_days, :metadata)
                """, data)
                
                logging.debug(f"Saved decision {decision.id}")
                return True
                
        except Exception as e:
            logging.error(f"Failed to save decision: {e}")
            return False
    
    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        """
        获取决策记录
        
        Args:
            decision_id: 决策ID
            
        Returns:
            决策记录对象，如果不存在返回 None
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_decision(row)
                return None
                
        except Exception as e:
            logging.error(f"Failed to get decision {decision_id}: {e}")
            return None
    
    def query_decisions(self,
                       symbol: Optional[str] = None,
                       timeframe: Optional[Timeframe] = None,
                       action: Optional[str] = None,
                       outcome: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       limit: Optional[int] = None,
                       order_by: str = "timestamp DESC") -> List[DecisionRecord]:
        """
        查询决策记录
        
        Args:
            symbol: 股票代码过滤
            timeframe: 时间尺度过滤
            action: 动作过滤
            outcome: 结果过滤
            start_time: 开始时间
            end_time: 结束时间
            limit: 最大返回数量
            order_by: 排序方式
            
        Returns:
            决策记录列表
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 构建查询
                query = "SELECT * FROM decisions WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                if timeframe:
                    query += " AND timeframe = ?"
                    params.append(timeframe.display_name)
                
                if action:
                    query += " AND action = ?"
                    params.append(action)
                
                if outcome:
                    query += " AND outcome = ?"
                    params.append(outcome)
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())
                
                query += f" ORDER BY {order_by}"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_decision(row) for row in rows]
                
        except Exception as e:
            logging.error(f"Failed to query decisions: {e}")
            return []
    
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
        try:
            # 先获取决策
            decision = self.get_decision(decision_id)
            if not decision:
                return False
            
            # 更新结果
            decision.update_outcome(exit_price, exit_time)
            
            # 保存
            return self.save_decision(decision)
            
        except Exception as e:
            logging.error(f"Failed to update decision outcome: {e}")
            return False
    
    def delete_decision(self, decision_id: str) -> bool:
        """
        删除决策记录
        
        Args:
            decision_id: 决策ID
            
        Returns:
            是否成功删除
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM decisions WHERE id = ?", (decision_id,))
                return cursor.rowcount > 0
                
        except Exception as e:
            logging.error(f"Failed to delete decision: {e}")
            return False
    
    # === 约束条件操作 ===
    
    def save_constraints(self,
                        timeframe: Timeframe,
                        constraints: HierarchicalConstraints,
                        symbol: Optional[str] = None) -> bool:
        """
        保存约束条件
        
        Args:
            timeframe: 时间尺度
            constraints: 约束条件对象
            symbol: 股票代码（None 表示全局约束）
            
        Returns:
            是否成功保存
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                data = {
                    'timeframe': timeframe.display_name,
                    'symbol': symbol,
                    'strategic': json.dumps(constraints.strategic) if constraints.strategic else None,
                    'campaign': json.dumps(constraints.campaign) if constraints.campaign else None,
                    'tactical': json.dumps(constraints.tactical) if constraints.tactical else None,
                }
                
                cursor.execute("""
                    INSERT OR REPLACE INTO constraints 
                    (timeframe, symbol, strategic, campaign, tactical, updated_at)
                    VALUES (:timeframe, :symbol, :strategic, :campaign, :tactical, CURRENT_TIMESTAMP)
                """, data)
                
                logging.debug(f"Saved constraints for {timeframe.display_name}")
                return True
                
        except Exception as e:
            logging.error(f"Failed to save constraints: {e}")
            return False
    
    def get_constraints(self,
                       timeframe: Timeframe,
                       symbol: Optional[str] = None) -> Optional[HierarchicalConstraints]:
        """
        获取约束条件
        
        Args:
            timeframe: 时间尺度
            symbol: 股票代码（None 获取全局约束）
            
        Returns:
            约束条件对象，如果不存在返回 None
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM constraints 
                    WHERE timeframe = ? AND (symbol = ? OR (symbol IS NULL AND ? IS NULL))
                    ORDER BY updated_at DESC LIMIT 1
                """, (timeframe.display_name, symbol, symbol))
                
                row = cursor.fetchone()
                
                if row:
                    return HierarchicalConstraints(
                        strategic=json.loads(row['strategic']) if row['strategic'] else None,
                        campaign=json.loads(row['campaign']) if row['campaign'] else None,
                        tactical=json.loads(row['tactical']) if row['tactical'] else None,
                    )
                return None
                
        except Exception as e:
            logging.error(f"Failed to get constraints: {e}")
            return None
    
    # === 统计和分析 ===
    
    def get_performance_stats(self,
                             symbol: Optional[str] = None,
                             timeframe: Optional[Timeframe] = None,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        获取性能统计
        
        Args:
            symbol: 股票代码过滤
            timeframe: 时间尺度过滤
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            统计信息字典
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 构建条件
                where_clause = "WHERE outcome IS NOT NULL"
                params = []
                
                if symbol:
                    where_clause += " AND symbol = ?"
                    params.append(symbol)
                
                if timeframe:
                    where_clause += " AND timeframe = ?"
                    params.append(timeframe.display_name)
                
                if start_time:
                    where_clause += " AND timestamp >= ?"
                    params.append(start_time.isoformat())
                
                if end_time:
                    where_clause += " AND timestamp <= ?"
                    params.append(end_time.isoformat())
                
                # 总体统计
                cursor.execute(f"""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN outcome = 'failure' THEN 1 ELSE 0 END) as losses,
                        AVG(pnl) as avg_pnl,
                        SUM(pnl) as total_pnl,
                        AVG(pnl_percent) as avg_return,
                        MAX(pnl_percent) as max_return,
                        MIN(pnl_percent) as min_return,
                        AVG(hold_duration_days) as avg_hold_days
                    FROM decisions {where_clause}
                """, params)
                
                row = cursor.fetchone()
                
                total_trades = row['total_trades'] or 0
                wins = row['wins'] or 0
                losses = row['losses'] or 0
                
                stats = {
                    'total_trades': total_trades,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': wins / total_trades if total_trades > 0 else 0,
                    'avg_pnl': row['avg_pnl'] or 0,
                    'total_pnl': row['total_pnl'] or 0,
                    'avg_return': row['avg_return'] or 0,
                    'max_return': row['max_return'] or 0,
                    'min_return': row['min_return'] or 0,
                    'avg_hold_days': row['avg_hold_days'] or 0,
                }
                
                return stats
                
        except Exception as e:
            logging.error(f"Failed to get performance stats: {e}")
            return {}
    
    def get_action_distribution(self,
                               symbol: Optional[str] = None,
                               timeframe: Optional[Timeframe] = None) -> Dict[str, int]:
        """
        获取动作分布统计
        
        Args:
            symbol: 股票代码过滤
            timeframe: 时间尺度过滤
            
        Returns:
            动作分布字典 {action: count}
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                where_clause = "WHERE 1=1"
                params = []
                
                if symbol:
                    where_clause += " AND symbol = ?"
                    params.append(symbol)
                
                if timeframe:
                    where_clause += " AND timeframe = ?"
                    params.append(timeframe.display_name)
                
                cursor.execute(f"""
                    SELECT action, COUNT(*) as count
                    FROM decisions {where_clause}
                    GROUP BY action
                """, params)
                
                return {row['action']: row['count'] for row in cursor.fetchall()}
                
        except Exception as e:
            logging.error(f"Failed to get action distribution: {e}")
            return {}
    
    def get_recent_decisions(self,
                            limit: int = 10,
                            symbol: Optional[str] = None) -> List[DecisionRecord]:
        """
        获取最近的决策
        
        Args:
            limit: 数量限制
            symbol: 股票代码过滤
            
        Returns:
            决策记录列表
        """
        return self.query_decisions(
            symbol=symbol,
            limit=limit,
            order_by="timestamp DESC"
        )
    
    # === 数据清理 ===
    
    def cleanup_old_decisions(self,
                             before_timestamp: datetime,
                             keep_outcomes: bool = True) -> int:
        """
        清理旧决策
        
        Args:
            before_timestamp: 删除此时间之前的数据
            keep_outcomes: 是否保留有结果的决策
            
        Returns:
            删除的记录数
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if keep_outcomes:
                    cursor.execute("""
                        DELETE FROM decisions 
                        WHERE timestamp < ? AND (outcome IS NULL OR outcome = 'ongoing')
                    """, (before_timestamp.isoformat(),))
                else:
                    cursor.execute("""
                        DELETE FROM decisions 
                        WHERE timestamp < ?
                    """, (before_timestamp.isoformat(),))
                
                count = cursor.rowcount
                logging.info(f"Cleaned up {count} old decisions")
                return count
                
        except Exception as e:
            logging.error(f"Failed to cleanup old decisions: {e}")
            return 0
    
    def vacuum(self):
        """压缩数据库文件"""
        try:
            with self._get_connection() as conn:
                conn.execute("VACUUM")
                logging.info("Database vacuumed")
        except Exception as e:
            logging.error(f"Failed to vacuum database: {e}")
    
    # === 辅助方法 ===
    
    def _row_to_decision(self, row: sqlite3.Row) -> DecisionRecord:
        """
        将数据库行转换为 DecisionRecord 对象
        
        Args:
            row: 数据库行
            
        Returns:
            DecisionRecord 对象
        """
        return DecisionRecord(
            id=row['id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            timeframe=Timeframe.from_string(row['timeframe']),
            symbol=row['symbol'],
            action=row['action'],
            quantity=row['quantity'],
            price=row['price'],
            reasoning=row['reasoning'],
            agent_name=row['agent_name'],
            conviction=row['conviction'],
            market_regime=row['market_regime'],
            technical_signals=json.loads(row['technical_signals']) if row['technical_signals'] else None,
            fundamental_data=json.loads(row['fundamental_data']) if row['fundamental_data'] else None,
            news_sentiment=json.loads(row['news_sentiment']) if row['news_sentiment'] else None,
            related_news_ids=json.loads(row['related_news_ids']) if row['related_news_ids'] else None,
            executed=bool(row['executed']),
            execution_price=row['execution_price'],
            execution_time=datetime.fromisoformat(row['execution_time']) if row['execution_time'] else None,
            outcome=row['outcome'],
            exit_price=row['exit_price'],
            exit_time=datetime.fromisoformat(row['exit_time']) if row['exit_time'] else None,
            pnl=row['pnl'],
            pnl_percent=row['pnl_percent'],
            hold_duration_days=row['hold_duration_days'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
        )
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            统计信息字典
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 决策记录统计
                cursor.execute("SELECT COUNT(*) as count FROM decisions")
                decisions_count = cursor.fetchone()['count']
                
                # 约束记录统计
                cursor.execute("SELECT COUNT(*) as count FROM constraints")
                constraints_count = cursor.fetchone()['count']
                
                # 数据库文件大小
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                return {
                    'db_path': self.db_path,
                    'db_size_mb': db_size / (1024 * 1024),
                    'decisions_count': decisions_count,
                    'constraints_count': constraints_count,
                }
                
        except Exception as e:
            logging.error(f"Failed to get database stats: {e}")
            return {}


# === 辅助函数 ===

def create_sql_store(db_path: str) -> SQLStore:
    """
    创建 SQL 数据库实例的便捷函数
    
    Args:
        db_path: 数据库文件路径
        
    Returns:
        SQLStore 实例
    """
    return SQLStore(db_path=db_path)
