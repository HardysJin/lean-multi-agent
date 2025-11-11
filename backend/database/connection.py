"""
Database connection and session management
"""

import os
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Generator
from sqlalchemy import create_engine, event, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base


class DatabaseManager:
    """数据库连接管理器"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        初始化数据库管理器
        
        Args:
            db_path: 数据库文件路径，默认为 Data/sql/trading_system.db
        """
        if db_path is None:
            # 默认路径
            project_root = Path(__file__).parent.parent.parent
            db_dir = project_root / 'Data' / 'sql'
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / 'trading_system.db')
        
        self.db_path = db_path
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        
    def _create_engine(self) -> Engine:
        """创建数据库引擎"""
        # SQLite连接字符串
        db_url = f'sqlite:///{self.db_path}'
        
        # 创建引擎
        # check_same_thread=False 允许多线程访问
        # StaticPool 用于测试环境
        if ':memory:' in self.db_path:
            # 内存数据库，用于测试
            engine = create_engine(
                db_url,
                connect_args={'check_same_thread': False},
                poolclass=StaticPool,
                echo=False,
            )
        else:
            # 文件数据库
            engine = create_engine(
                db_url,
                connect_args={'check_same_thread': False},
                echo=False,
            )
        
        # 启用外键约束（SQLite默认关闭）
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
            cursor.close()
        
        return engine
    
    def create_tables(self):
        """创建所有表"""
        Base.metadata.create_all(bind=self.engine)
        
    def drop_tables(self):
        """删除所有表（危险操作！）"""
        Base.metadata.drop_all(bind=self.engine)
        
    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        提供事务性会话上下文
        
        Usage:
            with db_manager.session_scope() as session:
                session.add(record)
                # 自动提交或回滚
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def close(self):
        """关闭数据库连接"""
        self.engine.dispose()
    
    def get_stats(self) -> dict:
        """获取数据库统计信息"""
        from .models import DecisionRecord, TradeRecord, BacktestResult, PortfolioSnapshot
        
        with self.session_scope() as session:
            stats = {
                'db_path': self.db_path,
                'db_size_mb': os.path.getsize(self.db_path) / 1024 / 1024 if os.path.exists(self.db_path) else 0,
                'decisions_count': session.query(DecisionRecord).count(),
                'trades_count': session.query(TradeRecord).count(),
                'backtest_results_count': session.query(BacktestResult).count(),
                'portfolio_snapshots_count': session.query(PortfolioSnapshot).count(),
            }
        
        return stats


# 全局数据库实例（单例模式）
_db_manager: Optional[DatabaseManager] = None


def get_db_manager(db_path: Optional[str] = None) -> DatabaseManager:
    """
    获取全局数据库管理器实例（单例）
    
    Args:
        db_path: 数据库路径，仅在首次调用时有效
        
    Returns:
        DatabaseManager实例
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(db_path)
    return _db_manager


def init_database(db_path: Optional[str] = None, drop_existing: bool = False):
    """
    初始化数据库
    
    Args:
        db_path: 数据库路径
        drop_existing: 是否删除现有表
    """
    db_manager = get_db_manager(db_path)
    
    if drop_existing:
        print("⚠️  删除现有表...")
        db_manager.drop_tables()
    
    print("✓ 创建数据库表...")
    db_manager.create_tables()
    
    print(f"✓ 数据库初始化完成: {db_manager.db_path}")
    
    return db_manager
