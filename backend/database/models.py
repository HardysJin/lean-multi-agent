"""
SQLAlchemy ORM models for persistence
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class DecisionRecord(Base):
    """LLM决策记录"""
    __tablename__ = 'decisions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # 决策参数
    symbol = Column(String(20), nullable=False, index=True)
    lookback_days = Column(Integer, nullable=False)
    forecast_days = Column(Integer, nullable=False)
    
    # LLM配置
    llm_provider = Column(String(20), nullable=False)  # openai/anthropic/deepseek
    llm_model = Column(String(50), nullable=False)
    
    # 决策结果
    action = Column(String(20), nullable=False)  # buy/sell/hold
    confidence = Column(Float)  # 0-1
    reasoning = Column(Text)  # LLM推理过程
    
    # 策略推荐
    recommended_strategy = Column(String(50))
    strategy_reasoning = Column(Text)
    
    # 市场数据
    current_price = Column(Float)
    position_size = Column(Float)  # 建议仓位大小
    
    # 风控信息
    risk_score = Column(Float)
    max_drawdown = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    
    # 元数据
    execution_time_ms = Column(Float)  # 执行耗时
    is_executed = Column(Boolean, default=False)  # 是否已执行
    
    # 关联交易记录
    trades = relationship("TradeRecord", back_populates="decision")
    
    # 索引
    __table_args__ = (
        Index('idx_decisions_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_decisions_action_timestamp', 'action', 'timestamp'),
    )
    
    def __repr__(self):
        return (f"<DecisionRecord(id={self.id}, symbol={self.symbol}, "
                f"action={self.action}, timestamp={self.timestamp})>")


class TradeRecord(Base):
    """交易记录"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # 关联决策
    decision_id = Column(Integer, ForeignKey('decisions.id'), nullable=True)
    decision = relationship("DecisionRecord", back_populates="trades")
    
    # 交易信息
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(String(10), nullable=False)  # buy/sell
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    
    # 成本
    commission = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)
    total_cost = Column(Float, nullable=False)  # 总成本（含佣金滑点）
    
    # 盈亏（如果是卖出）
    profit_loss = Column(Float)
    profit_loss_pct = Column(Float)
    
    # 策略信息
    strategy = Column(String(50))
    
    # 元数据
    is_backtest = Column(Boolean, default=False)  # 是否来自回测
    backtest_id = Column(Integer, ForeignKey('backtest_results.id'), nullable=True)
    
    # 索引
    __table_args__ = (
        Index('idx_trades_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_trades_strategy', 'strategy'),
    )
    
    def __repr__(self):
        return (f"<TradeRecord(id={self.id}, symbol={self.symbol}, "
                f"action={self.action}, quantity={self.quantity}, price={self.price})>")


class BacktestResult(Base):
    """回测结果记录"""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # 回测配置
    strategy_name = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    
    # 策略参数（JSON字符串）
    strategy_params = Column(Text)
    
    # 性能指标
    final_equity = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    total_return_pct = Column(Float, nullable=False)
    
    # 基准对比
    benchmark_return_pct = Column(Float)
    excess_return_pct = Column(Float)
    
    # 风险指标
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    max_drawdown_pct = Column(Float)
    volatility = Column(Float)
    
    # 交易统计
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float)
    
    # 盈亏统计
    avg_profit = Column(Float)
    avg_loss = Column(Float)
    profit_factor = Column(Float)
    largest_win = Column(Float)
    largest_loss = Column(Float)
    
    # 持仓统计
    avg_bars_held = Column(Float)
    
    # 元数据
    execution_time_ms = Column(Float)
    notes = Column(Text)
    
    # 索引
    __table_args__ = (
        Index('idx_backtest_strategy_symbol', 'strategy_name', 'symbol'),
        Index('idx_backtest_date_range', 'start_date', 'end_date'),
    )
    
    def __repr__(self):
        return (f"<BacktestResult(id={self.id}, strategy={self.strategy_name}, "
                f"symbol={self.symbol}, return={self.total_return_pct:.2f}%)>")


class PortfolioSnapshot(Base):
    """投资组合快照"""
    __tablename__ = 'portfolio_snapshots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # 组合价值
    total_equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)
    
    # 收益指标
    total_return = Column(Float)
    total_return_pct = Column(Float)
    daily_return = Column(Float)
    daily_return_pct = Column(Float)
    
    # 持仓详情（JSON字符串）
    positions = Column(Text)  # {symbol: {quantity, price, value, weight}}
    
    # 风险指标
    current_drawdown = Column(Float)
    current_drawdown_pct = Column(Float)
    
    # 元数据
    notes = Column(Text)
    
    # 索引
    __table_args__ = (
        Index('idx_portfolio_timestamp', 'timestamp'),
    )
    
    def __repr__(self):
        return (f"<PortfolioSnapshot(id={self.id}, timestamp={self.timestamp}, "
                f"equity={self.total_equity:.2f})>")
