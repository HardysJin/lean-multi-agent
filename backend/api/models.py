"""
Pydantic models for API requests and responses
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# Decision Models
# ============================================================================

class DecisionRequest(BaseModel):
    """触发决策请求"""
    symbol: str = Field(..., description="交易标的", example="SPY")
    lookback_days: int = Field(7, description="回看天数", ge=1, le=365)
    forecast_days: int = Field(7, description="预测天数", ge=1, le=90)
    llm_provider: str = Field("openai", description="LLM提供商")
    llm_model: str = Field("gpt-4o", description="LLM模型")


class DecisionResponse(BaseModel):
    """决策响应"""
    decision_id: int
    symbol: str
    action: str
    confidence: Optional[float]
    reasoning: str
    recommended_strategy: Optional[str]
    current_price: Optional[float]
    position_size: Optional[float]
    timestamp: datetime
    execution_time_ms: Optional[float]


class DecisionQuery(BaseModel):
    """决策查询参数"""
    symbol: Optional[str] = None
    action: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)


class DecisionStats(BaseModel):
    """决策统计"""
    total_decisions: int
    buy_count: int
    sell_count: int
    hold_count: int
    avg_confidence: Optional[float]
    avg_execution_time_ms: Optional[float]


# ============================================================================
# Backtest Models
# ============================================================================

class BacktestRequest(BaseModel):
    """回测请求"""
    strategy_name: str = Field(..., description="策略名称")
    symbol: str = Field(..., description="交易标的")
    start_date: datetime = Field(..., description="开始日期")
    end_date: datetime = Field(..., description="结束日期")
    initial_capital: float = Field(100000.0, description="初始资金", gt=0)
    strategy_params: Optional[Dict[str, Any]] = Field(None, description="策略参数")


class BacktestResponse(BaseModel):
    """回测响应"""
    backtest_id: int
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_equity: float
    total_return: float
    total_return_pct: float
    benchmark_return_pct: Optional[float]
    excess_return_pct: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown_pct: Optional[float]
    total_trades: int
    win_rate: Optional[float]
    timestamp: datetime


class BacktestQuery(BaseModel):
    """回测查询参数"""
    strategy_name: Optional[str] = None
    symbol: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)


class StrategyComparisonResponse(BaseModel):
    """策略对比响应"""
    symbol: str
    strategies: List[BacktestResponse]
    best_strategy: str
    best_metric: str


# ============================================================================
# Portfolio Models
# ============================================================================

class PortfolioSnapshot(BaseModel):
    """投资组合快照"""
    snapshot_id: int
    timestamp: datetime
    total_equity: float
    cash: float
    positions_value: float
    positions: Dict[str, Dict[str, Any]]
    total_return_pct: Optional[float]
    current_drawdown_pct: Optional[float]


class PortfolioQuery(BaseModel):
    """组合查询参数"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(100, ge=1, le=1000)


class EquityCurveResponse(BaseModel):
    """权益曲线响应"""
    data: List[Dict[str, Any]]
    total_points: int
    start_date: datetime
    end_date: datetime


# ============================================================================
# Trade Models
# ============================================================================

class TradeRecord(BaseModel):
    """交易记录"""
    trade_id: int
    timestamp: datetime
    symbol: str
    action: str
    quantity: int
    price: float
    commission: float
    total_cost: float
    profit_loss: Optional[float]
    profit_loss_pct: Optional[float]
    strategy: Optional[str]


# ============================================================================
# System Models
# ============================================================================

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    database: str
    timestamp: datetime


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
