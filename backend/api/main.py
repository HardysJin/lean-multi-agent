"""
FastAPI Main Application
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Optional
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.api.models import (
    DecisionRequest, DecisionResponse, DecisionQuery, DecisionStats,
    BacktestRequest, BacktestResponse, BacktestQuery, StrategyComparisonResponse,
    PortfolioSnapshot, PortfolioQuery, EquityCurveResponse,
    TradeRecord, HealthResponse, ErrorResponse
)
from backend.database import DecisionStore, BacktestStore, PortfolioStore
from backend.database.connection import get_db_manager, init_database

# 创建FastAPI应用
app = FastAPI(
    title="LLM量化交易决策系统 API",
    description="提供决策查询、回测管理、组合监控等功能",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化数据库
init_database()


# ============================================================================
# 健康检查
# ============================================================================

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    db_manager = get_db_manager()
    stats = db_manager.get_stats()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        database=f"{stats['decisions_count']} decisions, {stats['backtest_results_count']} backtests",
        timestamp=datetime.utcnow()
    )


# ============================================================================
# 决策相关API
# ============================================================================

@app.post("/api/decisions/trigger", response_model=DecisionResponse, tags=["Decisions"])
async def trigger_decision(request: DecisionRequest):
    """
    触发LLM决策
    
    注意：此接口仅保存决策记录，实际决策逻辑需要调用 Coordinator
    """
    # TODO: 实际调用 CoordinatorAgent 生成决策
    # 这里仅作为示例，保存一条测试决策
    store = DecisionStore()
    
    decision = store.save_decision(
        symbol=request.symbol,
        action="hold",  # 临时占位
        lookback_days=request.lookback_days,
        forecast_days=request.forecast_days,
        llm_provider=request.llm_provider,
        llm_model=request.llm_model,
        reasoning="API触发决策（待实现完整逻辑）",
        confidence=0.0,
    )
    
    return DecisionResponse(
        decision_id=decision.id,
        symbol=decision.symbol,
        action=decision.action,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
        recommended_strategy=decision.recommended_strategy,
        current_price=decision.current_price,
        position_size=decision.position_size,
        timestamp=decision.timestamp,
        execution_time_ms=decision.execution_time_ms
    )


@app.get("/api/decisions", response_model=List[DecisionResponse], tags=["Decisions"])
async def query_decisions(
    symbol: Optional[str] = Query(None, description="筛选标的"),
    action: Optional[str] = Query(None, description="筛选动作"),
    limit: int = Query(100, ge=1, le=1000, description="返回数量"),
    offset: int = Query(0, ge=0, description="偏移量")
):
    """查询决策记录"""
    store = DecisionStore()
    
    decisions = store.query_decisions(
        symbol=symbol,
        action=action,
        limit=limit,
        offset=offset
    )
    
    return [
        DecisionResponse(
            decision_id=d.id,
            symbol=d.symbol,
            action=d.action,
            confidence=d.confidence,
            reasoning=d.reasoning,
            recommended_strategy=d.recommended_strategy,
            current_price=d.current_price,
            position_size=d.position_size,
            timestamp=d.timestamp,
            execution_time_ms=d.execution_time_ms
        )
        for d in decisions
    ]


@app.get("/api/decisions/{decision_id}", response_model=DecisionResponse, tags=["Decisions"])
async def get_decision(decision_id: int):
    """获取单个决策"""
    store = DecisionStore()
    decision = store.get_decision(decision_id)
    
    if not decision:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")
    
    return DecisionResponse(
        decision_id=decision.id,
        symbol=decision.symbol,
        action=decision.action,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
        recommended_strategy=decision.recommended_strategy,
        current_price=decision.current_price,
        position_size=decision.position_size,
        timestamp=decision.timestamp,
        execution_time_ms=decision.execution_time_ms
    )


@app.get("/api/decisions/stats/{symbol}", response_model=DecisionStats, tags=["Decisions"])
async def get_decision_stats(
    symbol: str,
    days: Optional[int] = Query(None, description="统计天数")
):
    """获取决策统计"""
    store = DecisionStore()
    stats = store.get_decision_stats(symbol=symbol, days=days)
    
    return DecisionStats(**stats)


# ============================================================================
# 回测相关API
# ============================================================================

@app.post("/api/backtests", response_model=BacktestResponse, tags=["Backtests"])
async def run_backtest(request: BacktestRequest):
    """
    运行回测
    
    注意：此接口仅保存回测记录，实际回测需要调用 BacktestEngine
    """
    # TODO: 实际调用 BacktestEngine 运行回测
    raise HTTPException(status_code=501, detail="Backtest execution not implemented yet")


@app.get("/api/backtests", response_model=List[BacktestResponse], tags=["Backtests"])
async def query_backtests(
    strategy_name: Optional[str] = Query(None, description="策略名称"),
    symbol: Optional[str] = Query(None, description="交易标的"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """查询回测结果"""
    store = BacktestStore()
    
    results = store.query_backtest_results(
        strategy_name=strategy_name,
        symbol=symbol,
        limit=limit,
        offset=offset
    )
    
    return [
        BacktestResponse(
            backtest_id=r.id,
            strategy_name=r.strategy_name,
            symbol=r.symbol,
            start_date=r.start_date,
            end_date=r.end_date,
            initial_capital=r.initial_capital,
            final_equity=r.final_equity,
            total_return=r.total_return,
            total_return_pct=r.total_return_pct,
            benchmark_return_pct=r.benchmark_return_pct,
            excess_return_pct=r.excess_return_pct,
            sharpe_ratio=r.sharpe_ratio,
            max_drawdown_pct=r.max_drawdown_pct,
            total_trades=r.total_trades,
            win_rate=r.win_rate,
            timestamp=r.timestamp
        )
        for r in results
    ]


@app.get("/api/backtests/{backtest_id}", response_model=BacktestResponse, tags=["Backtests"])
async def get_backtest(backtest_id: int):
    """获取单个回测结果"""
    store = BacktestStore()
    result = store.get_backtest_result(backtest_id)
    
    if not result:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")
    
    return BacktestResponse(
        backtest_id=result.id,
        strategy_name=result.strategy_name,
        symbol=result.symbol,
        start_date=result.start_date,
        end_date=result.end_date,
        initial_capital=result.initial_capital,
        final_equity=result.final_equity,
        total_return=result.total_return,
        total_return_pct=result.total_return_pct,
        benchmark_return_pct=result.benchmark_return_pct,
        excess_return_pct=result.excess_return_pct,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown_pct=result.max_drawdown_pct,
        total_trades=result.total_trades,
        win_rate=result.win_rate,
        timestamp=result.timestamp
    )


@app.get("/api/backtests/best/{symbol}", response_model=BacktestResponse, tags=["Backtests"])
async def get_best_strategy(
    symbol: str,
    metric: str = Query("sharpe_ratio", description="评价指标")
):
    """获取最佳策略"""
    store = BacktestStore()
    result = store.get_best_strategy(symbol, metric=metric)
    
    if not result:
        raise HTTPException(status_code=404, detail=f"No backtest found for {symbol}")
    
    return BacktestResponse(
        backtest_id=result.id,
        strategy_name=result.strategy_name,
        symbol=result.symbol,
        start_date=result.start_date,
        end_date=result.end_date,
        initial_capital=result.initial_capital,
        final_equity=result.final_equity,
        total_return=result.total_return,
        total_return_pct=result.total_return_pct,
        benchmark_return_pct=result.benchmark_return_pct,
        excess_return_pct=result.excess_return_pct,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown_pct=result.max_drawdown_pct,
        total_trades=result.total_trades,
        win_rate=result.win_rate,
        timestamp=result.timestamp
    )


@app.get("/api/backtests/comparison/{symbol}", response_model=StrategyComparisonResponse, tags=["Backtests"])
async def compare_strategies(symbol: str, days: Optional[int] = Query(None)):
    """策略对比"""
    store = BacktestStore()
    results = store.get_strategy_comparison(symbol, days=days)
    
    if not results:
        raise HTTPException(status_code=404, detail=f"No backtests found for {symbol}")
    
    # 找出最佳策略
    best = max(results, key=lambda r: r.sharpe_ratio or 0)
    
    return StrategyComparisonResponse(
        symbol=symbol,
        strategies=[
            BacktestResponse(
                backtest_id=r.id,
                strategy_name=r.strategy_name,
                symbol=r.symbol,
                start_date=r.start_date,
                end_date=r.end_date,
                initial_capital=r.initial_capital,
                final_equity=r.final_equity,
                total_return=r.total_return,
                total_return_pct=r.total_return_pct,
                benchmark_return_pct=r.benchmark_return_pct,
                excess_return_pct=r.excess_return_pct,
                sharpe_ratio=r.sharpe_ratio,
                max_drawdown_pct=r.max_drawdown_pct,
                total_trades=r.total_trades,
                win_rate=r.win_rate,
                timestamp=r.timestamp
            )
            for r in results
        ],
        best_strategy=best.strategy_name,
        best_metric="sharpe_ratio"
    )


@app.get("/api/backtests/{backtest_id}/trades", response_model=List[TradeRecord], tags=["Backtests"])
async def get_backtest_trades(backtest_id: int):
    """获取回测的交易记录"""
    store = BacktestStore()
    trades = store.get_backtest_trades(backtest_id)
    
    return [
        TradeRecord(
            trade_id=t.id,
            timestamp=t.timestamp,
            symbol=t.symbol,
            action=t.action,
            quantity=t.quantity,
            price=t.price,
            commission=t.commission,
            total_cost=t.total_cost,
            profit_loss=t.profit_loss,
            profit_loss_pct=t.profit_loss_pct,
            strategy=t.strategy
        )
        for t in trades
    ]


# ============================================================================
# 组合相关API
# ============================================================================

@app.get("/api/portfolio/latest", response_model=PortfolioSnapshot, tags=["Portfolio"])
async def get_latest_portfolio():
    """获取最新组合快照"""
    store = PortfolioStore()
    snapshot = store.get_latest_snapshot()
    
    if not snapshot:
        raise HTTPException(status_code=404, detail="No portfolio snapshot found")
    
    import json
    positions = json.loads(snapshot.positions) if snapshot.positions else {}
    
    return PortfolioSnapshot(
        snapshot_id=snapshot.id,
        timestamp=snapshot.timestamp,
        total_equity=snapshot.total_equity,
        cash=snapshot.cash,
        positions_value=snapshot.positions_value,
        positions=positions,
        total_return_pct=snapshot.total_return_pct,
        current_drawdown_pct=snapshot.current_drawdown_pct
    )


@app.get("/api/portfolio/snapshots", response_model=List[PortfolioSnapshot], tags=["Portfolio"])
async def query_portfolio_snapshots(
    limit: int = Query(100, ge=1, le=1000),
    days: Optional[int] = Query(None, description="查询天数")
):
    """查询组合快照历史"""
    store = PortfolioStore()
    
    from datetime import timedelta
    start_date = datetime.utcnow() - timedelta(days=days) if days else None
    
    snapshots = store.query_snapshots(start_date=start_date, limit=limit)
    
    import json
    return [
        PortfolioSnapshot(
            snapshot_id=s.id,
            timestamp=s.timestamp,
            total_equity=s.total_equity,
            cash=s.cash,
            positions_value=s.positions_value,
            positions=json.loads(s.positions) if s.positions else {},
            total_return_pct=s.total_return_pct,
            current_drawdown_pct=s.current_drawdown_pct
        )
        for s in snapshots
    ]


@app.get("/api/portfolio/equity-curve", response_model=EquityCurveResponse, tags=["Portfolio"])
async def get_equity_curve(days: int = Query(30, ge=1, le=365)):
    """获取权益曲线"""
    store = PortfolioStore()
    curve_data = store.get_equity_curve(days=days)
    
    if not curve_data:
        raise HTTPException(status_code=404, detail="No equity curve data found")
    
    return EquityCurveResponse(
        data=curve_data,
        total_points=len(curve_data),
        start_date=curve_data[0]['timestamp'],
        end_date=curve_data[-1]['timestamp']
    )


@app.get("/api/portfolio/stats", tags=["Portfolio"])
async def get_portfolio_stats(days: Optional[int] = Query(None)):
    """获取组合性能统计"""
    store = PortfolioStore()
    stats = store.get_performance_stats(days=days)
    
    return stats


# ============================================================================
# 数据库管理API
# ============================================================================

@app.get("/api/database/stats", tags=["Database"])
async def get_database_stats():
    """获取数据库统计信息"""
    db_manager = get_db_manager()
    return db_manager.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
