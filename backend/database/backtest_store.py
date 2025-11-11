"""
Backtest results storage layer
"""

import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import desc, and_
from sqlalchemy.orm import Session

from .models import BacktestResult, TradeRecord
from .connection import get_db_manager


class BacktestStore:
    """回测结果存储层"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
    
    def save_backtest_result(
        self,
        strategy_name: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        final_equity: float,
        total_return: float,
        total_return_pct: float,
        strategy_params: Optional[Dict] = None,
        benchmark_return_pct: Optional[float] = None,
        excess_return_pct: Optional[float] = None,
        sharpe_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        max_drawdown_pct: Optional[float] = None,
        volatility: Optional[float] = None,
        total_trades: int = 0,
        winning_trades: int = 0,
        losing_trades: int = 0,
        win_rate: Optional[float] = None,
        avg_profit: Optional[float] = None,
        avg_loss: Optional[float] = None,
        profit_factor: Optional[float] = None,
        largest_win: Optional[float] = None,
        largest_loss: Optional[float] = None,
        avg_bars_held: Optional[float] = None,
        execution_time_ms: Optional[float] = None,
        notes: Optional[str] = None,
        **kwargs
    ) -> BacktestResult:
        """
        保存回测结果
        
        Args:
            strategy_name: 策略名称
            symbol: 交易标的
            start_date: 回测开始日期
            end_date: 回测结束日期
            initial_capital: 初始资金
            final_equity: 最终权益
            total_return: 总收益
            total_return_pct: 总收益率
            其他性能指标...
            
        Returns:
            保存的回测结果记录
        """
        with self.db_manager.session_scope() as session:
            result = BacktestResult(
                timestamp=datetime.utcnow(),
                strategy_name=strategy_name,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                strategy_params=json.dumps(strategy_params) if strategy_params else None,
                final_equity=final_equity,
                total_return=total_return,
                total_return_pct=total_return_pct,
                benchmark_return_pct=benchmark_return_pct,
                excess_return_pct=excess_return_pct,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_pct=max_drawdown_pct,
                volatility=volatility,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_profit=avg_profit,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                largest_win=largest_win,
                largest_loss=largest_loss,
                avg_bars_held=avg_bars_held,
                execution_time_ms=execution_time_ms,
                notes=notes,
            )
            
            session.add(result)
            session.flush()
            session.refresh(result)
            session.expunge(result)
        
        return result
    
    def save_backtest_trades(
        self,
        backtest_id: int,
        trades: List[Dict[str, Any]]
    ) -> List[TradeRecord]:
        """
        保存回测的交易记录
        
        Args:
            backtest_id: 回测结果ID
            trades: 交易记录列表，每个元素包含:
                    {timestamp, symbol, action, quantity, price, 
                     commission, slippage, total_cost, profit_loss, 
                     profit_loss_pct, strategy}
                     
        Returns:
            保存的交易记录列表
        """
        trade_records = []
        
        with self.db_manager.session_scope() as session:
            for trade_data in trades:
                trade = TradeRecord(
                    timestamp=trade_data.get('timestamp', datetime.utcnow()),
                    symbol=trade_data['symbol'],
                    action=trade_data['action'],
                    quantity=trade_data['quantity'],
                    price=trade_data['price'],
                    commission=trade_data.get('commission', 0.0),
                    slippage=trade_data.get('slippage', 0.0),
                    total_cost=trade_data['total_cost'],
                    profit_loss=trade_data.get('profit_loss'),
                    profit_loss_pct=trade_data.get('profit_loss_pct'),
                    strategy=trade_data.get('strategy'),
                    is_backtest=True,
                    backtest_id=backtest_id,
                )
                session.add(trade)
                trade_records.append(trade)
            
            session.flush()
            
            # 刷新对象
            for trade in trade_records:
                session.refresh(trade)
        
        return trade_records
    
    def get_backtest_result(self, backtest_id: int) -> Optional[BacktestResult]:
        """根据ID获取回测结果"""
        with self.db_manager.session_scope() as session:
            result = session.query(BacktestResult).filter(
                BacktestResult.id == backtest_id
            ).first()
            
            if result:
                session.expunge(result)
        
        return result
    
    def query_backtest_results(
        self,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[BacktestResult]:
        """
        查询回测结果
        
        Args:
            strategy_name: 筛选策略名
            symbol: 筛选标的
            start_date: 筛选时间范围起始
            end_date: 筛选时间范围结束
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            回测结果列表
        """
        with self.db_manager.session_scope() as session:
            query = session.query(BacktestResult)
            
            if strategy_name:
                query = query.filter(BacktestResult.strategy_name == strategy_name)
            if symbol:
                query = query.filter(BacktestResult.symbol == symbol)
            if start_date:
                query = query.filter(BacktestResult.timestamp >= start_date)
            if end_date:
                query = query.filter(BacktestResult.timestamp <= end_date)
            
            query = query.order_by(desc(BacktestResult.timestamp))
            query = query.limit(limit).offset(offset)
            
            results = query.all()
            
            for result in results:
                session.expunge(result)
        
        return results
    
    def get_backtest_trades(self, backtest_id: int) -> List[TradeRecord]:
        """获取回测的交易记录"""
        with self.db_manager.session_scope() as session:
            trades = session.query(TradeRecord).filter(
                TradeRecord.backtest_id == backtest_id
            ).order_by(TradeRecord.timestamp).all()
            
            for trade in trades:
                session.expunge(trade)
        
        return trades
    
    def get_best_strategy(
        self,
        symbol: str,
        metric: str = 'sharpe_ratio',
        days: Optional[int] = None,
    ) -> Optional[BacktestResult]:
        """
        获取最佳策略
        
        Args:
            symbol: 交易标的
            metric: 评价指标 (sharpe_ratio/total_return_pct/win_rate)
            days: 可选时间范围
            
        Returns:
            最佳策略的回测结果
        """
        with self.db_manager.session_scope() as session:
            query = session.query(BacktestResult).filter(
                BacktestResult.symbol == symbol
            )
            
            if days:
                start_date = datetime.utcnow() - timedelta(days=days)
                query = query.filter(BacktestResult.timestamp >= start_date)
            
            # 根据指标排序
            if metric == 'sharpe_ratio':
                query = query.order_by(desc(BacktestResult.sharpe_ratio))
            elif metric == 'total_return_pct':
                query = query.order_by(desc(BacktestResult.total_return_pct))
            elif metric == 'win_rate':
                query = query.order_by(desc(BacktestResult.win_rate))
            else:
                query = query.order_by(desc(BacktestResult.sharpe_ratio))
            
            result = query.first()
            
            if result:
                session.expunge(result)
        
        return result
    
    def get_strategy_comparison(
        self,
        symbol: str,
        days: Optional[int] = None,
    ) -> List[BacktestResult]:
        """
        获取策略对比数据
        
        Args:
            symbol: 交易标的
            days: 可选时间范围
            
        Returns:
            所有策略的最新回测结果
        """
        with self.db_manager.session_scope() as session:
            # 获取每个策略的最新结果
            query = session.query(BacktestResult).filter(
                BacktestResult.symbol == symbol
            )
            
            if days:
                start_date = datetime.utcnow() - timedelta(days=days)
                query = query.filter(BacktestResult.timestamp >= start_date)
            
            all_results = query.all()
            
            # 按策略分组，取最新的
            strategy_results = {}
            for result in all_results:
                strategy = result.strategy_name
                if strategy not in strategy_results:
                    strategy_results[strategy] = result
                elif result.timestamp > strategy_results[strategy].timestamp:
                    strategy_results[strategy] = result
            
            results = list(strategy_results.values())
            
            for result in results:
                session.expunge(result)
        
        return results
    
    def delete_old_results(self, days: int = 365) -> int:
        """
        删除旧的回测结果
        
        Args:
            days: 保留天数
            
        Returns:
            删除的记录数
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        with self.db_manager.session_scope() as session:
            deleted_count = session.query(BacktestResult).filter(
                BacktestResult.timestamp < cutoff_date
            ).delete()
        
        return deleted_count
