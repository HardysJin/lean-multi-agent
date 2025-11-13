"""
Portfolio Manager
统一管理投资组合状态，包括持仓、现金、交易历史、决策历史等
保证状态变更的原子性和一致性
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import threading
from dataclasses import dataclass, field, asdict
from copy import deepcopy

from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    shares: int
    total_cost: float  # 总成本
    entry_price: float  # 平均入场价格
    
    @property
    def avg_cost(self) -> float:
        """平均成本"""
        return self.total_cost / self.shares if self.shares > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'shares': self.shares,
            'total_cost': self.total_cost,
            'entry_price': self.entry_price,
            'avg_cost': self.avg_cost
        }


@dataclass
class Trade:
    """交易记录"""
    date: str
    symbol: str
    action: str  # 'buy' or 'sell'
    shares: int
    price: float
    cost: float  # 正数表示支出，负数表示收入
    strategy: str
    cash_after: float
    position_after: int
    profit: Optional[float] = None  # 仅卖出时有值
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class Decision:
    """决策记录"""
    date: str
    decision: Dict[str, Any]
    daily_executions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class PortfolioSnapshot:
    """组合快照"""
    date: str
    cash: float
    positions: Dict[str, Position]
    total_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'date': self.date,
            'cash': self.cash,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'total_value': self.total_value
        }


class PortfolioManager:
    """
    投资组合管理器
    
    职责：
    1. 管理现金、持仓状态
    2. 记录交易历史
    3. 记录决策历史
    4. 记录组合价值历史
    5. 保证所有状态变更的原子性
    6. 提供线程安全的读写接口
    """
    
    def __init__(self, initial_capital: float):
        """
        初始化 Portfolio Manager
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        
        # 核心状态（需要线程安全保护）
        self._lock = threading.RLock()
        self._cash = initial_capital
        self._positions: Dict[str, Position] = {}  # {symbol: Position}
        
        # 历史记录
        self._trades: List[Trade] = []
        self._decisions: List[Decision] = []
        self._portfolio_values: List[Dict[str, Any]] = []
        
        logger.info(f"Portfolio Manager 初始化完成，初始资金: ${initial_capital:,.2f}")
    
    # ==================== 查询接口 ====================
    
    @property
    def cash(self) -> float:
        """获取当前现金"""
        with self._lock:
            return self._cash
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        获取指定标的的持仓
        
        Args:
            symbol: 标的代码
            
        Returns:
            持仓信息，如果没有持仓则返回 None
        """
        with self._lock:
            return deepcopy(self._positions.get(symbol))
    
    def get_position_shares(self, symbol: str) -> int:
        """
        获取指定标的的持仓股数
        
        Args:
            symbol: 标的代码
            
        Returns:
            持仓股数，如果没有持仓则返回 0
        """
        with self._lock:
            pos = self._positions.get(symbol)
            return pos.shares if pos else 0
    
    def get_position_entry_price(self, symbol: str) -> float:
        """
        获取指定标的的入场价格
        
        Args:
            symbol: 标的代码
            
        Returns:
            入场价格，如果没有持仓则返回 0
        """
        with self._lock:
            pos = self._positions.get(symbol)
            return pos.entry_price if pos else 0.0
    
    def has_position(self, symbol: str) -> bool:
        """
        检查是否持有指定标的
        
        Args:
            symbol: 标的代码
            
        Returns:
            是否持有
        """
        with self._lock:
            return symbol in self._positions and self._positions[symbol].shares > 0
    
    def get_all_positions(self) -> Dict[str, Position]:
        """获取所有持仓"""
        with self._lock:
            return deepcopy(self._positions)
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        计算当前组合价值
        
        Args:
            current_prices: {symbol: price} 当前价格字典
            
        Returns:
            组合总价值
        """
        with self._lock:
            position_value = 0.0
            for symbol, pos in self._positions.items():
                if symbol in current_prices:
                    position_value += pos.shares * current_prices[symbol]
            return self._cash + position_value
    
    def get_trades(self) -> List[Trade]:
        """获取所有交易记录"""
        with self._lock:
            return deepcopy(self._trades)
    
    def get_decisions(self) -> List[Decision]:
        """获取所有决策记录"""
        with self._lock:
            return deepcopy(self._decisions)
    
    def get_portfolio_history(self) -> List[Dict[str, Any]]:
        """获取组合价值历史"""
        with self._lock:
            return deepcopy(self._portfolio_values)
    
    def get_last_buy_price(self) -> Optional[float]:
        """
        获取最后一次买入的价格
        
        Returns:
            最后一次买入价格，如果没有买入记录则返回 None
        """
        with self._lock:
            for trade in reversed(self._trades):
                if trade.action == 'buy':
                    return trade.price
            return None
    
    # ==================== 状态变更接口（原子性保证）====================
    
    def execute_buy(
        self,
        symbol: str,
        shares: int,
        price: float,
        commission: float,
        date: str,
        strategy: str
    ) -> bool:
        """
        执行买入操作（原子性）
        
        Args:
            symbol: 标的代码
            shares: 买入股数
            price: 买入价格
            commission: 手续费率
            date: 交易日期
            strategy: 策略名称
            
        Returns:
            是否执行成功
        """
        with self._lock:
            # 计算成本
            cost = shares * price * (1 + commission)
            
            # 检查现金充足性
            if cost > self._cash:
                logger.warning(f"现金不足，无法买入 {shares} 股 {symbol}")
                return False
            
            # 扣除现金
            self._cash -= cost
            
            # 更新持仓
            if symbol not in self._positions:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    shares=shares,
                    total_cost=cost,
                    entry_price=price
                )
            else:
                pos = self._positions[symbol]
                new_shares = pos.shares + shares
                new_cost = pos.total_cost + cost
                pos.shares = new_shares
                pos.total_cost = new_cost
                pos.entry_price = new_cost / new_shares  # 更新平均入场价
            
            # 记录交易
            trade = Trade(
                date=date,
                symbol=symbol,
                action='buy',
                shares=shares,
                price=price,
                cost=cost,
                strategy=strategy,
                cash_after=self._cash,
                position_after=self._positions[symbol].shares
            )
            self._trades.append(trade)
            
            logger.info(f"✓ 买入 {shares} 股 {symbol} @ ${price:.2f}, 成本: ${cost:.2f}")
            return True
    
    def execute_sell(
        self,
        symbol: str,
        shares: int,
        price: float,
        commission: float,
        date: str,
        strategy: str
    ) -> bool:
        """
        执行卖出操作（原子性）
        
        Args:
            symbol: 标的代码
            shares: 卖出股数
            price: 卖出价格
            commission: 手续费率
            date: 交易日期
            strategy: 策略名称
            
        Returns:
            是否执行成功
        """
        with self._lock:
            # 检查持仓
            if symbol not in self._positions:
                logger.warning(f"无 {symbol} 持仓，无法卖出")
                return False
            
            pos = self._positions[symbol]
            if pos.shares < shares:
                logger.warning(f"{symbol} 持仓不足，无法卖出 {shares} 股（当前持有 {pos.shares} 股）")
                return False
            
            # 计算收入和盈亏
            proceeds = shares * price * (1 - commission)
            sold_cost = (pos.total_cost / pos.shares) * shares
            profit = proceeds - sold_cost
            
            # 增加现金
            self._cash += proceeds
            
            # 更新持仓
            pos.shares -= shares
            pos.total_cost -= sold_cost
            
            # 如果清仓，删除持仓记录
            if pos.shares == 0:
                del self._positions[symbol]
                remaining_shares = 0
            else:
                remaining_shares = pos.shares
            
            # 记录交易
            trade = Trade(
                date=date,
                symbol=symbol,
                action='sell',
                shares=-shares,  # 负数表示卖出
                price=price,
                cost=-proceeds,  # 负数表示收入
                strategy=strategy,
                cash_after=self._cash,
                position_after=remaining_shares,
                profit=profit
            )
            self._trades.append(trade)
            
            logger.info(f"✓ 卖出 {shares} 股 {symbol} @ ${price:.2f}, 收入: ${proceeds:.2f}, 盈亏: ${profit:+,.2f}")
            return True
    
    def record_decision(self, date: str, decision: Dict[str, Any]) -> None:
        """
        记录决策
        
        Args:
            date: 决策日期
            decision: 决策内容
        """
        with self._lock:
            dec = Decision(
                date=date,
                decision=decision,
                daily_executions=[]
            )
            self._decisions.append(dec)
            logger.debug(f"记录决策: {date}")
    
    def add_daily_execution(self, execution: Dict[str, Any]) -> None:
        """
        向最近的决策添加每日执行记录
        
        Args:
            execution: 执行记录
        """
        with self._lock:
            if self._decisions:
                self._decisions[-1].daily_executions.append(execution)
    
    def record_portfolio_value(
        self,
        date: str,
        current_prices: Dict[str, float]
    ) -> None:
        """
        记录组合价值快照
        
        Args:
            date: 日期
            current_prices: {symbol: price} 当前价格字典
        """
        with self._lock:
            total_value = self.get_portfolio_value(current_prices)
            
            snapshot = {
                'date': date,
                'value': total_value,
                'cash': self._cash,
                'positions': {k: v.to_dict() for k, v in self._positions.items()}
            }
            self._portfolio_values.append(snapshot)
            logger.debug(f"记录组合价值: {date}, ${total_value:,.2f}")
    
    # ==================== 统计分析接口 ====================
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取组合汇总信息
        
        Returns:
            汇总字典
        """
        with self._lock:
            total_trades = len([t for t in self._trades if t.action in ['buy', 'sell']])
            sell_trades = [t for t in self._trades if t.action == 'sell' and t.profit is not None]
            winning_trades = sum(1 for t in sell_trades if t.profit > 0)
            win_rate = winning_trades / len(sell_trades) if sell_trades else 0
            
            return {
                'initial_capital': self.initial_capital,
                'current_cash': self._cash,
                'positions': {k: v.to_dict() for k, v in self._positions.items()},
                'total_trades': total_trades,
                'sell_trades': len(sell_trades),
                'winning_trades': winning_trades,
                'win_rate': win_rate
            }
    
    def get_portfolio_snapshot(
        self,
        symbol: str,
        current_price: float
    ) -> Dict[str, Any]:
        """
        获取组合快照（用于传递给 Coordinator）
        
        Args:
            symbol: 标的代码
            current_price: 当前价格
            
        Returns:
            组合快照字典
        """
        with self._lock:
            holdings = {}
            if symbol in self._positions and self._positions[symbol].shares > 0:
                pos = self._positions[symbol]
                holdings[symbol] = {
                    'shares': pos.shares,
                    'current_price': current_price,
                    'market_value': pos.shares * current_price,
                    'avg_cost': pos.avg_cost,
                    'unrealized_pnl': (current_price - pos.avg_cost) * pos.shares
                }
            
            total_value = self._cash + sum(h['market_value'] for h in holdings.values())
            
            return {
                'cash': self._cash,
                'holdings': holdings,
                'total_value': total_value
            }
    
    def calculate_last_period_pnl(self) -> float:
        """
        计算上期盈亏
        
        Returns:
            上期盈亏金额
        """
        with self._lock:
            if len(self._portfolio_values) < 2:
                return 0.0
            
            current = self._portfolio_values[-1]['value']
            previous = self._portfolio_values[-2]['value']
            return current - previous
    
    # ==================== 重置接口 ====================
    
    def reset(self, initial_capital: Optional[float] = None) -> None:
        """
        重置组合状态（用于新的回测）
        
        Args:
            initial_capital: 新的初始资金，如果为 None 则使用原始初始资金
        """
        with self._lock:
            if initial_capital is not None:
                self.initial_capital = initial_capital
            
            self._cash = self.initial_capital
            self._positions.clear()
            self._trades.clear()
            self._decisions.clear()
            self._portfolio_values.clear()
            
            logger.info(f"Portfolio Manager 已重置，初始资金: ${self.initial_capital:,.2f}")
