"""
Execution Logger - 执行日志系统

多层级可配置的日志系统，支持：
- 5个日志级别（CRITICAL/ERROR/WARNING/INFO/DEBUG）
- 多种输出目标（Console/File/Database）
- 结构化日志格式
- 性能追踪
- 可视化执行轨迹

使用示例：
    # 创建logger
    logger = ExecutionLogger(
        level=LogLevel.INFO,
        enable_console=True,
        enable_file=True,
        enable_database=False
    )
    
    # 记录决策
    logger.log_decision(
        agent_name="meta_agent",
        symbol="AAPL",
        action="BUY",
        conviction=8,
        details={...}
    )
    
    # 记录工具调用
    logger.log_tool_call(
        agent_name="technical",
        tool_name="calculate_indicators",
        arguments={...},
        result={...},
        execution_time_ms=123.45
    )
    
    # 查看执行轨迹
    trace = logger.get_execution_trace(symbol="AAPL", date=datetime(2024, 10, 15))
    logger.visualize_trace(trace)
"""

import os
import json
import logging
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from pathlib import Path
import sqlite3
from collections import defaultdict


class LogLevel(IntEnum):
    """
    日志级别枚举
    
    级别越高，输出的信息越少
    """
    DEBUG = 10      # 详细调试信息（所有参数、返回值）
    INFO = 20       # 重要业务信息（决策、工具调用）⭐ 推荐默认
    WARNING = 30    # 警告信息（非致命问题）
    ERROR = 40      # 错误信息（需要关注）
    CRITICAL = 50   # 严重错误（系统级问题）
    
    @classmethod
    def from_string(cls, level_str: str) -> 'LogLevel':
        """从字符串创建"""
        level_map = {
            'DEBUG': cls.DEBUG,
            'INFO': cls.INFO,
            'WARNING': cls.WARNING,
            'ERROR': cls.ERROR,
            'CRITICAL': cls.CRITICAL
        }
        return level_map.get(level_str.upper(), cls.INFO)


class LogCategory(Enum):
    """日志类别"""
    DECISION = "decision"           # 决策相关
    TOOL_CALL = "tool_call"        # 工具调用
    CACHE = "cache"                # 缓存操作
    MEMORY = "memory"              # Memory操作
    TIMEFRAME = "timeframe"        # 时间尺度切换
    ESCALATION = "escalation"      # 反向传导
    ERROR = "error"                # 错误
    PERFORMANCE = "performance"    # 性能统计


@dataclass
class LogEntry:
    """
    日志条目
    
    结构化的日志记录
    """
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    agent_name: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    # 性能相关
    execution_time_ms: Optional[float] = None
    
    # 回测相关
    backtest_date: Optional[datetime] = None  # 回测时间点
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    
    # 追踪相关
    session_id: Optional[str] = None
    parent_log_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['level'] = self.level.name
        data['category'] = self.category.value
        if self.backtest_date:
            data['backtest_date'] = self.backtest_date.isoformat()
        return data
    
    def to_json(self) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)
    
    def to_console_string(self, colored: bool = True) -> str:
        """
        转换为控制台输出格式
        
        Args:
            colored: 是否使用颜色（需要colorama库）
        """
        timestamp_str = self.timestamp.strftime('%H:%M:%S.%f')[:-3]
        level_str = self.level.name.ljust(8)
        agent_str = self.agent_name.ljust(20)
        category_str = f"[{self.category.value}]".ljust(15)
        
        # 颜色代码（ANSI）
        if colored:
            level_colors = {
                LogLevel.DEBUG: '\033[36m',      # Cyan
                LogLevel.INFO: '\033[32m',       # Green
                LogLevel.WARNING: '\033[33m',    # Yellow
                LogLevel.ERROR: '\033[31m',      # Red
                LogLevel.CRITICAL: '\033[1;31m'  # Bold Red
            }
            reset = '\033[0m'
            color = level_colors.get(self.level, '')
            
            line = f"{color}{timestamp_str}{reset} | {color}{level_str}{reset} | {agent_str} | {category_str} | {self.message}"
        else:
            line = f"{timestamp_str} | {level_str} | {agent_str} | {category_str} | {self.message}"
        
        # 添加执行时间
        if self.execution_time_ms is not None:
            line += f" ({self.execution_time_ms:.2f}ms)"
        
        return line


class ExecutionLogger:
    """
    执行日志器
    
    核心功能：
    - 多级别日志（DEBUG/INFO/WARNING/ERROR/CRITICAL）
    - 多输出目标（Console/File/Database）
    - 结构化日志
    - 执行轨迹追踪
    - 性能分析
    """
    
    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_database: bool = False,
        log_dir: str = "Data/logs",
        db_path: str = "Data/logs/execution.db",
        colored_console: bool = True,
        session_id: Optional[str] = None
    ):
        """
        初始化日志器
        
        Args:
            level: 日志级别（只记录>=此级别的日志）
            enable_console: 是否输出到控制台
            enable_file: 是否输出到文件
            enable_database: 是否存储到数据库
            log_dir: 日志文件目录
            db_path: 数据库路径
            colored_console: 控制台输出是否使用颜色
            session_id: 会话ID（用于关联日志）
        """
        self.level = level
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_database = enable_database
        self.colored_console = colored_console
        self.session_id = session_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 内存缓存（用于快速查询）
        self.logs: List[LogEntry] = []
        
        # 创建日志目录
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件日志
        if self.enable_file:
            self.log_file = self.log_dir / f"execution_{self.session_id}.log"
            self._init_file_logging()
        
        # 数据库日志
        if self.enable_database:
            self.db_path = db_path
            self._init_database()
        
        # 性能统计
        self.performance_stats = defaultdict(lambda: {
            'count': 0,
            'total_time_ms': 0.0,
            'min_time_ms': float('inf'),
            'max_time_ms': 0.0
        })
        
        self._log_system(LogLevel.INFO, "ExecutionLogger initialized", {
            'level': level.name,
            'session_id': self.session_id,
            'console': enable_console,
            'file': enable_file,
            'database': enable_database
        })
    
    def _init_file_logging(self):
        """初始化文件日志"""
        # 写入文件头
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"# Execution Log - Session {self.session_id}\n")
            f.write(f"# Started at: {datetime.now().isoformat()}\n")
            f.write(f"# Log Level: {self.level.name}\n")
            f.write(f"{'='*100}\n\n")
    
    def _init_database(self):
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS execution_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                category TEXT NOT NULL,
                agent_name TEXT,
                message TEXT,
                details JSON,
                execution_time_ms REAL,
                backtest_date TEXT,
                symbol TEXT,
                timeframe TEXT,
                parent_log_id TEXT
            )
        """)
        
        # 创建索引
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session 
            ON execution_logs(session_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_date 
            ON execution_logs(symbol, backtest_date)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_category 
            ON execution_logs(category)
        """)
        
        conn.commit()
        conn.close()
    
    def _should_log(self, level: LogLevel) -> bool:
        """判断是否应该记录此级别的日志"""
        return level >= self.level
    
    def _log(self, entry: LogEntry):
        """
        内部日志方法
        
        根据配置输出到不同目标
        """
        if not self._should_log(entry.level):
            return
        
        # 添加到内存缓存
        self.logs.append(entry)
        
        # 控制台输出
        if self.enable_console:
            print(entry.to_console_string(self.colored_console))
        
        # 文件输出
        if self.enable_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(entry.to_json() + '\n')
        
        # 数据库存储
        if self.enable_database:
            self._store_to_database(entry)
    
    def _store_to_database(self, entry: LogEntry):
        """存储日志到数据库"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO execution_logs 
            (session_id, timestamp, level, category, agent_name, message, 
             details, execution_time_ms, backtest_date, symbol, timeframe, parent_log_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.session_id,
            entry.timestamp.isoformat(),
            entry.level.name,
            entry.category.value,
            entry.agent_name,
            entry.message,
            json.dumps(entry.details, default=str),
            entry.execution_time_ms,
            entry.backtest_date.isoformat() if entry.backtest_date else None,
            entry.symbol,
            entry.timeframe,
            entry.parent_log_id
        ))
        conn.commit()
        conn.close()
    
    def _log_system(self, level: LogLevel, message: str, details: Dict = None):
        """记录系统级日志"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            category=LogCategory.PERFORMANCE,
            agent_name="system",
            message=message,
            details=details or {},
            session_id=self.session_id
        )
        self._log(entry)
    
    # === 公共日志接口 ===
    
    def log_decision(
        self,
        agent_name: str,
        symbol: str,
        action: str,
        conviction: float,
        reasoning: str,
        timeframe: str = None,
        backtest_date: datetime = None,
        details: Dict = None
    ):
        """
        记录决策
        
        Args:
            agent_name: Agent名称
            symbol: 股票代码
            action: 决策动作（BUY/SELL/HOLD）
            conviction: 信心度（1-10）
            reasoning: 决策理由
            timeframe: 时间尺度
            backtest_date: 回测日期
            details: 额外详情
        """
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            category=LogCategory.DECISION,
            agent_name=agent_name,
            message=f"Decision: {action} {symbol} (conviction={conviction})",
            details={
                'action': action,
                'conviction': conviction,
                'reasoning': reasoning,
                **(details or {})
            },
            symbol=symbol,
            timeframe=timeframe,
            backtest_date=backtest_date,
            session_id=self.session_id
        )
        self._log(entry)
    
    def log_tool_call(
        self,
        agent_name: str,
        tool_name: str,
        arguments: Dict,
        result: Any = None,
        execution_time_ms: float = None,
        symbol: str = None,
        backtest_date: datetime = None,
        error: str = None
    ):
        """
        记录工具调用
        
        Args:
            agent_name: Agent名称
            tool_name: 工具名称
            arguments: 工具参数
            result: 执行结果
            execution_time_ms: 执行时间（毫秒）
            symbol: 股票代码
            backtest_date: 回测日期
            error: 错误信息（如果有）
        """
        level = LogLevel.ERROR if error else LogLevel.INFO
        
        # 结果摘要（避免过长）
        result_summary = None
        if result:
            result_str = str(result)
            result_summary = result_str[:200] + "..." if len(result_str) > 200 else result_str
        
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            category=LogCategory.TOOL_CALL,
            agent_name=agent_name,
            message=f"Tool call: {tool_name}",
            details={
                'tool_name': tool_name,
                'arguments': arguments,
                'result_summary': result_summary,
                'error': error
            },
            execution_time_ms=execution_time_ms,
            symbol=symbol,
            backtest_date=backtest_date,
            session_id=self.session_id
        )
        self._log(entry)
        
        # 更新性能统计
        if execution_time_ms:
            self._update_performance_stats(f"{agent_name}.{tool_name}", execution_time_ms)
    
    def log_cache_hit(
        self,
        cache_type: str,
        key: str,
        symbol: str = None,
        saved_time_ms: float = None,
        backtest_date: datetime = None
    ):
        """记录缓存命中"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.DEBUG,
            category=LogCategory.CACHE,
            agent_name="cache",
            message=f"Cache hit: {cache_type}",
            details={
                'cache_type': cache_type,
                'key': key,
                'saved_time_ms': saved_time_ms
            },
            symbol=symbol,
            backtest_date=backtest_date,
            session_id=self.session_id
        )
        self._log(entry)
    
    def log_memory_operation(
        self,
        operation: str,
        timeframe: str,
        symbol: str = None,
        details: Dict = None,
        backtest_date: datetime = None
    ):
        """记录Memory操作"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.DEBUG,
            category=LogCategory.MEMORY,
            agent_name="memory",
            message=f"Memory {operation}: {timeframe}",
            details=details or {},
            symbol=symbol,
            timeframe=timeframe,
            backtest_date=backtest_date,
            session_id=self.session_id
        )
        self._log(entry)
    
    def log_timeframe_switch(
        self,
        from_timeframe: str,
        to_timeframe: str,
        reason: str,
        symbol: str = None,
        backtest_date: datetime = None
    ):
        """记录时间尺度切换"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            category=LogCategory.TIMEFRAME,
            agent_name="scheduler",
            message=f"Timeframe switch: {from_timeframe} → {to_timeframe}",
            details={'reason': reason},
            symbol=symbol,
            timeframe=to_timeframe,
            backtest_date=backtest_date,
            session_id=self.session_id
        )
        self._log(entry)
    
    def log_escalation(
        self,
        from_timeframe: str,
        to_timeframe: str,
        trigger: str,
        impact_score: float,
        symbol: str = None,
        backtest_date: datetime = None,
        details: Dict = None
    ):
        """记录反向传导"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.WARNING,
            category=LogCategory.ESCALATION,
            agent_name="escalation",
            message=f"Escalation: {from_timeframe} → {to_timeframe} (trigger: {trigger})",
            details={
                'trigger': trigger,
                'impact_score': impact_score,
                **(details or {})
            },
            symbol=symbol,
            timeframe=to_timeframe,
            backtest_date=backtest_date,
            session_id=self.session_id
        )
        self._log(entry)
    
    def log_error(
        self,
        agent_name: str,
        error_message: str,
        exception: Exception = None,
        details: Dict = None
    ):
        """记录错误"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            category=LogCategory.ERROR,
            agent_name=agent_name,
            message=error_message,
            details={
                'exception_type': type(exception).__name__ if exception else None,
                'exception_message': str(exception) if exception else None,
                **(details or {})
            },
            session_id=self.session_id
        )
        self._log(entry)
    
    def debug(self, agent_name: str, message: str, details: Dict = None):
        """DEBUG级别日志"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.DEBUG,
            category=LogCategory.PERFORMANCE,
            agent_name=agent_name,
            message=message,
            details=details or {},
            session_id=self.session_id
        )
        self._log(entry)
    
    def info(self, agent_name: str, message: str, details: Dict = None):
        """INFO级别日志"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            category=LogCategory.PERFORMANCE,
            agent_name=agent_name,
            message=message,
            details=details or {},
            session_id=self.session_id
        )
        self._log(entry)
    
    def warning(self, agent_name: str, message: str, details: Dict = None):
        """WARNING级别日志"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.WARNING,
            category=LogCategory.PERFORMANCE,
            agent_name=agent_name,
            message=message,
            details=details or {},
            session_id=self.session_id
        )
        self._log(entry)
    
    # === 性能统计 ===
    
    def _update_performance_stats(self, key: str, execution_time_ms: float):
        """更新性能统计"""
        stats = self.performance_stats[key]
        stats['count'] += 1
        stats['total_time_ms'] += execution_time_ms
        stats['min_time_ms'] = min(stats['min_time_ms'], execution_time_ms)
        stats['max_time_ms'] = max(stats['max_time_ms'], execution_time_ms)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能统计摘要"""
        summary = {}
        for key, stats in self.performance_stats.items():
            if stats['count'] > 0:
                summary[key] = {
                    'count': stats['count'],
                    'total_ms': stats['total_time_ms'],
                    'avg_ms': stats['total_time_ms'] / stats['count'],
                    'min_ms': stats['min_time_ms'],
                    'max_ms': stats['max_time_ms']
                }
        return summary
    
    def print_performance_summary(self):
        """打印性能统计"""
        summary = self.get_performance_summary()
        
        print("\n" + "="*80)
        print("Performance Summary")
        print("="*80)
        
        for key, stats in sorted(summary.items(), key=lambda x: x[1]['total_ms'], reverse=True):
            print(f"\n{key}:")
            print(f"  Calls: {stats['count']}")
            print(f"  Total: {stats['total_ms']:.2f}ms")
            print(f"  Avg:   {stats['avg_ms']:.2f}ms")
            print(f"  Min:   {stats['min_ms']:.2f}ms")
            print(f"  Max:   {stats['max_ms']:.2f}ms")
    
    # === 执行轨迹查询 ===
    
    def get_execution_trace(
        self,
        symbol: str = None,
        backtest_date: datetime = None,
        category: LogCategory = None,
        timeframe: str = None
    ) -> List[LogEntry]:
        """
        获取执行轨迹
        
        Args:
            symbol: 过滤股票代码
            backtest_date: 过滤回测日期
            category: 过滤日志类别
            timeframe: 过滤时间尺度
        
        Returns:
            符合条件的日志列表
        """
        filtered = self.logs
        
        if symbol:
            filtered = [log for log in filtered if log.symbol == symbol]
        
        if backtest_date:
            date_str = backtest_date.strftime('%Y-%m-%d')
            filtered = [
                log for log in filtered 
                if log.backtest_date and log.backtest_date.strftime('%Y-%m-%d') == date_str
            ]
        
        if category:
            filtered = [log for log in filtered if log.category == category]
        
        if timeframe:
            filtered = [log for log in filtered if log.timeframe == timeframe]
        
        return filtered
    
    def visualize_trace(
        self,
        symbol: str = None,
        backtest_date: datetime = None,
        max_entries: int = 100
    ):
        """
        可视化执行轨迹
        
        Args:
            symbol: 股票代码
            backtest_date: 回测日期
            max_entries: 最大显示条目数
        """
        trace = self.get_execution_trace(symbol=symbol, backtest_date=backtest_date)
        
        if not trace:
            print("No logs found for the specified criteria.")
            return
        
        # 限制显示数量
        if len(trace) > max_entries:
            print(f"Found {len(trace)} logs, showing first {max_entries}...")
            trace = trace[:max_entries]
        
        print(f"\n{'='*100}")
        print(f"Execution Trace")
        if symbol:
            print(f"Symbol: {symbol}")
        if backtest_date:
            print(f"Date: {backtest_date.strftime('%Y-%m-%d')}")
        print(f"Total Entries: {len(trace)}")
        print(f"{'='*100}\n")
        
        for i, entry in enumerate(trace, 1):
            timestamp = entry.timestamp.strftime('%H:%M:%S.%f')[:-3]
            level_icon = {
                LogLevel.DEBUG: '🔍',
                LogLevel.INFO: 'ℹ️',
                LogLevel.WARNING: '⚠️',
                LogLevel.ERROR: '❌',
                LogLevel.CRITICAL: '🔥'
            }.get(entry.level, '•')
            
            category_icon = {
                LogCategory.DECISION: '🎯',
                LogCategory.TOOL_CALL: '🔧',
                LogCategory.CACHE: '💾',
                LogCategory.MEMORY: '🧠',
                LogCategory.TIMEFRAME: '⏱️',
                LogCategory.ESCALATION: '⬆️',
                LogCategory.ERROR: '❌'
            }.get(entry.category, '•')
            
            print(f"{i:3d}. [{timestamp}] {level_icon} {category_icon} {entry.agent_name:20s} | {entry.message}")
            
            # 显示关键详情
            if entry.category == LogCategory.DECISION:
                print(f"      └─ Action: {entry.details.get('action')}")
                print(f"      └─ Conviction: {entry.details.get('conviction')}/10")
                reasoning = entry.details.get('reasoning', '')
                if reasoning:
                    print(f"      └─ Reasoning: {reasoning[:80]}...")
            
            elif entry.category == LogCategory.TOOL_CALL:
                print(f"      └─ Tool: {entry.details.get('tool_name')}")
                if entry.execution_time_ms:
                    print(f"      └─ Time: {entry.execution_time_ms:.2f}ms")
                if entry.details.get('error'):
                    print(f"      └─ Error: {entry.details['error']}")
            
            elif entry.category == LogCategory.ESCALATION:
                print(f"      └─ Trigger: {entry.details.get('trigger')}")
                print(f"      └─ Impact: {entry.details.get('impact_score')}")
            
            print()
    
    # === 配置管理 ===
    
    def set_level(self, level: LogLevel):
        """动态修改日志级别"""
        old_level = self.level
        self.level = level
        self._log_system(LogLevel.INFO, f"Log level changed: {old_level.name} → {level.name}")
    
    def enable_category(self, category: LogCategory):
        """启用特定类别的日志（未来功能）"""
        pass
    
    def disable_category(self, category: LogCategory):
        """禁用特定类别的日志（未来功能）"""
        pass
    
    def save_summary(self, output_file: str = None):
        """
        保存执行摘要
        
        Args:
            output_file: 输出文件路径（默认：logs/summary_{session_id}.txt）
        """
        if not output_file:
            output_file = self.log_dir / f"summary_{self.session_id}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Execution Summary - Session {self.session_id}\n")
            f.write(f"{'='*80}\n\n")
            
            # 基本统计
            f.write(f"Total Logs: {len(self.logs)}\n")
            f.write(f"Log Level: {self.level.name}\n\n")
            
            # 按类别统计
            category_counts = defaultdict(int)
            for log in self.logs:
                category_counts[log.category] += 1
            
            f.write("Logs by Category:\n")
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {category.value:15s}: {count:5d}\n")
            
            f.write("\n")
            
            # 性能统计
            f.write("Performance Statistics:\n")
            summary = self.get_performance_summary()
            for key, stats in sorted(summary.items(), key=lambda x: x[1]['total_ms'], reverse=True):
                f.write(f"\n{key}:\n")
                f.write(f"  Calls: {stats['count']}\n")
                f.write(f"  Avg:   {stats['avg_ms']:.2f}ms\n")
        
        print(f"✓ Summary saved to: {output_file}")


# === 全局日志器（单例模式）===

_global_logger: Optional[ExecutionLogger] = None


def get_logger() -> ExecutionLogger:
    """获取全局日志器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = ExecutionLogger()
    return _global_logger


def set_global_logger(logger: ExecutionLogger):
    """设置全局日志器"""
    global _global_logger
    _global_logger = logger


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_database: bool = False,
    **kwargs
) -> ExecutionLogger:
    """
    配置全局日志系统（便捷函数）
    
    Args:
        level: 日志级别
        enable_console: 是否输出到控制台
        enable_file: 是否输出到文件
        enable_database: 是否存储到数据库
        **kwargs: 其他参数传递给ExecutionLogger
    
    Returns:
        配置好的日志器
    """
    logger = ExecutionLogger(
        level=level,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_database=enable_database,
        **kwargs
    )
    set_global_logger(logger)
    return logger


# === 示例用法 ===

if __name__ == "__main__":
    # 创建logger（INFO级别，输出到控制台和文件）
    logger = ExecutionLogger(
        level=LogLevel.INFO,
        enable_console=True,
        enable_file=True,
        enable_database=False
    )
    
    # 记录决策
    logger.log_decision(
        agent_name="meta_agent",
        symbol="AAPL",
        action="BUY",
        conviction=8.0,
        reasoning="Strong technical signals with positive news sentiment",
        timeframe="tactical",
        backtest_date=datetime(2024, 10, 15)
    )
    
    # 记录工具调用
    logger.log_tool_call(
        agent_name="technical",
        tool_name="calculate_indicators",
        arguments={"symbol": "AAPL"},
        result={"rsi": 65.2, "macd": 1.23},
        execution_time_ms=123.45,
        symbol="AAPL"
    )
    
    # 记录缓存命中
    logger.log_cache_hit(
        cache_type="signal_cache",
        key="AAPL_2024-10-15_tactical",
        symbol="AAPL",
        saved_time_ms=5000.0
    )
    
    # 记录反向传导
    logger.log_escalation(
        from_timeframe="tactical",
        to_timeframe="strategic",
        trigger="market_shock",
        impact_score=9.5,
        symbol="AAPL",
        details={"price_drop": -5.2}
    )
    
    # 记录错误
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.log_error(
            agent_name="test_agent",
            error_message="Test error occurred",
            exception=e
        )
    
    # 查看执行轨迹
    logger.visualize_trace(symbol="AAPL")
    
    # 性能统计
    logger.print_performance_summary()
    
    # 保存摘要
    logger.save_summary()
    
    print("\n✓ Logger test complete!")
