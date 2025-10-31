"""
Memory Schemas - 数据结构定义

定义多时间尺度分层记忆系统的核心数据结构：
- Timeframe: 时间尺度枚举
- DecisionRecord: 决策记录
- MemoryDocument: 向量文档
- HierarchicalConstraints: 分层约束
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import json


class Timeframe(Enum):
    """
    时间尺度枚举
    
    定义5个时间尺度层级，从快到慢：
    - REALTIME: 实时层（5分钟特征周期）
    - EXECUTION: 执行层（1小时特征周期）
    - TACTICAL: 战术层（1天特征周期）- 主要交易决策层
    - CAMPAIGN: 战役层（1周特征周期）
    - STRATEGIC: 战略层（30天特征周期）
    
    每个时间尺度包含：
    - name: 显示名称
    - characteristic_period_seconds: 特征周期（秒）
    """
    
    # 格式: (name, characteristic_period_seconds)
    REALTIME = ("realtime", 300)        # 5分钟
    EXECUTION = ("execution", 3600)      # 1小时
    TACTICAL = ("tactical", 86400)       # 1天
    CAMPAIGN = ("campaign", 604800)      # 1周 (7天)
    STRATEGIC = ("strategic", 2592000)   # 30天 (约3个月)
    
    def __init__(self, display_name: str, seconds: int):
        self.display_name = display_name
        self.characteristic_period_seconds = seconds
    
    @property
    def seconds(self) -> int:
        """返回特征周期秒数"""
        return self.characteristic_period_seconds
    
    @property
    def name_display(self) -> str:
        """返回显示名称"""
        return self.display_name
    
    def __str__(self) -> str:
        return self.display_name
    
    def __lt__(self, other) -> bool:
        """比较时间尺度（用于排序和层级判断）"""
        if not isinstance(other, Timeframe):
            return NotImplemented
        return self.characteristic_period_seconds < other.characteristic_period_seconds
    
    def __le__(self, other) -> bool:
        if not isinstance(other, Timeframe):
            return NotImplemented
        return self.characteristic_period_seconds <= other.characteristic_period_seconds
    
    def __gt__(self, other) -> bool:
        if not isinstance(other, Timeframe):
            return NotImplemented
        return self.characteristic_period_seconds > other.characteristic_period_seconds
    
    def __ge__(self, other) -> bool:
        if not isinstance(other, Timeframe):
            return NotImplemented
        return self.characteristic_period_seconds >= other.characteristic_period_seconds
    
    @classmethod
    def from_string(cls, name: str) -> 'Timeframe':
        """从字符串创建Timeframe"""
        name_lower = name.lower()
        for tf in cls:
            if tf.display_name == name_lower or tf.name.lower() == name_lower:
                return tf
        raise ValueError(f"Unknown timeframe: {name}")
    
    @classmethod
    def get_all_ordered(cls) -> List['Timeframe']:
        """获取所有时间尺度，按从快到慢排序"""
        return sorted(list(cls), key=lambda x: x.characteristic_period_seconds)
    
    @classmethod
    def get_higher_timeframes(cls, current: 'Timeframe') -> List['Timeframe']:
        """获取比当前时间尺度更高（更慢）的所有时间尺度"""
        return [tf for tf in cls if tf > current]
    
    @classmethod
    def get_lower_timeframes(cls, current: 'Timeframe') -> List['Timeframe']:
        """获取比当前时间尺度更低（更快）的所有时间尺度"""
        return [tf for tf in cls if tf < current]


@dataclass
class DecisionRecord:
    """
    决策记录的标准结构
    
    记录一次交易决策的完整信息，用于：
    1. 存储到记忆系统
    2. 回溯分析决策质量
    3. 学习成功/失败案例
    """
    
    # === 基础信息 ===
    id: str                          # 唯一标识符
    timestamp: datetime              # 决策时间
    timeframe: Timeframe             # 决策的时间尺度
    symbol: str                      # 股票代码
    
    # === 决策内容 ===
    action: str                      # 动作: BUY/SELL/HOLD/ADD/REDUCE
    quantity: int                    # 数量（股数）
    price: float                     # 决策时的价格
    reasoning: str                   # 决策理由（可以很长，用于embedding）
    
    # === Agent信息 ===
    agent_name: str                  # 做出决策的Agent名称
    conviction: float                # 信心度 (1-10)
    
    # === 上下文信息 ===
    market_regime: Optional[str] = None              # 市场状态（来自战略层）
    technical_signals: Optional[Dict[str, Any]] = None  # 技术指标信号
    fundamental_data: Optional[Dict[str, Any]] = None   # 基本面数据
    news_sentiment: Optional[Dict[str, Any]] = None     # 新闻情绪
    related_news_ids: Optional[List[str]] = None        # 相关新闻ID列表
    
    # === 执行跟踪 ===
    executed: bool = False                          # 是否已执行
    execution_price: Optional[float] = None         # 实际成交价格
    execution_time: Optional[datetime] = None       # 实际成交时间
    
    # === 结果跟踪 ===
    outcome: Optional[str] = None                   # 结果: success/failure/ongoing
    exit_price: Optional[float] = None              # 退出价格
    exit_time: Optional[datetime] = None            # 退出时间
    pnl: Optional[float] = None                     # 盈亏金额
    pnl_percent: Optional[float] = None             # 盈亏百分比
    hold_duration_days: Optional[int] = None        # 持有天数
    
    # === 元数据 ===
    metadata: Dict[str, Any] = field(default_factory=dict)  # 其他元数据
    
    # === 回测时间控制（防止Look-Ahead Bias）===
    visible_data_end: Optional[datetime] = None  # 可见数据截止时间（回测模式专用）
    # 在回测模式下，这个字段确保决策只能基于 <= visible_data_end 的数据
    # 实盘模式下为 None（使用实时数据）
    
    # === 计算模式（性能优化）===
    computation_mode: str = 'full'  # 计算模式: full/hybrid/fast
    # - full: 完整Multi-Agent（所有Agent + LLM）
    # - hybrid: 混合模式（部分Agent + LLM）
    # - fast: 快速模式（仅规则引擎，无LLM）
    
    # === 缓存支持 ===
    cache_key: Optional[str] = None  # 信号缓存键
    # 用于标识可复用的计算结果，避免重复计算
    # 格式: f"{symbol}_{timeframe}_{strategy_hash}_{data_hash}"
    
    # === 反向传导（Escalation）===
    escalated_from: Optional[str] = None  # 如果是反向传导触发，记录来源时间尺度
    escalation_trigger: Optional[str] = None  # 触发反向传导的原因
    escalation_score: Optional[float] = None  # 触发评分（0-10）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于存储）"""
        data = asdict(self)
        # 转换datetime为ISO字符串
        data['timestamp'] = self.timestamp.isoformat()
        if self.execution_time:
            data['execution_time'] = self.execution_time.isoformat()
        if self.exit_time:
            data['exit_time'] = self.exit_time.isoformat()
        if self.visible_data_end:
            data['visible_data_end'] = self.visible_data_end.isoformat()
        # 转换Timeframe为字符串
        data['timeframe'] = self.timeframe.display_name
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionRecord':
        """从字典创建（用于加载）"""
        # 转换ISO字符串为datetime
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('execution_time'):
            data['execution_time'] = datetime.fromisoformat(data['execution_time'])
        if data.get('exit_time'):
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
        if data.get('visible_data_end'):
            data['visible_data_end'] = datetime.fromisoformat(data['visible_data_end'])
        # 转换字符串为Timeframe
        data['timeframe'] = Timeframe.from_string(data['timeframe'])
        return cls(**data)
    
    def to_text(self) -> str:
        """
        转换为文本描述（用于embedding）
        
        这个文本会被转换为向量，用于语义搜索
        """
        parts = [
            f"Decision on {self.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"Timeframe: {self.timeframe.display_name}",
            f"Symbol: {self.symbol}",
            f"Action: {self.action}",
            f"Conviction: {self.conviction}/10",
        ]
        
        if self.market_regime:
            parts.append(f"Market Regime: {self.market_regime}")
        
        parts.append(f"Reasoning: {self.reasoning}")
        
        if self.outcome:
            parts.append(f"Outcome: {self.outcome}")
            if self.pnl_percent is not None:
                parts.append(f"P&L: {self.pnl_percent:.2%}")
        
        return " | ".join(parts)
    
    def update_execution(self, execution_price: float, execution_time: datetime):
        """更新执行信息"""
        self.executed = True
        self.execution_price = execution_price
        self.execution_time = execution_time
    
    def update_outcome(self, exit_price: float, exit_time: datetime):
        """更新结果信息"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        
        # 计算持有天数
        if self.execution_time:
            self.hold_duration_days = (exit_time - self.execution_time).days
        
        # 计算盈亏
        if self.execution_price and self.quantity:
            if self.action in ['BUY', 'ADD']:
                self.pnl = (exit_price - self.execution_price) * self.quantity
                self.pnl_percent = (exit_price - self.execution_price) / self.execution_price
            elif self.action in ['SELL', 'REDUCE']:
                self.pnl = (self.execution_price - exit_price) * self.quantity
                self.pnl_percent = (self.execution_price - exit_price) / exit_price
        
        # 判断成功/失败
        if self.pnl_percent is not None:
            if self.pnl_percent > 0.02:  # >2% 认为成功
                self.outcome = 'success'
            elif self.pnl_percent < -0.02:  # <-2% 认为失败
                self.outcome = 'failure'
            else:
                self.outcome = 'neutral'
        else:
            self.outcome = 'ongoing'
    
    def is_backtest_mode(self) -> bool:
        """判断是否为回测模式"""
        return self.visible_data_end is not None
    
    def validate_data_timestamp(self, data_timestamp: datetime) -> bool:
        """
        验证数据时间戳是否在可见范围内（防止Look-Ahead）
        
        Args:
            data_timestamp: 要验证的数据时间戳
        
        Returns:
            True if data is visible (safe to use), False if future data (look-ahead)
        """
        if not self.is_backtest_mode():
            return True  # 实盘模式，所有实时数据都可用
        
        return data_timestamp <= self.visible_data_end
    
    def mark_as_escalated(self, from_timeframe: str, trigger: str, score: float):
        """
        标记为反向传导触发的决策
        
        Args:
            from_timeframe: 来源时间尺度（如 'tactical'）
            trigger: 触发原因（如 'market_shock', 'news_impact'）
            score: 触发评分（0-10，表示事件严重程度）
        """
        self.escalated_from = from_timeframe
        self.escalation_trigger = trigger
        self.escalation_score = score
        self.metadata['escalated'] = True
        self.metadata['escalation_details'] = {
            'from': from_timeframe,
            'trigger': trigger,
            'score': score,
            'escalated_at': datetime.now().isoformat()
        }
    
    def set_cache_key(self, strategy_version: str, data_hash: str):
        """
        设置缓存键
        
        Args:
            strategy_version: 策略版本（如 'v1.0.0'）
            data_hash: 数据哈希（唯一标识输入数据）
        """
        self.cache_key = f"{self.symbol}_{self.timeframe.display_name}_{strategy_version}_{data_hash}"


@dataclass
class MemoryDocument:
    """
    向量数据库中的文档结构
    
    用于ChromaDB存储，包含：
    - 文本内容（用于生成embedding）
    - embedding向量（可选，由ChromaDB自动生成）
    - 元数据（用于过滤查询）
    """
    
    id: str                                    # 文档唯一ID
    text: str                                  # 用于embedding的文本
    embedding: Optional[List[float]] = None    # embedding向量（可选）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def to_chroma_format(self) -> Dict[str, Any]:
        """转换为ChromaDB格式"""
        result = {
            'id': self.id,
            'document': self.text,
            'metadata': self.metadata.copy()
        }
        
        if self.embedding is not None:
            result['embedding'] = self.embedding
        
        return result
    
    @classmethod
    def from_decision(cls, decision: DecisionRecord) -> 'MemoryDocument':
        """从决策记录创建文档"""
        metadata = {
            'timestamp': decision.timestamp.isoformat(),
            'timeframe': decision.timeframe.display_name,
            'symbol': decision.symbol,
            'action': decision.action,
            'conviction': decision.conviction,
            'agent_name': decision.agent_name,
        }
        
        if decision.market_regime:
            metadata['market_regime'] = decision.market_regime
        
        if decision.outcome:
            metadata['outcome'] = decision.outcome
            if decision.pnl_percent is not None:
                metadata['pnl_percent'] = decision.pnl_percent
        
        return cls(
            id=decision.id,
            text=decision.to_text(),
            metadata=metadata
        )


@dataclass
class HierarchicalConstraints:
    """
    分层约束结构
    
    存储从上层时间尺度传递下来的硬约束和软建议
    下层决策必须遵守上层约束（军事化指挥原则）
    """
    
    # === 战略层约束（硬约束）===
    strategic: Optional[Dict[str, Any]] = None
    # 示例：
    # {
    #     'market_regime': 'bear',           # 市场状态: bull/bear/neutral
    #     'risk_budget': 0.5,                # 风险预算: 0-1
    #     'max_exposure': 0.6,               # 最大敞口
    #     'forbidden_sectors': ['crypto'],   # 禁止的行业
    #     'constraints_updated_at': datetime,
    # }
    
    # === 战役层配置（软建议）===
    campaign: Optional[Dict[str, Any]] = None
    # 示例：
    # {
    #     'sector_allocation': {             # 行业配置
    #         'tech': 0.4,
    #         'healthcare': 0.3,
    #         'finance': 0.3,
    #     },
    #     'rotation_signal': 'tech_to_value',  # 轮动信号
    #     'recommendation': 'reduce_tech',      # 建议
    # }
    
    # === 战术层历史（参考）===
    tactical: Optional[Dict[str, Any]] = None
    # 示例：
    # {
    #     'recent_decisions': [...],         # 近期决策
    #     'current_positions': {...},        # 当前持仓
    #     'win_rate': 0.65,                  # 胜率
    # }
    
    def __post_init__(self):
        """初始化默认值"""
        if self.strategic is None:
            self.strategic = {}
        if self.campaign is None:
            self.campaign = {}
        if self.tactical is None:
            self.tactical = {}
    
    def get_market_regime(self) -> str:
        """获取市场状态"""
        return self.strategic.get('market_regime', 'neutral')
    
    def get_risk_budget(self) -> float:
        """获取风险预算"""
        return self.strategic.get('risk_budget', 1.0)
    
    def get_max_exposure(self) -> float:
        """获取最大敞口"""
        return self.strategic.get('max_exposure', 1.0)
    
    def is_sector_forbidden(self, sector: str) -> bool:
        """检查行业是否被禁止"""
        forbidden = self.strategic.get('forbidden_sectors', [])
        return sector in forbidden
    
    def can_open_long(self) -> bool:
        """是否允许做多"""
        regime = self.get_market_regime()
        return regime in ['bull', 'neutral']
    
    def can_open_short(self) -> bool:
        """是否允许做空"""
        regime = self.get_market_regime()
        return regime in ['bear', 'neutral']
    
    def get_sector_allocation(self, sector: str) -> float:
        """获取行业配置比例"""
        allocations = self.campaign.get('sector_allocation', {})
        return allocations.get(sector, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'strategic': self.strategic,
            'campaign': self.campaign,
            'tactical': self.tactical,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HierarchicalConstraints':
        """从字典创建"""
        return cls(
            strategic=data.get('strategic'),
            campaign=data.get('campaign'),
            tactical=data.get('tactical'),
        )
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'HierarchicalConstraints':
        """从JSON字符串创建"""
        data = json.loads(json_str)
        return cls.from_dict(data)


# === 辅助函数 ===

def create_decision_id(symbol: str, timestamp: datetime, timeframe: Timeframe) -> str:
    """创建决策记录的唯一ID"""
    time_str = timestamp.strftime('%Y%m%d_%H%M%S')
    return f"decision_{symbol}_{timeframe.display_name}_{time_str}"


def create_memory_id(prefix: str, symbol: str, timestamp: datetime) -> str:
    """创建记忆文档的唯一ID"""
    time_str = timestamp.strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{symbol}_{time_str}"
