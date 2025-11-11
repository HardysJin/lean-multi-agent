"""
Configuration loader and management
配置加载和管理
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class SystemConfig(BaseModel):
    """系统配置"""
    lookback_days: int = 7     # 回看天数（收集历史数据）
    forecast_days: int = 7     # 预测天数（决策的有效期）
    backtest_start: str = "2020-01-01"
    backtest_end: str = "2024-12-31"
    initial_capital: float = 100000
    commission: float = 0.001
    slippage: float = 0.001


class AgentConfig(BaseModel):
    """Agent配置"""
    enabled: bool = True
    indicators: Optional[list] = None
    sources: Optional[list] = None
    api_key: Optional[str] = None
    top_headlines_count: int = 10


class LLMConfig(BaseModel):
    """LLM配置"""
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 2000
    can_suggest_positions: bool = True
    auto_execute: bool = False
    require_approval: bool = True
    api_key: Optional[str] = None


class RiskConfig(BaseModel):
    """风控配置"""
    max_single_position: float = 0.3
    min_cash_reserve: float = 0.2
    max_weekly_turnover: float = 0.5
    max_drawdown: float = 0.15
    stop_loss_pct: float = 0.05
    validate_before_execute: bool = True


class StrategyConfig(BaseModel):
    """策略配置"""
    available: list = ["grid_trading", "momentum", "mean_reversion", "hold"]
    default: str = "hold"


class DatabaseConfig(BaseModel):
    """数据库配置"""
    type: str = "sqlite"
    path: str = "./Data/sql/weekly_decisions.db"
    echo: bool = False


class Config(BaseSettings):
    """全局配置类"""
    system: SystemConfig = Field(default_factory=SystemConfig)
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategies: StrategyConfig = Field(default_factory=StrategyConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # 忽略额外的环境变量


def load_config(config_path: Optional[str] = None) -> Config:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，默认为backend/config/config.yaml
    
    Returns:
        Config: 配置对象
    """
    if config_path is None:
        # 默认配置文件路径
        config_path = Path(__file__).parent / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # 读取YAML文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # 替换环境变量
    config_data = _replace_env_vars(config_data)
    
    # 解析配置
    system_cfg = SystemConfig(**config_data.get('system', {}))
    
    agents_cfg = {}
    for agent_name, agent_data in config_data.get('agents', {}).items():
        agents_cfg[agent_name] = AgentConfig(**agent_data)
    
    llm_cfg = LLMConfig(**config_data.get('llm', {}))
    risk_cfg = RiskConfig(**config_data.get('risk', {}))
    strategies_cfg = StrategyConfig(**config_data.get('strategies', {}))
    database_cfg = DatabaseConfig(**config_data.get('database', {}))
    
    config = Config(
        system=system_cfg,
        agents=agents_cfg,
        llm=llm_cfg,
        risk=risk_cfg,
        strategies=strategies_cfg,
        database=database_cfg
    )
    
    return config


def _replace_env_vars(data: Any) -> Any:
    """
    递归替换配置中的环境变量
    ${VAR_NAME} -> os.getenv('VAR_NAME')
    """
    if isinstance(data, dict):
        return {k: _replace_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_replace_env_vars(item) for item in data]
    elif isinstance(data, str):
        # 替换 ${VAR_NAME} 格式
        if data.startswith('${') and data.endswith('}'):
            var_name = data[2:-1]
            return os.getenv(var_name, '')
        return data
    else:
        return data


# 全局配置实例
_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: Optional[str] = None):
    """重新加载配置"""
    global _config
    _config = load_config(config_path)
    return _config
