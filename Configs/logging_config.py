"""
日志配置文件

支持通过YAML/JSON配置日志行为
"""

# ==========================================
# 方式1: YAML配置 (推荐)
# ==========================================

logging_config_yaml = """
# Logging Configuration

# 全局日志级别
# 可选: DEBUG, INFO, WARNING, ERROR, CRITICAL
level: INFO

# 输出目标
outputs:
  console:
    enabled: true
    colored: true
    
  file:
    enabled: true
    directory: Data/logs
    # 文件名模式: {session_id}, {date}, {time}
    filename_pattern: "execution_{session_id}.log"
    
  database:
    enabled: false
    path: Data/logs/execution.db

# 类别过滤 (未来功能)
# 可以选择只记录特定类别的日志
categories:
  decision: true
  tool_call: true
  cache: true
  memory: true
  timeframe: true
  escalation: true
  error: true
  performance: true

# 性能追踪
performance:
  enabled: true
  # 慢操作阈值（毫秒），超过此值会警告
  slow_threshold_ms: 1000

# 会话配置
session:
  # 自动生成session_id
  auto_generate: true
  # 手动指定（如果auto_generate=false）
  id: "my_session_123"

# 回测特定配置
backtest:
  # 是否记录每个回测日期
  log_each_date: true
  # 是否在回测结束时生成摘要
  generate_summary: true
"""

# ==========================================
# 方式2: JSON配置
# ==========================================

logging_config_json = """
{
  "level": "INFO",
  "outputs": {
    "console": {
      "enabled": true,
      "colored": true
    },
    "file": {
      "enabled": true,
      "directory": "Data/logs",
      "filename_pattern": "execution_{session_id}.log"
    },
    "database": {
      "enabled": false,
      "path": "Data/logs/execution.db"
    }
  },
  "categories": {
    "decision": true,
    "tool_call": true,
    "cache": true,
    "memory": true,
    "timeframe": true,
    "escalation": true,
    "error": true,
    "performance": true
  },
  "performance": {
    "enabled": true,
    "slow_threshold_ms": 1000
  },
  "session": {
    "auto_generate": true,
    "id": "my_session_123"
  },
  "backtest": {
    "log_each_date": true,
    "generate_summary": true
  }
}
"""

# ==========================================
# 配置加载器
# ==========================================

import json
import yaml
from pathlib import Path
from typing import Dict, Any
from Utils.execution_logger import ExecutionLogger, LogLevel


class LoggerConfig:
    """日志配置加载器"""
    
    @staticmethod
    def from_yaml(file_path: str) -> Dict[str, Any]:
        """从YAML文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def from_json(file_path: str) -> Dict[str, Any]:
        """从JSON文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def from_dict(config: Dict[str, Any]) -> ExecutionLogger:
        """从字典创建logger"""
        # 解析日志级别
        level_str = config.get('level', 'INFO')
        level = LogLevel.from_string(level_str)
        
        # 解析输出配置
        outputs = config.get('outputs', {})
        enable_console = outputs.get('console', {}).get('enabled', True)
        enable_file = outputs.get('file', {}).get('enabled', True)
        enable_database = outputs.get('database', {}).get('enabled', False)
        colored_console = outputs.get('console', {}).get('colored', True)
        
        # 文件配置
        file_config = outputs.get('file', {})
        log_dir = file_config.get('directory', 'Data/logs')
        
        # 数据库配置
        db_config = outputs.get('database', {})
        db_path = db_config.get('path', 'Data/logs/execution.db')
        
        # 会话配置
        session_config = config.get('session', {})
        auto_generate_session = session_config.get('auto_generate', True)
        session_id = None if auto_generate_session else session_config.get('id')
        
        # 创建logger
        logger = ExecutionLogger(
            level=level,
            enable_console=enable_console,
            enable_file=enable_file,
            enable_database=enable_database,
            log_dir=log_dir,
            db_path=db_path,
            colored_console=colored_console,
            session_id=session_id
        )
        
        return logger
    
    @staticmethod
    def load(file_path: str) -> ExecutionLogger:
        """
        自动检测文件类型并加载
        
        Args:
            file_path: 配置文件路径 (.yaml, .yml, .json)
        
        Returns:
            配置好的ExecutionLogger
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        # 根据扩展名选择加载方式
        if path.suffix in ['.yaml', '.yml']:
            config = LoggerConfig.from_yaml(file_path)
        elif path.suffix == '.json':
            config = LoggerConfig.from_json(file_path)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        return LoggerConfig.from_dict(config)


# ==========================================
# 使用示例
# ==========================================

def example_usage():
    """配置文件使用示例"""
    
    # 方式1: 从YAML文件加载
    logger = LoggerConfig.load('Configs/logging.yaml')
    
    # 方式2: 从JSON文件加载
    # logger = LoggerConfig.load('Configs/logging.json')
    
    # 方式3: 从字典加载（程序动态生成）
    config = {
        'level': 'INFO',
        'outputs': {
            'console': {'enabled': True, 'colored': True},
            'file': {'enabled': True},
            'database': {'enabled': False}
        }
    }
    logger = LoggerConfig.from_dict(config)
    
    # 使用logger
    logger.log_decision(
        agent_name="meta_agent",
        symbol="AAPL",
        action="BUY",
        conviction=8.0,
        reasoning="Test"
    )


# ==========================================
# 预设配置（常用场景）
# ==========================================

PRESET_CONFIGS = {
    # 开发环境：详细日志，彩色输出
    'development': {
        'level': 'DEBUG',
        'outputs': {
            'console': {'enabled': True, 'colored': True},
            'file': {'enabled': True},
            'database': {'enabled': False}
        }
    },
    
    # 生产环境：只记录重要信息，持久化
    'production': {
        'level': 'INFO',
        'outputs': {
            'console': {'enabled': False},
            'file': {'enabled': True},
            'database': {'enabled': True}
        }
    },
    
    # 性能测试：只记录错误和性能数据
    'performance': {
        'level': 'ERROR',
        'outputs': {
            'console': {'enabled': True, 'colored': False},
            'file': {'enabled': False},
            'database': {'enabled': False}
        }
    },
    
    # 回测：完整日志，便于事后分析
    'backtest': {
        'level': 'INFO',
        'outputs': {
            'console': {'enabled': True, 'colored': True},
            'file': {'enabled': True},
            'database': {'enabled': True}
        },
        'backtest': {
            'log_each_date': True,
            'generate_summary': True
        }
    },
    
    # 静默模式：只记录严重错误
    'silent': {
        'level': 'CRITICAL',
        'outputs': {
            'console': {'enabled': False},
            'file': {'enabled': True},
            'database': {'enabled': False}
        }
    }
}


def get_preset_logger(preset_name: str) -> ExecutionLogger:
    """
    获取预设配置的logger
    
    Args:
        preset_name: 预设名称 (development/production/performance/backtest/silent)
    
    Returns:
        配置好的ExecutionLogger
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    
    config = PRESET_CONFIGS[preset_name]
    return LoggerConfig.from_dict(config)


# ==========================================
# 示例：在不同环境下使用
# ==========================================

if __name__ == "__main__":
    import os
    
    # 1. 从环境变量决定使用哪个预设
    env = os.getenv('ENVIRONMENT', 'development')
    logger = get_preset_logger(env)
    
    print(f"Using {env} logger configuration")
    print(f"Log level: {logger.level.name}")
    print(f"Console: {logger.enable_console}")
    print(f"File: {logger.enable_file}")
    print(f"Database: {logger.enable_database}")
    
    # 2. 测试日志
    logger.info(
        agent_name="test",
        message="Logger configuration test",
        details={'preset': env}
    )
    
    # 3. 显示所有预设
    print("\nAvailable presets:")
    for name, config in PRESET_CONFIGS.items():
        print(f"  - {name}: level={config['level']}")
