"""
Base Agent - 纯业务逻辑基类

所有核心 Agent 的基类，提供通用功能：
1. LLM 客户端管理（支持依赖注入）
2. 缓存机制
3. 日志功能
4. 工具方法

设计原则：
- 不依赖 MCP 协议
- 支持 LLM Mock（方便测试）
- 纯业务逻辑
"""

from typing import Dict, Any, Optional
from datetime import datetime
import logging


# ════════════════════════════════════════════════
# Base Agent - Pure Business Logic
# ════════════════════════════════════════════════
import hashlib
import json


class BaseAgent:
    """
    纯业务逻辑 Agent 基类
    
    职责：
    - LLM 客户端管理
    - 缓存管理
    - 日志记录
    
    不涉及：
    - MCP 协议
    - 网络通信
    - Server/Client 逻辑
    """
    
    def __init__(
        self,
        name: str,
        llm_client=None,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化 BaseAgent
        
        Args:
            name: Agent 名称
            llm_client: LLM 客户端实例（支持注入 Mock）
            enable_cache: 是否启用缓存
            cache_ttl: 缓存有效期（秒）
            logger: 日志记录器（可选）
        """
        self.name = name
        self._llm_client = llm_client
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        
        # 日志
        self.logger = logger or logging.getLogger(f"Agent.{name}")
        
        # 缓存
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"Initialized {name}")
    
    # ════════════════════════════════════════════════
    # LLM 客户端管理
    # ════════════════════════════════════════════════
    
    @property
    def llm(self):
        """
        获取 LLM 客户端
        
        Returns:
            LLM 客户端实例或 None
        """
        if self._llm_client is None:
            # 延迟加载：只在需要时才加载默认 LLM
            from Agents.utils.llm_config import get_default_llm
            try:
                self._llm_client = get_default_llm()
                self.logger.info(f"Loaded default LLM for {self.name}")
            except Exception as e:
                self.logger.warning(f"Failed to load LLM: {e}")
        return self._llm_client
    
    def set_llm(self, llm_client):
        """
        设置 LLM 客户端（用于测试 Mock）
        
        Args:
            llm_client: LLM 客户端实例
        """
        self._llm_client = llm_client
        self.logger.debug(f"LLM client set for {self.name}")
    
    def has_llm(self) -> bool:
        """检查是否有可用的 LLM"""
        return self._llm_client is not None or self._can_load_default_llm()
    
    def _can_load_default_llm(self) -> bool:
        """检查是否可以加载默认 LLM"""
        try:
            from Agents.utils.llm_config import get_default_llm
            get_default_llm()
            return True
        except:
            return False
    
    # ════════════════════════════════════════════════
    # 缓存管理
    # ════════════════════════════════════════════════
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            缓存键（字符串）
        """
        # 将参数序列化为字符串
        key_data = {
            'args': args,
            'kwargs': {k: v for k, v in kwargs.items() if k != 'force_refresh'}
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        
        # 生成 hash
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        从缓存获取数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存的数据或 None（如果不存在或过期）
        """
        if not self.enable_cache:
            return None
        
        if cache_key not in self._cache:
            return None
        
        cached = self._cache[cache_key]
        
        # 检查是否过期
        age = (datetime.now() - cached['timestamp']).total_seconds()
        if age > self.cache_ttl:
            del self._cache[cache_key]
            self.logger.debug(f"Cache expired for key: {cache_key[:8]}...")
            return None
        
        self.logger.debug(f"Cache hit for key: {cache_key[:8]}...")
        return cached['data']
    
    def _put_to_cache(self, cache_key: str, data: Any) -> None:
        """
        将数据放入缓存
        
        Args:
            cache_key: 缓存键
            data: 要缓存的数据
        """
        if not self.enable_cache:
            return
        
        self._cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        self.logger.debug(f"Cached data for key: {cache_key[:8]}...")
    
    def clear_cache(self) -> None:
        """清空缓存"""
        count = len(self._cache)
        self._cache.clear()
        self.logger.info(f"Cleared {count} cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计字典
        """
        return {
            'cached_items': len(self._cache),
            'cache_enabled': self.enable_cache,  # 保持测试兼容性
            'cache_ttl': self.cache_ttl,
            'cache_keys': list(self._cache.keys())
        }
    
    # ════════════════════════════════════════════════
    # 工具方法
    # ════════════════════════════════════════════════
    
    def _validate_arguments(self, arguments: Dict, required_keys: list) -> None:
        """
        验证参数是否包含必需的键
        
        Args:
            arguments: 参数字典
            required_keys: 必需的键列表
            
        Raises:
            ValueError: 如果缺少必需参数
        """
        missing = [key for key in required_keys if key not in arguments]
        if missing:
            raise ValueError(f"Missing required arguments: {missing}")
