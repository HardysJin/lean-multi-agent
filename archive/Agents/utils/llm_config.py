"""
LLM Configuration - 统一的LLM配置管理

支持多种LLM提供商：
- OpenAI (GPT-4, GPT-3.5)
- Anthropic Claude
- DeepSeek
- Local models via Ollama

使用LangChain统一接口，让所有agents可以选择性使用LLM。

Usage:
    # 方式1: 使用全局配置（从.env读取）
    from Agents.llm_config import get_default_llm
    llm = get_default_llm()
    
    # 方式2: 自定义配置
    from Agents.llm_config import LLMConfig
    config = LLMConfig(provider="claude", model="claude-3-5-sonnet-20241022")
    llm = config.get_llm()
    
    # 方式3: Agent内部使用
    class MyAgent(BaseMCPAgent):
        def __init__(self, use_llm=True, llm_config=None):
            self.llm = get_default_llm() if use_llm else None
"""

import os
from typing import Optional, Dict, Any, Union
from enum import Enum
import logging

# LangChain imports
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_community.chat_models import ChatOllama
    from langchain_core.language_models.chat_models import BaseChatModel
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseChatModel = object  # Fallback type


class LLMProvider(Enum):
    """支持的LLM提供商"""
    OPENAI = "openai"
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"  # 本地模型


class LLMConfig:
    """
    LLM配置类
    
    管理LLM的配置和实例化，支持多种提供商。
    """
    
    # 默认模型配置
    DEFAULT_MODELS = {
        LLMProvider.OPENAI: "gpt-4o-mini",
        LLMProvider.CLAUDE: "claude-3-5-sonnet-20241022",
        LLMProvider.DEEPSEEK: "deepseek-chat",
        LLMProvider.OLLAMA: "llama3.1:8b",
    }
    
    # 默认温度
    DEFAULT_TEMPERATURE = 0.0  # 交易决策需要确定性
    
    def __init__(
        self,
        provider: Optional[Union[str, LLMProvider]] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        初始化LLM配置
        
        Args:
            provider: LLM提供商 (openai/claude/deepseek/ollama)
            model: 模型名称（如果不提供，使用默认）
            api_key: API密钥（如果不提供，从环境变量读取）
            temperature: 温度参数 (0-2)
            max_tokens: 最大token数
            base_url: API base URL（用于自定义端点或代理）
            **kwargs: 其他传递给LangChain的参数
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Please install it with:\n"
                "pip install langchain langchain-openai langchain-anthropic langchain-community"
            )
        
        # 解析provider
        if provider is None:
            provider = self._detect_provider_from_env()
        elif isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        
        self.provider = provider
        self.model = model or self.DEFAULT_MODELS[provider]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.kwargs = kwargs
        
        # 获取API key
        self.api_key = api_key or self._get_api_key_from_env(provider)
        
        logging.info(f"LLM Config initialized: {provider.value}/{self.model}")
    
    def _detect_provider_from_env(self) -> LLMProvider:
        """从环境变量检测要使用的提供商"""
        # 优先级: OPENAI > CLAUDE > DEEPSEEK
        if os.getenv("OPENAI_API_KEY"):
            return LLMProvider.OPENAI
        elif os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
            return LLMProvider.CLAUDE
        elif os.getenv("DEEPSEEK_API_KEY"):
            return LLMProvider.DEEPSEEK
        else:
            # 默认使用Ollama（本地，不需要API key）
            logging.warning("No API keys found in environment. Falling back to Ollama (local).")
            return LLMProvider.OLLAMA
    
    def _get_api_key_from_env(self, provider: LLMProvider) -> Optional[str]:
        """从环境变量获取API key"""
        key_mapping = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.CLAUDE: ["CLAUDE_API_KEY", "ANTHROPIC_API_KEY"],
            LLMProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
            LLMProvider.OLLAMA: None,  # 本地不需要key
        }
        
        env_keys = key_mapping[provider]
        if env_keys is None:
            return None
        
        # 支持多个可能的环境变量名
        if isinstance(env_keys, list):
            for key in env_keys:
                value = os.getenv(key)
                if value:
                    return value
            return None
        else:
            return os.getenv(env_keys)
    
    def get_llm(self) -> BaseChatModel:
        """
        获取LangChain LLM实例
        
        Returns:
            LangChain BaseChatModel实例
            
        Raises:
            ValueError: 如果配置无效
        """
        if self.provider == LLMProvider.OPENAI:
            return self._get_openai_llm()
        elif self.provider == LLMProvider.CLAUDE:
            return self._get_claude_llm()
        elif self.provider == LLMProvider.DEEPSEEK:
            return self._get_deepseek_llm()
        elif self.provider == LLMProvider.OLLAMA:
            return self._get_ollama_llm()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _get_openai_llm(self) -> ChatOpenAI:
        """获取OpenAI LLM"""
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        kwargs = {
            "model": self.model,
            "api_key": self.api_key,
            "temperature": self.temperature,
            **self.kwargs
        }
        
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self.base_url:
            kwargs["base_url"] = self.base_url
        
        return ChatOpenAI(**kwargs)
    
    def _get_claude_llm(self) -> ChatAnthropic:
        """获取Claude LLM"""
        if not self.api_key:
            raise ValueError("Claude/Anthropic API key is required")
        
        kwargs = {
            "model": self.model,
            "anthropic_api_key": self.api_key,
            "temperature": self.temperature,
            **self.kwargs
        }
        
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self.base_url:
            kwargs["base_url"] = self.base_url
        
        return ChatAnthropic(**kwargs)
    
    def _get_deepseek_llm(self) -> ChatOpenAI:
        """
        获取DeepSeek LLM
        
        DeepSeek使用OpenAI兼容的API，所以用ChatOpenAI但指定base_url
        """
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")
        
        kwargs = {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url or "https://api.deepseek.com/v1",
            "temperature": self.temperature,
            **self.kwargs
        }
        
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        
        return ChatOpenAI(**kwargs)
    
    def _get_ollama_llm(self) -> ChatOllama:
        """获取Ollama本地LLM"""
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            **self.kwargs
        }
        
        if self.base_url:
            kwargs["base_url"] = self.base_url
        
        return ChatOllama(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMConfig':
        """从字典创建（用于反序列化）"""
        return cls(
            provider=data["provider"],
            model=data["model"],
            temperature=data.get("temperature", cls.DEFAULT_TEMPERATURE),
            max_tokens=data.get("max_tokens"),
            base_url=data.get("base_url"),
        )


# === 全局默认配置 ===

_default_llm_config: Optional[LLMConfig] = None
_default_llm: Optional[BaseChatModel] = None


def set_default_llm_config(config: LLMConfig) -> None:
    """
    设置全局默认LLM配置
    
    Args:
        config: LLMConfig实例
    """
    global _default_llm_config, _default_llm
    _default_llm_config = config
    _default_llm = None  # 重置缓存的LLM实例


def get_default_llm_config() -> LLMConfig:
    """
    获取全局默认LLM配置
    
    如果未设置，会自动从环境变量创建。
    
    Returns:
        LLMConfig实例
    """
    global _default_llm_config
    
    if _default_llm_config is None:
        # 从环境变量自动创建
        _default_llm_config = LLMConfig()
    
    return _default_llm_config


def get_default_llm(force_new: bool = False) -> BaseChatModel:
    """
    获取全局默认LLM实例
    
    使用单例模式，避免重复创建。
    
    Args:
        force_new: 是否强制创建新实例
        
    Returns:
        LangChain BaseChatModel实例
    """
    global _default_llm
    
    if _default_llm is None or force_new:
        config = get_default_llm_config()
        _default_llm = config.get_llm()
    
    return _default_llm


def reset_default_llm() -> None:
    """重置全局默认LLM配置和实例"""
    global _default_llm_config, _default_llm
    _default_llm_config = None
    _default_llm = None


# === 便捷函数 ===

def create_llm(
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs
) -> BaseChatModel:
    """
    便捷函数：快速创建LLM实例
    
    Args:
        provider: 提供商名称
        model: 模型名称
        **kwargs: 其他参数
        
    Returns:
        LangChain LLM实例
        
    Example:
        llm = create_llm("openai", "gpt-4o-mini")
        llm = create_llm("claude")
        llm = create_llm("deepseek")
    """
    config = LLMConfig(provider=provider, model=model, **kwargs)
    return config.get_llm()


def get_available_providers() -> Dict[str, bool]:
    """
    检查哪些LLM提供商可用（有API key）
    
    Returns:
        {provider: is_available} 字典
    """
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "claude": bool(os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")),
        "deepseek": bool(os.getenv("DEEPSEEK_API_KEY")),
        "ollama": True,  # 本地总是可用（假设已安装）
    }


# === 用于测试的Mock LLM ===

class MockLLM:
    """
    Mock LLM 用于测试
    
    提供完整的测试功能：
    - 固定响应或自定义响应函数
    - 调用计数和历史记录
    - 模拟同步/异步调用
    
    用法：
        # 固定响应
        mock = MockLLM(response="Mocked response")
        agent = MyAgent(llm_client=mock)
        
        # 自定义响应函数
        def custom_response(messages, **kwargs):
            return f"Received {len(messages)} messages"
        mock = MockLLM(response_func=custom_response)
    """
    
    def __init__(self, response: str = "Mock LLM response", response_func=None):
        """
        初始化 Mock LLM
        
        Args:
            response: 固定返回的响应（如果 response_func 为 None）
            response_func: 自定义响应函数 (messages, **kwargs) -> str
        """
        self.response = response
        self.response_func = response_func
        self.temperature = 0.0
        self.model_name = "mock-llm"
        
        # 测试辅助属性
        self.call_count = 0
        self.call_history = []
    
    def invoke(self, messages, **kwargs):
        """模拟LangChain的invoke接口"""
        from langchain_core.messages import AIMessage
        from datetime import datetime
        import json
        
        # 记录调用
        self.call_count += 1
        self.call_history.append({
            'messages': messages,
            'kwargs': kwargs,
            'timestamp': datetime.now()
        })
        
        # 生成响应
        if self.response_func:
            content = self.response_func(messages, **kwargs)
        else:
            # 智能检测是否需要JSON响应
            content = self._generate_smart_response(messages)
        
        return AIMessage(content=content)
    
    def _generate_smart_response(self, messages):
        """智能生成响应（检测是否需要JSON）"""
        import json
        
        # 提取消息内容
        message_text = ""
        if isinstance(messages, list):
            for msg in messages:
                if hasattr(msg, 'content'):
                    message_text += msg.content + "\n"
                elif isinstance(msg, dict) and 'content' in msg:
                    message_text += msg['content'] + "\n"
        elif hasattr(messages, 'content'):
            message_text = messages.content
        elif isinstance(messages, str):
            message_text = messages
        
        message_lower = message_text.lower()
        
        # 优先检测trading decision请求（因为可能包含macro等其他关键词）
        if any(word in message_lower for word in ['trading decision', 'buy', 'sell', 'hold', 'action:', 'analyze the trading']):
            # 简化逻辑：总是返回BUY用于测试
            # 这样可以验证信号生成流程是否工作
            action = "BUY"
            conviction = 8
            reasoning = "Mock response - TEST MODE: Always BUY to verify signal generation pipeline. Strong bullish indicators detected with positive momentum."
            
            return f"""ACTION: {action}
CONVICTION: {conviction}
REASONING: {reasoning}"""
        
        # 检测macro agent请求（需要JSON）
        if 'macro' in message_lower and ('json' in message_lower or 'market_regime' in message_lower):
            return json.dumps({
                "market_regime": "bull",
                "regime_confidence": 0.75,
                "interest_rate_trend": "stable",
                "current_rate": 5.25,
                "risk_level": 4.0,
                "volatility_level": "medium",
                "gdp_trend": "expanding",
                "inflation_level": "moderate",
                "market_sentiment": "greed",
                "vix_level": 18.5,
                "confidence_score": 0.7,
                "reasoning": "Mock macro analysis: Bull market with moderate volatility"
            }, indent=2)
        
        # 检测sector agent请求（需要JSON）
        if 'sector' in message_lower and 'json' in message_lower:
            return json.dumps({
                "trend": "bullish",
                "relative_strength": 0.65,
                "momentum": "accelerating",
                "sector_rotation_signal": "rotating_in",
                "avg_pe_ratio": 25.0,
                "avg_growth_rate": 0.15,
                "sentiment": "bullish",
                "confidence": 0.8,
                "recommendation": "overweight",
                "reasoning": "Mock sector analysis: Technology sector showing strong momentum with accelerating growth"
            }, indent=2)
        
        # 默认响应
        return self.response
    
    async def ainvoke(self, messages, **kwargs):
        """模拟异步调用"""
        return self.invoke(messages, **kwargs)


def get_mock_llm(response: str = "Mock response") -> MockLLM:
    """
    获取Mock LLM用于测试
    
    Args:
        response: 固定返回的响应
        
    Returns:
        MockLLM实例
    """
    return MockLLM(response)


if __name__ == "__main__":
    # 示例用法
    print("=== LLM Configuration Demo ===\n")
    
    # 1. 检查可用的提供商
    print("Available providers:")
    for provider, available in get_available_providers().items():
        print(f"  {provider}: {'✓' if available else '✗'}")
    print()
    
    # 2. 使用默认配置
    try:
        print("Creating default LLM...")
        llm = get_default_llm()
        print(f"✓ Default LLM: {llm}")
        print(f"  Config: {get_default_llm_config().to_dict()}")
    except Exception as e:
        print(f"✗ Failed to create default LLM: {e}")
    print()
    
    # 3. 创建特定配置
    examples = [
        ("openai", "gpt-4o-mini"),
        ("claude", "claude-3-5-sonnet-20241022"),
        ("deepseek", "deepseek-chat"),
    ]
    
    for provider, model in examples:
        try:
            print(f"Creating {provider} LLM with model {model}...")
            config = LLMConfig(provider=provider, model=model)
            llm = config.get_llm()
            print(f"✓ Success: {llm}")
        except Exception as e:
            print(f"✗ Failed: {e}")
        print()
    
    # 4. Mock LLM
    print("Creating Mock LLM for testing...")
    mock = get_mock_llm("This is a test response")
    print(f"✓ Mock LLM: {mock}")
    print(f"  Response: {mock.response}")
