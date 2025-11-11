"""
LLM Client wrapper for unified API access
支持 OpenAI, Anthropic, DeepSeek
"""

import json
import os
from typing import Dict, Any, Optional, List
from enum import Enum


class LLMProvider(Enum):
    """LLM提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"


class LLMClient:
    """统一的LLM客户端接口"""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        """
        初始化LLM客户端
        
        Args:
            provider: LLM提供商 (openai/anthropic/deepseek)
            model: 模型名称
            api_key: API密钥
            temperature: 温度参数
            max_tokens: 最大token数
        """
        self.provider = LLMProvider(provider)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 初始化对应的客户端
        if self.provider == LLMProvider.OPENAI:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        elif self.provider == LLMProvider.ANTHROPIC:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        elif self.provider == LLMProvider.DEEPSEEK:
            from openai import OpenAI
            # DeepSeek使用OpenAI兼容接口
            self.client = OpenAI(
                api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> str:
        """
        生成回复
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            response_format: 响应格式 {"type": "json_object"} 用于结构化输出
            **kwargs: 其他参数
        
        Returns:
            str: LLM生成的文本
        """
        if self.provider == LLMProvider.OPENAI or self.provider == LLMProvider.DEEPSEEK:
            return self._generate_openai(messages, response_format, **kwargs)
        elif self.provider == LLMProvider.ANTHROPIC:
            return self._generate_anthropic(messages, response_format, **kwargs)
    
    def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> str:
        """OpenAI/DeepSeek生成"""
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }
        
        # JSON mode
        if response_format:
            params["response_format"] = response_format
        
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
    
    def _generate_anthropic(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> str:
        """Anthropic生成"""
        # 提取system message
        system_message = None
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)
        
        params = {
            "model": self.model,
            "messages": user_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }
        
        if system_message:
            params["system"] = system_message
        
        # Anthropic目前不支持response_format，需要在prompt中要求JSON
        if response_format and response_format.get("type") == "json_object":
            # 在最后一条消息添加JSON格式要求
            if user_messages:
                user_messages[-1]["content"] += "\n\nRespond with valid JSON only."
        
        response = self.client.messages.create(**params)
        return response.content[0].text
    
    def generate_json(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成JSON响应
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
        
        Returns:
            Dict: 解析后的JSON对象
        """
        response_text = self.generate(
            messages,
            response_format={"type": "json_object"},
            **kwargs
        )
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            # JSON解析失败，尝试提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response_text}")
    
    def count_tokens(self, text: str) -> int:
        """
        估算token数量（简单实现）
        
        Args:
            text: 输入文本
        
        Returns:
            int: 估算的token数
        """
        # 简单估算：英文约4字符/token，中文约1.5字符/token
        # 这是粗略估算，实际应使用tiktoken
        return len(text) // 4


def create_llm_client(config: Optional[Dict[str, Any]] = None) -> LLMClient:
    """
    创建LLM客户端（工厂方法）
    
    Args:
        config: 配置字典，如果为None则从全局配置加载
    
    Returns:
        LLMClient: LLM客户端实例
    """
    if config is None:
        from backend.config import get_config
        cfg = get_config()
        config = {
            "provider": cfg.llm.provider,
            "model": cfg.llm.model,
            "api_key": cfg.llm.api_key,
            "temperature": cfg.llm.temperature,
            "max_tokens": cfg.llm.max_tokens
        }
    
    return LLMClient(**config)
