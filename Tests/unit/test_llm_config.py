"""
Test LLM Configuration - 测试LLM配置系统

测试覆盖：
1. LLMConfig初始化
2. 提供商检测
3. API key获取
4. LLM实例创建
5. 全局配置管理
6. Mock LLM
"""

import pytest
import os
from unittest.mock import patch, Mock

from Agents.llm_config import (
    LLMConfig, LLMProvider,
    get_default_llm_config, get_default_llm, set_default_llm_config, reset_default_llm,
    create_llm, get_available_providers, get_mock_llm,
    LANGCHAIN_AVAILABLE
)


# 跳过测试如果LangChain未安装
pytestmark = pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE,
    reason="LangChain not installed"
)


class TestLLMProvider:
    """测试LLMProvider枚举"""
    
    def test_provider_values(self):
        """测试提供商值"""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.CLAUDE.value == "claude"
        assert LLMProvider.DEEPSEEK.value == "deepseek"
        assert LLMProvider.OLLAMA.value == "ollama"
    
    def test_provider_from_string(self):
        """测试从字符串创建"""
        assert LLMProvider("openai") == LLMProvider.OPENAI
        assert LLMProvider("claude") == LLMProvider.CLAUDE


class TestLLMConfig:
    """测试LLMConfig类"""
    
    def test_initialization_with_defaults(self):
        """测试使用默认值初始化"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = LLMConfig()
            
            assert config.provider == LLMProvider.OPENAI
            assert config.model == LLMConfig.DEFAULT_MODELS[LLMProvider.OPENAI]
            assert config.temperature == LLMConfig.DEFAULT_TEMPERATURE
            assert config.api_key == "test-key"
    
    def test_initialization_with_custom_values(self):
        """测试使用自定义值初始化"""
        config = LLMConfig(
            provider="claude",
            model="claude-3-opus",
            api_key="custom-key",
            temperature=0.5,
            max_tokens=2000
        )
        
        assert config.provider == LLMProvider.CLAUDE
        assert config.model == "claude-3-opus"
        assert config.api_key == "custom-key"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
    
    def test_provider_detection_openai(self):
        """测试检测OpenAI提供商"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            config = LLMConfig()
            assert config.provider == LLMProvider.OPENAI
    
    def test_provider_detection_claude(self):
        """测试检测Claude提供商"""
        with patch.dict(os.environ, {"CLAUDE_API_KEY": "test-key"}, clear=True):
            config = LLMConfig()
            assert config.provider == LLMProvider.CLAUDE
    
    def test_provider_detection_anthropic(self):
        """测试检测Anthropic提供商（备用名称）"""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            config = LLMConfig()
            assert config.provider == LLMProvider.CLAUDE
    
    def test_provider_detection_fallback_ollama(self):
        """测试当无API key时fallback到Ollama"""
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig()
            assert config.provider == LLMProvider.OLLAMA
    
    def test_api_key_from_env_openai(self):
        """测试从环境变量获取OpenAI key"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-123"}):
            config = LLMConfig(provider="openai")
            assert config.api_key == "sk-test-123"
    
    def test_api_key_from_env_claude(self):
        """测试从环境变量获取Claude key"""
        with patch.dict(os.environ, {"CLAUDE_API_KEY": "claude-test-123"}):
            config = LLMConfig(provider="claude")
            assert config.api_key == "claude-test-123"
    
    def test_api_key_explicit_override(self):
        """测试显式提供的API key优先"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            config = LLMConfig(provider="openai", api_key="explicit-key")
            assert config.api_key == "explicit-key"
    
    def test_to_dict(self):
        """测试转换为字典"""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000
        )
        
        data = config.to_dict()
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4"
        assert data["temperature"] == 0.7
        assert data["max_tokens"] == 1000
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "provider": "claude",
            "model": "claude-3-opus",
            "temperature": 0.5,
            "max_tokens": 2000
        }
        
        config = LLMConfig.from_dict(data)
        assert config.provider == LLMProvider.CLAUDE
        assert config.model == "claude-3-opus"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000


class TestLLMCreation:
    """测试LLM实例创建"""
    
    @patch('Agents.llm_config.ChatOpenAI')
    def test_get_openai_llm(self, mock_chat):
        """测试创建OpenAI LLM"""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="test-key"
        )
        
        llm = config.get_llm()
        
        # 验证ChatOpenAI被正确调用
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["api_key"] == "test-key"
    
    @patch('Agents.llm_config.ChatAnthropic')
    def test_get_claude_llm(self, mock_chat):
        """测试创建Claude LLM"""
        config = LLMConfig(
            provider="claude",
            model="claude-3-sonnet",
            api_key="test-key"
        )
        
        llm = config.get_llm()
        
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["model"] == "claude-3-sonnet"
        assert call_kwargs["anthropic_api_key"] == "test-key"
    
    @patch('Agents.llm_config.ChatOpenAI')
    def test_get_deepseek_llm(self, mock_chat):
        """测试创建DeepSeek LLM（使用OpenAI接口）"""
        config = LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key="test-key"
        )
        
        llm = config.get_llm()
        
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["model"] == "deepseek-chat"
        assert call_kwargs["base_url"] == "https://api.deepseek.com/v1"
    
    @patch('Agents.llm_config.ChatOllama')
    def test_get_ollama_llm(self, mock_chat):
        """测试创建Ollama LLM"""
        config = LLMConfig(
            provider="ollama",
            model="llama3.1"
        )
        
        llm = config.get_llm()
        
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["model"] == "llama3.1"
    
    def test_missing_api_key_error(self):
        """测试缺少API key时报错"""
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig(
                provider="openai",
                model="gpt-4",
                api_key=None
            )
            
            with pytest.raises(ValueError, match="API key is required"):
                config.get_llm()


class TestGlobalConfig:
    """测试全局配置管理"""
    
    def setup_method(self):
        """每个测试前重置全局配置"""
        reset_default_llm()
    
    def teardown_method(self):
        """每个测试后重置全局配置"""
        reset_default_llm()
    
    def test_get_default_llm_config_auto_create(self):
        """测试自动创建默认配置"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = get_default_llm_config()
            
            assert isinstance(config, LLMConfig)
            assert config.provider == LLMProvider.OPENAI
    
    def test_set_and_get_default_config(self):
        """测试设置和获取默认配置"""
        custom_config = LLMConfig(
            provider="claude",
            model="claude-3-opus",
            api_key="test-key"
        )
        
        set_default_llm_config(custom_config)
        retrieved_config = get_default_llm_config()
        
        assert retrieved_config == custom_config
        assert retrieved_config.provider == LLMProvider.CLAUDE
    
    @patch('Agents.llm_config.ChatOpenAI')
    def test_get_default_llm_singleton(self, mock_chat):
        """测试默认LLM是单例"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            llm1 = get_default_llm()
            llm2 = get_default_llm()
            
            # 应该只创建一次
            assert mock_chat.call_count == 1
            # 返回同一个实例（如果实现了单例）
            # 注意：由于mock，实际上每次都创建新的mock对象
    
    @patch('Agents.llm_config.ChatOpenAI')
    def test_get_default_llm_force_new(self, mock_chat):
        """测试强制创建新LLM实例"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            llm1 = get_default_llm()
            llm2 = get_default_llm(force_new=True)
            
            # 应该创建两次
            assert mock_chat.call_count == 2
    
    def test_reset_default_llm(self):
        """测试重置默认配置"""
        custom_config = LLMConfig(provider="claude", api_key="test-key")
        set_default_llm_config(custom_config)
        
        reset_default_llm()
        
        # 重置后应该重新自动创建
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            new_config = get_default_llm_config()
            assert new_config.provider == LLMProvider.OPENAI


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    @patch('Agents.llm_config.ChatOpenAI')
    def test_create_llm_simple(self, mock_chat):
        """测试快速创建LLM"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            llm = create_llm("openai")
            
            mock_chat.assert_called_once()
    
    @patch('Agents.llm_config.ChatOpenAI')
    def test_create_llm_with_model(self, mock_chat):
        """测试指定模型创建LLM"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            llm = create_llm("openai", model="gpt-4")
            
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4"
    
    def test_get_available_providers(self):
        """测试检查可用提供商"""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "CLAUDE_API_KEY": "test-key",
        }, clear=True):
            available = get_available_providers()
            
            assert available["openai"] is True
            assert available["claude"] is True
            assert available["deepseek"] is False
            assert available["ollama"] is True  # 本地总是可用


class TestMockLLM:
    """测试Mock LLM"""
    
    def test_get_mock_llm(self):
        """测试创建Mock LLM"""
        mock = get_mock_llm("Test response")
        
        assert mock.response == "Test response"
        assert mock.temperature == 0.0
        assert mock.model_name == "mock-llm"
    
    def test_mock_llm_invoke(self):
        """测试Mock LLM调用"""
        mock = get_mock_llm("Test response")
        
        # 模拟调用
        result = mock.invoke(["test message"])
        
        # 验证返回的是AIMessage
        assert hasattr(result, 'content')
        assert result.content == "Test response"
    
    def test_mock_llm_default_response(self):
        """测试Mock LLM默认响应"""
        mock = get_mock_llm()
        
        # get_mock_llm()的默认参数是"Mock response"
        assert mock.response == "Mock response"


class TestIntegration:
    """集成测试"""
    
    def setup_method(self):
        """测试前重置"""
        reset_default_llm()
    
    def teardown_method(self):
        """测试后清理"""
        reset_default_llm()
    
    @patch('Agents.llm_config.ChatOpenAI')
    def test_full_workflow(self, mock_chat):
        """测试完整工作流"""
        # 1. 设置全局配置
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            temperature=0.5
        )
        set_default_llm_config(config)
        
        # 2. 获取LLM
        llm = get_default_llm()
        
        # 3. 验证配置正确传递
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["temperature"] == 0.5
    
    def test_config_serialization(self):
        """测试配置序列化和反序列化"""
        original = LLMConfig(
            provider="claude",
            model="claude-3-opus",
            temperature=0.7,
            max_tokens=2000
        )
        
        # 序列化
        data = original.to_dict()
        
        # 反序列化
        restored = LLMConfig.from_dict(data)
        
        # 验证
        assert restored.provider == original.provider
        assert restored.model == original.model
        assert restored.temperature == original.temperature
        assert restored.max_tokens == original.max_tokens


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
