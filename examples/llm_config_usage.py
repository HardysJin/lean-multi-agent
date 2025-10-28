"""
LLM Configuration Usage Examples
展示如何在agents中使用统一的LLM配置

示例涵盖：
1. 全局配置使用
2. Agent级别的可选LLM
3. 不同提供商的切换
4. 测试时使用Mock LLM
"""

import asyncio
from Agents.llm_config import (
    get_default_llm,
    get_default_llm_config,
    set_default_llm_config,
    LLMConfig,
    create_llm,
    get_available_providers,
    get_mock_llm
)


# === 示例1: 全局默认配置 ===

def example_1_default_config():
    """使用全局默认配置（从.env自动检测）"""
    print("=== Example 1: Default Global Config ===\n")
    
    # 自动从环境变量检测
    config = get_default_llm_config()
    print(f"Detected provider: {config.provider.value}")
    print(f"Model: {config.model}")
    print(f"Config: {config.to_dict()}")
    
    # 获取LLM实例
    llm = get_default_llm()
    print(f"LLM instance: {llm}\n")


# === 示例2: 设置全局配置 ===

def example_2_set_global_config():
    """手动设置全局配置"""
    print("=== Example 2: Set Global Config ===\n")
    
    # 创建自定义配置
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.2
    )
    
    # 设置为全局默认
    set_default_llm_config(config)
    
    # 之后所有get_default_llm()都会使用这个配置
    llm = get_default_llm()
    print(f"Global LLM now uses: {config.to_dict()}\n")


# === 示例3: Agent可选使用LLM ===

class TradingAgent:
    """示例Agent，可选使用LLM"""
    
    def __init__(self, use_llm: bool = True, llm_config: LLMConfig = None):
        """
        Args:
            use_llm: 是否使用LLM
            llm_config: 自定义LLM配置（如果为None，使用全局默认）
        """
        self.use_llm = use_llm
        
        if use_llm:
            if llm_config:
                self.llm = llm_config.get_llm()
            else:
                self.llm = get_default_llm()
        else:
            self.llm = None
    
    def analyze(self, data: str):
        """分析数据"""
        if self.llm:
            # 使用LLM增强分析
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=f"Analyze this trading data: {data}")]
            response = self.llm.invoke(messages)
            return f"LLM Analysis: {response.content}"
        else:
            # 纯规则基础分析
            return f"Rule-based Analysis: {data}"


def example_3_optional_llm():
    """Agent可选使用LLM"""
    print("=== Example 3: Optional LLM in Agent ===\n")
    
    # Agent 1: 使用全局默认LLM
    agent1 = TradingAgent(use_llm=True)
    print("Agent 1 (with default LLM):")
    print(f"  LLM: {agent1.llm}")
    
    # Agent 2: 不使用LLM
    agent2 = TradingAgent(use_llm=False)
    print("\nAgent 2 (without LLM):")
    print(f"  LLM: {agent2.llm}")
    
    # Agent 3: 使用自定义LLM配置
    custom_config = LLMConfig(provider="claude", temperature=0.5)
    agent3 = TradingAgent(use_llm=True, llm_config=custom_config)
    print("\nAgent 3 (with custom Claude config):")
    print(f"  LLM: {agent3.llm}")
    print(f"  Config: {custom_config.to_dict()}\n")


# === 示例4: 快速创建不同提供商的LLM ===

def example_4_different_providers():
    """快速切换不同LLM提供商"""
    print("=== Example 4: Different Providers ===\n")
    
    # 检查可用的提供商
    available = get_available_providers()
    print("Available providers:")
    for provider, is_available in available.items():
        status = "✓" if is_available else "✗ (no API key)"
        print(f"  {provider}: {status}")
    print()
    
    # 快速创建不同提供商的LLM
    providers_to_try = ["openai", "claude", "deepseek"]
    
    for provider in providers_to_try:
        try:
            llm = create_llm(provider=provider)
            print(f"✓ Created {provider} LLM: {llm}")
        except Exception as e:
            print(f"✗ Failed to create {provider} LLM: {e}")
    print()


# === 示例5: 测试时使用Mock LLM ===

def example_5_mock_llm():
    """测试时使用Mock LLM"""
    print("=== Example 5: Mock LLM for Testing ===\n")
    
    # 创建Mock LLM
    mock_llm = get_mock_llm(response="BUY AAPL at $150")
    
    # 在Agent中使用
    class TestAgent:
        def __init__(self, llm):
            self.llm = llm
        
        def decide(self):
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content="What should I do?")]
            response = self.llm.invoke(messages)
            return response.content
    
    agent = TestAgent(mock_llm)
    decision = agent.decide()
    print(f"Mock LLM response: {decision}\n")


# === 示例6: MetaAgent集成 ===

async def example_6_meta_agent_integration():
    """MetaAgent使用LLM配置"""
    print("=== Example 6: MetaAgent with LLM Config ===\n")
    
    from Agents.meta_agent import MetaAgent
    from Agents.technical_agent import TechnicalAnalysisAgent
    
    # 方式1: 使用全局默认LLM
    meta1 = MetaAgent()  # 内部会调用get_default_llm()
    print(f"MetaAgent with default LLM: {meta1.llm_client}")
    
    # 方式2: 传入自定义配置
    custom_config = LLMConfig(provider="openai", model="gpt-4o-mini")
    # MetaAgent需要更新构造函数支持llm_config参数
    # meta2 = MetaAgent(llm_config=custom_config)
    
    # 连接TechnicalAgent（不需要LLM）
    technical = TechnicalAnalysisAgent()
    await meta1.connect_to_agent("technical", technical)
    
    print(f"Connected agents: {meta1.list_agents()}\n")


# === 示例7: 实际使用场景 ===

async def example_7_real_usage():
    """实际使用场景：完整工作流"""
    print("=== Example 7: Real Usage Workflow ===\n")
    
    # 1. 设置全局LLM配置（应用启动时）
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,  # 交易决策需要确定性
        max_tokens=4096
    )
    set_default_llm_config(config)
    print(f"✓ Global LLM config set: {config.to_dict()}")
    
    # 2. 创建Agents
    from Agents.technical_agent import TechnicalAnalysisAgent
    from Agents.meta_agent import MetaAgent
    
    # TechnicalAgent不需要LLM（纯指标计算）
    technical = TechnicalAnalysisAgent(algorithm=None)
    print("✓ TechnicalAgent created (no LLM)")
    
    # MetaAgent需要LLM（决策大脑）
    meta = MetaAgent()  # 自动使用全局LLM配置
    await meta.connect_to_agent("technical", technical)
    print(f"✓ MetaAgent created with LLM: {config.provider.value}/{config.model}")
    
    # 3. 执行工具并获取数据
    indicators = await meta.execute_tool(
        agent_name="technical",
        tool_name="calculate_indicators",
        arguments={"symbol": "AAPL"}
    )
    print(f"✓ Got indicators for AAPL: RSI={indicators['indicators']['rsi']['value']:.2f}")
    
    # 4. LLM决策（如果有API key）
    # decision = await meta.analyze_and_decide(
    #     symbol="AAPL",
    #     query="Should I buy AAPL based on current technical indicators?"
    # )
    # print(f"✓ Decision: {decision.action} (conviction: {decision.conviction})")
    
    print("\n✓ Workflow completed successfully")


# === 主函数 ===

async def main():
    """运行所有示例"""
    examples = [
        ("Example 1: Default Config", example_1_default_config),
        ("Example 2: Set Global Config", example_2_set_global_config),
        ("Example 3: Optional LLM", example_3_optional_llm),
        ("Example 4: Different Providers", example_4_different_providers),
        ("Example 5: Mock LLM", example_5_mock_llm),
        ("Example 6: MetaAgent Integration", example_6_meta_agent_integration),
        ("Example 7: Real Usage", example_7_real_usage),
    ]
    
    for name, func in examples:
        print(f"\n{'='*60}")
        print(f"{name}")
        print('='*60)
        try:
            if asyncio.iscoroutinefunction(func):
                await func()
            else:
                func()
        except Exception as e:
            print(f"Error: {e}")
        print()


if __name__ == "__main__":
    # 运行所有示例
    asyncio.run(main())
