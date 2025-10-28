"""
快速验证NewsAgent功能
"""
import asyncio
import json
from Agents.news_agent import NewsAgent
from Agents.meta_agent import MetaAgent


async def main():
    print("=" * 60)
    print("NewsAgent 快速功能验证")
    print("=" * 60)
    
    # 创建NewsAgent
    print("\n1. 创建NewsAgent...")
    news_agent = NewsAgent()
    print(f"   ✓ Agent名称: {news_agent.name}")
    print(f"   ✓ LLM可用: {news_agent.has_llm()}")
    print(f"   ✓ 工具数量: {len(news_agent.get_tools())}")
    print(f"   ✓ 资源数量: {len(news_agent.get_resources())}")
    
    # 测试工具
    print("\n2. 测试工具 - 获取最新新闻...")
    result = await news_agent.handle_tool_call(
        "get_latest_news",
        {"symbol": "AAPL", "limit": 3}
    )
    print(f"   ✓ 获取到 {result['count']} 篇新闻")
    print(f"   ✓ 第一篇: {result['articles'][0]['title'][:60]}...")
    
    # 测试情绪分析
    print("\n3. 测试工具 - 生成情绪报告...")
    summary = await news_agent.handle_tool_call(
        "get_news_summary",
        {"symbol": "TSLA"}
    )
    print(f"   ✓ 整体情绪: {summary['overall_sentiment']}")
    print(f"   ✓ 情绪分数: {summary['sentiment_score']:.2f}")
    print(f"   ✓ 分析文章数: {summary['articles_analyzed']}")
    print(f"   ✓ 关键主题: {', '.join(summary['key_themes'][:3])}")
    
    # 测试与MetaAgent集成
    print("\n4. 测试与MetaAgent集成...")
    meta = MetaAgent()
    await meta.connect_to_agent(
        agent_name="news",
        agent_instance=news_agent,
        description="News and sentiment analysis specialist"
    )
    print(f"   ✓ NewsAgent已连接到MetaAgent")
    print(f"   ✓ MetaAgent工具总数: {len(meta.get_all_tools())}")
    
    # 通过MetaAgent调用NewsAgent工具
    print("\n5. 通过MetaAgent调用NewsAgent...")
    result = await meta.execute_tool(
        agent_name="news",
        tool_name="get_latest_news",
        arguments={"symbol": "MSFT", "limit": 2}
    )
    print(f"   ✓ 成功调用，获取 {result['count']} 篇新闻")
    
    print("\n" + "=" * 60)
    print("✅ 所有功能验证通过！")
    print("=" * 60)
    
    # 显示一个完整的新闻情绪报告示例
    print("\n【完整示例：AAPL新闻情绪报告】")
    print("-" * 60)
    full_report = await news_agent.handle_tool_call(
        "get_news_summary",
        {"symbol": "AAPL", "days_back": 7}
    )
    print(json.dumps(full_report, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
