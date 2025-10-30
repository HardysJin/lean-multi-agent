"""
Tests for MetaAgent with Macro/Sector Context Support
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from Agents.meta_agent import MetaAgent
from Agents.macro_agent import MacroAgent, MacroContext
from Agents.sector_agent import SectorAgent, SectorContext


class TestMetaAgentWithContext:
    """测试MetaAgent支持宏观和行业背景"""
    
    @pytest.mark.asyncio
    async def test_analyze_with_macro_context(self):
        """测试传入宏观背景"""
        meta = MetaAgent(enable_memory=False)
        
        macro_context = {
            'market_regime': 'bull',
            'risk_level': 3.0,
            'constraints': {
                'allow_long': True,
                'allow_short': False,
                'max_risk': 0.02
            }
        }
        
        # 模拟decision
        decision = await meta.analyze_and_decide(
            symbol='AAPL',
            macro_context=macro_context
        )
        
        assert decision.symbol == 'AAPL'
        assert decision.action in ['BUY', 'SELL', 'HOLD']
    
    @pytest.mark.asyncio
    async def test_analyze_with_sector_context(self):
        """测试传入行业背景"""
        meta = MetaAgent(enable_memory=False)
        
        sector_context = {
            'sector': 'Technology',
            'trend': 'bullish',
            'relative_strength': 0.5,
            'recommendation': 'overweight'
        }
        
        decision = await meta.analyze_and_decide(
            symbol='AAPL',
            sector_context=sector_context
        )
        
        assert decision.symbol == 'AAPL'
        assert decision.action in ['BUY', 'SELL', 'HOLD']
    
    @pytest.mark.asyncio
    async def test_analyze_with_both_contexts(self):
        """测试同时传入宏观和行业背景"""
        meta = MetaAgent(enable_memory=False)
        
        macro_context = {
            'market_regime': 'bull',
            'risk_level': 3.0
        }
        
        sector_context = {
            'sector': 'Technology',
            'trend': 'bullish'
        }
        
        decision = await meta.analyze_and_decide(
            symbol='AAPL',
            macro_context=macro_context,
            sector_context=sector_context
        )
        
        assert decision.symbol == 'AAPL'
        assert decision.action in ['BUY', 'SELL', 'HOLD']


class TestMetaAgentConstraints:
    """测试约束条件功能"""
    
    @pytest.mark.asyncio
    async def test_bear_market_constraint(self):
        """测试熊市禁止做多"""
        meta = MetaAgent(enable_memory=False)
        
        constraints = {
            'allow_long': False,
            'allow_short': False  # 极端情况：禁止所有交易
        }
        
        decision = await meta.analyze_and_decide(
            symbol='AAPL',
            constraints=constraints
        )
        
        # 应该返回HOLD
        assert decision.action == 'HOLD'
        assert '禁止' in decision.reasoning or 'prohibit' in decision.reasoning.lower()
        assert decision.conviction == 10  # 高确信度HOLD
    
    @pytest.mark.asyncio
    async def test_allow_long_constraint(self):
        """测试允许做多约束"""
        meta = MetaAgent(enable_memory=False)
        
        constraints = {
            'allow_long': True,
            'allow_short': False,
            'max_risk_per_trade': 0.02
        }
        
        decision = await meta.analyze_and_decide(
            symbol='AAPL',
            constraints=constraints
        )
        
        # 不应该因为约束返回HOLD（允许做多）
        assert decision.action in ['BUY', 'HOLD']  # 可能买或持有，但不能是强制HOLD


class TestMetaAgentIntegration:
    """集成测试：与MacroAgent和SectorAgent协同"""
    
    @pytest.mark.asyncio
    async def test_integration_with_macro_agent(self):
        """测试与MacroAgent集成"""
        # 创建agents
        macro_agent = MacroAgent()
        meta_agent = MetaAgent(enable_memory=False)
        
        # MacroAgent分析宏观环境
        macro_context = await macro_agent.analyze_macro_environment()
        
        # MetaAgent使用宏观背景做决策
        decision = await meta_agent.analyze_and_decide(
            symbol='AAPL',
            macro_context=macro_context.to_dict(),
            constraints=macro_context.constraints
        )
        
        assert decision.symbol == 'AAPL'
        
        # 如果宏观环境禁止做多，decision不应该是BUY
        if not macro_context.constraints.get('allow_long', True):
            assert decision.action != 'BUY'
    
    @pytest.mark.asyncio
    async def test_integration_with_sector_agent(self):
        """测试与SectorAgent集成"""
        # 创建agents
        sector_agent = SectorAgent()
        meta_agent = MetaAgent(enable_memory=False)
        
        # SectorAgent分析行业
        sector_context = await sector_agent.analyze_sector('Technology')
        
        # MetaAgent使用行业背景做决策
        decision = await meta_agent.analyze_and_decide(
            symbol='AAPL',
            sector_context=sector_context.to_dict()
        )
        
        assert decision.symbol == 'AAPL'
    
    @pytest.mark.asyncio
    async def test_full_integration(self):
        """完整集成测试：Macro + Sector + Meta"""
        # 创建所有agents
        macro_agent = MacroAgent()
        sector_agent = SectorAgent()
        meta_agent = MetaAgent(enable_memory=False)
        
        # 1. 宏观分析
        macro_context = await macro_agent.analyze_macro_environment()
        
        # 2. 行业分析
        sector_context = await sector_agent.analyze_sector('Technology')
        
        # 3. 个股决策
        decision = await meta_agent.analyze_and_decide(
            symbol='AAPL',
            macro_context=macro_context.to_dict(),
            sector_context=sector_context.to_dict(),
            constraints=macro_context.constraints
        )
        
        assert decision.symbol == 'AAPL'
        assert decision.action in ['BUY', 'SELL', 'HOLD']
        assert 1 <= decision.conviction <= 10
        
        # 验证decision的evidence包含上下文信息
        assert 'macro' in str(decision.evidence) or len(decision.evidence) > 0


class TestBackwardsCompatibility:
    """向后兼容性测试"""
    
    @pytest.mark.asyncio
    async def test_old_api_still_works(self):
        """测试旧API（不传macro/sector）仍然工作"""
        meta = MetaAgent(enable_memory=False)
        
        # 旧的调用方式（不传macro_context等）
        decision = await meta.analyze_and_decide(symbol='AAPL')
        
        assert decision.symbol == 'AAPL'
        assert decision.action in ['BUY', 'SELL', 'HOLD']
    
    @pytest.mark.asyncio
    async def test_old_api_with_additional_context(self):
        """测试旧API（使用additional_context）仍然工作"""
        meta = MetaAgent(enable_memory=False)
        
        decision = await meta.analyze_and_decide(
            symbol='AAPL',
            additional_context={'custom_data': 'test'}
        )
        
        assert decision.symbol == 'AAPL'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
