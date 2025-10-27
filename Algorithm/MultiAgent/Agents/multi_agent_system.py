import json
import os
from datetime import datetime, timedelta

class MultiAgentSystem:
    """Multi-Agent分析系统"""
    
    def __init__(self, claude_api_key='', news_api_key='', 
                 use_local_llm=False, debug_mode=False):
        
        self.debug = debug_mode
        self.claude = None
        self.newsapi = None
        
        # 初始化Claude
        if claude_api_key:
            try:
                import anthropic
                self.claude = anthropic.Anthropic(api_key=claude_api_key)
                self._log("✅ Claude API初始化成功")
            except Exception as e:
                self._log(f"⚠️ Claude初始化失败: {e}")
        
        # 初始化NewsAPI
        if news_api_key:
            try:
                from newsapi import NewsApiClient
                self.newsapi = NewsApiClient(api_key=news_api_key)
                self._log("✅ NewsAPI初始化成功")
            except Exception as e:
                self._log(f"⚠️ NewsAPI初始化失败: {e}")
        
        self.cache = {}
    
    def analyze(self, symbol, data):
        """完整分析流程"""
        
        results = {}
        
        # 1. 新闻分析
        if self.claude and self.newsapi:
            results['news'] = self._analyze_news(symbol)
        else:
            results['news'] = {'score': 0, 'confidence': 0, 'reasoning': '新闻Agent未启用'}
        
        # 2. 技术分析
        results['technical'] = self._analyze_technical(data.get('technical', {}))
        
        # 3. Meta决策
        final = self._meta_decision(symbol, results)
        
        return final
    
    def _analyze_news(self, symbol):
        """新闻分析"""
        
        try:
            # 获取新闻
            articles = self._fetch_news(symbol)
            
            if not articles:
                return {'score': 0, 'confidence': 0, 'reasoning': '无新闻'}
            
            # 构建prompt
            news_text = "\n".join([f"- {a['title']}" for a in articles[:5]])
            
            prompt = f"""分析{symbol}新闻情绪（简短）:

{news_text}

返回JSON:
{{"score": <-10到10>, "confidence": <0到1>, "reasoning": "<一句话>"}}

只返回JSON。"""
            
            response = self.claude.messages.create(
                model="claude-sonnet-4",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return json.loads(response.content[0].text)
            
        except Exception as e:
            self._log(f"新闻分析失败: {e}")
            return {'score': 0, 'confidence': 0, 'reasoning': f'失败:{str(e)[:30]}'}
    
    def _analyze_technical(self, technical):
        """技术分析"""
        
        rsi = technical.get('rsi')
        
        if not rsi:
            return {'score': 0, 'confidence': 0, 'reasoning': '技术指标未就绪'}
        
        if rsi < 30:
            return {'score': 7, 'confidence': 0.8, 'reasoning': f'RSI超卖{rsi:.1f}'}
        elif rsi > 70:
            return {'score': -7, 'confidence': 0.8, 'reasoning': f'RSI超买{rsi:.1f}'}
        else:
            return {'score': 0, 'confidence': 0.5, 'reasoning': f'RSI中性{rsi:.1f}'}
    
    def _meta_decision(self, symbol, results):
        """综合决策"""
        
        weights = {'news': 0.5, 'technical': 0.5}
        
        total_score = sum(
            results[agent]['score'] * weights[agent] * results[agent]['confidence']
            for agent in weights
        )
        
        avg_conf = sum(results[agent]['confidence'] * weights[agent] for agent in weights) / sum(weights.values())
        
        if total_score > 4:
            action = 'buy'
        elif total_score < -4:
            action = 'sell'
        else:
            action = 'hold'
        
        reasoning = '; '.join([
            f"{agent}:{results[agent]['reasoning']}"
            for agent in weights
        ])
        
        return {
            'action': action,
            'score': total_score,
            'confidence': avg_conf,
            'reasoning': reasoning
        }
    
    def _fetch_news(self, symbol):
        """获取新闻"""
        
        if not self.newsapi:
            return []
        
        try:
            today = datetime.now()
            week_ago = today - timedelta(days=7)
            
            response = self.newsapi.get_everything(
                q=symbol,
                from_param=week_ago.strftime('%Y-%m-%d'),
                to=today.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=5
            )
            
            return response.get('articles', [])
        except:
            return []
    
    def _log(self, message):
        if self.debug:
            print(f"[MultiAgent] {message}")
