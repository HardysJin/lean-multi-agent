#!/bin/bash

set -e  # 遇到错误立即退出

echo "=============================================="
echo "   LEAN Multi-Agent 量化系统环境搭建"
echo "=============================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查Docker
echo -e "${YELLOW}[1/7]${NC} 检查Docker..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker未安装${NC}"
    echo "请先安装Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose未安装${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker和Docker Compose已就绪${NC}"
echo ""

# 创建目录结构
echo -e "${YELLOW}[2/7]${NC} 创建项目目录结构..."
mkdir -p Algorithm/MultiAgent/{Agents,Utils}
mkdir -p Data/cache
mkdir -p Logs
mkdir -p Results
echo -e "${GREEN}✅ 目录创建完成${NC}"
echo ""

# 创建docker-compose.yml
echo -e "${YELLOW}[3/7]${NC} 创建docker-compose.yml..."
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  lean:
    image: quantconnect/lean:latest
    container_name: lean-multi-agent
    volumes:
      - ./Algorithm:/Lean/Algorithm:rw
      - ./Data:/Lean/Data:rw
      - ./config.json:/Lean/Launcher/config.json:ro
      - ./Results:/Results:rw
      - ./Logs:/Lean/Logs:rw
    environment:
      - CLAUDE_API_KEY=${CLAUDE_API_KEY:-}
      - NEWS_API_KEY=${NEWS_API_KEY:-}
      - ALPHAVANTAGE_API_KEY=${ALPHAVANTAGE_API_KEY:-}
    working_dir: /Lean/Launcher
    command: >
      bash -c "
        echo '安装Python依赖...' &&
        pip install --quiet anthropic newsapi-python yfinance requests &&
        echo '✅ 依赖安装完成' &&
        echo '启动LEAN引擎...' &&
        dotnet QuantConnect.Lean.Launcher.dll
      "
    restart: unless-stopped
EOF
echo -e "${GREEN}✅ docker-compose.yml创建完成${NC}"
echo ""

# 创建config.json
echo -e "${YELLOW}[4/7]${NC} 创建config.json..."
cat > config.json << 'EOF'
{
    "algorithm-type-name": "ProductionMultiAgent",
    "algorithm-language": "Python",
    "algorithm-location": "/Lean/Algorithm/MultiAgent/main.py",
    
    "data-folder": "/Lean/Data",
    "results-destination-folder": "/Results",
    "log-handler": "ConsoleLogHandler",
    
    "environment": "backtesting",
    "close-automatically": true,
    
    "parameters": {
        "ema-fast": "10",
        "ema-slow": "20"
    },
    
    "job-user-id": "0",
    "api-access-token": "",
    "job-organization-id": ""
}
EOF
echo -e "${GREEN}✅ config.json创建完成${NC}"
echo ""

# 创建主策略文件
echo -e "${YELLOW}[5/7]${NC} 创建主策略文件..."
cat > Algorithm/MultiAgent/main.py << 'EOFMAIN'
from AlgorithmImports import *
import sys
import os
sys.path.append('/Lean/Algorithm/MultiAgent/Agents')
sys.path.append('/Lean/Algorithm/MultiAgent/Utils')

try:
    from multi_agent_system import MultiAgentSystem
    AGENT_AVAILABLE = True
except:
    AGENT_AVAILABLE = False

class ProductionMultiAgent(QCAlgorithm):
    """Multi-Agent量化策略"""
    
    def Initialize(self):
        """初始化"""
        
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)
        
        # 股票池
        self.symbols = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'TSLA']
        for symbol in self.symbols:
            equity = self.AddEquity(symbol, Resolution.Daily)
            equity.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
        
        # 初始化Multi-Agent系统
        self.agent_enabled = False
        if AGENT_AVAILABLE:
            try:
                claude_key = os.environ.get('CLAUDE_API_KEY', '')
                news_key = os.environ.get('NEWS_API_KEY', '')
                
                self.agent_system = MultiAgentSystem(
                    claude_api_key=claude_key,
                    news_api_key=news_key,
                    use_local_llm=False,
                    debug_mode=True
                )
                
                self.agent_enabled = True
                self.Debug("✅ Multi-Agent系统初始化成功")
                
            except Exception as e:
                self.Error(f"⚠️ Multi-Agent初始化失败: {e}")
        else:
            self.Debug("⚠️ Multi-Agent模块未找到，使用技术指标策略")
        
        # 技术指标
        self.indicators = {}
        for symbol in self.symbols:
            self.indicators[symbol] = {
                'rsi': self.RSI(symbol, 14, Resolution.Daily),
                'macd': self.MACD(symbol, 12, 26, 9, Resolution.Daily),
                'sma50': self.SMA(symbol, 50, Resolution.Daily),
                'sma200': self.SMA(symbol, 200, Resolution.Daily)
            }
        
        # 定时任务
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketClose(self.symbols[0], 30),
            self.DailyAnalysis
        )
        
        self.daily_signals = {}
        self.positions_info = {}
        self.trade_count = 0
        
        self.Debug(f"\n{'='*60}")
        self.Debug(f"策略初始化完成")
        self.Debug(f"监控股票: {', '.join(self.symbols)}")
        self.Debug(f"Multi-Agent: {'启用' if self.agent_enabled else '未启用（使用技术指标）'}")
        self.Debug(f"{'='*60}\n")
    
    def DailyAnalysis(self):
        """每日分析"""
        
        self.Debug(f"\n{'='*60}")
        self.Debug(f"📊 日期: {self.Time.strftime('%Y-%m-%d')}")
        self.Debug(f"{'='*60}")
        
        for symbol in self.symbols:
            try:
                data = self._prepare_data(symbol)
                
                if self.agent_enabled:
                    # Multi-Agent分析
                    decision = self.agent_system.analyze(symbol, data)
                else:
                    # 技术指标fallback
                    decision = self._technical_analysis(symbol)
                
                self.daily_signals[symbol] = decision
                
                # 输出分析结果
                action_emoji = {'buy': '🟢', 'sell': '🔴', 'hold': '⚪'}
                emoji = action_emoji.get(decision['action'], '⚪')
                
                self.Debug(f"\n{emoji} {symbol}:")
                self.Debug(f"  动作: {decision['action'].upper()}")
                self.Debug(f"  得分: {decision['score']:.2f}/10")
                self.Debug(f"  置信度: {decision['confidence']:.1%}")
                self.Debug(f"  理由: {decision['reasoning'][:80]}")
                
            except Exception as e:
                self.Error(f"❌ 分析{symbol}失败: {e}")
                self.daily_signals[symbol] = {
                    'action': 'hold',
                    'score': 0,
                    'confidence': 0,
                    'reasoning': f'分析失败: {str(e)}'
                }
        
        self.Debug(f"\n{'='*60}\n")
    
    def OnData(self, data):
        """执行交易"""
        
        # 每天开盘后1分钟执行
        if self.Time.hour != 9 or self.Time.minute != 31:
            return
        
        if not self.daily_signals:
            return
        
        for symbol, decision in self.daily_signals.items():
            if not data.ContainsKey(symbol):
                continue
            
            self._execute_decision(symbol, decision)
    
    def _prepare_data(self, symbol):
        """准备分析数据"""
        
        history = self.History(symbol, 60, Resolution.Daily)
        security = self.Securities[symbol]
        indicators = self.indicators.get(symbol, {})
        
        return {
            'symbol': symbol,
            'history': history,
            'current_price': security.Price,
            'current_position': self.Portfolio[symbol].Invested,
            'technical': {
                'rsi': indicators['rsi'].Current.Value if indicators['rsi'].IsReady else None,
                'macd': indicators['macd'].Current.Value if indicators['macd'].IsReady else None,
                'sma50': indicators['sma50'].Current.Value if indicators['sma50'].IsReady else None,
                'sma200': indicators['sma200'].Current.Value if indicators['sma200'].IsReady else None,
            }
        }
    
    def _technical_analysis(self, symbol):
        """纯技术指标分析"""
        
        indicators = self.indicators.get(symbol, {})
        
        if not indicators['rsi'].IsReady:
            return {'action': 'hold', 'score': 0, 'confidence': 0, 'reasoning': '指标未就绪'}
        
        rsi = indicators['rsi'].Current.Value
        price = self.Securities[symbol].Price
        sma50 = indicators['sma50'].Current.Value if indicators['sma50'].IsReady else price
        sma200 = indicators['sma200'].Current.Value if indicators['sma200'].IsReady else price
        
        score = 0
        reasons = []
        
        # RSI评分
        if rsi < 30:
            score += 3
            reasons.append(f"RSI超卖({rsi:.1f})")
        elif rsi > 70:
            score -= 3
            reasons.append(f"RSI超买({rsi:.1f})")
        
        # 均线评分
        if price > sma50 > sma200:
            score += 2
            reasons.append("多头排列")
        elif price < sma50 < sma200:
            score -= 2
            reasons.append("空头排列")
        
        # 决策
        if score >= 4:
            action = 'buy'
        elif score <= -4:
            action = 'sell'
        else:
            action = 'hold'
        
        return {
            'action': action,
            'score': score,
            'confidence': min(abs(score) / 5.0, 1.0),
            'reasoning': '; '.join(reasons) if reasons else 'RSI中性'
        }
    
    def _execute_decision(self, symbol, decision):
        """执行交易"""
        
        if decision['action'] == 'buy':
            if not self.Portfolio[symbol].Invested and decision['confidence'] > 0.6:
                target_weight = min(decision['confidence'] * 0.15, 0.12)
                self.SetHoldings(symbol, target_weight)
                
                self.positions_info[symbol] = {
                    'entry_time': self.Time,
                    'entry_price': self.Securities[symbol].Price,
                    'entry_reason': decision['reasoning']
                }
                
                self.trade_count += 1
                self.Debug(f"✅ 买入 {symbol} ({target_weight:.1%}): {decision['reasoning'][:50]}")
        
        elif decision['action'] == 'sell':
            if self.Portfolio[symbol].Invested:
                pnl = self.Portfolio[symbol].UnrealizedProfit
                pnl_pct = self.Portfolio[symbol].UnrealizedProfitPercent
                
                self.Liquidate(symbol)
                
                if symbol in self.positions_info:
                    entry_info = self.positions_info[symbol]
                    hold_days = (self.Time - entry_info['entry_time']).days
                    
                    self.Debug(f"❌ 卖出 {symbol}: 持有{hold_days}天, "
                             f"盈亏${pnl:.2f} ({pnl_pct:.1%})")
                    del self.positions_info[symbol]
                
                self.trade_count += 1
    
    def OnEndOfAlgorithm(self):
        """回测结束统计"""
        
        total_return = (self.Portfolio.TotalPortfolioValue / 100000 - 1) * 100
        
        self.Debug(f"\n{'='*60}")
        self.Debug(f"📈 回测结果汇总")
        self.Debug(f"{'='*60}")
        self.Debug(f"初始资金: $100,000")
        self.Debug(f"最终权益: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Debug(f"总收益率: {total_return:.2f}%")
        self.Debug(f"总交易次数: {self.trade_count}")
        self.Debug(f"{'='*60}\n")
EOFMAIN
echo -e "${GREEN}✅ 主策略文件创建完成${NC}"
echo ""

# 创建Multi-Agent系统
echo -e "${YELLOW}[6/7]${NC} 创建Multi-Agent系统..."
cat > Algorithm/MultiAgent/Agents/multi_agent_system.py << 'EOFAGENT'
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
EOFAGENT

# 创建空的__init__.py
touch Algorithm/MultiAgent/Agents/__init__.py
touch Algorithm/MultiAgent/Utils/__init__.py

echo -e "${GREEN}✅ Multi-Agent系统创建完成${NC}"
echo ""

# 创建.env模板
echo -e "${YELLOW}[7/7]${NC} 创建环境变量模板..."
cat > .env.template << 'EOF'
# API Keys配置
# 复制此文件为.env并填入真实的API Keys

# Claude API Key (可选，用于新闻分析)
# 获取地址: https://console.anthropic.com/
CLAUDE_API_KEY=sk-ant-api03-xxxxx

# News API Key (可选，用于获取新闻)
# 获取地址: https://newsapi.org/
NEWS_API_KEY=xxxxx

# Alpha Vantage API Key (可选，用于基本面数据)
# 获取地址: https://www.alphavantage.co/
ALPHAVANTAGE_API_KEY=xxxxx
EOF

cat > .env << 'EOF'
# 默认空配置（不使用LLM Agent，纯技术指标策略）
CLAUDE_API_KEY=
NEWS_API_KEY=
ALPHAVANTAGE_API_KEY=
EOF

echo -e "${GREEN}✅ 环境变量模板创建完成${NC}"
echo ""

# 创建.gitignore
cat > .gitignore << 'EOF'
.env
Results/
Logs/
Data/cache/
__pycache__/
*.pyc
.DS_Store
EOF

echo -e "${GREEN}✅ 所有文件创建完成！${NC}"
echo ""
echo "=============================================="
echo "   环境搭建完成！"
echo "=============================================="
echo ""
echo "📁 项目结构:"
echo "   lean-multi-agent/"
echo "   ├── Algorithm/MultiAgent/    # 策略代码"
echo "   ├── Data/                    # 数据目录"
echo "   ├── Logs/                    # 日志"
echo "   ├── Results/                 # 回测结果"
echo "   ├── docker-compose.yml       # Docker配置"
echo "   ├── config.json              # LEAN配置"
echo "   └── .env                     # API Keys"
echo ""
echo "📝 下一步:"
echo "   1. (可选) 编辑 .env 添加API Keys"
echo "   2. 运行: ./run.sh"
echo ""
echo "💡 提示:"
echo "   - 不配置API Keys也可以运行（使用纯技术指标策略）"
echo "   - 配置Claude API后可使用完整Multi-Agent功能"
echo "   - 首次运行需要下载Docker镜像（约2GB）"
echo ""