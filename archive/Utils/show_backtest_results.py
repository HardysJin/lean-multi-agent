#!/usr/bin/env python3
import json
import sys
from datetime import datetime
from tabulate import tabulate

def analyze_backtest_results(order_events_file, summary_file):
    """综合分析回测结果：持仓变化 + 收益计算 + 汇总统计"""
    
    # 读取订单事件
    with open(order_events_file, 'r') as f:
        events = json.load(f)
    
    # 读取汇总统计
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    stats = summary.get('statistics', {})
    runtime_stats = summary.get('runtimeStatistics', {})
    algo_config = summary.get('algorithmConfiguration', {})
    
    print("\n" + "=" * 100)
    print(" " * 35 + "回测结果分析报告")
    print("=" * 100)
    
    # ==================== 基本信息 ====================
    print("\n【基本信息】")
    print(f"  算法名称: {algo_config.get('name', 'N/A')}")
    start_date = algo_config.get('startDate', 'N/A')
    end_date = algo_config.get('endDate', 'N/A')
    print(f"  回测期间: {start_date[:10]} 至 {end_date[:10]}")
    print(f"  初始资金: ${stats.get('Start Equity', 'N/A')}")
    
    # ==================== 持仓变化与收益计算 ====================
    print("\n" + "=" * 100)
    print(" " * 35 + "交易明细与收益分析")
    print("=" * 100)
    
    holdings = {}  # 当前持仓 {symbol: quantity}
    positions = {}  # 持仓成本 {symbol: {'qty': 0, 'cost': 0}}
    total_pnl = 0.0  # 累计已实现盈亏
    total_fees = 0.0  # 累计手续费
    trade_count = 0  # 交易次数
    
    # 收集所有交易记录
    trade_records = []
    
    for event in events:
        if event['status'] == 'filled':
            time_str = datetime.fromtimestamp(event['time']).strftime('%Y-%m-%d %H:%M')
            symbol = event['symbolValue']
            direction = event['direction']
            quantity = event['fillQuantity']
            price = event['fillPrice']
            fee = event.get('orderFeeAmount', 0)
            
            total_fees += fee
            
            # 初始化持仓记录
            if symbol not in holdings:
                holdings[symbol] = 0
                positions[symbol] = {'qty': 0, 'cost': 0, 'avg_price': 0}
            
            # 计算本次交易盈亏
            trade_pnl = 0.0
            trade_type = ""  # 交易类型：开仓/加仓/减仓/平仓
            
            if direction == "buy":
                # 买入操作
                if holdings[symbol] < 0:
                    # 平空头仓位
                    close_qty = min(abs(quantity), abs(holdings[symbol]))
                    avg_cost = positions[symbol]['avg_price']
                    trade_pnl = (avg_cost - price) * close_qty - fee
                    total_pnl += trade_pnl
                    
                    if abs(quantity) > abs(holdings[symbol]):
                        trade_type = "平空+开多"
                    else:
                        trade_type = "减仓(平空)" if holdings[symbol] + quantity != 0 else "平仓(空)"
                    
                    # 更新持仓
                    holdings[symbol] += quantity
                    if holdings[symbol] > 0:
                        # 反手做多
                        remaining_qty = holdings[symbol]
                        positions[symbol] = {
                            'qty': remaining_qty,
                            'cost': price * remaining_qty + fee,
                            'avg_price': (price * remaining_qty + fee) / remaining_qty
                        }
                    elif holdings[symbol] == 0:
                        # 完全平仓
                        positions[symbol] = {'qty': 0, 'cost': 0, 'avg_price': 0}
                    else:
                        # 减仓，保持均价不变
                        positions[symbol]['qty'] = holdings[symbol]
                        positions[symbol]['cost'] = abs(holdings[symbol]) * avg_cost
                else:
                    # 开多头仓位或加仓
                    if holdings[symbol] == 0:
                        trade_type = "开仓(多)"
                    else:
                        trade_type = "加仓(多)"
                    
                    old_cost = positions[symbol]['cost']
                    old_qty = positions[symbol]['qty']
                    new_cost = old_cost + price * quantity + fee
                    new_qty = old_qty + quantity
                    
                    holdings[symbol] = new_qty
                    positions[symbol] = {
                        'qty': new_qty,
                        'cost': new_cost,
                        'avg_price': new_cost / new_qty if new_qty != 0 else 0
                    }
            
            else:  # sell
                # 卖出操作
                if holdings[symbol] > 0:
                    # 平多头仓位
                    close_qty = min(abs(quantity), holdings[symbol])
                    avg_cost = positions[symbol]['avg_price']
                    trade_pnl = (price - avg_cost) * close_qty - fee
                    total_pnl += trade_pnl
                    
                    if abs(quantity) > holdings[symbol]:
                        trade_type = "平多+开空"
                    else:
                        trade_type = "减仓(平多)" if holdings[symbol] + quantity != 0 else "平仓(多)"
                    
                    # 更新持仓
                    holdings[symbol] += quantity
                    if holdings[symbol] < 0:
                        # 反手做空
                        remaining_qty = abs(holdings[symbol])
                        positions[symbol] = {
                            'qty': holdings[symbol],
                            'cost': abs(price * holdings[symbol]) + fee,
                            'avg_price': (abs(price * holdings[symbol]) + fee) / remaining_qty
                        }
                    elif holdings[symbol] == 0:
                        # 完全平仓
                        positions[symbol] = {'qty': 0, 'cost': 0, 'avg_price': 0}
                    else:
                        # 减仓，保持均价不变
                        positions[symbol]['qty'] = holdings[symbol]
                        # 持仓均价不变，但要调整总成本
                        positions[symbol]['cost'] = holdings[symbol] * avg_cost
                else:
                    # 开空头仓位或加仓
                    if holdings[symbol] == 0:
                        trade_type = "开仓(空)"
                    else:
                        trade_type = "加仓(空)"
                    
                    old_cost = positions[symbol]['cost']
                    old_qty = abs(positions[symbol]['qty'])
                    new_cost = old_cost + abs(price * quantity) + fee
                    new_qty = old_qty + abs(quantity)
                    
                    holdings[symbol] = holdings[symbol] + quantity
                    positions[symbol] = {
                        'qty': holdings[symbol],
                        'cost': new_cost,
                        'avg_price': new_cost / new_qty if new_qty != 0 else 0
                    }
            
            trade_count += 1
            
            # 收集交易记录
            direction_cn = "买入" if direction == "buy" else "卖出"
            
            # 计算持仓均价
            position_avg_price = positions[symbol]['avg_price'] if holdings[symbol] != 0 else 0
            
            # 判断是否显示交易盈亏
            if "平仓" in trade_type or "减仓" in trade_type:
                trade_pnl_str = f"${trade_pnl:+,.2f}"
                cumulative_pnl_str = f"${total_pnl:+,.2f}"
            else:
                trade_pnl_str = "-"
                cumulative_pnl_str = "-" if total_pnl == 0 else f"${total_pnl:+,.2f}"
            
            trade_records.append([
                time_str,
                symbol,
                direction_cn,
                trade_type,
                f"{abs(quantity):.0f}",
                f"${price:.2f}",
                f"${fee:.2f}",
                f"{holdings[symbol]:.0f}",
                f"${position_avg_price:.2f}" if position_avg_price > 0 else "-",
                trade_pnl_str,
                cumulative_pnl_str
            ])
    
    # 打印交易明细表格
    headers = ["时间", "股票", "操作", "类型", "数量", "价格", "手续费", "持仓", "持仓均价", "交易盈亏", "累计收益"]
    print("\n" + tabulate(trade_records, headers=headers, tablefmt="grid"))
    
    # ==================== 最终持仓 ====================
    print("\n" + "=" * 100)
    print("【最终持仓状态】")
    if any(qty != 0 for qty in holdings.values()):
        position_records = []
        for symbol, qty in holdings.items():
            if qty != 0:
                status = "多头" if qty > 0 else "空头"
                avg_price = positions[symbol]['avg_price']
                position_records.append([
                    symbol,
                    f"{qty:.0f}",
                    status,
                    f"${avg_price:.2f}",
                    "持仓中"
                ])
        position_headers = ["股票", "持仓数量", "持仓方向", "成本均价", "状态"]
        print("\n" + tabulate(position_records, headers=position_headers, tablefmt="grid"))
    else:
        print("  所有仓位已平仓")
    
    # ==================== 收益汇总 ====================
    print("\n" + "=" * 100)
    print(" " * 38 + "收益汇总")
    print("=" * 100)
    
    print("\n【收益指标】")
    net_profit_pct = stats.get('Net Profit', 'N/A')
    net_profit_amt = runtime_stats.get('Net Profit', 'N/A')
    print(f"  净收益率: {net_profit_pct}")
    print(f"  净收益额: {net_profit_amt}")
    print(f"  已实现盈亏: ${total_pnl:.2f}")
    print(f"  未实现盈亏: {runtime_stats.get('Unrealized', 'N/A')}")
    print(f"  年化收益率: {stats.get('Compounding Annual Return', 'N/A')}")
    print(f"  最终资金: {runtime_stats.get('Equity', 'N/A')}")
    
    print("\n【风险指标】")
    print(f"  最大回撤: {stats.get('Drawdown', 'N/A')}")
    print(f"  Sharpe比率: {stats.get('Sharpe Ratio', 'N/A')}")
    print(f"  Sortino比率: {stats.get('Sortino Ratio', 'N/A')}")
    print(f"  概率Sharpe比: {stats.get('Probabilistic Sharpe Ratio', 'N/A')}")
    print(f"  年化波动率: {stats.get('Annual Standard Deviation', 'N/A')}")
    
    print("\n【交易统计】")
    print(f"  总订单数: {stats.get('Total Orders', 'N/A')}")
    print(f"  成交笔数: {trade_count}")
    print(f"  胜率: {stats.get('Win Rate', 'N/A')}")
    print(f"  平均盈利: {stats.get('Average Win', 'N/A')}")
    print(f"  平均亏损: {stats.get('Average Loss', 'N/A')}")
    print(f"  盈亏比: {stats.get('Profit-Loss Ratio', 'N/A')}")
    
    print("\n【成本费用】")
    print(f"  总手续费: {stats.get('Total Fees', 'N/A')} (计算: ${total_fees:.2f})")
    print(f"  换手率: {stats.get('Portfolio Turnover', 'N/A')}")
    
    print("\n【策略容量】")
    print(f"  预估容量: {stats.get('Estimated Strategy Capacity', 'N/A')}")
    print(f"  最小容量资产: {stats.get('Lowest Capacity Asset', 'N/A')}")
    
    print("\n" + "=" * 100 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python3 show_backtest_results.py <order-events.json> <summary.json>")
        sys.exit(1)
    
    analyze_backtest_results(sys.argv[1], sys.argv[2])
