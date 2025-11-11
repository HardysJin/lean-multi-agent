#!/usr/bin/env python3
"""
数据库初始化脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.database.connection import init_database, get_db_manager
from tabulate import tabulate


def main():
    """初始化数据库"""
    print("=" * 80)
    print("数据库初始化")
    print("=" * 80)
    print()
    
    # 初始化数据库（不删除现有表）
    db_manager = init_database(drop_existing=False)
    
    # 显示数据库信息
    print()
    print("数据库信息:")
    print("-" * 80)
    
    stats = db_manager.get_stats()
    
    info_table = [
        ['数据库路径', stats['db_path']],
        ['数据库大小', f"{stats['db_size_mb']:.2f} MB"],
        ['决策记录数', stats['decisions_count']],
        ['交易记录数', stats['trades_count']],
        ['回测结果数', stats['backtest_results_count']],
        ['组合快照数', stats['portfolio_snapshots_count']],
    ]
    
    print(tabulate(info_table, headers=['项目', '值'], tablefmt='presto'))
    
    print()
    print("=" * 80)
    print("数据库初始化完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
