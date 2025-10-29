"""
LEAN Multi-Agent Trading System
基于 LEAN 引擎的多智能体量化交易系统
"""

from setuptools import setup, find_packages
import os

# 读取 README 作为长描述
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# 读取 requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith('#')
            ]
    return []

setup(
    name="lean-multi-agent",
    version="0.1.0",
    author="HardysJin",
    author_email="",
    description="基于 LEAN 引擎的多智能体量化交易系统",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/HardysJin/lean-multi-agent",
    project_urls={
        "Bug Tracker": "https://github.com/HardysJin/lean-multi-agent/issues",
        "Documentation": "https://github.com/HardysJin/lean-multi-agent#readme",
        "Source Code": "https://github.com/HardysJin/lean-multi-agent",
    },
    
    # 包配置
    packages=find_packages(exclude=['Tests', 'Tests.*', 'Lean', 'Lean.*', 'docs', 'examples']),
    py_modules=['lean_multi_agent'],  # 包含顶层模块文件
    python_requires=">=3.10",
    
    # 依赖配置
    install_requires=read_requirements(),
    
    # 额外依赖（可选）
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'pytest-asyncio>=0.21.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'docs': [
            'sphinx>=6.0.0',
            'sphinx-rtd-theme>=1.2.0',
        ],
    },
    
    # 分类信息
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    
    # 关键词
    keywords=[
        "trading", 
        "quantitative", 
        "multi-agent", 
        "AI", 
        "machine-learning",
        "LEAN",
        "MCP",
        "LangChain",
        "memory-system",
    ],
    
    # 包数据
    package_data={
        'Configs': ['*.json'],
        'Data': ['README.md'],
    },
    
    # 入口点（命令行工具）
    entry_points={
        'console_scripts': [
            # 可以在这里添加命令行工具
            # 'lean-agent=Agents.cli:main',
        ],
    },
    
    # 包含非Python文件
    include_package_data=True,
    
    # Zip安全
    zip_safe=False,
)
