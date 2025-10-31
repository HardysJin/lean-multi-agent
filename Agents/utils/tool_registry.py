"""
Tool Registry - Automatic Tool Discovery Mechanism

自动工具发现机制，用于从Agent类中自动提取工具定义
替代硬编码的if-elif分支
"""

import inspect
import asyncio
from typing import Dict, List, Any, Optional, Callable, get_type_hints, Union
from dataclasses import dataclass
from functools import wraps


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    method: Callable


class ToolRegistry:
    """
    工具注册表
    
    使用装饰器标记可导出的方法，自动发现并生成工具定义
    """
    
    # 存储每个方法的元数据
    _tool_metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def tool(
        cls,
        description: Optional[str] = None,
        name: Optional[str] = None
    ) -> Callable:
        """
        装饰器：标记方法为可导出工具
        
        Args:
            description: 工具描述（如果不提供，从docstring提取）
            name: 工具名称（如果不提供，使用方法名）
            
        Example:
            @ToolRegistry.tool(description="Calculate technical indicators")
            def calculate_indicators(self, symbol: str, timeframe: str = "1d"):
                pass
        """
        def decorator(func: Callable) -> Callable:
            # 提取元数据
            tool_name = name or func.__name__
            tool_desc = description or cls._extract_description(func)
            
            # 存储元数据
            func_id = id(func)
            cls._tool_metadata[func_id] = {
                'name': tool_name,
                'description': tool_desc,
                'original_func': func
            }
            
            # 标记函数为工具
            func.__is_tool__ = True  # type: ignore
            func.__tool_name__ = tool_name  # type: ignore
            func.__tool_description__ = tool_desc  # type: ignore
            
            # 保持原函数行为不变
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            # 复制工具标记到wrapper
            wrapper.__is_tool__ = True  # type: ignore
            wrapper.__tool_name__ = tool_name  # type: ignore
            wrapper.__tool_description__ = tool_desc  # type: ignore
            
            return wrapper
        
        return decorator
    
    @classmethod
    def discover_tools(cls, agent_instance: Any) -> List[Dict[str, Any]]:
        """
        自动发现agent的所有工具
        
        Args:
            agent_instance: Agent实例
            
        Returns:
            工具定义列表（MCP协议格式）
        """
        tools = []
        
        # 遍历agent的所有方法
        for name, method in inspect.getmembers(agent_instance, predicate=inspect.ismethod):
            # 检查是否标记为工具
            if hasattr(method, '__is_tool__') and method.__is_tool__:
                tool_def = cls._create_tool_definition(name, method)
                if tool_def:
                    tools.append(tool_def)
        
        return tools
    
    @classmethod
    def _create_tool_definition(cls, method_name: str, method: Callable) -> Optional[Dict[str, Any]]:
        """
        创建工具定义
        
        Args:
            method_name: 方法名
            method: 方法对象
            
        Returns:
            MCP协议格式的工具定义
        """
        try:
            # 获取工具元数据
            tool_name = getattr(method, '__tool_name__', method_name)
            description = getattr(method, '__tool_description__', '')
            
            # 提取参数schema
            input_schema = cls._extract_parameters(method)
            
            return {
                'name': tool_name,
                'description': description,
                'inputSchema': input_schema
            }
        except Exception as e:
            # 如果提取失败，记录警告但不中断
            print(f"Warning: Failed to create tool definition for {method_name}: {e}")
            return None
    
    @classmethod
    def _extract_parameters(cls, method: Callable) -> Dict[str, Any]:
        """
        从方法签名提取参数schema
        
        Args:
            method: 方法对象
            
        Returns:
            JSON Schema格式的参数定义
        """
        # 获取方法签名
        sig = inspect.signature(method)
        
        # 获取类型提示
        try:
            type_hints = get_type_hints(method)
        except Exception:
            type_hints = {}
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            # 跳过self/cls
            if param_name in ('self', 'cls'):
                continue
            
            # 获取参数类型
            param_type = type_hints.get(param_name, Any)
            json_type = cls._python_type_to_json_type(param_type)
            
            # 获取参数描述（从docstring）
            param_desc = cls._extract_param_description(method, param_name)
            
            # 构建参数定义
            param_def = {
                'type': json_type,
                'description': param_desc
            }
            
            # 添加默认值
            if param.default is not inspect.Parameter.empty:
                param_def['default'] = param.default
            else:
                # 无默认值 = 必需参数
                required.append(param_name)
            
            properties[param_name] = param_def
        
        schema = {
            'type': 'object',
            'properties': properties
        }
        
        if required:
            schema['required'] = required
        
        return schema
    
    @classmethod
    def _python_type_to_json_type(cls, py_type: Any) -> str:
        """
        将Python类型转换为JSON Schema类型
        
        Args:
            py_type: Python类型
            
        Returns:
            JSON Schema类型字符串
        """
        # 处理Union类型（如Optional）
        origin = getattr(py_type, '__origin__', None)
        
        if origin is Union:
            # Optional[X] = Union[X, None]，取第一个非None类型
            args = getattr(py_type, '__args__', ())
            for arg in args:
                if arg is not type(None):
                    return cls._python_type_to_json_type(arg)
            return 'string'  # 默认
        
        if origin is list or py_type is list:
            return 'array'
        
        if origin is dict or py_type is dict:
            return 'object'
        
        # 基本类型映射
        type_mapping = {
            str: 'string',
            int: 'integer',
            float: 'number',
            bool: 'boolean',
            list: 'array',
            dict: 'object',
            List: 'array',
            Dict: 'object',
        }
        
        # 尝试直接映射
        if py_type in type_mapping:
            return type_mapping[py_type]
        
        # 尝试按名称映射
        type_name = getattr(py_type, '__name__', str(py_type))
        if 'str' in type_name.lower():
            return 'string'
        elif 'int' in type_name.lower():
            return 'integer'
        elif 'float' in type_name.lower() or 'number' in type_name.lower():
            return 'number'
        elif 'bool' in type_name.lower():
            return 'boolean'
        elif 'list' in type_name.lower() or 'array' in type_name.lower():
            return 'array'
        elif 'dict' in type_name.lower() or 'object' in type_name.lower():
            return 'object'
        
        # 默认为string
        return 'string'
    
    @classmethod
    def _extract_description(cls, func: Callable) -> str:
        """
        从docstring提取描述（第一行）
        
        Args:
            func: 函数对象
            
        Returns:
            描述文本
        """
        doc = inspect.getdoc(func)
        if doc:
            # 返回第一行非空文本
            lines = [line.strip() for line in doc.split('\n') if line.strip()]
            if lines:
                return lines[0]
        return func.__name__.replace('_', ' ').title()
    
    @classmethod
    def _extract_param_description(cls, func: Callable, param_name: str) -> str:
        """
        从docstring提取参数描述
        
        Args:
            func: 函数对象
            param_name: 参数名
            
        Returns:
            参数描述
        """
        doc = inspect.getdoc(func)
        if not doc:
            return ''
        
        # 查找Args部分
        lines = doc.split('\n')
        in_args_section = False
        
        for line in lines:
            stripped = line.strip()
            
            # 检测Args部分开始
            if stripped.lower().startswith('args:'):
                in_args_section = True
                continue
            
            # 检测下一个section（结束Args）
            if in_args_section and stripped.endswith(':') and not stripped.startswith(param_name):
                break
            
            # 在Args部分查找参数
            if in_args_section and stripped.startswith(f"{param_name}:"):
                # 提取描述
                desc = stripped[len(param_name) + 1:].strip()
                return desc
        
        return param_name.replace('_', ' ').title()


# 便捷导出
tool = ToolRegistry.tool
discover_tools = ToolRegistry.discover_tools
