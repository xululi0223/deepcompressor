# -*- coding: utf-8 -*-
"""Monkey-patching utilities."""

import copy
import functools
import types
import typing

import torch.nn as nn

__all__ = ["copy_func", "get_module_parents_map"]


def copy_func(f: types.FunctionType, globals: dict[str, typing.Any] | None = None):
    """
    用于复制一个已有的函数f。
    复制后的函数g拥有与原函数 f 相同的代码、默认参数和闭包，但可以在不同的上下文中使用。
    此函数在进行猴子补丁（monkey-patching）或动态函数修改时非常有用。
    Copied from https://stackoverflow.com/a/13503277/2988730 (@unutbu)
    and https://github.com/spcl/QuaRot/blob/main/fake_quant/monkeypatch.py.

    Copy a function.

    Args:
        f (`types.FunctionType`):
            Function to be copied.
        globals (`dict[str, typing.Any]` or `None`, *optional*, defaults to `None`):
            Globals.

    Returns:
        `types.FunctionType`:
            Copied function.
    """
    # 处理globals参数
    if globals is None:
        globals = f.__globals__     # 将globals设置为原函数f的全局命名空间
    # 创建新函数对象g，其代码、默认参数和闭包与原函数f相同
    g = types.FunctionType(f.__code__, globals, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    # 更新包装器属性
    g = functools.update_wrapper(g, f)
    # 设置模块属性
    g.__module__ = f.__module__
    # 复制关键字默认值
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)  # type: ignore
    return g


def get_module_parents_map(
    module: nn.Module, name: str = "", parents_map: dict[nn.Module, list[tuple[str, nn.Module, str]]] | None = None
) -> dict[nn.Module, list[tuple[str, nn.Module, str]]]:
    """
    用于创建一个映射（parents_map），其中每个子模块（child_module）对应一个列表，
    列表中的每个元组包含父模块的名称、父模块对象以及子模块在父模块中的名称。
    此函数通过递归遍历 PyTorch 的 nn.Module 结构，记录每个子模块的父模块信息。
    Get module parents map.

    Args:
        module (`nn.Module`):
            Module.
        name (`str`, *optional*, defaults to `""`):
            Name.
        parents_map (`dict[nn.Module, list[tuple[str, nn.Module, str]]]`, *optional*, defaults to `None`):
            Parents map.

    Returns:
        `dict[nn.Module, list[tuple[str, nn.Module, str]]]`:
            Module parents map. The key is the child module and the value is a list of tuples.
            Each tuple contains the name of the parent module, the parent module,
            and the child module name in the parent module.
    """
    # 初始化parents_map
    if parents_map is None:
        parents_map = {}
    # 遍历子模块
    for child_name, child_module in module._modules.items():
        # 跳过空的子模块
        if child_module is None:
            continue
        # 记录子模块的父模块信息
        parents_map.setdefault(child_module, []).append((name, module, child_name))
        # 递归遍历子模块
        get_module_parents_map(child_module, f"{name}.{child_name}" if name else child_name, parents_map)
    return parents_map
