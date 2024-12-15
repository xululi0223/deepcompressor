# -*- coding: utf-8 -*-
"""Common utilities."""

import typing as tp

import numpy as np
import torch

__all__ = [
    "join_name",
    "join_names",
    "num2str",
    "split_sequence",
    "tree_map",
    "tree_copy_with_ref",
    "tree_split",
    "tree_collate",
    "hash_str_to_int",
]


def join_name(prefix: str, name: str, sep: str = ".", relative: bool = True) -> str:
    """
    用于将一个前缀prefix和一个名称name使用指定的分隔符sep连接起来。
    它还支持处理相对名称的情况，通过relative参数控制。
    Join a prefix and a name with a separator.

    Args:
        prefix (`str`): Prefix.
        name (`str`): Name.
        sep (`str`, *optional*, defaults to `.`): Separator.
        relative (`bool`, *optional*, defaults to `True`):
            Whether to resolve relative name.

    Returns:
        `str`: Joined name.
    """
    # 前缀prefix不为空
    if prefix:
        # 确保prefix不以分隔符sep结尾
        assert not prefix.endswith(sep), f"prefix={prefix} ends with sep={sep}"
        # 名称name不为空
        if name:
            # 如果名称name以分隔符sep开头且relative为True
            if name.startswith(sep) and relative:
                # Remove leading separator
                # 移除name开头的分隔符
                prefix_names = prefix.split(sep)    # 分割前缀
                unsep_name = name.lstrip(sep)       # 移除name开头的分隔符
                num_leading_seps = len(name) - len(unsep_name)      # 计算前导分隔符数量
                # 如果前导分隔符数量大于前缀名称数量，则通过重复分隔符sep来构建新的前缀prefix
                if num_leading_seps > len(prefix_names):
                    prefix = sep * (num_leading_seps - len(prefix_names) - 1)
                # 否则，将prefix_names列表中除最后num_leading_seps个元素外的所有元素连接起来，形成新的prefix
                else:
                    prefix = sep.join(prefix_names[:-num_leading_seps])
                return f"{prefix}{sep}{unsep_name}"
            else:
                return f"{prefix}{sep}{name}"
        else:
            return prefix
    else:
        return name


def join_names(*names: str, sep: str = ".", relative: bool = True) -> str:
    """
    用于将多个名称使用指定的分隔符连接起来。
    它内部调用 join_name 函数处理每一对前缀和名称，支持相对名称解析。
    Join multiple names with a separator.

    Args:
        names (`str`): Names.
        sep (`str`, *optional*, defaults to `.`): Separator.
        relative (`bool`, *optional*, defaults to `True`):
            Whether to resolve relative name.

    Returns:
        `str`: Joined name.
    """
    # 如果names为空，则返回空字符串
    if not names:
        return ""
    # 初始化前缀prefix
    prefix = ""
    # 遍历所有传入的名称names
    for name in names:
        # 调用join_name函数处理前缀prefix和名称name，得到新的前缀prefix
        prefix = join_name(prefix, name, sep=sep, relative=relative)
    return prefix


def num2str(num: int | float) -> str:
    """
    用于将一个整数或浮点数转换为特定格式的字符串。
    主要用于将负号替换为 n，将小数点替换为 p，以适应某些命名或标识的需求。
    Convert a number to a string.

    Args:
        num (`int` or `float`): Number to convert.

    Returns:
        str: Converted string.
    """
    # 数字转字符串并替换负号为 n
    s = str(num).replace("-", "n")
    # 通过小数点分割整数部分和小数部分
    us = s.split(".")
    # 如果小数部分为空或为0，则直接返回整数部分
    if len(us) == 1 or int(us[1]) == 0:
        return us[0]
    # 否则，返回整数部分加上小数部分，小数点替换为 p
    else:
        return us[0] + "p" + us[1]


def split_sequence(lst: tp.Sequence[tp.Any], splits: tp.Sequence[int]) -> list[list[tp.Any]]:
    """
    用于将一个序列 lst 按照给定的索引 splits 切分成多个子序列。
    返回一个包含切分后子序列的列表。
    Split a sequence into multiple sequences.

    Args:
        lst (`Sequence`):
            Sequence to split.
        splits (`Sequence`):
            Split indices.

    Returns:
        `list[list]`:
            Splitted sequences.
    """
    # 初始化返回列表
    ret = []
    # 初始化起始索引
    start = 0
    # 遍历切分索引
    for end in splits:
        # 切分序列并添加到返回列表
        ret.append(lst[start:end])
        # 更新起始索引
        start = end
    # 切分剩余部分并添加到返回列表
    ret.append(lst[start:])
    return ret


def tree_map(func: tp.Callable[[tp.Any], tp.Any], tree: tp.Any) -> tp.Any:
    """
    用于对树状结构的数据（如嵌套的字典和列表）中的每一个叶子节点应用一个指定的函数 func。
    支持的树状结构包括字典、列表、元组、PyTorch 张量和 NumPy 数组。
    Apply a function to tree-structured data.
    
    Args:
        func: 要应用的函数，接受一个参数并返回一个值。
        tree: 树状结构的数据。
    """
    # 如果tree是字典，则对每个键值对应用函数
    if isinstance(tree, dict):
        return {k: tree_map(func, v) for k, v in tree.items()}
    # 如果tree是列表或元组，则对每个元素应用函数
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(func, v) for v in tree)
    # 如果tree是 PyTorch 张量或 NumPy 数组，则直接应用函数
    elif isinstance(tree, (torch.Tensor, np.ndarray)):
        return func(tree)
    # 否则，直接返回tree
    else:
        return tree


def tree_copy_with_ref(
    tree: tp.Any, /, ref: tp.Any, copy_func: tp.Callable[[tp.Any, tp.Any], tp.Any] | None = None
) -> tp.Any:
    """
    用于根据参考树 ref，复制树状结构的数据 tree。
    在复制过程中，如果 tree 和 ref 中某个节点满足特定条件（如引用相同或数值接近），则使用 ref 中的节点代替 tree 中的节点。
    可选地，可以提供自定义的复制函数 copy_func。
    Copy tree-structured data with reference.
    
    Args:
        tree: 要复制的树状结构数据。
        ref: 参考树，用于决定如何复制 tree。
        copy_func: 自定义复制函数，接受tree和ref的对应节点，并返回复制后的节点。
    """
    # 如果tree是字典，则对每个键值对应用函数
    if isinstance(tree, dict):
        return {k: tree_copy_with_ref(v, ref[k]) for k, v in tree.items()}
    # 如果tree是列表或元组，则对每个元素应用函数
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_copy_with_ref(v, ref[i]) for i, v in enumerate(tree))
    # 如果tree是 PyTorch 张量，则直接返回tree或ref
    elif isinstance(tree, torch.Tensor):
        # 确保ref也是一个Pytorch张量
        assert isinstance(ref, torch.Tensor), f"source is a tensor but reference is not: {type(ref)}"
        # 确保tree和ref的形状相同
        assert tree.shape == ref.shape, f"source.shape={tree.shape} != reference.shape={ref.shape}"
        # 如果tree和ref的数据指针相同或数值接近，则返回ref，否则返回tree
        if tree.data_ptr() == ref.data_ptr() or tree.allclose(ref):
            return ref
        else:
            return tree
    # 否则，如果提供了自定义复制函数，则使用它
    elif copy_func is not None:
        return copy_func(tree, ref)
    # 否则，直接返回tree
    else:
        return tree


def tree_split(tree: tp.Any) -> list[tp.Any]:
    """
    用于将树状结构的数据 tree 按照批次（batch）进行拆分，生成一个包含多个数据样本的列表。
    适用于处理批次数据，如深度学习中的训练数据。
    Split tree-structured data into a list of data samples.
    
    Args:
        tree: 要拆分的树状结构数据。
    """

    def get_batch_size(tree: tp.Any) -> int | None:
        """
        内部函数。确定树状结构数据中的批次大小，即第一个具有批次维度（通常为第一个维度）的张量的大小。
        
        Args:
            tree: 要确定批次大小的树状结构数据。
        """
        # 如果tree是字典，则递归调用get_batch_size函数，找到第一个具有批次维度的张量
        if isinstance(tree, dict):
            for v in tree.values():
                b = get_batch_size(v)
                if b is not None:
                    return b
        # 如果tree是列表或元组，则递归调用get_batch_size函数，找到第一个具有批次维度的张量
        elif isinstance(tree, (list, tuple)):
            for samples in tree:
                b = get_batch_size(samples)
                if b is not None:
                    return b
        # 如果tree是 PyTorch 张量，则返回第一个维度的大小
        elif isinstance(tree, torch.Tensor) and tree.ndim > 0:
            return tree.shape[0]
        # 否则，返回None，表示无法确定批次大小
        return None

    def get_batch(tree: tp.Any, batch_id: int) -> tp.Any:
        """
        内部函数。根据批次索引 batch_id，获取树状结构数据 tree 中的一个数据样本。
        
        Args:
            tree: 要获取数据样本的树状结构数据。
            batch_id: 批次索引。
        """
        # 如果tree是字典，则递归调用get_batch函数，获取每个键值对应的数据样本
        if isinstance(tree, dict):
            return {k: get_batch(v, batch_id) for k, v in tree.items()}
        # 如果tree是列表或元组，则递归调用get_batch函数，获取每个元素对应的数据样本
        elif isinstance(tree, (list, tuple)):
            return [get_batch(samples, batch_id) for samples in tree]
        # 如果tree是 PyTorch 张量，则根据批次索引获取数据样本
        elif isinstance(tree, torch.Tensor) and tree.ndim > 0:
            return tree[batch_id : batch_id + 1]
        # 否则，直接返回tree，不进行任何处理
        else:
            return tree

    # 初始化返回列表，用于存储拆分后的数据样本
    ret = []
    # 获取批次大小
    batch_size = get_batch_size(tree)
    # 确保批次大小不为空
    assert batch_size is not None, "Cannot determine batch size"
    # 遍历每个批次，获取数据样本并添加到返回列表
    for i in range(batch_size):
        ret.append(get_batch(tree, i))
    return ret


def tree_collate(batch: list[tp.Any] | tuple[tp.Any, ...]) -> tp.Any:
    """
    用于将一批树状结构的数据 batch 合并成单一结构。
    这在数据加载过程中（如批处理）非常有用，能够将多个数据样本合并为一个批次。
    Collate function for tree-structured data.
    
    Args:
        batch: 要合并的批次数据，通常是一组数据样本组成的列表或元组。
    """
    # 对于字典类型，递归调用tree_collate函数，对每个键值对应用函数
    if isinstance(batch[0], dict):
        return {k: tree_collate([d[k] for d in batch]) for k in batch[0]}
    # 对于列表或元组类型，递归调用tree_collate函数，对每个元素应用函数
    elif isinstance(batch[0], (list, tuple)):
        return [tree_collate(samples) for samples in zip(*batch, strict=True)]
    # 对于 PyTorch 张量，使用 torch.cat 函数将多个张量合并为一个张量
    elif isinstance(batch[0], torch.Tensor):
        return torch.cat(batch)
    # 否则，直接返回第一个元素，不进行任何处理
    else:
        return batch[0]


def hash_str_to_int(s: str) -> int:
    """
    用于将一个字符串 s 哈希为一个整数。
    采用的哈希算法类似于 Java 中的字符串哈希，通过乘以一个基数（如 31）并加上字符的 ASCII 值来计算哈希值，
    同时使用大素数 10**9 + 7 作为模数以防止整数溢出。
    Hash a string to an integer.
    
    Args:
        s: 要哈希的字符串。
    """
    # 大素数模数定义
    modulus = 10**9 + 7  # Large prime modulus
    # 初始化哈希值
    hash_int = 0
    # 遍历字符串 s 中的每个字符，计算哈希值
    for char in s:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int
