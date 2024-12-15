# -*- coding: utf-8 -*-
"""Math utility functions."""

import torch

__all__ = ["is_pow2", "root_"]


def is_pow2(n: int) -> bool:
    """
    用于判断一个整数 n 是否是 2 的幂次方。
    它通过位运算的方式高效地进行判断，适用于需要快速确定数字是否为 2 的幂的场景。
    Check if a number is a power of 2.

    Args:
        n (`int`):
            The number to check.

    Returns:
        `bool`:
            Whether the number is a power of 2.
    """
    return (n & (n - 1) == 0) and (n > 0)


def root_(y: torch.Tensor, index: float) -> torch.Tensor:
    """
    用于对 PyTorch 张量 y 的每个元素进行逐元素开方运算，计算指定指数的根。
    该操作是“原地”进行的，即直接修改输入张量 y，而不是创建新的张量。
    这在需要节省内存或优化性能的场景中特别有用。
    In-place compute the root of a tensor element-wise.

    Args:
        y (`torch.Tensor`):
            The input tensor.
        index (`float`):
            The root index.

    Returns:
        `torch.Tensor`:
            The output tensor.
    """
    return y.pow_(1 / index) if index != 2 else y.sqrt_()
