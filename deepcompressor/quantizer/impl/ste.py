# -*- coding: utf-8 -*-
"""Simple quantization functions."""

import typing as tp

import torch

__all__ = ["ste"]


class STEFunction(torch.autograd.Function):
    """
    实现了一个自定义的自动微分函数，用于量化过程中的前向和反向传播操作。
    ste 函数是该自定义函数的简便接口。
    STEFunction for quantization."""

    @staticmethod
    def forward(ctx: tp.Any, tensor: torch.Tensor, fn: tp.Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        前向传播方法。
        Forward pass for DtypeSTEFunction.
        
        Args:
            ctx: 上下文对象。用于存储在前向传播中保存的信息，可在反向传播中使用。
            tensor: 输入张量。
            fn: 一个可调用对象，接受张量并返回张量，通常用于量化操作。
        """
        # 应用函数fn处理输入张量
        return fn(tensor)

    @staticmethod
    def backward(ctx: tp.Any, grad_output: torch.Tensor) -> tp.Tuple[torch.Tensor, None]:
        """
        反向传播方法。
        Backward pass for DtypeSTEFunction.
        
        Args:
            ctx: 上下文对象。
            grad_output: 上一层反向传播传递的梯度。
        """
        # 直接将grad_output传递给下一层，实现STE的直通特性
        return grad_output, None


def ste(tensor: torch.Tensor, fn: tp.Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    STEFunction的简便接口。
    STE function.
    
    Args:
        tensor: 输入张量。
        fn: 一个可调用对象，接受张量并返回张量，通常用于量化操作。
    """
    # 调用STEFunction的apply方法，执行自定义的前向和反向传播操作
    return STEFunction.apply(tensor, fn)  # type: ignore
