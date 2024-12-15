# -*- coding: utf-8 -*-
"""Type hints used in deepcompressor."""

import torch
import torch.nn.functional as F

__all__ = [
    "ReshapeFn",
    "LinearReshapeFn",
    "ConvInputReshapeFn",
    "ConvOutputReshapedFn",
    "AttentionInputReshapeFn",
]


class ReshapeFn:
    """
    基础的重塑函数类，用于调整输入张量的形状。
    它定义了一个可调用的方法，但在基类中并不进行任何实际的重塑操作，仅返回原始张量。
    其他具体的重塑功能将在子类中实现。
    Reshape function.
    """

    def __call__(self, x: torch.Tensor, /, ic_last: bool = True) -> torch.Tensor:
        """
        调用方法。
        Reshape input tensor to the desired shape used for GEMM.

        Args:
            x (`torch.Tensor`):
                Input tensor.
            ic_last (`bool`, *optional*, defaults to `True`):
                Whether input channel is the last dimension.

        Returns:
            `torch.Tensor`:
                Reshaped tensor.
        """
        return x


class LinearReshapeFn(ReshapeFn):
    """
    用于线性层（全连接层）的输入重塑函数。
    它将输入张量重塑为适合矩阵乘法（GEMM）的二维形状。
    Inputs reshape function for linear layers.
    """

    def __call__(self, x: torch.Tensor, /, ic_last: bool = True) -> torch.Tensor:
        """
        可调用方法。
        Reshape input tensor to the desired 2D shape used for GEMM.

        Args:
            x (`torch.Tensor`):
                Input tensor.
            ic_last (`bool`, *optional*, defaults to `True`):
                Whether input channel is the last dimension.

        Returns:
            `torch.Tensor`:
                Reshaped tensor.
        """
        # 重塑张量，使输出张量形状为(-1, input_channels)
        return x.view(-1, x.shape[-1]).permute(int(not ic_last), int(ic_last))


class ConvInputReshapeFn(ReshapeFn):
    """
    用于卷积层输入的重塑函数。
    它利用 torch.nn.functional.unfold 函数将输入张量展开为适合矩阵乘法（GEMM）的二维形状。
    Inputs reshape function for convolutional layers.
    """

    def __init__(
        self, kernel_size: tuple[int, ...], padding: tuple[int, ...], stride: tuple[int, ...], dilation: tuple[int, ...]
    ) -> None:
        """Initialize the reshape function.

        Args:
            kernel_size (`tuple[int, ...]`):
                Kernel size.
            padding (`tuple[int, ...]`):
                Padding.
            stride (`tuple[int, ...]`):
                Stride.
            dilation (`tuple[int, ...]`):
                Dilation.
        """
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def __call__(self, x: torch.Tensor, /, ic_last: bool = True) -> torch.Tensor:
        """
        可调用方法。
        Reshape input tensor to the desired 2D shape used for GEMM.

        Args:
            x (`torch.Tensor`):
                Input tensor.
            ic_last (`bool`, *optional*, defaults to `True`):
                Whether input channel is the last dimension.

        Returns:
            `torch.Tensor`:
                Reshaped tensor.
        """
        # 将输入张量x展开为若干卷积核大小的重叠块
        x = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
        )
        # 获取输入通道数
        ic = x.shape[1]
        # 重塑张量，使输出张量形状为(-1, input_channels)
        if ic_last:
            return x.permute(0, 2, 1).reshape(-1, ic)
        else:
            return x.permute(1, 0, 2).reshape(ic, -1)


class ConvOutputReshapedFn(ReshapeFn):
    """
    用于卷积层输出的重塑函数。
    它将卷积层的输出张量重塑为适合矩阵乘法（GEMM）的二维形状。
    Outputs reshape function for convolutional layers.
    """

    def __call__(self, x: torch.Tensor, /, ic_last: bool = True) -> torch.Tensor:
        """Reshape output tensor to the desired shape.

        Args:
            x (`torch.Tensor`):
                Input tensor.
            ic_last (`bool`, *optional*, defaults to `True`):
                Whether input channel is the last dimension.

        Returns:
            `torch.Tensor`:
                Reshaped tensor.
        """
        # 获取输入通道数
        ic = x.shape[1]
        # 重塑张量，使输出张量形状为(N, input_channels, -1)
        x = x.view(x.shape[0], ic, -1)
        # 重塑张量，使输出张量形状为(-1, input_channels)
        if ic_last:
            return x.permute(0, 2, 1).reshape(-1, ic)
        else:
            return x.permute(1, 0, 2).reshape(ic, -1)


class AttentionInputReshapeFn(ReshapeFn):
    """
    用于注意力层输入的重塑函数。
    它将输入张量重塑为适合矩阵乘法（GEMM）的二维形状，特别适用于多维输入数据的注意力机制。
    Inputs reshape function for attention layer.
    """

    def __init__(self, channels_dim: int) -> None:
        """Initialize the reshape function.

        Args:
            channels_dim (`int`):
                The dimension of the channels.
        """
        self.channels_dim = channels_dim

    def __call__(self, x: torch.Tensor, /, ic_last: bool = True) -> torch.Tensor:
        """Reshape input tensor to the desired 2D shape used for GEMM.

        Args:
            x (`torch.Tensor`):
                Input tensor.
            ic_last (`bool`, *optional*, defaults to `True`):
                Whether input channel is the last dimension.

        Returns:
            `torch.Tensor`:
                Reshaped tensor.
        """
        # 获取通道数
        num_channels = x.shape[self.channels_dim]
        # 获取通道维度之前的所有维度大小
        shape_before = x.shape[: self.channels_dim]
        # 获取通道维度之后的所有维度大小
        shape_after = x.shape[self.channels_dim + 1 :]
        # 将输入张量x重塑为三维张量，形状为（A, C, B）
        x = x.view(shape_before.numel(), num_channels, shape_after.numel())
        # 根据ic_last标志位，重塑张量
        if ic_last:
            return x.permute(0, 2, 1).reshape(-1, num_channels)
        else:
            return x.permute(1, 0, 2).reshape(num_channels, -1)
