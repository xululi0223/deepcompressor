# -*- coding: utf-8 -*-
"""Channel-wise metric calculation module."""

import typing as tp

import torch

from ..data.utils.shape import infer_view_shape

__all__ = ["ChannelMetric"]


class ChannelMetric:
    """
    用于按通道计算各种指标的工具类。
    该类包含多个静态方法，用于对张量进行归一化、计算绝对最大值、绝对平均值、归一化绝对平均值以及均方根等操作。
    这些方法在量化校准过程中用于评估和优化模型权重。
    Channel-wise metric.
    """

    @staticmethod
    def _normalize(
        tensor: torch.Tensor,
        group_shape: tp.Sequence[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        对输入张量进行归一化处理，使其按指定的分组形状进行标准化。

        Args:
            tensor: 待归一化的输入张量。
            group_shape: 分组形状，用于确定如何重塑张量进行归一化。
            dtype: 目标数据类型，归一化后张量的数据类型。
        """
        # 获取形状和维度
        shape, ndim = tensor.shape, tensor.ndim
        # 推断视图形状
        view_shape = infer_view_shape(tensor.shape, group_shape)
        # (d0, d1, d2, ...) -> (#g0, gs0, #g1, gs1, #g2, gs2, ...)
        # 重塑和类型转换
        tensor = tensor.view(view_shape).to(dtype=dtype)
        # 计算绝对最大值
        tensor_max = tensor.abs().amax(dim=list(range(1, ndim * 2, 2)), keepdim=True)
        # 防止除零错误
        tensor_max[tensor_max == 0] = 1
        # 归一化
        tensor = tensor / tensor_max
        # 恢复原始形状
        return tensor.view(shape)

    @staticmethod
    def _abs_max(
        tensor: torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int]:
        """
        计算张量按通道的绝对最大值。

        Args:
            tensor: 输入张量。
            num_channels: 通道数。
            group_shape: 分组形状。
            device: 目标设备。
            dtype: 目标数据类型。
        """
        return (
            tensor.view(tensor.shape[0], num_channels, -1)      # 重塑张量
            .abs()
            .amax(dim=(0, 2))                                   # 计算绝对最大值
            .view(-1)
            .to(dtype=dtype, device=device),                    # 调整形状和类型
            1,
        )

    @staticmethod
    def _abs_sum(
        tensor: torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int]:
        """
        计算张量按通道的绝对值总和。

        Args:
            tensor: 输入张量。
            num_channels: 通道数。
            group_shape: 分组形状。
            device: 目标设备。
            dtype: 目标数据类型。
        """
        # 重塑张量
        tensor = tensor.view(tensor.shape[0], num_channels, -1)
        # 计算元素数量
        cnt = tensor.shape[0] * tensor.shape[2]
        # 计算绝对值总和
        return tensor.abs().to(dtype=dtype).sum(dim=(0, 2)).view(-1).to(device=device), cnt

    @staticmethod
    def _abs_normalize_sum(
        tensor: torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int]:
        """
        先对张量进行归一化，然后计算归一化后张量按通道的绝对值总和。

        Args:
            tensor: 输入张量。
            num_channels: 通道数。
            group_shape: 分组形状。
            device: 目标设备。
            dtype: 目标数据类型。
        """
        return ChannelMetric._abs_sum(
            ChannelMetric._normalize(tensor, group_shape, dtype=dtype),
            num_channels,
            group_shape,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _square_sum(
        tensor: torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int]:
        """
        计算张量按通道的平方和。

        Args:
            tensor: 输入张量。
            num_channels: 通道数。
            group_shape: 分组形状。
            device: 目标设备。
            dtype: 目标数据类型。
        """
        # 重塑张量
        tensor = tensor.view(tensor.shape[0], num_channels, -1)
        # 计算元素数量
        cnt = tensor.shape[0] * tensor.shape[2]
        # 计算平方和
        return tensor.to(dtype=dtype).pow(2).sum(dim=(0, 2)).view(-1).to(device=device), cnt

    @staticmethod
    def _max_reduce(
        fn: tp.Callable[
            [torch.Tensor, int, tp.Sequence[int], torch.device, torch.dtype],
            tuple[torch.Tensor, torch.Tensor | int | float],
        ],
        tensors: tp.Sequence[torch.Tensor] | torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor | int | float]:
        """
        递归地对一组张量应用指定的最大化函数 fn，并结合多个张量的结果。

        Args:
            fn: 最大化函数。
            tensors: 输入张量。
            num_channels: 通道数。
            group_shape: 分组形状。
            device: 目标设备。
            dtype: 目标数据类型。
        """
        # 单个张量处理
        if isinstance(tensors, torch.Tensor):
            device = device or tensors.device
            return fn(tensors, num_channels, group_shape, device, dtype)
        # 多个张量处理
        # 1. 初始结果（对第一个张量应用 fn）
        # 2. 遍历剩余张量，对每个张量应用 fn 并更新结果
        else:
            rst_0, rst_1 = ChannelMetric._max_reduce(fn, tensors[0], num_channels, group_shape, device, dtype)
            for tensor in tensors[1:]:
                _rst_0, _rst_1 = ChannelMetric._max_reduce(fn, tensor, num_channels, group_shape, device, dtype)
                rst_0 = torch.maximum(rst_0, _rst_0.to(device=rst_0.device))
                if isinstance(rst_1, torch.Tensor):
                    rst_1 = torch.maximum(rst_1, _rst_1.to(device=rst_1.device))
                else:
                    rst_1 = max(rst_1, _rst_1)
            return rst_0, rst_1

    @staticmethod
    def _sum_reduce(
        fn: tp.Callable[
            [torch.Tensor, int, tp.Sequence[int], torch.device, torch.dtype],
            tuple[torch.Tensor, torch.Tensor | int | float],
        ],
        tensors: tp.Sequence[torch.Tensor] | torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor | int | float]:
        """
        递归地对一组张量应用指定的求和函数 fn，并累加多个张量的结果。

        Args:
            fn: 求和函数。
            tensors: 输入张量。
            num_channels: 通道数。
            group_shape: 分组形状。
            device: 目标设备。
            dtype: 目标数据类型。
        """
        # 单个张量处理
        if isinstance(tensors, torch.Tensor):
            device = device or tensors.device
            return fn(tensors.to(device), num_channels, group_shape, device, dtype)
        # 多个张量处理
        # 1. 初始累加结果（对第一个张量应用 fn）
        # 2. 遍历剩余张量，对每个张量应用 fn 并累加结果
        else:
            assert isinstance(tensors, (list, tuple))
            rst_0, rst_1 = ChannelMetric._sum_reduce(fn, tensors[0], num_channels, group_shape, device, dtype)
            for tensor in tensors[1:]:
                _rst_0, _rst_1 = ChannelMetric._sum_reduce(fn, tensor, num_channels, group_shape, device, dtype)
                rst_0 += _rst_0.to(device=rst_0.device)
                if isinstance(rst_1, torch.Tensor):
                    rst_1 += _rst_1.to(device=rst_1.device)
                else:
                    rst_1 += _rst_1
            return rst_0, rst_1

    @staticmethod
    def abs_max(
        tensors: tp.Iterable[torch.Tensor] | torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        device: torch.device | str = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        获取一组张量按通道的绝对最大值。
        Get the absolute maximum of the tensors, where `R[i] = AbsMax(T[i, :])`.

        Args:
            tensors: 输入张量。
            num_channels: 通道数。
            group_shape: 分组形状。
            device: 目标设备。
            dtype: 目标数据类型。
        """
        return ChannelMetric._max_reduce(
            ChannelMetric._abs_max, tensors, num_channels, group_shape, device=device, dtype=dtype
        )[0]

    @staticmethod
    def abs_mean(
        tensors: tp.Iterable[torch.Tensor] | torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        device: torch.device | str = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        获取一组张量按通道的绝对平均值。
        Get the absolute mean of the tensors, where `R[i] = AbsMean(T[i, :])`.

        Args:
            tensors: 输入张量。
            num_channels: 通道数。
            group_shape: 分组形状。
            device: 目标设备。
            dtype: 目标数据类型。
        """
        rst, cnt = ChannelMetric._sum_reduce(
            ChannelMetric._abs_sum, tensors, num_channels, group_shape, device=device, dtype=dtype
        )
        return rst.div_(cnt)

    @staticmethod
    def abs_normalize_mean(
        tensors: tp.Iterable[torch.Tensor] | torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        device: torch.device | str = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        获取一组张量按通道的绝对值归一化平均值。
        Get the absolute group normalized mean of the tensors, where `R[i] = Mean(U[i, :])`
        and `U[i,j] = Abs(T[i, j]) / AbsMax(T[:, j]))`.
        
        Args:
            tensors: 输入张量。
            num_channels: 通道数。
            group_shape: 分组形状。
            device: 目标设备。
            dtype: 目标数据类型。
        """
        rst, cnt = ChannelMetric._sum_reduce(
            ChannelMetric._abs_normalize_sum, tensors, num_channels, group_shape, device=device, dtype=dtype
        )
        return rst.div_(cnt)

    @staticmethod
    def root_mean_square(
        tensors: tp.Iterable[torch.Tensor] | torch.Tensor,
        num_channels: int,
        group_shape: tp.Sequence[int],
        device: torch.device | str = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        获取一组张量按通道的均方根值。
        Get the root mean square of the tensors, where `R[i] = Root(Mean(T[i, :]^2))`.
        
        Args:
            tensors: 输入张量。
            num_channels: 通道数。
            group_shape: 分组形状。
            device: 目标设备。
            dtype: 目标数据类型。
        """
        rst, cnt = ChannelMetric._sum_reduce(
            ChannelMetric._square_sum, tensors, num_channels, group_shape, device=device, dtype=dtype
        )
        return rst.div_(cnt).sqrt_()
