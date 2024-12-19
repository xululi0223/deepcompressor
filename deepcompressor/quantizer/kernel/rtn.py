# -*- coding: utf-8 -*-
"""Round-to-nearest (RTN) quantization module."""

import torch

from ...data.dtype import QuantDataType
from ...data.range import QuantRange
from ...data.zero import ZeroPointDomain
from ..config.kernel import BaseQuantKernel
from ..impl.simple import simple_quantize

__all__ = ["QuantRtnKernel", "rtn_quantize"]


class QuantRtnKernel(BaseQuantKernel):
    """
    实现了最近邻（Round-to-Nearest, RTN）量化内核，继承自 BaseQuantKernel。
    该类主要负责将输入张量按照最近邻策略进行量化。
    其核心方法是 quantize，该方法调用了 rtn_quantize 函数，执行具体的量化逻辑。
    Round-to-nearest (RTN) Quantization kernel.
    """

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        view_shape: torch.Size,
        quant_dtype: QuantDataType,
        zero_domain: ZeroPointDomain | None,
        scale: torch.Tensor,
        zero: torch.Tensor,
        quant_range: QuantRange | None = None,
        round_delta: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        调用 rtn_quantize 函数，将输入张量按照 RTN 策略进行量化。
        Quantize the tensor.

        Args:
            tensor (`torch.Tensor`):
                The tensor to quantize.
            view_shape (`torch.Size`):
                The view shape when quantizing the tensor.
            quant_dtype (`QuantDataType`):
                The quantization data type.
            zero_domain (`ZeroPointDomain` or `None`):
                The zero point domain.
            scale (`torch.Tensor`):
                The scale tensor.
            zero (`torch.Tensor`):
                The zero point tensor.
            quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                The quantization range.
            round_delta (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                The rounding delta.
            **kwargs: Other keyword arguments.

        Returns:
            `torch.Tensor`:
                The quantized tensor in the shape of ``view_shape``.
        """
        # 调用 rtn_quantize 函数，执行具体的量化逻辑
        return rtn_quantize(
            tensor,
            view_shape=view_shape,
            quant_dtype=quant_dtype,
            zero_domain=zero_domain,
            scale=scale,
            zero=zero,
            quant_range=quant_range,
            round_delta=round_delta,
        )


def rtn_quantize(
    tensor: torch.Tensor,
    *,
    view_shape: torch.Size,
    quant_dtype: QuantDataType,
    zero_domain: ZeroPointDomain | None,
    scale: torch.Tensor,
    zero: torch.Tensor,
    quant_range: QuantRange | None = None,
    round_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    实现了最近邻（RTN）量化算法，用于将输入张量按照指定的量化策略进行量化处理。
    Quantize the tensor using the RTN quantization kernel.

    Args:
        tensor (`torch.Tensor`):
            The tensor to quantize.
        view_shape (`torch.Size`):
            The view shape when quantizing the tensor.
        quant_dtype (`QuantDataType`):
            The quantization data type.
        zero_domain (`ZeroPointDomain` or `None`):
            The zero point domain.
        scale (`torch.Tensor`):
            The scale tensor.
        zero (`torch.Tensor`):
            The zero point tensor.
        quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
            The quantization range.
        round_delta (`torch.Tensor` or `None`, *optional*, defaults to `None`):
            The rounding delta.

    Returns:
        `torch.Tensor`:
            The quantized tensor in the shape of ``view_shape``.
    """
    # 重塑张量
    qtensor = tensor.view(view_shape)
    # 重塑舍入偏移量张量，以便与张量形状匹配
    round_delta = round_delta.view(view_shape) if round_delta is not None else None
    # 应用零点偏移和缩放
    if zero_domain == ZeroPointDomain.PostScale:
        qtensor = qtensor.add_(zero)
    qtensor = qtensor.div(scale)
    if zero_domain == ZeroPointDomain.PreScale:
        qtensor = qtensor.add_(zero)
        
    # 调用 simple_quantize 函数，对张量进行量化
    qtensor = simple_quantize(
        qtensor,
        quant_dtype=quant_dtype,
        has_zero_point=zero_domain is not None,
        quant_range=quant_range,
        round_delta=round_delta,
    )
    return qtensor
