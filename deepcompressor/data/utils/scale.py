# -*- coding: utf-8 -*-
"""Utility functions for quantization scale."""

import typing as tp

import torch

from ..dtype import QuantDataType

__all__ = ["infer_scale_dtypes", "infer_scale_quant_spans", "infer_exponent_scale_level"]


def infer_scale_dtypes(
    scale_dtypes: tp.Sequence[torch.dtype | QuantDataType | None], default_dtype: torch.dtype | QuantDataType
) -> list[torch.dtype | QuantDataType]:
    """
    用于根据给定的量化数据类型序列和默认数据类型，返回一个新列表，其中任何为 None 的数据类型都会被替换为默认数据类型。
    Get the scale dtypes for the given tensor dtype.

    Args:
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]`):
            The scale dtypes.
        default_dtype (`torch.dtype`):
            The default scale dtype.

    Returns:
        `list[torch.dtype | QuantDataType]`:
            The scale dtypes.
    """
    # 确保默认数据类型是 torch.dtype 或 QuantDataType 类型
    assert isinstance(
        default_dtype, (torch.dtype, QuantDataType)
    ), f"dtype must be torch.dtype or QuantDataType, got {default_dtype}"
    # 如果 scale_dtypes 中的数据类型为 None，则替换为默认数据类型
    return [s_dtype or default_dtype for s_dtype in scale_dtypes]


def infer_scale_quant_spans(scale_dtypes: tp.Sequence[QuantDataType], base: int = 1) -> list[float]:
    """
    用于根据给定的量化数据类型序列和基数，推断每个量化层级的跨度（span），返回一个包含浮点数的列表。

    Args:
        scale_dtypes: 类行为Sequence[QuantDataType]的数据类型序列。
        base: 表示初始的基数。
    """
    # 初始化量化跨度列表，第一个量化层级的跨度为基数
    quant_spans: list[float] = [base]
    # 从最后一个量化数据类型开始，逐个计算每个量化层级的跨度
    for s_dtype in reversed(scale_dtypes[1:]):
        # 确保 s_dtype 是 QuantDataType
        assert isinstance(s_dtype, QuantDataType), f"s_dtype must be QuantDataType, got {s_dtype}"
        # 计算当前量化层级的跨度，并添加到列表中
        quant_spans.append(s_dtype.max_value * quant_spans[-1])
    # 返回反转后的量化跨度列表
    return list(reversed(quant_spans))


def infer_exponent_scale_level(scale_dtypes: tp.Sequence[torch.dtype | QuantDataType]) -> int:
    """
    用于确定量化数据类型序列中，哪一级别的量化类型具有指数特性，并返回其所在的级别。
    如果没有量化类型具有指数特性，则返回序列的长度。
    Get the exponent scaling level.

    Args:
        scale_dtypes (`Sequence[torch.dtype | QuantDataType]`):
            The scale data types.

    Returns:
        `int`: The exponent scaling level.
    """
    # 遍历量化数据类型序列，找到具有指数特性的量化数据类型所在的级别
    for level, scale_dtype in enumerate(scale_dtypes):
        if isinstance(scale_dtype, QuantDataType) and scale_dtype.is_exponent:
            return level
    # 如果没有量化数据类型具有指数特性，则返回序列的长度
    return len(scale_dtypes)
