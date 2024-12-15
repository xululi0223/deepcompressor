# -*- coding: utf-8 -*-
"""Common uantization data."""

import enum

__all__ = ["TensorType"]


class TensorType(enum.Enum):
    """
    枚举类，用于定义张量（Tensor）的类型。
    枚举（Enum）是一种特殊的数据类型，允许将一组相关的常量组合在一起，提高代码的可读性和可维护性。
    The tensor type.
    """
    # 自动分配一个唯一的值给枚举成员
    Weights = enum.auto()
    Inputs = enum.auto()
    Outputs = enum.auto()
