# -*- coding: utf-8 -*-
"""Zero-point for quantization."""

import enum

__all__ = ["ZeroPointDomain"]


class ZeroPointDomain(enum.Enum):
    """
    枚举类，用于定义量化过程中零点（Zero Point）的域（Domain）。
    在量化过程中，零点用于调整量化后的整数值与原始浮点值之间的偏移。
    根据不同的量化策略，零点可以应用在缩放（Scale）之前或之后。
    Zero-point domain.
    """
    # 零点应用在缩放之前
    PreScale = enum.auto()
    # 零点应用在缩放之后
    PostScale = enum.auto()
