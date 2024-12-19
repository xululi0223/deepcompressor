# -*- coding: utf-8 -*-
"""Quantized tensor module."""

import torch

from .scale import QuantScale

__all__ = ["QuantTensor"]


class QuantTensor:
    """
    用于表示量化张量的模块。
    QuantTensor 类封装了量化张量的相关信息，包括量化和反量化后的数据、缩放因子、零点以及视图形状等。
    Quantized tensor.
    """

    # 存储反量化后的浮点张量数据
    _dequantized: torch.Tensor | None
    # 存储量化后的整数张量数据
    _quantized: torch.Tensor | None
    # 存储量化缩放因子
    scale: QuantScale | None
    # 存储量化零点
    zero: torch.Tensor | float | None
    # 存储张量的视图形状（用于重塑张量）
    view_shape: torch.Size | None

    def __init__(
        self,
        dequantized: torch.Tensor | None = None,
        quantized: torch.Tensor | None = None,
        scale: QuantScale | None = None,
        zero: torch.Tensor | float | None = None,
        view_shape: torch.Size | None = None,
    ):
        """
        Initialize the quantized tensor.
        
        Args:
            dequantized: 反量化后的浮点张量数据。
            quantized: 量化后的整数张量数据。
            scale: 量化缩放因子。
            zero: 量化零点。
            view_shape: 张量的视图形状。
        """
        # 参数验证，确保反量化或量化张量至少有一个被提供
        assert (
            dequantized is not None or quantized is not None
        ), "Either the dequantized or quantized tensor must be provided."
        self.view_shape = view_shape
        self._dequantized = dequantized
        self._quantized = quantized
        self.scale = scale
        self.zero = zero

    @property
    def data(self) -> torch.Tensor | None:
        """
        返回反量化后的浮点张量
        Get the dequantized tensor.
        """
        return self._dequantized

    @property
    def qdata(self) -> torch.Tensor | None:
        """
        返回量化后的整数张量
        Get the quantized tensor.
        """
        return self._quantized
