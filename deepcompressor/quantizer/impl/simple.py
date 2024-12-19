# -*- coding: utf-8 -*-
"""Simple quantization functions."""

import torch

from ...data.dtype import QuantDataType
from ...data.range import LogQuantRange, QuantRange
from .ste import ste

__all__ = ["simple_quantize"]


def simple_quantize(
    tensor: torch.Tensor,
    *,
    quant_dtype: torch.dtype | QuantDataType,
    has_zero_point: bool,
    quant_range: QuantRange | None = None,
    round_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    用于量化张量的简单函数。
    它根据提供的量化数据类型 (quant_dtype) 和其他配置参数，对输入的 PyTorch 张量 (tensor) 进行量化处理。
    该函数支持不同类型的量化，包括标准定点量化、指数量化和浮点量化，并根据需要进行截断和梯度传递的处理。
    Simple quantization function.
    
    Args:
        tensor: 待量化的张量。
        quant_dtype: 量化数据类型。
        has_zero_point: 是否包含零点，用于定义量化的偏移量。
        quant_range: 量化范围，用于限制量化结果的取值范围。
        round_delta: 用于量化过程中的偏移量，通常用于误差补偿。
    """
    # 获取梯度需求
    requires_grad = tensor.requires_grad

    # 根据量化数据类型进行量化处理
    if isinstance(quant_dtype, torch.dtype):
        # 如果量化数据类型是 torch.dtype，则执行标准的定点量化逻辑
        # 保存原始数据类型
        dtype = tensor.dtype
        # 将张量转换为指定的量化数据类型，然后转换回原始数据类型，类似于量化和反量化的过程
        tensor = tensor.to(dtype=quant_dtype).to(dtype=dtype)
        # 如果提供了偏移量，则加到量化后的张量上
        if round_delta is not None:
            tensor = tensor.add_(round_delta)
        # 如果提供了量化范围，则对量化后的张量进行截断
        if quant_range is not None and quant_range.is_set():
            tensor = torch.clamp(tensor, min=quant_range.min, max=quant_range.max)
        return tensor
    
    elif isinstance(quant_dtype, QuantDataType):
        # 如果量化数据类型是 QuantDataType，则根据不同的量化类型执行相应的量化逻辑
        if quant_dtype.is_exponent:
            # 如果是指数量化，则执行指数量化逻辑
            # 指数量化不支持 round_delta
            assert round_delta is None, "round_delta is not supported for exponential quantization"
            # 构建指数量化范围
            quant_range = LogQuantRange.construct(quant_dtype, quant_range)
            # 对张量进行指数量化处理
            tensor = ste(tensor.log2(), lambda x: x.floor()) if requires_grad else tensor.log2_().floor_()
            # 截断量化结果，并恢复到原始的指数尺度
            return tensor.clamp_(min=quant_range.min, max=quant_range.max).exp2_()

        
        elif quant_dtype.is_float_point:
            # 如果是浮点量化，则执行浮点量化逻辑
            # 浮点量化不支持 round_delta
            assert round_delta is None, "round_delta is not supported for float quantization"
            # 截断张量到浮点量化范围
            tensor = torch.clamp(tensor, min=quant_dtype.min_value, max=quant_dtype.max_value)
            # 对张量进行浮点量化处理
            tensor = ste(tensor, lambda x: quant_dtype.get_codebook(normalize=False, device=tensor.device).quantize(x))
            # 截断量化范围
            if quant_range is not None and quant_range.is_set():
                tensor = tensor.clamp_(min=quant_range.min, max=quant_range.max)
            return tensor
        
        else:
            # 处理其他类型的量化（标准定点量化）
            # 构建量化范围
            quant_range = QuantRange.construct(quant_dtype, has_zero_point=has_zero_point, quant_range=quant_range)
            # 对张量进行量化处理
            # 如果提供了偏移量，则执行标准的四舍五入量化逻辑
            if round_delta is None:
                tensor = ste(tensor, lambda x: torch.round(x)) if requires_grad else tensor.round_()
            # 如果未提供偏移量，则执行向下取整量化逻辑，并将量化结果加上偏移量
            else:
                tensor = ste(tensor, lambda x: torch.floor(x)) if requires_grad else tensor.floor_()
                tensor = tensor.add_(round_delta)
            # 截断量化范围
            return tensor.clamp_(min=quant_range.min, max=quant_range.max)
    else:
        raise TypeError(
            f"quant_dtype must be either torch.dtype or QuantDataType, got {quant_dtype} ({type(quant_dtype)})"
        )
