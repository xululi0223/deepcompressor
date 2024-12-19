# -*- coding: utf-8 -*-
"""Quantization scale module."""

import math
import typing as tp
from dataclasses import dataclass, field

import torch

from ...data.dtype import QuantDataType
from ...data.range import DynamicRange, QuantRange, RangeBound
from ...data.scale import QuantScale
from ...data.utils import ScaleUtils
from ...data.zero import ZeroPointDomain
from .simple import simple_quantize

__all__ = ["quantize_scale", "QuantScaleInfo"]


def quantize_scale(
    s: torch.Tensor,
    /,
    *,
    quant_dtypes: tp.Sequence[QuantDataType],
    quant_spans: tp.Sequence[float],
    view_shapes: tp.Sequence[torch.Size],
) -> QuantScale:
    """
    对给定的缩放张量 s 进行量化处理。
    它根据提供的量化数据类型 (quant_dtypes)、量化跨度 (quant_spans) 和视图形状 (view_shapes) 将缩放张量分阶段量化，
    并返回一个 QuantScale 对象，包含量化后的缩放信息。
    Quantize the scale tensor.

    Args:
        s (`torch.Tensor`):
            The scale tensor.
        quant_dtypes (`Sequence[QuantDataType]`):
            The quantization dtypes of the scale tensor.
        quant_spans (`Sequence[float]`):
            The quantization spans of the scale tensor.
        view_shapes (`Sequence[torch.Size]`):
            The view shapes of the scale tensor.

    Returns:
        `QuantScale`:
            The quantized scale tensor.
    """
    scale = QuantScale()            # 初始化量化缩放对象
    s = s.abs()                     # 取绝对值，确保缩放张量为正
    
    # 遍历并量化每一阶段的缩放张量（除最后一阶段外）
    for view_shape, quant_dtype, quant_span in zip(view_shapes[:-1], quant_dtypes[:-1], quant_spans[:-1], strict=True):
        # 将缩放张量s按照视图形状view_shape进行重塑，便于分组和处理
        s = s.view(view_shape)  # (#g0, rs0, #g1, rs1, #g2, rs2, ...)
        # 计算指定维度的最大值，得到动态跨度ss
        ss = s.amax(dim=list(range(1, len(view_shape), 2)), keepdim=True)  # i.e., s_dynamic_span
        # 将动态跨度除以当前的量化跨度，得到缩放因子
        ss = simple_quantize(
            ss / quant_span, has_zero_point=False, quant_dtype=quant_dtype
        )  # i.e., s_scale = s_dynamic_span / s_quant_span
        # 将张量s除以量化后的缩放因子ss，调整缩放后的张量值
        s = s / ss
        # 将量化后的缩放因子ss添加到缩放列表中
        scale.append(ss)
        
    # 处理最后一阶段的缩放张量
    # 将张量s重塑为最后一阶段的视图形状
    view_shape = view_shapes[-1]
    s = s.view(view_shape)
    # 判断是否需要计算动态跨度
    if any(v != 1 for v in view_shape[1::2]):       # 判断view_shape中的每隔一个维度的值是否为1
        # 如果存在不为1的维度
        # 计算指定维度的最大值，得到动态跨度ss
        ss = s.amax(dim=list(range(1, len(view_shape), 2)), keepdim=True)
        # 对动态跨度ss进行量化处理，得到最终的缩放因子
        ss = simple_quantize(ss / quant_spans[-1], has_zero_point=False, quant_dtype=quant_dtypes[-1])
    else:
        # 如果所有维度的值都为1
        # 确保最后一个量化跨度为1
        assert quant_spans[-1] == 1, "The last quant span must be 1."
        # 直接将张量s作为最终的缩放因子
        ss = simple_quantize(s, has_zero_point=False, quant_dtype=quant_dtypes[-1])
    # 将最终的缩放因子ss添加到缩放列表中
    scale.append(ss)
    scale.remove_zero()
    return scale


@dataclass
class QuantScaleInfo:
    """
    用于存储和管理量化缩放相关的配置信息和参数。
    它涵盖了张量的视图形状、量化数据类型、零点域、量化范围等信息，以及线性和指数缩放的具体配置。
    """
    # region tensor information
    # 张量信息区域
    # 张量的视图形状，用于量化过程中的重塑操作
    tensor_view_shape: torch.Size
    # 张量的量化数据类型
    tensor_quant_dtype: torch.dtype | QuantDataType
    # 张量的零点域，决定零点偏移的方式
    tensor_zero_domain: ZeroPointDomain | None
    # 张量的量化范围，定义量化后的值域
    tensor_quant_range: QuantRange
    # 张量的范围边界，用于限制动态范围
    tensor_range_bound: RangeBound | None
    # endregion
    
    # 默认量化数据类型与缩放配置
    # 默认的量化数据类型，用于推断缩放数据类型
    default_quant_dtype: torch.dtype | QuantDataType
    # 每个缩放阶段的视图形状列表
    scale_view_shapes: list[torch.Size]
    # 每个缩放阶段的量化数据类型列表
    scale_quant_dtypes: list[torch.dtype | QuantDataType]
    # 指数缩放的级别
    exponent_scale_level: int = field(init=False)
    # 零点的量化数据类型
    zero_quant_dtype: torch.dtype | QuantDataType = field(init=False)
    
    # region linear scale information
    # 线性缩放信息区域
    # 线性缩放的量化跨度
    linear_tensor_quant_span: float = field(init=False)
    # 线性缩放阶段的量化数据类型列表
    linear_scale_quant_dtypes: list[torch.dtype | QuantDataType] = field(init=False)
    # 线性缩放阶段的视图形状列表
    linear_scale_view_shapes: list[torch.Size] = field(init=False)
    # 线性缩放阶段的量化跨度列表
    linear_scale_quant_spans: list[float] = field(init=False)
    # endregion
    
    # region exponent scale information
    # 指数缩放信息区域
    # 指数缩放的量化跨度
    exponent_tensor_quant_span: float = field(init=False)
    # 指数缩放阶段的量化数据类型列表
    exponent_scale_quant_dtypes: list[torch.dtype | QuantDataType] = field(init=False)
    # 指数缩放阶段的视图形状列表
    exponent_scale_view_shapes: list[torch.Size] = field(init=False)
    # 指数缩放阶段的量化跨度列表
    exponent_scale_quant_spans: list[float] = field(init=False)
    # endregion

    @property
    def has_zero_point(self) -> bool:
        """
        判断当前量化配置是否包含零点偏移。
        """
        return self.tensor_zero_domain is not None

    def __post_init__(self):
        """
        在数据类初始化后执行，完成属性的进一步设置和推断。
        """
        # 量化数据类型检查，确保数据类型为 torch.dtype
        if isinstance(self.tensor_quant_dtype, torch.dtype):
            raise NotImplementedError("torch.dtype is not supported yet.")
        # 构建量化范围对象
        self.tensor_quant_range = QuantRange.construct(
            self.tensor_quant_dtype, has_zero_point=self.has_zero_point, quant_range=self.tensor_quant_range
        )
        # 推断缩放数据类型
        self.scale_quant_dtypes = ScaleUtils.infer_scale_dtypes(self.scale_quant_dtypes, self.default_quant_dtype)
        # 推断指数缩放级别
        self.exponent_scale_level = ScaleUtils.infer_exponent_scale_level(self.scale_quant_dtypes)
        # 处理零点量化数据类型和量化跨度
        if self.has_zero_point:
            if self.tensor_zero_domain == ZeroPointDomain.PreScale:
                self.zero_quant_dtype = self.tensor_quant_dtype
            elif self.tensor_zero_domain == ZeroPointDomain.PostScale:
                # TODO: fix zero quant dtype (signed or unsigned)
                self.zero_quant_dtype = self.scale_quant_dtypes[-1]
                if isinstance(self.zero_quant_dtype, QuantDataType) and self.zero_quant_dtype.is_exponent:
                    self.zero_quant_dtype = self.default_quant_dtype
            else:
                raise ValueError(f"Unsupported zero point domain: {self.tensor_zero_domain}")
            self.linear_tensor_quant_span = self.tensor_quant_range.max - self.tensor_quant_range.min   # 线性缩放的量化跨度
            self.exponent_tensor_quant_span = 2 ** int(                                                 # 指数缩放的量化跨度
                math.log2(self.tensor_quant_range.max) + int(self.tensor_quant_dtype.signed)
            )
        else:
            self.zero_quant_dtype = None
            self.linear_tensor_quant_span = self.tensor_quant_range.max
            self.exponent_tensor_quant_span = 2 ** int(math.log2(self.tensor_quant_range.max))
            
        # 分离线性和指数缩放的配置
        if self.exponent_scale_level >= 0 and self.exponent_scale_level < len(self.scale_quant_dtypes):
            # 量化数据类型分离
            lin_s_dtypes = self.scale_quant_dtypes[: self.exponent_scale_level]
            exp_s_dtypes = self.scale_quant_dtypes[self.exponent_scale_level :]
            # 视图形状分离
            lin_s_view_shapes = self.scale_view_shapes[: self.exponent_scale_level]
            exp_s_view_shapes = self.scale_view_shapes[self.exponent_scale_level :]
            # 量化跨度分离
            exp_s_spans = ScaleUtils.infer_scale_quant_spans(exp_s_dtypes)
            lin_s_spans = ScaleUtils.infer_scale_quant_spans(lin_s_dtypes, base=exp_s_spans[-1]) if lin_s_dtypes else []
        else:
            lin_s_dtypes, exp_s_dtypes = self.scale_quant_dtypes, []
            lin_s_view_shapes, exp_s_view_shapes = self.scale_view_shapes, []
            lin_s_spans, exp_s_spans = ScaleUtils.infer_scale_quant_spans(lin_s_dtypes), []
            
        # 设置线性和指数缩放属性
        self.linear_scale_quant_dtypes = lin_s_dtypes
        self.linear_scale_view_shapes = lin_s_view_shapes
        self.linear_scale_quant_spans = lin_s_spans
        self.exponent_scale_quant_dtypes = exp_s_dtypes
        self.exponent_scale_view_shapes = exp_s_view_shapes
        self.exponent_scale_quant_spans = exp_s_spans

    def quantize(
        self,
        *,
        # scale-based quantization related arguments
        scale: torch.Tensor | None = None,
        zero: torch.Tensor | None = None,
        # range-based quantization related arguments
        tensor: torch.Tensor | None = None,
        dynamic_range: DynamicRange | None = None,
    ) -> tuple[QuantScale, torch.Tensor]:
        """
        根据提供的参数获取待量化张量的缩放因子和零点。
        Get the quantization scale and zero point of the tensor to be quantized.

        Args:
            scale (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                The scale tensor.
            zero (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                The zero point tensor.
            tensor (`torch.Tensor` or `None`, *optional*, defaults to `None`):
                Ten tensor to be quantized. This is only used for range-based quantization.
            dynamic_range (`DynamicRange` or `None`, *optional*, defaults to `None`):
                The dynamic range of the tensor to be quantized.

        Returns:
            `tuple[QuantScale, torch.Tensor]`:
                The scale and the zero point.
        """
        # region step 1: get the dynamic span for range-based scale or the scale tensor
        # 获取动态跨度或缩放张量
        if scale is None:
            # 采用基于范围的量化
            range_based = True
            assert isinstance(tensor, torch.Tensor), "View tensor must be a tensor."
            dynamic_range = dynamic_range or DynamicRange()     # 初始化动态范围对象
            dynamic_range = dynamic_range.measure(              # 测量张量的动态范围
                tensor.view(self.tensor_view_shape),
                zero_domain=self.tensor_zero_domain,
                is_float_point=self.tensor_quant_dtype.is_float_point,
            )
            dynamic_range = dynamic_range.intersect(self.tensor_range_bound)    # 将动态范围与范围边界取交集，确保量化范围合理
            dynamic_span = (dynamic_range.max - dynamic_range.min) if self.has_zero_point else dynamic_range.max    # 计算动态跨度
        else:
            # 采用基于缩放的量化
            range_based = False
            scale = scale.view(self.scale_view_shapes[-1])      # 重塑缩放张量为最后一个缩放阶段的视图形状
            assert isinstance(scale, torch.Tensor), "Scale must be a tensor."
        # endregion
        
        # region step 2: get the scale
        # 获取缩放因子
        # 线性缩放处理
        if self.linear_scale_quant_dtypes:
            if range_based:
                # 基于范围的量化
                linear_scale = dynamic_span / self.linear_tensor_quant_span
            elif self.exponent_scale_quant_dtypes:
                # 基于缩放且存在指数缩放
                linear_scale = scale.mul(self.exponent_tensor_quant_span).div(self.linear_tensor_quant_span)
            else:
                # 基于缩放且不存在指数缩放
                linear_scale = scale
            lin_s = quantize_scale(             # 对线性缩放因子进行量化处理
                linear_scale,
                quant_dtypes=self.linear_scale_quant_dtypes,
                quant_spans=self.linear_scale_quant_spans,
                view_shapes=self.linear_scale_view_shapes,
            )
            assert lin_s.data is not None, "Linear scale tensor is None."
            assert not lin_s.data.isnan().any(), "Linear scale tensor contains NaN."
            assert not lin_s.data.isinf().any(), "Linear scale tensor contains Inf."
        else:
            lin_s = QuantScale()
            
        # 指数缩放处理
        if self.exponent_scale_quant_dtypes:
            if range_based:
                # 基于范围的量化
                exp_scale = dynamic_span / self.exponent_tensor_quant_span
            else:
                # 基于缩放的量化
                exp_scale = scale
            if lin_s.data is not None:
                lin_s.data = lin_s.data.expand(self.linear_scale_view_shapes[-1]).reshape(self.scale_view_shapes[-1])   # 扩展并重塑线性缩放张量，与缩放视图形状对齐
                exp_scale = exp_scale / lin_s.data          # 更新指数缩放因子
            exp_s = quantize_scale(                         # 对指数缩放因子进行量化处理
                exp_scale,
                quant_dtypes=self.exponent_scale_quant_dtypes,
                quant_spans=self.exponent_scale_quant_spans,
                view_shapes=self.exponent_scale_view_shapes,
            )
            assert exp_s.data is not None, "Exponential scale tensor is None."
            assert not exp_s.data.isnan().any(), "Exponential scale tensor contains NaN."
            assert not exp_s.data.isinf().any(), "Exponential scale tensor contains Inf."
            s = exp_s if lin_s.data is None else lin_s.extend(exp_s)    # 合并线性和指数缩放因子
        else:
            s = lin_s
        assert s.data is not None, "Scale tensor is None."
        assert not s.data.isnan().any(), "Scale tensor contains NaN."
        assert not s.data.isinf().any(), "Scale tensor contains Inf."
        # endregion
        
        # region step 3: get the zero point
        # 获取零点
        # 处理零点偏移
        if self.has_zero_point:
            if range_based:
                # 基于范围的量化
                if self.tensor_zero_domain == ZeroPointDomain.PreScale:
                    zero = self.tensor_quant_range.min - dynamic_range.min / s.data
                else:
                    zero = self.tensor_quant_range.min * s.data - dynamic_range.min
            assert isinstance(zero, torch.Tensor), "Zero point must be a tensor."
            z = simple_quantize(zero, has_zero_point=True, quant_dtype=self.zero_quant_dtype)   # 对零点进行量化处理
        else:
            z = torch.tensor(0, dtype=s.data.dtype, device=s.data.device)
        assert not z.isnan().any(), "Zero point tensor contains NaN."
        assert not z.isinf().any(), "Zero point tensor contains Inf."
        # endregion
        return s, z
