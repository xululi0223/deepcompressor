# -*- coding: utf-8 -*-
"""Quantizer."""

import typing as tp
from dataclasses import _MISSING_TYPE, MISSING, dataclass

import torch

from ..data.range import DynamicRange, QuantRange, RangeBound
from ..data.tensor import QuantTensor
from ..nn.patch.lowrank import LowRankBranch
from ..utils.common import tree_map
from ..utils.config import KeyEnableConfig
from ..utils.hooks import BaseInputPackager, BaseOutputPackager, BaseTensorProcessor
from .config.kernel import BaseKeyEnableQuantKernelConfig, BaseQuantKernel, BaseQuantKernelConfig
from .config.lowrank import QuantLowRankConfig
from .impl.base import QuantizerImpl
from .impl.info import QuantInfo

__all__ = ["Quantizer"]


@dataclass
class Quantizer(QuantizerImpl, BaseTensorProcessor):
    """
    深度学习模型中用于张量量化的核心模块。
    它继承自 QuantizerImpl 和 BaseTensorProcessor，结合了量化的实现细节和张量处理的功能。
    该类支持各种量化配置，包括动态范围、量化范围、低秩分支等，通过不同的配置和方法，实现对张量的高效量化与反量化操作。
    Quantizer class.

    Args:
        config (`BasicQuantizerConfig` or `None`):
            The quantizer configuration.
        key (`str`, *optional*, defaults to `""`):
            The key of the quantizer.
        kernel (`BaseKeyEnableQuantKernelConfig` or `BaseQuantKernelConfig` or `BaseQuantKernel` or `None`,
                *optional*, defaults to `None`):
            The quantizer kernel configuration.
        channels_dim (`int` or `None`, *optional*, defaults to `None`):
            The dimension of channels.
        scale (`torch.Tensor` or `Sequence[torch.Tensor]` or `None`, *optional*, defaults to `None`):
            The scale tensor.
        zero (`torch.Tensor` or `None`, *optional*, defaults to `None`):
            The zero point tensor.
        dynamic_range (`DynamicRange` or `Sequence[DynamicRange]` or `None`, *optional*, defaults to `None`):
            The dynamic range.
        range_bound (`RangeBound` or `None`, *optional*, defaults to `None`):
            The dynamic range bound.
        quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
            The quantization range.
        default_dtype (`torch.dtype` or `None`, *optional*, defaults to `None`):
            The default scale dtype
        develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The quantization development dtype.

        low_rank (`QuantLowRankConfig` or `None`, *optional*, defaults to `None`):
            The quantization low-rank branch configuration.
        input_packager (`BaseInputPackager` or `None`, *optional*, defaults to `None`):
            The input packager, used for unpacking and repacking the input tensor(s).
        output_packager (`BaseOutputPackager` or `None`, *optional*, defaults to `None`):
            The output packager, used for unpacking and repacking the output tensor(s).
    """

    # region keyword arguments' defaults
    # 关键词参数默认值区域
    # 量化内核的配置或实例，支持多种配置类型
    kernel: BaseKeyEnableQuantKernelConfig | BaseQuantKernelConfig | BaseQuantKernel | None = None
    # 指定张量的通道维度
    channels_dim: int | None = None
    # 缩放因子张量或其序列，用于量化过程
    scale: torch.Tensor | tp.Sequence[torch.Tensor] | None = None
    # 零点张量，用于量化过程
    zero: torch.Tensor | None = None
    # 动态范围对象或其序列，控制量化过程中动态范围的调整
    dynamic_range: DynamicRange | tp.Sequence[DynamicRange] | None = None
    # 范围边界对象，限制动态范围
    range_bound: RangeBound | None = None
    # 量化范围对象，定义量化后的张量值范围
    quant_range: QuantRange | None = None
    # 默认的缩放因子数据类型
    default_dtype: torch.dtype | None = None
    # 量化开发时使用的数据类型
    develop_dtype: torch.dtype = torch.float32
    # endregion

    # region hook-related attributes
    # 钩子相关属性区域
    # 低秩分支的量化配置
    low_rank: QuantLowRankConfig | None = None
    # 输入数据的包装器，用于处理输入张量
    input_packager: BaseInputPackager | None = None
    # 输出数据的包装器，用于处理输出张量
    output_packager: BaseOutputPackager | None = None
    # endregion

    def is_enabled_low_rank(self) -> bool:
        """
        判断低秩分支是否被启用。
        """
        # 如果低秩分支配置未启用，则返回 False
        if self.low_rank is None:
            return False
        # 如果低秩分支配置是KeyEnableConfig类型，则根据key判断是否启用
        if isinstance(self.low_rank, KeyEnableConfig):
            return self.low_rank.is_enabled_for(self.key)
        # 否则，根据分支配置判断是否启用
        return self.low_rank.is_enabled()

    def get_input_packager(self) -> BaseInputPackager | None:
        """
        获取输入数据的包装器。
        """
        return self.input_packager

    def get_output_packager(self) -> BaseOutputPackager | None:
        """
        获取输出数据的包装器。
        """
        return self.output_packager

    def process(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        对输入的张量进行量化处理，只返回量化后的数据部分。
        """
        return self.quantize(tensor).data

    def quantize(
        self,
        tensor: torch.Tensor,
        /,
        *,
        return_with_dequant: bool = True,
        return_with_quant: bool = False,
        kernel: (
            BaseKeyEnableQuantKernelConfig | BaseQuantKernelConfig | BaseQuantKernel | None | _MISSING_TYPE
        ) = MISSING,
        channels_dim: int | None | _MISSING_TYPE = MISSING,
        # scale-based quantization arguments
        scale: torch.Tensor | tp.Sequence[torch.Tensor] | None | _MISSING_TYPE = MISSING,
        zero: torch.Tensor | None | _MISSING_TYPE = MISSING,
        # range-based quantization arguments
        dynamic_range: DynamicRange | tp.Sequence[DynamicRange] | None | _MISSING_TYPE = MISSING,
        range_bound: RangeBound | None | _MISSING_TYPE = MISSING,
        # other arguments
        quant_range: QuantRange | None | _MISSING_TYPE = MISSING,
        default_dtype: torch.dtype | None | _MISSING_TYPE = MISSING,
        develop_dtype: torch.dtype | _MISSING_TYPE = MISSING,
        **kwargs,
    ) -> QuantTensor:
        """
        对单个或多个张量进行量化操作，根据配置和传入参数，返回量化结果和相关信息。
        Quantize a tensor.

        Args:
            tensor (`torch.Tensor`):
                The tensor to quantize.
            return_with_dequant (`bool`, *optional*, defaults to `True`):
                Whether to return the dequantized tensor.
            return_with_quant (`bool`, *optional*, defaults to `False`):
                Whether to return the quantized tensor.
            kernel (`BaseKeyEnableQuantKernelConfig` or `BaseQuantKernelConfig` or `BaseQuantKernel`
                    or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The quantization kernel configuration.
            channels_dim (`int` or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The dimension of channels.
            scale (`torch.Tensor` or `Sequence[torch.Tensor]` or `None` or `_MISSING_TYPE`,
                   *optional*, defaults to `MISSING`):
                The scale tensor.
            zero (`torch.Tensor` or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The zero point tensor.
            dynamic_range (`DynamicRange` or `Sequence[DynamicRange]` or `None` or `_MISSING_TYPE`,
                           *optional*, defaults to `MISSING`):
                The dynamic range.
            range_bound (`RangeBound` or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The dynamic range bound.
            quant_range (`QuantRange` or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The quantization range.
            default_dtype (`torch.dtype` or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The default scale dtype.
            develop_dtype (`torch.dtype` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The quantization development dtype.
            **kwargs:
                Other keyword arguments for the quantization kernel. For example,
                ``inputs`` for the input tensors in GPTQ kernel,
                ``round_delta`` for the rounding delta in the RTN kernel.

        Returns:
            QuantTensor: The quantized tensor.
        """
        # 参数处理
        channels_dim = self.channels_dim if channels_dim is MISSING else channels_dim
        scale = self.scale if scale is MISSING else scale
        zero = self.zero if zero is MISSING else zero
        dynamic_range = self.dynamic_range if dynamic_range is MISSING else dynamic_range
        range_bound = self.range_bound if range_bound is MISSING else range_bound
        quant_range = self.quant_range if quant_range is MISSING else quant_range
        default_dtype = self.default_dtype if default_dtype is MISSING else default_dtype
        develop_dtype = self.develop_dtype if develop_dtype is MISSING else develop_dtype
        # 内核配置处理
        if kernel is MISSING:
            kernel = self.kernel
        if isinstance(kernel, BaseKeyEnableQuantKernelConfig):
            kernel = kernel.specialize_for(self.key)
        elif isinstance(kernel, KeyEnableConfig):
            kernel = kernel if kernel.is_enabled_for(self.key) else None
        assert isinstance(kernel, (BaseQuantKernel, BaseQuantKernelConfig, type(None)))
        # 量化操作
        return super().quantize(
            tensor,
            kernel=kernel,
            channels_dim=channels_dim,
            scale=scale,
            zero=zero,
            dynamic_range=dynamic_range,
            range_bound=range_bound,
            quant_range=quant_range,
            return_with_dequant=return_with_dequant,
            return_with_quant=return_with_quant,
            default_dtype=default_dtype,
            develop_dtype=develop_dtype,
            **kwargs,
        )

    def update(
        self,
        tensor_shape: torch.Size,
        default_dtype: torch.dtype | _MISSING_TYPE = MISSING,
        quant_range: QuantRange | None | _MISSING_TYPE = MISSING,
        range_bound: RangeBound | None | _MISSING_TYPE = MISSING,
    ) -> QuantInfo | None:
        """
        更新量化信息，根据当前张量的形状和配置参数，确保量化过程的准确性和一致性。
        Update the quantization information.

        Args:
            tensor_shape (`torch.Size`):
                The shape of the tensor.
            default_dtype (`torch.dtype` or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The default scale dtype.
            quant_range (`QuantRange` or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The quantization range.
            range_bound (`RangeBound` or `None` or `_MISSING_TYPE`, *optional*, defaults to `MISSING`):
                The dynamic range bound

        Returns:
            `QuantInfo` or `None`:
                The updated quantization. If the quantizer is disabled, return `None`.
        """
        # 调用基类的更新方法，更新量化信息
        return super().update(
            tensor_shape,
            default_dtype=self.default_dtype if default_dtype is MISSING else default_dtype,
            quant_range=self.quant_range if quant_range is MISSING else quant_range,
            range_bound=self.range_bound if range_bound is MISSING else range_bound,
        )

    def quantize_with_low_rank(
        self,
        tensors: torch.Tensor | tp.Sequence[torch.Tensor],
        /,
        *,
        return_with_dequant: bool = True,
        return_with_quant: bool = False,
        kernel: (
            BaseKeyEnableQuantKernelConfig | BaseQuantKernelConfig | BaseQuantKernel | None | _MISSING_TYPE
        ) = MISSING,
        channels_dim: int | None | _MISSING_TYPE = MISSING,
        # scale-based quantization arguments
        scale: torch.Tensor | tp.Sequence[torch.Tensor] | None | _MISSING_TYPE = MISSING,
        zero: torch.Tensor | None | _MISSING_TYPE = MISSING,
        # range-based quantization arguments
        dynamic_range: DynamicRange | tp.Sequence[DynamicRange] | None | _MISSING_TYPE = MISSING,
        range_bound: RangeBound | None | _MISSING_TYPE = MISSING,
        # other arguments
        quant_range: QuantRange | None | _MISSING_TYPE = MISSING,
        default_dtype: torch.dtype | None | _MISSING_TYPE = MISSING,
        develop_dtype: torch.dtype | _MISSING_TYPE = MISSING,
        **kwargs,
    ) -> tuple[list[QuantTensor], list[LowRankBranch] | None]:
        """
        对多个张量进行量化，并结合低秩分支优化量化过程，返回量化结果和相应的低秩分支对象。
        """
        # 张量处理，确保输入为张量序列
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
            
        # 量化参数汇总
        qkwargs = dict(
            return_with_dequant=return_with_dequant,
            return_with_quant=return_with_quant,
            kernel=kernel,
            channels_dim=channels_dim,
            scale=scale,
            zero=zero,
            dynamic_range=dynamic_range,
            range_bound=range_bound,
            quant_range=quant_range,
            default_dtype=default_dtype,
            develop_dtype=develop_dtype,
            **kwargs,
        )
        # 低秩分支处理
        if self.is_enabled_low_rank():
            qtensors: list[QuantTensor] = []                    # 用于存储量化结果
            branches: list[LowRankBranch] = []                  # 用于存储低秩分支对象
            
            if len(tensors) == 1 or self.low_rank.exclusive:
                # 单个张量或独占模式
                if self.low_rank.compensate:
                    # 补偿模式
                    qkwargs["return_with_dequant"] = True       # 返回反量化结果
                    for t in tensors:
                        qt = self.quantize(t.data, **qkwargs)   # 量化张量
                        lb = LowRankBranch(t.shape[1], t.shape[0], rank=self.low_rank.rank, weight=t.data - qt.data)    # 创建低秩分支对象
                        qtensors.append(qt)                     # 存储量化结果
                        branches.append(lb)                     # 存储低秩分支对象
                else:
                    # 非补偿模式
                    for t in tensors:
                        lb = LowRankBranch(t.shape[1], t.shape[0], rank=self.low_rank.rank, weight=t.data)              # 创建低秩分支对象
                        qt = self.quantize(t.data - lb.get_effective_weight(), **qkwargs)                               # 量化张量
                        qtensors.append(qt)                     # 存储量化结果
                        branches.append(lb)                     # 存储低秩分支对象
                return qtensors, branches
            else:
                # 多个张量且非独占模式
                st = torch.cat([t.data for t in tensors], dim=0)                        # 将多个张量拼接为一个张量
                if self.low_rank.compensate:
                    # 补偿模式
                    qkwargs["return_with_dequant"] = True                               # 返回反量化结果
                    for t in tensors:
                        qt = self.quantize(t.data, **qkwargs)                           # 量化张量
                        qtensors.append(qt)                                             # 存储量化结果
                    sl = LowRankBranch(                                                 # 创建低秩分支对象
                        st.shape[1],
                        st.shape[0],
                        rank=self.low_rank.rank,
                        weight=st - torch.cat([q.data for q in qtensors], dim=0),
                    )
                    del st
                    i = 0
                    # 遍历每个张量，创建对应的低秩分支对象
                    for t in tensors:
                        lb = LowRankBranch(t.shape[1], t.shape[0], rank=self.low_rank.rank) # 创建低秩分支对象
                        lb.a = sl.a                                                         # 共享低秩分支的a参数
                        lb.b.to(dtype=t.dtype, device=t.device)
                        lb.b.weight.copy_(sl.b.weight[i : i + t.shape[0]])                  # 拷贝低秩分支的b参数的对应部分
                        branches.append(lb)                                                 # 存储低秩分支对象
                        i += t.shape[0]
                    return qtensors, branches
                else:
                    # 非补偿模式
                    sl = LowRankBranch(st.shape[1], st.shape[0], rank=self.low_rank.rank, weight=st)    # 创建低秩分支对象
                    del st
                    i = 0
                    # 遍历每个张量，创建对应的低秩分支对象和量化结果
                    for t in tensors:
                        lb = LowRankBranch(t.shape[1], t.shape[0], rank=self.low_rank.rank) # 创建低秩分支对象
                        lb.a = sl.a                                                         # 共享低秩分支的a参数
                        lb.b.to(dtype=t.dtype, device=t.device)
                        lb.b.weight.copy_(sl.b.weight[i : i + t.shape[0]])                  # 拷贝低秩分支的b参数的对应部分
                        qt = self.quantize(t.data - lb.get_effective_weight(), **qkwargs)   # 量化张量
                        qtensors.append(qt)                                                 # 存储量化结果
                        branches.append(lb)                                                 # 存储低秩分支对象
                        i += t.shape[0]
                    return qtensors, branches
        else:
            # 未启用低秩分支，直接量化
            return [self.quantize(t.data, **qkwargs) for t in tensors], None

    def state_dict(self, device: torch.device | str = "cpu") -> dict[str, tp.Any]:
        """
        获取量化器的状态字典，将当前量化器的配置和参数保存为可序列化的字典，用于模型保存和加载。
        Get the state dictionary of the quantizer.

        Args:
            device (`torch.device` or `str`, *optional*, defaults to `"cpu"`):
                The device to store the state dictionary.

        Returns:
            `dict[str, Any]`:
                The state dictionary.
        """
        # 初始化空的状态字典
        state_dict = {}

        # 辅助函数，用于将张量移动到指定设备并克隆
        def _copy_to(x):
            return x.to(device).clone()

        # 保存关键属性
        state_dict["channels_dim"] = self.channels_dim                                  # 保存通道维度信息
        state_dict["scale"] = tree_map(_copy_to, self.scale)                            # 保存缩放因子张量
        state_dict["zero"] = _copy_to(self.zero) if self.zero is not None else None     # 保存零点张量
        # 保存动态范围信息
        if self.dynamic_range is None:
            state_dict["dynamic_range"] = None
        elif isinstance(self.dynamic_range, DynamicRange):
            state_dict["dynamic_range"] = tree_map(_copy_to, self.dynamic_range.to_dict())
        else:
            state_dict["dynamic_range"] = tree_map(_copy_to, tuple(d.to_dict() for d in self.dynamic_range))
        state_dict["range_bound"] = self.range_bound.to_dict() if self.range_bound is not None else None    # 保存范围边界信息
        state_dict["quant_range"] = self.quant_range.to_dict() if self.quant_range is not None else None    # 保存量化范围信息
        return state_dict

    def load_state_dict(self, state_dict: dict[str, tp.Any], device: torch.device | str = "cpu"):
        """
        加载之前保存的状态字典，恢复量化器的配置和参数，确保模型的可重复性和持续性。
        Load the state dictionary.

        Args:
            state_dict (`dict[str, Any]`):
                The state dictionary.
            device (`torch.device` or `str`, *optional*, defaults to `"cpu"`):
                The device to load the state dictionary.
        """
        # 辅助函数，用于将张量移动到指定设备
        def _move_to(x):
            return x.to(device)

        # 加载状态字典中的关键属性
        self.channels_dim = state_dict["channels_dim"]                                                      # 加载通道维度信息
        self.scale = tree_map(_move_to, state_dict["scale"])                                                # 加载缩放因子张量
        self.zero = _move_to(state_dict["zero"]) if state_dict["zero"] is not None else None                # 加载零点张量
        # 加载动态范围信息
        if state_dict["dynamic_range"] is None:
            self.dynamic_range = None
        elif isinstance(state_dict["dynamic_range"], dict):
            self.dynamic_range = DynamicRange.from_dict(tree_map(_move_to, state_dict["dynamic_range"]))
        else:
            self.dynamic_range = tuple(
                DynamicRange.from_dict(tree_map(_move_to, d)) for d in state_dict["dynamic_range"]
            )
        self.range_bound = RangeBound.from_dict(state_dict["range_bound"])                                  # 加载范围边界信息
        self.quant_range = QuantRange.from_dict(state_dict["quant_range"])                                  # 加载量化范围信息
