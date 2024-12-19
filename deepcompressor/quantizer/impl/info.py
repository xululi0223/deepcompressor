# -*- coding: utf-8 -*-
"""Quantization information class."""

from dataclasses import dataclass, field

import torch

from ...data.dtype import QuantDataType
from ...data.range import ProtectiveQuantRange, QuantRange, RangeBound
from ...data.utils import ShapeUtils
from ...data.zero import ZeroPointDomain
from ..config.base import DecomposedQuantizerConfig, QuantizerConfig
from .scale import QuantScaleInfo

__all__ = ["QuantScaleInfo", "QuantStepInfo", "QuantInfo"]


@dataclass
class QuantStepInfo:
    """
    旨在存储和管理单个量化步骤的配置信息和相关参数。
    它包括量化数据类型、零点域、分组形状、缩放数据类型、量化范围、范围边界等信息。
    此外，该类还包含张量的形状信息以及视图形状，用于在量化过程中进行张量的重塑和分组。
    """
    # region config
    # 配置区域属性定义
    # 当前步骤使用的数据类型
    quant_dtype: QuantDataType
    # 零点域，决定了零点偏移的应用方式
    zero_domain: ZeroPointDomain | None
    # 张量分组的形状。支持分组量化，定义每个分组的具体形状
    group_shapes: tuple[tuple[int, ...], ...]
    # 各个缩放阶段使用的数据类型。支持多阶段缩放，以适应不同的量化需求
    scale_dtypes: tuple[torch.dtype | QuantDataType | None, ...]
    # 量化的范围，限制量化后张量的值域
    quant_range: QuantRange | None
    # 范围边界，用于进一步限制动态范围，防止量化超出预期范围
    range_bound: RangeBound | None
    # 默认的数据类型，用于推断未明确指定的数据类型
    default_dtype: torch.dtype
    # endregion
    
    
    # region information
    # 信息区域属性定义
    # 原始张量的形状，用于量化前的基础形状信息
    tensor_shape: torch.Size
    """the shape is a torch.Size (s0, s1, ...)"""
    # 每个分组的而形状，支持分组量化
    tensor_group_shapes: list[torch.Size]
    """each group shape is a torch.Size (gs0, gs1, ...)"""
    # 张量在量化过程中重塑后的视图形状，通常用于并行计算或分组处理
    tensor_view_shape: torch.Size
    """the view shape is a torch.Size (#g0, gs0, #g1, gs1, ...)"""
    # endregion
    
    # 存储量化缩放相关的信息
    scale: QuantScaleInfo = field(init=False)

    def __post_init__(self):
        """
        在数据类实例化后，进一步初始化 scale 属性。
        """
        # 创建量化缩放信息对象
        self.scale = QuantScaleInfo(
            tensor_view_shape=self.tensor_view_shape,
            tensor_quant_dtype=self.quant_dtype,
            tensor_zero_domain=self.zero_domain,
            tensor_quant_range=self.quant_range,
            tensor_range_bound=self.range_bound,
            scale_view_shapes=ShapeUtils.infer_scale_view_shapes(self.tensor_group_shapes, shape=self.tensor_shape),
            scale_quant_dtypes=self.scale_dtypes,
            default_quant_dtype=self.default_dtype,
        )

    @property
    def tensor_zero_domain(self) -> ZeroPointDomain | None:
        """
        获取scale对象中的tensor_zero_domain属性。
        """
        return self.scale.tensor_zero_domain

    @property
    def tensor_quant_range(self) -> QuantRange:
        """
        获取scale对象中的tensor_quant_range属性。
        The intersection of the quant_range and quant_dtype.
        """
        return self.scale.tensor_quant_range

    @property
    def tensor_range_bound(self) -> RangeBound | None:
        """
        获取scale对象中的tensor_range_bound属性。
        """
        return self.scale.tensor_range_bound

    def to_config(self) -> QuantizerConfig:
        """
        将当前 QuantStepInfo 实例转换为 QuantizerConfig 配置对象。
        """
        return QuantizerConfig(
            dtype=self.quant_dtype,
            zero_point=self.zero_domain,
            group_shapes=self.tensor_group_shapes,
            scale_dtypes=self.scale.scale_quant_dtypes,
        )

    @staticmethod
    def construct(
        config: QuantizerConfig,
        tensor_shape: torch.Size,
        default_dtype: torch.dtype,
        quant_range: QuantRange | None = None,
        range_bound: RangeBound | None = None,
    ) -> "QuantStepInfo":
        """
        根据给定的配置和张量形状信息构建一个新的 QuantStepInfo 实例。
        """
        # 根据配置信息推断分组形状和视图形状
        tensor_group_shapes = ShapeUtils.infer_group_shapes(config.group_shapes, shape=tensor_shape)
        tensor_view_shape = ShapeUtils.infer_view_shape(tensor_shape, group_shape=tensor_group_shapes[-1])
        # 创建新的 QuantStepInfo 实例
        return QuantStepInfo(
            quant_dtype=config.dtype,
            zero_domain=config.zero_point,
            group_shapes=config.group_shapes,
            scale_dtypes=config.scale_dtypes,
            quant_range=quant_range,
            range_bound=range_bound,
            default_dtype=default_dtype,
            tensor_shape=tensor_shape,
            tensor_group_shapes=tensor_group_shapes,
            tensor_view_shape=tensor_view_shape,
        )


@dataclass
class QuantInfo:
    """
    用于存储和管理整个量化过程的配置信息。
    它包含多个量化步骤 (QuantStepInfo)，并提供方法来检查量化信息是否过时、获取特定步骤的信息以及根据配置构建量化信息。
    """
    
    # 存储多个QuantStepInfo对象实例，表示量化过程中的各个量化步骤
    steps: tuple[QuantStepInfo, ...]
    # 标识是否需要在反量化时进行饱和操作
    needs_dequant_saturation: bool = False

    @property
    def num_steps(self) -> int:
        """
        返回当前量化信息中的步骤数量。
        """
        return len(self.steps)

    def get_child(self, idx: int) -> QuantStepInfo:
        """
        获取指定索引的量化步骤信息。
        
        Args:
            idx: 量化步骤的索引
        """
        return self.steps[idx]

    def is_outdated(
        self,
        config: DecomposedQuantizerConfig,
        tensor_shape: torch.Size,
        default_dtype: torch.dtype,
        quant_range: QuantRange | None = None,
        range_bound: RangeBound | None = None,
    ) -> bool:
        """
        检查当前的量化信息是否已经果实，需要重新生成。
        Check if the current quantization information is outdated.
        
        Args:
            config: 分解后的量化配置对象，包含多个量化步骤的配置信息
            tensor_shape: 当前张量的形状
            default_dtype: 默认的数据类型
            quant_range: 当前的量化范围
            range_bound: 当前的范围边界
        """
        # 步骤数量检查
        if self.num_steps != config.num_steps:
            return True

        # 逐步检查每个量化步骤的配置信息，检查量化数据类型、分组形状、缩放数据类型是否一致
        for step_info, step_config in zip(self.steps, config.steps, strict=True):
            if step_info.quant_dtype != step_config.quant_dtype:
                return True
            if step_info.group_shapes != step_config.group_shapes:
                return True
            if step_info.scale_dtypes != step_config.scale_dtypes:
                return True
        
        # 张量属性检查
        if self.num_steps > 0:
            # 检查第一个步骤的张量形状、默认数据类型、范围边界是否一致
            first_step = self.steps[0]
            if first_step.tensor_shape != tensor_shape:
                return True
            if first_step.default_dtype != default_dtype:
                return True
            if first_step.range_bound != range_bound:
                return True

            # 检查最后一个步骤的量化范围是否一致
            if self.steps[-1].quant_range != quant_range:
                return True

            # 检查是否需要在反量化时进行饱和操作
            if self.num_steps > 1 and self.needs_dequant_saturation != config.needs_dequant_saturation:
                return True
        return False

    @staticmethod
    def construct(
        config: DecomposedQuantizerConfig,
        tensor_shape: torch.Size,
        default_dtype: torch.dtype,
        quant_range: QuantRange | None = None,
        range_bound: RangeBound | None = None,
    ) -> "QuantInfo":
        """
        根据给定的量化配置和张量形状信息构建一个新的 QuantInfo 实例，包含多个量化步骤的信息。
        
        Args:
            config: 分解后的量化配置对象，包含多个量化步骤的配置信息
            tensor_shape: 当前张量的形状
            default_dtype: 默认的数据类型
            quant_range: 当前的量化范围
            range_bound: 当前的范围边界
        """
        # 初始化量化步骤列表
        steps: list[QuantStepInfo] = []

        # 初始化步骤数量、默认数据类型、范围边界
        num_steps = config.num_steps
        step_default_dtype = default_dtype
        step_range_bound = range_bound
        
        # 遍历每个量化步骤，构建对应的量化步骤信息
        for step, step_config in enumerate(config.steps):
            assert step_config.quant_dtype is not None, f"quant_dtype is required for step {step}"
            # 处理最后一个步骤的量化范围
            if step == num_steps - 1:
                step_quant_range = quant_range
            # 处理除最后两个步骤外的量化范围
            elif step < num_steps - 2 or config.needs_dequant_saturation:
                step_quant_range = None
            # 仅倒数第二步的量化步骤可以不需要饱和操作
            else:  # ! only second last step quantization can be protected without saturation in the computation
                # 构建保护性量化范围
                step_quant_range = ProtectiveQuantRange.construct(
                    outer_dtype=step_config.quant_dtype,
                    inner_dtype=config.steps[-1].quant_dtype,
                    zero_domain=config.steps[-1].zero_domain,
                    inner_quant_range=quant_range,
                )
            # 构建量化步骤信息，并添加到步骤列表中
            steps.append(
                QuantStepInfo.construct(
                    step_config,
                    tensor_shape=tensor_shape,
                    default_dtype=step_default_dtype,
                    quant_range=step_quant_range,
                    range_bound=step_range_bound,
                )
            )
            # 更新默认数据类型和范围边界
            step_default_dtype = step_config.quant_dtype
            step_range_bound = steps[-1].scale.tensor_quant_range
        # 返回新的量化信息实例
        return QuantInfo(steps=tuple(steps), needs_dequant_saturation=config.needs_dequant_saturation)
