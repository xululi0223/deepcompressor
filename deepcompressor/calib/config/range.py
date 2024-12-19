# -*- coding: utf-8 -*-
"""Quantization dynamic range calibration configuration."""

from dataclasses import dataclass

from omniconfig import configclass

from ...utils.common import num2str
from ...utils.config import SkipBasedConfig
from .search import SearchBasedCalibConfig, SearchBasedCalibStrategy

__all__ = ["DynamicRangeCalibConfig", "SkipBasedDynamicRangeCalibConfig"]


@configclass
@dataclass
class DynamicRangeCalibConfig(SearchBasedCalibConfig):
    """
    继承自 SearchBasedCalibConfig，用于配置量化动态范围校准过程。
    通过添加特定于动态范围校准的参数，扩展了基类的功能，以支持线性范围搜索和动态缩放等特性。
    Configuration for quantization dynamic range calibration.

    Args:
        degree (`int`, *optional*, default=`2`):
            The power degree for the quantization error. Defaults to `2`.
        objective (`SearchBasedCalibObjective`, *optional*, default=`SearchBasedCalibObjective.OutputsError`):
            The objective for quantization calibration.
        strategy (`SearchBasedCalibStrategy`, *optional*, default=`SearchBasedCalibStrategy.Manual`):
            The strategy for quantization calibration.
        granularity (`SearchBasedCalibGranularity`, *optional*, default=`SearchBasedCalibGranularity.Layer`):
            The granularity for quantization calibration.
        element_batch_size (`int`, *optional*, default=`-1`):
            The element batch size for calibration.
        sample_batch_size (`int`, *optional*, default=`-1`):
            The samples batch size for calibration.
        element_size (`int`, *optional*, default=`-1`):
            The calibration element size.
        sample_size (`int`, *optional*, default=`-1`):
            The calibration sample size.
        pre_reshape (`bool`, *optional*, default=`True`):
            Whether to enable reshaping the tensor before calibration.
        outputs_device (`str`, *optional*, default=`"cpu"`):
            The device to store the precomputed outputs of the module.
        ratio (`float`, *optional*, default=`1.0`):
            The dynamic range ratio.
        max_shrink (`float`, *optional*, default=`0.2`):
            Maximum shrinkage ratio.
        max_expand (`float`, *optional*, default=`1.0`):
            Maximum expansion ratio.
        num_grids (`int`, *optional*, default=`80`):
            Number of grids for linear range search.
        allow_scale (`bool`, *optional*, default=`False`):
            Whether to allow range dynamic scaling.
    """

    ratio: float = 1.0              # 动态范围比例，用于手动策略时设置特定的调整比例
    max_shrink: float = 0.2         # 最大收缩比例，用于网格搜索时，限制动态范围的最小缩减程度
    max_expand: float = 1.0         # 最大扩张比例，用于网格搜索时，限制动态范围的最大扩展程度
    num_grids: int = 80             # 线性搜索的网格数量，决定搜索的精细程度
    allow_scale: bool = False       # 是否允许范围动态缩放，启用后将调整动态范围以适应特定需求

    def get_linear_ratios(self) -> list[float]:
        """
        生成用于线性范围搜索的动态范围比例列表。
        Get the ratios for linear range search.

        Returns:
            `list[float]`:
                The dynamic range ratio candidates for linear range search.
        """
        # 变量提取
        num_grids, max_shrink, max_expand = self.num_grids, self.max_shrink, self.max_expand
        # 收缩比例验证
        assert max_shrink < 1, "maximal shrinkage ratio must be less than 1"
        # 生成收缩比例
        ratios = [1 - grid / num_grids * (1 - max_shrink) for grid in range(1, num_grids + 1)]
        # 生成扩张比例
        if max_expand > 1:
            ratios += [1 + grid / num_grids * (max_expand - 1) for grid in range(1, num_grids + 1)]
        return ratios

    def get_ratios(self) -> list[list[float]]:
        """
        根据当前的搜索策略，返回对应的动态范围比例列表。
        Get the ratios for linear range search.

        Returns:
            `list[list[float]]`:
                The dynamic range ratio candidates for linear range search.
        """
        # 如果是手动策略，直接返回当前的动态范围比例
        if self.strategy == SearchBasedCalibStrategy.Manual:
            return [[self.ratio]]
        # 如果是网格搜索策略，返回线性搜索的动态范围比例
        elif self.strategy == SearchBasedCalibStrategy.GridSearch:
            return [[1.0], self.get_linear_ratios()]
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """
        根据当前配置生成唯一的目录名称，用于组织和存储校准结果或缓存数据。
        Generate the directory names of the configuration.

        Args:
            prefix (`str`, *optional*, default=`""`):
                The prefix of the directory.

        Returns:
            `list[str]`:
                The directory names.
        """
        # 调用基类方法获取积累生成的目录名称列表
        names = super().generate_dirnames(**kwargs)

        # 根据策略生成名称
        if self.strategy == SearchBasedCalibStrategy.Manual:
            name = f"r.[{num2str(self.ratio)}]"
        elif self.strategy == SearchBasedCalibStrategy.GridSearch:
            name = f"r.[{num2str(self.max_shrink)}.{num2str(self.max_expand)}].g{self.num_grids}"
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

        # 是否允许动态缩放
        if self.allow_scale:
            name += ".scale"
        names.append(name)

        # 添加前缀
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names


@configclass
@dataclass
class SkipBasedDynamicRangeCalibConfig(SkipBasedConfig, DynamicRangeCalibConfig):
    """
    继承自 SkipBasedConfig 和 DynamicRangeCalibConfig，用于配置在动态范围校准过程中需要跳过特定模块的设置。
    通过多重继承，该类结合了跳过配置和动态范围校准的功能。
    Configuration for quantization dynamic range calibration.

    Args:
        degree (`int`, *optional*, default=`2`):
            The power degree for the quantization error. Defaults to `2`.
        objective (`SearchBasedCalibObjective`, *optional*, default=`SearchBasedCalibObjective.OutputsError`):
            The objective for quantization calibration.
        strategy (`SearchBasedCalibStrategy`, *optional*, default=`SearchBasedCalibStrategy.Manual`):
            The strategy for quantization calibration.
        granularity (`SearchBasedCalibGranularity`, *optional*, default=`SearchBasedCalibGranularity.Layer`):
            The granularity for quantization calibration.
        element_batch_size (`int`, *optional*, default=`-1`):
            The element batch size for calibration.
        sample_batch_size (`int`, *optional*, default=`-1`):
            The samples batch size for calibration.
        element_size (`int`, *optional*, default=`-1`):
            The calibration element size.
        sample_size (`int`, *optional*, default=`-1`):
            The calibration sample size.
        pre_reshape (`bool`, *optional*, default=`True`):
            Whether to enable reshaping the tensor before calibration.
        outputs_device (`str`, *optional*, default=`"cpu"`):
            The device to store the precomputed outputs of the module.
        ratio (`float`, *optional*, default=`1.0`):
            The dynamic range ratio.
        max_shrink (`float`, *optional*, default=`0.2`):
            Maximum shrinkage ratio.
        max_expand (`float`, *optional*, default=`1.0`):
            Maximum expansion ratio.
        num_grids (`int`, *optional*, default=`80`):
            Number of grids for linear range search.
        allow_scale (`bool`, *optional*, default=`False`):
            Whether to allow range dynamic scaling.
        skips (`list[str]`, *optional*, default=`[]`):
            The keys of the modules to skip.
    """

    pass
