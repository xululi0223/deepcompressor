# -*- coding: utf-8 -*-
"""Quantization SVD calibration configuration."""

from dataclasses import dataclass, field

from omniconfig import configclass

from ...quantizer.config import QuantLowRankConfig
from ...utils.common import num2str
from ...utils.config import SkipBasedConfig
from .search import SearchBasedCalibConfig, SearchBasedCalibGranularity, SearchBasedCalibStrategy

__all__ = ["QuantLowRankCalibConfig", "SkipBasedQuantLowRankCalibConfig"]


@configclass
@dataclass
class QuantLowRankCalibConfig(SearchBasedCalibConfig, QuantLowRankConfig):
    """
    继承自 SearchBasedCalibConfig 和 QuantLowRankConfig，用于配置量化过程中低秩分支的校准参数。
    该类结合了搜索策略配置和低秩分支特有的配置选项，通过数据类和配置类装饰器简化了配置的定义，并提供属性验证和后处理逻辑，确保配置的有效性和一致性。
    Configuration for quantization low-rank branch calibration.

    Args:
        rank (`int`, *optional*, defaults to `32`):
            The rank of the low-rank branch.
        exclusive (`bool`, *optional*, defaults to `False`):
            Whether to use exclusive low-rank branch for each weight sharing the inputs.
        compensate (`bool`, *optional*, defaults to `False`):
            Whether the low-rank branch compensates the quantization error.
        degree (`int`, *optional*, default=`2`):
            The power degree for the quantization error. Defaults to `2`.
        objective (`SearchBasedCalibObjective`, *optional*, default=`SearchBasedCalibObjective.OutputsError`):
            The objective for quantization calibration.
        sample_batch_size (`int`, *optional*, default=`-1`):
            The samples batch size for calibration.
        sample_size (`int`, *optional*, default=`-1`):
            The calibration sample size.
        outputs_device (`str`, *optional*, default=`"cpu"`):
            The device to store the precomputed outputs of the module.
        num_iters (`int`, *optional*, default=`1`):
            The number of iterations.
        early_stop (`bool`, *optional*, default=`False`):
            Whether to stop the calibration early.
    """

    granularity: SearchBasedCalibGranularity = field(init=False, default=SearchBasedCalibGranularity.Layer) # 量化校准的粒度级别
    element_batch_size: int = field(init=False, default=-1)             # 校准时元素的批次大小
    element_size: int = field(init=False, default=-1)                   # 校准元素的大小
    pre_reshape: bool = field(init=False, default=True)                 # 是否在校准前对张量进行重塑
    num_iters: int = 1                                                  # 迭代次数
    early_stop: bool = False                                            # 是否提前停止校准

    def __post_init__(self):
        """
        在数据类实例化后执行，进行额外的属性验证和调整，确保配置参数的有效性和一致性。
        """
        # 策略设置
        if self.strategy != SearchBasedCalibStrategy.Manual:
            self.strategy = SearchBasedCalibStrategy.GridSearch
        
        # 独占性设置
        if self.compensate and self.num_iters <= 1:
            self.exclusive = True
        super().__post_init__()

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """
        根据当前配置生成唯一的目录名称，用于组织和存储校准结果或缓存数据。
        Generate the directory names of the configuration.

        Returns:
            list[str]: The directory names.
        """
        # 调用基类方法获取积累生成的目录名称列表
        names = super().generate_dirnames(**kwargs)
        
        # 构建额外的名称
        name = f"i{num2str(self.num_iters)}.r{num2str(self.rank)}"

        # 添加独占性、补偿性和早停标识
        if self.exclusive:
            name += ".exclusive"
        if self.compensate:
            name += ".compensate"
        if self.early_stop and self.num_iters > 1:
            name += ".earlystop"
        names.append(name)
        
        # 添加前缀
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names


@configclass
@dataclass
class SkipBasedQuantLowRankCalibConfig(SkipBasedConfig, QuantLowRankCalibConfig):
    """
    通过多重继承结合了 SkipBasedConfig 和 QuantLowRankCalibConfig 的功能，用于配置量化低秩分支的校准过程，同时支持跳过特定模块的设置。
    该类允许用户在进行低秩分支校准时，灵活排除不需要校准的模块，以优化校准过程的效率和效果。
    Configuration for Quantization Low-Rank Branch calibration.

    Args:
        rank (`int`, *optional*, defaults to `32`):
            The rank of the low-rank branch.
        exclusive (`bool`, *optional*, defaults to `False`):
            Whether to use exclusive low-rank branch for each weight sharing the inputs.
        compensate (`bool`, *optional*, defaults to `False`):
            Whether the low-rank branch compensates the quantization error.
        degree (`int`, *optional*, default=`2`):
            The power degree for the quantization error. Defaults to `2`.
        objective (`SearchBasedCalibObjective`, *optional*, default=`SearchBasedCalibObjective.OutputsError`):
            The objective for quantization calibration.
        sample_batch_size (`int`, *optional*, default=`-1`):
            The samples batch size for calibration.
        sample_size (`int`, *optional*, default=`-1`):
            The calibration sample size.
        outputs_device (`str`, *optional*, default=`"cpu"`):
            The device to store the precomputed outputs of the module.
        num_iters (`int`, *optional*, default=`1`):
            The number of iterations.
        early_stop (`bool`, *optional*, default=`False`):
            Whether to stop the calibration early.
        skips (`list[str]`, *optional*, default=`[]`):
            The keys of the modules to skip.
    """

    pass
