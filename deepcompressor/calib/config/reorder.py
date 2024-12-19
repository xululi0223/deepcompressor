# -*- coding: utf-8 -*-
"""Channel reorder configuration."""

import enum
from dataclasses import dataclass, field

from omniconfig import configclass

from ...utils.config import SkipBasedConfig
from .search import (
    SearchBasedCalibConfig,
    SearchBasedCalibGranularity,
    SearchBasedCalibObjective,
    SearchBasedCalibStrategy,
)

__all__ = ["ChannelOrderCalibConfig", "SkipBasedChannelOrderConfig"]


@configclass
@dataclass
class ChannelOrderCalibConfig(SearchBasedCalibConfig):
    """
    继承自 SearchBasedCalibConfig，用于配置组量化中的通道顺序校准。
    该类通过组合搜索策略配置和通道顺序校准特有的配置选项，利用数据类和配置类装饰器简化配置定义，并提供属性验证和后处理逻辑，确保配置的有效性和一致性。
    Configuration for channel order calibration in group quantization.

    Args:
        degree (`int`, *optional*, default=`2`):
            The power degree for the quantization error. Defaults to `2`.
        strategy (`SearchBasedCalibStrategy`, *optional*, default=`SearchBasedCalibStrategy.Manual`):
            The strategy for quantization calibration.
        sample_batch_size (`int`, *optional*, default=`-1`):
            The samples batch size for calibration.
        sample_size (`int`, *optional*, default=`-1`):
            The calibration sample size.
        allow_x_quant (`bool`, *optional*, default=`True`):
            Whether to allow input quantization during calibration.
        allow_w_quant (`bool`, *optional*, default=`True`):
            Whether to allow weight quantization during calibration.
        channel_metric (`ChannelMetricMode`, *optional*, default=`ChannelMetricMode.AbsNormalizedMean`):
            The mode for computing the channel importance.
        channel_index (`ChannelIndexMode`, *optional*, default=`ChannelIndexMode.Sequential`):
            The mode for ranking the channel importance.
        dynamic (`bool`, *optional*, default=`False`):
            Whether to enable dynamic channel reorder.
    """

    class ChannelMetric(enum.Enum):
        """
        枚举类，定义了通道重要性计算的模式。
        The mode for computing the channel importance.
        """

        InputsAbsMax = "xMax"               # 输入的绝对最大值
        InputsAbsMean = "xAvg"              # 输入的绝对均值
        InputsRootMeanSquare = "xRms"       # 输入的均方根
        WeightsAbsMax = "wMax"              # 权重的绝对最大值
        WeightsAbsMean = "wAvg"             # 权重的绝对均值
        WeightsRootMeanSquare = "wRms"      # 权重的均方根
        AbsMaxProduct = "pMax"              # 乘积的绝对最大值
        AbsMeanProduct = "pAvg"             # 乘积的绝对均值
        RootMeanSquareProduct = "pRms"      # 乘积的均方根

    class ChannelIndex(enum.Enum):
        """
        定义了排名通道重要性的模式。
        The mode for ranking the channel importance.
        """

        Sequential = "Seq"                  # 顺序排名
        Transpose = "Trp"                   # 转置排名

    objective: SearchBasedCalibObjective = field(init=False, default=SearchBasedCalibObjective.OutputsError)    # 量化校准的目标
    granularity: SearchBasedCalibGranularity = field(init=False, default=SearchBasedCalibGranularity.Layer)     # 量化校准的粒度级别
    element_batch_size: int = field(init=False, default=-1)                                                     # 校准时元素的批次大小
    element_size: int = field(init=False, default=-1)                                                           # 校准元素的大小
    pre_reshape: bool = field(init=False, default=True)                                                         # 是否在校准前对张量进行重塑
    allow_x_quant: bool = True                                                                                  # 是否允许校准时对输入进行量化
    allow_w_quant: bool = True                                                                                  # 是否允许校准时对权重进行量化
    channel_metric: ChannelMetric = ChannelMetric.InputsAbsMax                                                  # 通道重要性计算的模式
    channel_index: ChannelIndex = ChannelIndex.Sequential                                                       # 通道重要性排名的模式
    dynamic: bool = False                                                                                       # 是否启用动态通道重排序

    def __post_init__(self) -> None:
        """
        在数据类实例化后执行的后处理方法，用于验证和调整属性值，确保配置参数的有效性和一致性。
        """
        # 策略设置
        if self.strategy != SearchBasedCalibStrategy.Manual:
            self.strategy = SearchBasedCalibStrategy.GridSearch
        super().__post_init__()

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
            name = f"{self.channel_metric.name}.{self.channel_index.name}"
        else:
            name = "search"

        # 处理动态通道重排序
        if self.dynamic:
            name += ".dynamic"
        
        # 添加到名称列表
        names.append(name)

        # 处理不允许量化的情况
        disallows = []
        if not self.allow_x_quant:
            disallows.append("x")
        if not self.allow_w_quant:
            disallows.append("w")
        if disallows:
            names.append(f"disallow.[{'+'.join(disallows)}]")

        # 添加前缀
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names


@configclass
@dataclass
class SkipBasedChannelOrderConfig(SkipBasedConfig, ChannelOrderCalibConfig):
    """Configuration for channel order calibration in group quantization.

    Args:
        degree (`int`, *optional*, default=`2`):
            The power degree for the quantization error. Defaults to `2`.
        strategy (`SearchBasedCalibStrategy`, *optional*, default=`SearchBasedCalibStrategy.Manual`):
            The strategy for quantization calibration.
        sample_batch_size (`int`, *optional*, default=`-1`):
            The samples batch size for calibration.
        sample_size (`int`, *optional*, default=`-1`):
            The calibration sample size.
        allow_x_quant (`bool`, *optional*, default=`True`):
            Whether to allow input quantization during calibration.
        allow_w_quant (`bool`, *optional*, default=`True`):
            Whether to allow weight quantization during calibration.
        channel_metric (`ChannelMetricMode`, *optional*, default=`ChannelMetricMode.AbsNormalizedMean`):
            The mode for computing the channel importance.
        channel_index (`ChannelIndexMode`, *optional*, default=`ChannelIndexMode.Sequential`):
            The mode for ranking the channel importance.
        dynamic (`bool`, *optional*, default=`False`):
            Whether to enable dynamic channel reorder.
        skips (`list[str]`, *optional*, default=`[]`):
            The keys of the modules to skip.
    """

    pass
