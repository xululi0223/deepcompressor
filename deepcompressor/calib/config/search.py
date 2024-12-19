# -*- coding: utf-8 -*-
"""Quantization calibrator configurations."""

import enum
from dataclasses import dataclass

from omniconfig import configclass

from ...utils.common import num2str

__all__ = [
    "SearchBasedCalibStrategy",
    "SearchBasedCalibGranularity",
    "SearchBasedCalibObjective",
    "SearchBasedCalibConfig",
]


class SearchBasedCalibStrategy(enum.Enum):
    """
    枚举类，定义了基于搜索的量化校准策略。
    该类继承自 enum.Enum，用于限制策略的取值范围，确保只有预定义的策略可用。
    The strategy for search-based quantization calibration.
    """

    Manual = enum.auto()                    # 手动策略
    GridSearch = enum.auto()                # 网格搜索策略
    # RandomSearch = enum.auto()
    # Bayesian = enum.auto()
    # EvolutionaryAlgorithm = enum.auto()
    # EvolutionaryStrategy = enum.auto()


class SearchBasedCalibGranularity(enum.Enum):
    """
    枚举类，定义了基于搜索的量化校准的粒度级别。
    继承自 enum.Enum，该类限制了可选的粒度类型，确保策略配置的一致性和有效性。
    The granularity for search-based quantization calibration.
    """

    Group = enum.auto()                     # 按组粒度进行校准
    ChannelGroup = enum.auto()              # 按通道组粒度进行校准
    Layer = enum.auto()                     # 按层粒度进行校准


class SearchBasedCalibObjective(enum.Enum):
    """
    枚举类，定义了基于搜索的量化校准的目标。
    继承自 enum.Enum，该类确保校准目标的取值在预定义的选项内。
    The objective for search-based quantization calibration.
    """

    TensorError = enum.auto()               # 最小化张量的量化误差
    """minimize the quantization error of the tensor."""
    ProductsError = enum.auto()             # 最小化乘积的误差
    """minimize the error of the the multiplication products."""
    OutputsError = enum.auto()              # 最小化评估模块输出的误差
    """minimize the error of the outputs of the evaluation module."""


@configclass
@dataclass
class SearchBasedCalibConfig:
    """
    数据类，用于配置基于搜索的量化校准过程。
    该类通过装饰器 @configclass 和 @dataclass 简化了配置类的定义，并结合了数据校验和默认值设置，确保配置的一致性和有效性。
    The base configuration for search-based quantization calibration.

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
    """

    degree: int = 2                                                                 # 量化误差的幂次，控制误差计算的敏感度
    objective: SearchBasedCalibObjective = SearchBasedCalibObjective.OutputsError   # 量化校准的目标，决定优化的误差类型
    strategy: SearchBasedCalibStrategy = SearchBasedCalibStrategy.Manual            # 量化校准的策略，决定搜索方法
    granularity: SearchBasedCalibGranularity = SearchBasedCalibGranularity.Layer    # 量化校准的粒度级别，决定参数调整的范围
    element_batch_size: int = -1                                                    # 校准时元素的批次大小，控制每次处理的元素数量
    sample_batch_size: int = -1                                                     # 校准时样本的批次大小，影响校准效率和内存使用
    element_size: int = -1                                                          # 校准元素的大小，控制每个元素的数据量
    sample_size: int = -1                                                           # 校准样本的大小，控制每个样本的数据量
    pre_reshape: bool = True                                                        # 是否在校准前对张量进行重塑，影响数据处理的形式
    outputs_device: str = "cpu"                                                     # 用于存储模块预计算输出的设备，决定数据存储的位置

    def __post_init__(self) -> None:
        """
        在数据类初始化后执行，进行额外的属性验证和调整。
        """
        # outputs_device 设置
        if self.outputs_device != "cpu":
            self.outputs_device = None
        
        # 样本和元素大小验证
        if self.element_size != 0 or self.sample_size != 0:
            assert self.element_batch_size != 0, "element_batch_size must not be zero"
            assert self.sample_batch_size != 0, "sample_batch_size must not be zero"
            assert self.element_size != 0, "element_size must not be zero"
            assert self.sample_size != 0, "sample_size must not be zero"
        else:
            assert self.objective == SearchBasedCalibObjective.TensorError

        # 目标与粒度关联
        if self.objective == SearchBasedCalibObjective.TensorError:
            pass
        elif self.granularity == SearchBasedCalibGranularity.Layer:
            self.objective = SearchBasedCalibObjective.OutputsError
            self.element_batch_size = -1
            self.element_size = -1

    @property
    def needs_search(self) -> bool:
        """
        判断是否启用搜索功能。
        Whether the search is enabled.
        """
        return self.strategy != SearchBasedCalibStrategy.Manual

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """
        生成基于当前配置的目录名称，用于组织和存储校准结果或缓存数据。
        Generate the directory names of the configuration.

        Args:
            prefix (`str`, *optional*, default=`""`):
                The prefix of the directory.

        Returns:
            `list[str]`:
                The directory names.
        """
        # 构建基本名称
        name = f"{self.objective.name}.{self.strategy.name}.{self.granularity.name}.d{num2str(self.degree)}"
        name += f".e{num2str(self.element_size)}.s{num2str(self.sample_size)}"
        # 添加前缀
        if prefix:
            name = f"{prefix}.{name}"
        return [name]
