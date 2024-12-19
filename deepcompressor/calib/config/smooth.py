# -*- coding: utf-8 -*-
"""Smooth quantization configuration."""

import enum
from dataclasses import dataclass, field

import omniconfig
from omniconfig import configclass

from ...utils.common import num2str
from ...utils.config import SkipBasedConfig
from .search import (
    SearchBasedCalibConfig,
    SearchBasedCalibGranularity,
    SearchBasedCalibObjective,
    SearchBasedCalibStrategy,
)

__all__ = [
    "SmoothSpanMode",
    "SmoothCalibConfig",
    "SmoothAttentionCalibConfig",
    "SkipBasedSmoothCalibConfig",
    "SmoothTransfomerConfig",
]


class SmoothSpanMode(enum.Enum):
    """
    枚举类，继承自 enum.Enum，用于定义在平滑量化过程中计算跨度（span）所使用的方法模式。
    该类提供了两种计算模式：AbsMax 和 RootMeanSquare。
    The mode for computing the span used in smoothing scale calculation.
    """

    AbsMax = enum.auto()                    # 绝对最大值
    RootMeanSquare = enum.auto()            # 均方根


@configclass
@dataclass
class SmoothCalibConfig(SearchBasedCalibConfig):
    """
    继承自 SearchBasedCalibConfig，用于配置平滑量化（Smooth Quantization）。
    该类通过数据类和配置类装饰器简化配置定义，结合搜索策略配置和特定的平滑量化参数，实现对量化过程的精细控制。
    Configuration for smooth quantization.

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
        allow_a_quant (`bool`, *optional*, default=`True`):
            Whether to allow the quantization for alpha tensor.
        allow_b_quant (`bool`, *optional*, default=`True`):
            Whether to allow the quantization for beta tensor.
        spans (`list[tuple[SmoothSpanMode, SmoothSpanMode]]`, *optional*, default=`[]`):
            The span combinations. The first element is for the alpha and the second element is for the beta.
        alpha (`float`, *optional*, default=`0.5`):
            The smoothing alpha.
        beta (`float`, *optional*, default=`-1`):
            The smoothing beta.
        num_grids (`int`, *optional*, default=`20`):
            The number of grids for grid search.
        allow_low_rank (`bool`, *optional*, default=`False`):
            Whether to allow quantization low-rank branch during calibration.
    """

    allow_a_quant: bool = True                                                                      # 是否允许 alpha 张量的量化
    allow_b_quant: bool = True                                                                      # 是否允许 beta 张量的量化
    spans: list[tuple[SmoothSpanMode, SmoothSpanMode]] = field(                                     # 跨度组合，第一个元素用于 alpha，第二个元素用于 beta
        default_factory=list,
        metadata={
            omniconfig.ARGPARSE_KWARGS: {
                "nargs": "+",
                "type": lambda s: tuple(SmoothSpanMode[x.split(".")[-1]] for x in s.split(",")),
            }
        },
    )
    a_spans: list[SmoothSpanMode] = field(default_factory=list, init=False)                         # 处理后的 alpha 跨度列表
    b_spans: list[SmoothSpanMode] = field(default_factory=list, init=False)                         # 处理后的 beta 跨度列表
    alpha: float = 0.5                                                                              # 平滑参数 alpha
    beta: float = -1                                                                                # 平滑参数 beta
    num_grids: int = 20                                                                             # 网格搜索的网格数量
    allow_low_rank: bool = False                                                                    # 是否允许在校准过程中进行低秩分支量化

    def __post_init__(self) -> None:  # noqa: C901
        """
        在数据类实例化后执行，用于对属性进行验证和调整，确保配置参数的有效性和一致性。
        """
        # region remove duplicates of ranges
        # 处理跨度组合，去除重复的组合
        _spans, _spanset, _a_spanset, _b_spanset = [], set(), set(), set()      # 初始化临时变量
        self.a_spans, self.b_spans = [], []                                     # 清空原始跨度列表
        for a_span, b_span in self.spans:
            # 将跨度字符串转换为 SmoothSpanMode 枚举类型
            if isinstance(a_span, str):
                a_span = SmoothSpanMode[a_span]
            if isinstance(b_span, str):
                b_span = SmoothSpanMode[b_span]
            assert isinstance(a_span, SmoothSpanMode), f"Invalid span mode used for alpha: {a_span}"
            assert isinstance(b_span, SmoothSpanMode), f"Invalid span mode used for beta: {b_span}"
            _span = (a_span, b_span)                                            # 构建跨度组合
            if _span in _spanset:                                               # 如果组合已存在，则跳过
                continue
            _spans.append(_span)                                                # 添加到临时列表
            _spanset.add(_span)                                                 # 添加到集合
            # 单独处理 alpha 和 beta 的跨度
            if a_span not in _a_spanset:
                _a_spanset.add(a_span)
                self.a_spans.append(a_span)
            if b_span not in _b_spanset:
                _b_spanset.add(b_span)
                self.b_spans.append(b_span)
        self.spans = _spans
        # endregion
        
        # 手动策略下的配置验证
        if self.strategy == SearchBasedCalibStrategy.Manual:
            assert len(self.spans) == 1, "Only one span combination is allowed in manual mode"
            assert self.alpha != 0 or self.beta != 0, "alpha and beta cannot be both zero"
            self.alpha, self.beta = self.get_alpha_beta_pairs()[0]
        
        # 粒度调整
        if self.granularity == SearchBasedCalibGranularity.Group:
            self.granularity = SearchBasedCalibGranularity.ChannelGroup
        if self.allow_low_rank:
            self.granularity = SearchBasedCalibGranularity.Layer
            
        # 参数范围验证
        assert -3 <= self.alpha <= 1, "alpha must be less than or equal to 1"
        assert -3 <= self.beta <= 1, "beta must be less than or equal to 1"
        super().__post_init__()

    def get_alpha_beta_pairs(self) -> list[tuple[float, float]]:  # noqa: C901
        """
        根据当前配置生成用于平滑量化的 alpha 和 beta 值对。
        Get the alpha and beta pairs for smooth quantization.

        Returns:
            `list[tuple[float, float]]`:
                The alpha and beta pair candidates.
        """
        # 手动策略
        if self.strategy == SearchBasedCalibStrategy.Manual:
            # 根据 alpha 和 beta 的值调整并生成对应的值对
            if self.beta < 0:
                assert 0 <= self.alpha <= 1, "alpha must be in [0, 1]"
                return [(self.alpha, 1 - self.alpha)]
            elif self.alpha < 0:
                assert 0 <= self.beta <= 1, "beta must be in [0, 1]"
                return [(1 - self.beta, self.beta)]
            else:
                assert 0 <= self.alpha <= 1, "alpha must be in [0, 1]"
                assert 0 <= self.beta <= 1, "beta must be in [0, 1]"
                return [(self.alpha, self.beta)]
        
        # 其他策略
        choices = [i / self.num_grids for i in range(1, self.num_grids)]        # 生成网格搜索的选择列表
        # 根据 alpha 和 beta 的不同取值组合生成对应的值对
        if self.alpha > 0:
            if self.beta > 0:
                return [(0, 0)] + [(alpha, alpha) for alpha in choices]
            if self.beta == 0:
                return [(0, 0)] + [(alpha, 0) for alpha in choices]
            if self.beta == -1:
                return [(0, 0)] + [(alpha, 1 - alpha) for alpha in choices]
            if self.beta == -2:
                return [(0, 0)] + [(alpha, 0) for alpha in choices] + [(alpha, 1 - alpha) for alpha in choices]
            return (
                [(0, 0)] + [(alpha, 0) for alpha in choices] + [(alpha, beta) for alpha in choices for beta in choices]
            )
        if self.alpha == 0:
            if self.beta > 0:
                return [(0, 0)] + [(0, beta) for beta in choices]
            if self.beta == 0:
                return [(0, 0)] + [(alpha, 0) for alpha in choices] + [(0, beta) for beta in choices]
            if self.beta == -1:
                return [(0, 0)] + [(0, beta) for beta in choices] + [(alpha, 1 - alpha) for alpha in choices]
            if self.beta == -2:
                return (
                    [(0, 0)]
                    + [(alpha, 0) for alpha in choices]
                    + [(0, beta) for beta in choices]
                    + [(alpha, 1 - alpha) for alpha in choices]
                )
            return (
                [(0, 0)]
                + [(alpha, 0) for alpha in choices]
                + [(0, beta) for beta in choices]
                + [(alpha, beta) for alpha in choices for beta in choices]
            )
        if self.alpha == -1:
            if self.beta > 0 or self.beta == -1:
                return [(0, 0)] + [(alpha, 1 - alpha) for alpha in choices]
            if self.beta == 0 or self.beta == -2:
                return [(0, 0)] + [(alpha, 0) for alpha in choices] + [(alpha, 1 - alpha) for alpha in choices]
            return (
                [(0, 0)] + [(alpha, 0) for alpha in choices] + [(alpha, beta) for alpha in choices for beta in choices]
            )
        if self.alpha == -2:
            if self.beta > 0 or self.beta == -1:
                return [(0, 0)] + [(0, beta) for beta in choices] + [(alpha, 1 - alpha) for alpha in choices]
            if self.beta == 0 or self.beta == -2:
                return (
                    [(0, 0)]
                    + [(alpha, 0) for alpha in choices]
                    + [(0, beta) for beta in choices]
                    + [(alpha, 1 - alpha) for alpha in choices]
                )
            return (
                [(0, 0)]
                + [(alpha, 0) for alpha in choices]
                + [(0, beta) for beta in choices]
                + [(alpha, beta) for alpha in choices for beta in choices]
            )
        if self.alpha == -3:
            if self.beta > 0:
                return (
                    [(0, 0)]
                    + [(0, beta) for beta in choices]
                    + [(alpha, beta) for alpha in choices for beta in choices]
                )
            return (
                [(0, 0)]
                + [(0, beta) for beta in choices]
                + [(alpha, 0) for alpha in choices]
                + [(alpha, beta) for alpha in choices for beta in choices]
            )
        raise ValueError("Invalid alpha and beta values")

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """
        根据当前配置生成用于组织和存储平滑量化结果或缓存数据的唯一目录名称。
        Get the directory names of the smooth quantization configuration.

        Args:
            prefix (`str`, *optional*, default=`""`):
                The prefix of the directory.

        Returns:
            `list[str]`:
                The directory names of the configuration.
        """
        # 调用基类方法获取积累生成的目录名称列表
        names = super().generate_dirnames(**kwargs)
        # 添加跨度组合名称
        names.append("[{}]".format("+".join(f"a.{a_span.name}.b.{b_span.name}" for a_span, b_span in self.spans)))
        # 添加 alpha 和 beta 值对名称
        alpha, beta = num2str(self.alpha), num2str(self.beta)
        if self.strategy == SearchBasedCalibStrategy.Manual:
            names.append(f"a{alpha}.b{beta}")
        elif self.alpha > 0:
            names.append(f"g{self.num_grids}.b{beta}")
        elif self.beta > 0:
            names.append(f"g{self.num_grids}.a{alpha}")
        else:
            names.append(f"g{self.num_grids}.a{alpha}.b{beta}")
            
        # 添加低秩标识
        if self.allow_low_rank:
            names[-1] += ".lr"
        
        # 处理不允许的量化选项
        disallows = []
        if not self.allow_a_quant:
            disallows.append("a")
        if not self.allow_b_quant:
            disallows.append("b")
        if disallows:
            names.append(f"disallow.[{'+'.join(disallows)}]")
        # 添加前缀
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names


@configclass
@dataclass
class SkipBasedSmoothCalibConfig(SkipBasedConfig, SmoothCalibConfig):
    """
    通过多重继承结合了 SkipBasedConfig 和 SmoothCalibConfig 的功能，用于配置平滑量化过程，同时支持跳过特定模块。
    该类允许用户在进行平滑量化时，灵活排除不需要校准的模块，以优化量化过程的效率和效果。
    Configuration for smooth quantization.

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
        allow_a_quant (`bool`, *optional*, default=`True`):
            Whether to allow the quantization for alpha tensor.
        allow_b_quant (`bool`, *optional*, default=`True`):
            Whether to allow the quantization for beta tensor.
        spans (`list[tuple[SmoothSpanMode, SmoothSpanMode]]`, *optional*, default=`[]`):
            The span combinations. The first element is for the alpha and the second element is for the beta.
        alpha (`float`, *optional*, default=`0.5`):
            The smoothing alpha.
        beta (`float`, *optional*, default=`-1`):
            The smoothing beta.
        num_grids (`int`, *optional*, default=`20`):
            The number of grids for grid search.
        allow_low_rank (`bool`, *optional*, default=`False`):
            Whether to allow quantization SVD during calibration.
        skips (`list[str]`, *optional*, default=`[]`):
            The keys of the modules to skip.
    """

    pass


@configclass
@dataclass
class SmoothAttentionCalibConfig(SmoothCalibConfig):
    """
    继承自 SmoothCalibConfig，专门用于配置平滑量化在注意力机制中的应用。
    通过继承，将 SmoothCalibConfig 的通用配置参数应用于注意力部分的平滑量化，确保对注意力模块的量化过程进行精细控制。
    Configuration for smooth quantization.

    Args:
        degree (`int`, *optional*, default=`2`):
            The power degree for the quantization error. Defaults to `2`.
        strategy (`SearchBasedCalibStrategy`, *optional*, default=`SearchBasedCalibStrategy.Manual`):
            The strategy for quantization calibration.
        sample_batch_size (`int`, *optional*, default=`-1`):
            The samples batch size for calibration.
        sample_size (`int`, *optional*, default=`-1`):
            The calibration sample size.
        outputs_device (`str`, *optional*, default=`"cpu"`):
            The device to store the precomputed outputs of the module.
        allow_a_quant (`bool`, *optional*, default=`True`):
            Whether to allow the quantization for alpha tensor.
        allow_b_quant (`bool`, *optional*, default=`True`):
            Whether to allow the quantization for beta tensor.
        spans (`list[tuple[SmoothSpanMode, SmoothSpanMode]]`, *optional*, default=`[]`):
            The span combinations. The first element is for the alpha and the second element is for the beta.
        alpha (`float`, *optional*, default=`0.5`):
            The smoothing alpha.
        beta (`float`, *optional*, default=`-1`):
            The smoothing beta.
        num_grids (`int`, *optional*, default=`20`):
            The number of grids for grid search.
    """

    objective: SearchBasedCalibObjective = field(init=False, default=SearchBasedCalibObjective.OutputsError)    # 量化校准的目标
    granularity: SearchBasedCalibGranularity = field(init=False, default=SearchBasedCalibGranularity.Layer)     # 量化校准的粒度级别
    element_batch_size: int = field(init=False, default=-1)                                                     # 校准时元素的批次大小
    element_size: int = field(init=False, default=-1)                                                           # 校准元素的大小
    pre_reshape: bool = field(init=False, default=True)                                                         # 是否在校准前对张量进行重塑
    allow_low_rank: bool = field(init=False, default=False)                                                     # 是否允许在校准过程中进行低秩分支量化


@configclass
@dataclass
class SmoothTransfomerConfig:
    """
    用于配置基于 Transformer 模型的平滑量化。
    该类包含两个主要部分：投影层 (proj) 和注意力层 (attn) 的平滑量化配置。
    通过分别配置这两部分，用户可以对 Transformer 模型中的关键组件进行精细的量化调整。
    Configuration for smooth quantization of transformer-based models.

    Args:
        proj (`SkipBasedSmoothCalibConfig` or `None`, *optional*, default=`None`):
            The smooth configuration for projections.
        attn (`SmoothAttentionCalibConfig` or `None`, *optional*, default=`None`):
            The smooth configuration for attentions.
    """

    proj: SkipBasedSmoothCalibConfig | None = None      # 投影层的平滑量化配置
    attn: SmoothAttentionCalibConfig | None = None      # 注意力层的平滑量化配置

    @property
    def enabled_proj(self) -> bool:
        """
        判断是否为投影层启用了平滑量化配置。
        Whether the smooth quantization is enabled for projections.
        """
        return self.proj is not None

    @property
    def enabled_attn(self) -> bool:
        """
        判断是否为注意力层启用了平滑量化配置。
        Whether the smooth quantization is enabled for attentions.
        """
        return self.attn is not None

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """
        根据当前配置生成平滑量化配置的目录名称，用于组织和存储相关的量化结果或缓存数据。
        Get the names of the smooth quantization configuration.

        Args:
            prefix (`str`, *optional*, default=`""`):
                The prefix of the directory.

        Returns:
            `list[str]`:
                The names of the smooth quantization configuration
        """
        proj_names = self.proj.generate_dirnames(prefix="proj") if self.proj is not None else []    # 生成投影层的目录名称
        attn_names = self.attn.generate_dirnames(prefix="attn") if self.attn is not None else []    # 生成注意力层的目录名称
        num_names = max(len(proj_names), len(attn_names))

        # 遍历组合投影层和注意力层的目录名称
        names = []
        for index in range(num_names):
            name = []
            if index < len(proj_names):
                name.append(proj_names[index])
            if index < len(attn_names):
                name.append(attn_names[index])
            names.append("-".join(name))
            
        # 添加前缀
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names
