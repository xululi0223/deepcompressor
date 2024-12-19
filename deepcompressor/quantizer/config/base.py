# -*- coding: utf-8 -*-
"""Quantization kernel config."""

import typing as tp
from abc import abstractmethod
from dataclasses import dataclass, field

import omniconfig
import torch
from omniconfig import configclass

from ...data.dtype import QuantDataType
from ...data.utils import DtypeUtils, ScaleUtils, ShapeUtils
from ...data.zero import ZeroPointDomain
from ...utils.config import EnableConfig

__all__ = [
    "BaseQuantizerConfig",
    "DecomposedQuantizerConfig",
    "QuantizerConfig",
    "ProgressiveQuantizerConfig",
]


class BaseQuantizerConfig(EnableConfig):
    """
    抽象基类，定义了量化配置的基本接口和属性。
    它继承自 EnableConfig，并使用 Python 的 abc（抽象基类）模块来强制子类实现特定的方法和属性。
    该类旨在为不同类型的量化配置提供统一的接口和基本功能。
    Base Quantizer configuration.
    """

    @property
    @abstractmethod
    def quant_dtype(self) -> QuantDataType | None:
        """
        获取量化数据类型。
        The quantization data type.
        """
        ...

    @property
    @abstractmethod
    def zero_domain(self) -> ZeroPointDomain | None:
        """
        获取零点域。
        The zero-point domain.
        """
        ...

    @property
    @abstractmethod
    def largest_group_shape(self) -> tp.Sequence[int]:
        """
        获取最大的组形状。
        The shape of the largest group.
        """
        ...

    @property
    @abstractmethod
    def smallest_group_shape(self) -> tp.Sequence[int]:
        """
        获取最小的组形状。
        The shape of the smallest group.
        """
        ...

    def is_enabled(self) -> bool:
        """
        检查量化配置是否启用。
        Whether the quantization configuration is enabled.
        """
        return self.quant_dtype is not None

    @abstractmethod
    def decompose(self) -> "DecomposedQuantizerConfig":
        """
        将当前复杂的量化配置分解为更简单的配置列表。
        Decompose the configuration to a list of simple configurations.
        """
        ...

    def generate_dirnames(
        self,
        *,
        prefix: str = "",
        shape: torch.Size | tuple[int, ...] = (4096, 4096),
        default_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> list[str]:
        """
        生成量化配置的目录名称。
        Generate the directory names of the quantization configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix for the directory names.
            shape (`torch.Size` or `tuple[int, ...]`, *optional*, defaults to `(4096, 4096)`):
                The shape of the tensor to be quantized.

        Returns:
            `list[str]`:
                The names of the quantization configuration.
                    - The number of effective bits.
                    - The name of the quantization data type.
                    - The name of the group shapes.
        """
        # 调用decompose方法将复杂配置分解为DecomposedQuantizerConfig对象
        # 调用分解后的配置的generate_dirnames方法生成目录名称列表
        return self.decompose().generate_dirnames(
            prefix=prefix, shape=torch.Size(shape), default_dtype=default_dtype, **kwargs
        )


@dataclass(frozen=True)
class DecomposedQuantizerConfig(BaseQuantizerConfig):
    """
    表示已分解的量化配置。
    它封装了多个 QuantizerConfig 步骤，每个步骤定义了不同的量化设置。
    此类支持用于复杂量化策略的配置管理，如逐步量化或多阶段量化过程。
    """
    # 表示量化配置的步骤列表
    steps: tuple["QuantizerConfig", ...]
    # 标记是否需要进行反量化饱和处理
    needs_dequant_saturation: bool = False

    @property
    def quant_dtype(self) -> QuantDataType | None:
        """
        返回最后一个步骤的量化数据类型。
        """
        return self.steps[-1].dtype if self.steps else None

    @property
    def zero_domain(self) -> ZeroPointDomain | None:
        """
        返回最后一个步骤的零点域。
        """
        return self.steps[-1].zero_point if self.steps else None

    @property
    def largest_group_shape(self) -> tp.Sequence[int]:
        """
        返回第一个步骤的最大组形状。
        """
        return self.steps[0].largest_group_shape if self.steps else (-1, -1, -1)

    @property
    def smallest_group_shape(self) -> tp.Sequence[int]:
        """
        返回最后一个步骤的最小组形状。
        """
        return self.steps[-1].smallest_group_shape if self.steps else (-1, -1, -1)

    @property
    def num_steps(self) -> int:
        """
        返回量化配置中步骤的数量。
        """
        return len(self.steps)

    def decompose(self) -> "DecomposedQuantizerConfig":
        """
        返回自身。因为DecomposedQuantizerConfig已经是分解后的配置。
        """
        return self

    def __eq__(self, value: object) -> bool:
        """
        定义相等性比较。
        """
        # 如果 value 不是 DecomposedQuantizerConfig 类型，则返回 False
        if not isinstance(value, DecomposedQuantizerConfig):
            return False

        # 如果步骤数量不同，则返回 False
        if self.num_steps != value.num_steps:
            return False

        # 逐一比较每个步骤的配置是否相等，具体比较dtype、group_shapes、scale_dtypes
        for rhs, lhs in zip(self.steps, value.steps, strict=True):
            # ! we only compare the dtype, group_shapes, and scale_dtypes
            if rhs.dtype != lhs.dtype:
                return False
            if rhs.group_shapes != lhs.group_shapes:
                return False
            if rhs.scale_dtypes != lhs.scale_dtypes:
                return False
            
        # 如果需要反量化饱和处理的标志不同，则返回 False
        if self.num_steps > 1:
            if self.needs_dequant_saturation != value.needs_dequant_saturation:
                return False
        return True

    def _get_effective_bits(
        self, *, shape: torch.Size | tuple[int, ...] = (4096, 4096), default_dtype: torch.dtype = torch.float16
    ) -> float:
        """
        计算量化配置的有效位数。
        Get the effective bits of the quantization.

        Args:
            shape (`torch.Size` or `tuple[int, ...]`, *optional*, defaults to `(4096, 4096)`):
                The shape of the tensor to be quantized.
            dtype (torch.dtype, *optional*, defaults to `torch.float16`):
                The dtype of the tensor to be quantized.

        Returns:
            `float`:
                The effective bits.
        """
        # 将 shape 转换为 torch.Size 类型
        shape = torch.Size(shape)
        
        # 如果 quant_dtype 为 None，则返回 default_dtype 的位数
        if self.quant_dtype is None:
            return DtypeUtils.infer_dtype_bits(default_dtype)

        # 初始化 bits 为 quant_dtype 的总位数
        bits = self.quant_dtype.total_bits

        # 逐一遍历每个步骤的配置
        for step_config in self.steps:
            # 推断组形状和缩放数据类型
            group_shapes = ShapeUtils.infer_group_shapes(step_config.group_shapes, shape=shape)
            scale_dtypes = ScaleUtils.infer_scale_dtypes(step_config.scale_dtypes, default_dtype=default_dtype)
            # 对每个组形状和缩放数据类型，计算缩放数据类型的位数除以组中元素数量，然后累加到 bits 中
            for group_shape, scale_dtype in zip(group_shapes, scale_dtypes, strict=True):
                bits += DtypeUtils.infer_dtype_bits(scale_dtype) / group_shape.numel()
                
        # 零点域调整
        if self.zero_domain == ZeroPointDomain.PreScale:
            # 如果零点域为 PreScale，则 bits 加上 quant_dtype 的总位数除以最后一个组的元素数量
            bits += self.quant_dtype.total_bits / group_shapes[-1].numel()
        elif self.zero_domain == ZeroPointDomain.PostScale:
            # 如果零点域为 PostScale，则 bits 加上 quant_dtype 的总位数除以组的元素数量
            bits += DtypeUtils.infer_dtype_bits(scale_dtype) / group_shape.numel()
        return bits

    def _get_dtype_name(self, default_dtype: torch.dtype = torch.float16) -> str:
        """
        获取量化数据类型的名称字符串。
        Get the name of the quantization data type.

        Args:
            default_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
                The default_dtype dtype of the input tensor.

        Returns:
            `str`:
                The name of the quantization data type.
        """
        # 如果 quant_dtype 为 None，则返回 default_dtype 的名称
        if self.quant_dtype is None:
            return DtypeUtils.infer_dtype_name(default_dtype)

        # 获取 quant_dtype 的名称
        name = DtypeUtils.infer_dtype_name(self.quant_dtype)

        # 零点域后缀添加
        if self.zero_domain == ZeroPointDomain.PreScale:
            # 如果零点域为 PreScale，则添加 .z 后缀
            name += ".z"
        elif self.zero_domain == ZeroPointDomain.PostScale:
            # 如果零点域为 PostScale，则添加 .zp 后缀
            name += ".zp"
        return name

    def _get_group_shapes_name(self, default_dtype: torch.dtype = torch.float16) -> str:
        """
        获取组形状的名称字符串。
        Get the name of the group shapes.

        Args:
            default_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
                The default_dtype dtype of the input tensor.

        Returns:
            str: The name of the group shapes.
        """
        # 如果 quant_dtype 为 None，则返回默认 dtype 的名称
        if self.quant_dtype is None:
            return f"tnsr.{DtypeUtils.infer_dtype_name(default_dtype)}"
        
        # 初始化
        num_steps = len(self.steps)     # 获取步骤数量
        names = []                      # 初始化名称列表
        step_default_dtype = default_dtype  # 设置step_default_dtype为default_dtype
        
        # 逐一遍历每个步骤的配置
        for step, step_config in enumerate(self.steps):
            # 初始化step_names列表
            step_names = []
            # 遍历每个group_shape和scale_dtype
            for group_shape, sdtype in zip(step_config.group_shapes, step_config.scale_dtypes, strict=True):
                # 获取组形状名称和缩放数据类型名称，然后添加到step_names列表中
                name = f"{ShapeUtils.infer_group_shape_name(group_shape)}"
                name += f".{DtypeUtils.infer_dtype_name(sdtype or step_default_dtype)}"
                step_names.append(name)
            # 将step_names列表反转，然后用 . 连接，添加到step_name列表中
            step_name = ".".join(reversed(step_names))
            # 添加 [step_name] 或 step_name 到names列表中
            names.append(f"[{step_name}]" if step < num_steps - 2 else step_name)
            # 更新step_default_dtype为step_config.dtype
            step_default_dtype = step_config.dtype
            assert step_default_dtype is not None, "step_default_dtype must not be None"
        return ".".join(reversed(names))

    def generate_dirnames(
        self,
        *,
        prefix: str = "",
        shape: torch.Size | tuple[int, ...] = (4096, 4096),
        default_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> list[str]:
        """
        生成量化配置的目录名称列表，用于文件系统或配置管理。
        Generate the directory names of the quantization configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix for the directory names.
            shape (`torch.Size` or `tuple[int, ...]`, *optional*, defaults to `(4096, 4096)`):
                The shape of the tensor to be quantized.
            default_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
                The dtype of the tensor to be quantized.

        Returns:
            `list[str]`:
                The names of the quantization configuration.
                    - The number of effective bits.
                    - The name of the quantization data type.
                    - The name of the group shapes.
        """
        # 将 shape 转换为 torch.Size 类型
        shape = torch.Size(shape)

        # 有效位数、量化数据类型名称、组形状名称
        bits_str = str(int(self._get_effective_bits(shape=shape, default_dtype=default_dtype)))
        dtype_str = self._get_dtype_name(default_dtype=default_dtype)
        group_str = self._get_group_shapes_name(default_dtype=default_dtype)

        # 名称列表构建
        names = [bits_str, dtype_str, group_str]
        # 添加前缀
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names


@configclass
@dataclass
class QuantizerConfig(BaseQuantizerConfig):
    """
    表示一种具体的量化配置。
    它使用了 @dataclass 装饰器，使得类的实例能够自动生成常用方法，如 __init__ 等。
    该类定义了一系列与量化相关的配置参数，如量化数据类型、零点域、组形状等，并提供了初始化后处理方法来格式化配置。
    Quantizer configuration.

    Args:
        dtype (`QuantDataType` or `None`, *optional*, defaults to `None`):
            The quantization data type.
        zero_point (`ZeroPointDomain` or `None`, *optional*, defaults to `None`):
            The zero-point domain.
        group_shapes (`Sequence[Sequence[int]]`, *optional*, defaults to `((-1, -1, -1),)`):
            The shapes for per-group quantization.
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]`, *optional*, defaults to `(None,)`):
            The quantization scale data type for per-group quantization.
    """

    # 量化数据类型
    dtype: QuantDataType | None = None
    # 零点域
    zero_point: ZeroPointDomain | None = None
    # 每组量化的形状序列
    group_shapes: tp.Sequence[tp.Sequence[int]] = field(
        default=((-1, -1, -1),),
        metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": lambda s: [int(n) for n in s.split(",")]}},
    )
    # 每组量化的缩放数据类型序列
    scale_dtypes: tp.Sequence[torch.dtype | QuantDataType | None] = field(
        default=(None,), metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": DtypeUtils.eval_dtype}}
    )

    def __post_init__(self) -> None:
        """
        在数据类的初始化方法之后执行，用于对初始配置进行处理和格式化。
        """
        # 格式化组形状和缩放数据类型
        self.group_shapes, self.scale_dtypes = ShapeUtils.format_group_configs(
            group_shapes=self.group_shapes, scale_dtypes=self.scale_dtypes
        )
        # 如果量化数据类型为 None，则重新设置组形状为 (-1, -1, -1) 和缩放数据类型为 None
        if self.dtype is None:
            self.group_shapes, self.scale_dtypes = ((-1, -1, -1),), (None,)

    @property
    def quant_dtype(self) -> QuantDataType | None:
        """
        返回最终的量化数据类型
        The final quantization data type.
        """
        return self.dtype

    @property
    def zero_domain(self) -> ZeroPointDomain | None:
        """
        返回最终的零点域
        The final zero-point domain.
        """
        return self.zero_point

    @property
    def largest_group_shape(self) -> tp.Sequence[int]:
        """
        返回最大的组形状，即 self.group_shapes 的第一个元素。
        The shape of the largest group.
        """
        return self.group_shapes[0]

    @property
    def smallest_group_shape(self) -> tp.Sequence[int]:
        """
        返回最小的组形状，即 self.group_shapes 的最后一个元素。
        The shape of the smallest group."""
        return self.group_shapes[-1]

    def decompose(self) -> DecomposedQuantizerConfig:
        """
        将当前配置分解为 DecomposedQuantizerConfig 实例。
        Decompose the configuration to a list of simple configurations.
        """
        return DecomposedQuantizerConfig(steps=(self,) if self.dtype is not None else ())


@configclass
@dataclass
class ProgressiveQuantizerConfig(QuantizerConfig):
    """
    表示一种渐进式量化配置。
    该类引入了中间量化数据类型和中间量化级别，支持更复杂的渐进量化策略。
    通过定义多个中间步骤，使量化过程更加灵活和细致，适用于需要逐步优化的量化场景。
    Progressive Quantizer configuration.

    Args:
        dtype (`QuantDataType` or `None`, *optional*, defaults to `None`):
            The quantization data type.
        zero_point (`ZeroPointDomain` or `None`, *optional*, defaults to `None`):
            The zero-point domain.
        group_shapes (`Sequence[Sequence[int]]`, *optional*, defaults to `((-1, -1, -1),)`):
            The shapes for per-group quantization.
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]`, *optional*, defaults to `(None,)`):
            The quantization scale data type for per-group quantization.
        intermediate_dtypes (`Sequence[QuantDataType]`, *optional*, defaults to `()`):
            The intermediate quantization data types.
        intermediate_levels (Sequence[int], *optional*, defaults to `()`):
            The intermediate quantization levels.
        needs_dequant_saturation (`bool`, *optional*, defaults to `False`):
            Whether the dequantization needs saturation.
    """

    # 中间步骤的量化数据类型序列
    intermediate_dtypes: tp.Sequence[QuantDataType] = field(
        default_factory=tuple, metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": QuantDataType.from_str}}
    )
    # 中间量化级别序列，每个级别对应一个量化步骤的分组级别
    intermediate_levels: tp.Sequence[int] = field(
        default_factory=tuple, metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "+", "type": int}}
    )
    # 是否需要反量化饱和处理
    needs_dequant_saturation: bool = False

    def __post_init__(self) -> None:
        """
        初始化后处理方法，用于对渐进量化配置进行格式化和验证。
        """
        # 调用父类的初始化后处理方法
        super().__post_init__()

        # 如果量化数据类型为 None，则设置中间量化数据类型和中间量化级别为空
        if self.dtype is None:
            self.intermediate_dtypes = ()
            self.intermediate_levels = ()
            self.needs_dequant_saturation = False
            return

        # 获取量化级别数量
        num_levels = len(self.group_shapes)

        # 如果中间量化数据类型为QuantDataType类型，则转换为元组
        if isinstance(self.intermediate_dtypes, QuantDataType):
            self.intermediate_dtypes = (self.intermediate_dtypes,)
        # 如果中间量化级别为int类型，则转换为元组
        if isinstance(self.intermediate_levels, int):
            self.intermediate_levels = (self.intermediate_levels,)
        # 将中间量化数据类型转换为元组
        self.intermediate_dtypes = tuple(self.intermediate_dtypes)
        # 将中间量化级别转换为元组，确保每个级别小于量化级别数量
        self.intermediate_levels = tuple(level % num_levels for level in self.intermediate_levels)
        # 如果中间量化数据类型为空，则中间量化级别也为空
        if len(self.intermediate_dtypes) == 0:
            self.intermediate_levels = ()
            self.needs_dequant_saturation = False

        # 验证中间量化数据类型和中间量化级别
        assert len(self.intermediate_dtypes) == len(self.intermediate_levels)
        assert len(self.intermediate_levels) < num_levels
        assert all(isinstance(dtype, QuantDataType) for dtype in self.intermediate_dtypes)
        assert all(level < num_levels - 1 for level in self.intermediate_levels)

    def decompose(self) -> DecomposedQuantizerConfig:
        """
        将渐进量化配置分解为 DecomposedQuantizerConfig 实例，包含多个简单的 QuantizerConfig 步骤。
        Decompose the configuration to a list of simple configurations.
        """
        # 如果量化数据类型为 None，则直接返回空的 DecomposedQuantizerConfig 实例
        if self.dtype is None:
            return DecomposedQuantizerConfig(steps=())
        # 如果中间量化数据类型为空，则返回包含当前配置的 DecomposedQuantizerConfig 实例
        elif len(self.intermediate_dtypes) == 0:
            return DecomposedQuantizerConfig(steps=(self,))
        # 处理有中间步骤的情况
        else:
            # 初始化步骤列表
            steps = []
            prev_level = 0
            # 遍历中间量化级别和中间量化数据类型
            for level, dtype in zip(self.intermediate_levels, self.intermediate_dtypes, strict=True):
                # 创建 QuantizerConfig 实例，包含中间量化级别和中间量化数据类型，然后添加到步骤列表中
                steps.append(
                    QuantizerConfig(
                        dtype=dtype,
                        zero_point=None,
                        group_shapes=self.group_shapes[prev_level : level + 1],
                        scale_dtypes=self.scale_dtypes[prev_level : level + 1],
                    )
                )
                # 更新 prev_level 为当前 level
                prev_level = level + 1
            # 创建最后一个 QuantizerConfig 实例，包含最后一个量化级别和量化数据类型，然后添加到步骤列表中
            steps.append(
                QuantizerConfig(
                    dtype=self.dtype,
                    zero_point=self.zero_point,
                    group_shapes=self.group_shapes[prev_level:],
                    scale_dtypes=self.scale_dtypes[prev_level:],
                )
            )
            # 返回包含步骤列表的 DecomposedQuantizerConfig 实例
            return DecomposedQuantizerConfig(steps=tuple(steps), needs_dequant_saturation=self.needs_dequant_saturation)
