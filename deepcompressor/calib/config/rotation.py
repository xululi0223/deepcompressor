# -*- coding: utf-8 -*-
"""Quantization Rotation configuration."""

import typing as tp
from dataclasses import dataclass, field

import omniconfig
from omniconfig import configclass

__all__ = ["QuantRotationConfig"]


@configclass
@dataclass
class QuantRotationConfig:
    """
    用于配置旋转量化（Rotation Quantization）。
    该类通过 @dataclass 和 @configclass 装饰器简化配置定义，利用数据类特性自动生成初始化方法，并集成 omniconfig 的配置管理功能。
    该配置类允许用户指定是否使用随机哈达玛旋转矩阵以及明确使用哈达玛变换的模块。
    Configuration for rotation quantization.

    Args:
        random (`bool`, *optional*, default=`False`):
            Whether to use random hadamard sample as rotation matrix.
        transforms (`list[str]`, *optional*, default=`[]`):
            The module keys using explicit hadamard transform.
    """

    random: bool = False                                    # 是否使用随机哈达玛旋转矩阵
    transforms: list[str] = field(default_factory=list)     # 使用显式哈达玛变换的模块键列表

    def __post_init__(self) -> None:
        """
        在数据类实例化后执行，用于对 transforms 属性进行后处理，确保其唯一性并按字母顺序排序。
        """
        self.transforms = sorted(set(self.transforms or []))

    @property
    def with_hadamard_transform(self) -> bool:
        """
        判断是否存在哈达玛变换。
        """
        return len(self.transforms) > 0

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """
        根据当前配置生成唯一的目录名称，用于组织和存储旋转量化的结果或缓存数据。
        Get the directory names of the rotation quantization configuration.

        Returns:
            list[str]: The directory names of the rotation quantization configuration.
        """
        # 基础名称生成
        name = "random" if self.random else "hadamard"
        # 添加显式哈达玛变换的模块键
        if self.with_hadamard_transform:
            name += f".[{'+'.join(self.transforms)}]"
        # 添加前缀
        return [f"{prefix}.{name}" if prefix else name]

    @classmethod
    def update_get_arguments(
        cls: type["QuantRotationConfig"],
        *,
        overwrites: dict[str, tp.Callable[[omniconfig.Arguments], None] | None] | None = None,
        defaults: dict[str, tp.Any] | None = None,
    ) -> tuple[dict[str, tp.Callable[[omniconfig.Arguments], None] | None], dict[str, tp.Any]]:
        """
        更新并获取旋转量化配置的命令行参数，用于与 omniconfig 配置管理工具集成。
        Get the arguments for the rotation quantization configuration.
        
        Args:
            overwrites: The overwrites of the arguments.
            defaults: The default values of the arguments.
        """
        # 初始化参数
        overwrites = overwrites or {}
        defaults = defaults or {}

        # 收集布尔字段，用于添加带有前缀的布尔字段
        collect_fn = omniconfig.ADD_PREFIX_BOOL_FIELDS("transform", **defaults)

        def add_transforms_argument(parser):
            """
            内部函数，添加 transforms 参数。
            """
            collect_fn(parser)              # 添加带有前缀的布尔字段
            parser.add_argument("--transforms", nargs="+", default=[], help="The keys of the modules to transform.")

        overwrites.setdefault("transforms", add_transforms_argument)    # 设置默认的参数覆盖
        return overwrites, defaults

    @classmethod
    def update_from_dict(
        cls: type["QuantRotationConfig"], *, parsed_args: dict[str, tp.Any], overwrites: dict[str, tp.Any]
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """
        从解析后的参数字典中创建旋转量化配置，处理带前缀的布尔字段，并合并到 transforms 列表中。
        Create a rotation quantization configuration from the parsed arguments.
        
        Args:
            parsed_args: The parsed arguments.
            overwrites: The overwrites of the arguments.
        """
        # 收集带前缀的布尔字段
        parsed_args.setdefault("transforms", []).extend(omniconfig.COLLECT_PREFIX_BOOL_FIELDS(parsed_args, "transform"))
        return parsed_args, overwrites
