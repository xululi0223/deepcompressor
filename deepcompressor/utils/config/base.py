# -*- coding: utf-8 -*-

import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import omniconfig
from omniconfig import configclass

__all__ = ["EnableConfig", "KeyEnableConfig", "SkipBasedConfig", "IncludeBasedConfig"]


class EnableConfig(ABC):
    """
    抽象基类，继承自abc.ABC，用于定义接口。
    """
    @abstractmethod
    def is_enabled(self) -> bool:
        """
        判断配置是否启用。
        Whether the configuration is enabled.
        """
        return True


class KeyEnableConfig(ABC):
    """
    抽象基类，继承自abc.ABC，用于定义接口。
    """
    @abstractmethod
    def is_enabled_for(self, key: str) -> bool:
        """
        判断配置是否对给定的key启用。
        Whether the configuration is enabled for the given key.
        
        Args:
            key: 字符串参数。
        """
        return True


@configclass
@dataclass
class SkipBasedConfig(KeyEnableConfig, EnableConfig):
    """
    基于跳过的配置。
    Skip-based configration.

    Args:
        skips (`list[str]`, *optional*, defaults to `[]`):
            The keys of the modules to skip.
    """

    # 存储要跳过的模块键
    skips: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        初始化函数。
        """
        __post_init__ = getattr(super(), "__post_init__", None)
        if __post_init__:
            __post_init__()
        # 对skips列表进行排序和去重
        self.skips = sorted(set(self.skips or []))

    def is_enabled(self) -> bool:
        """
        判断配置是否启用。
        Whether the configuration is enabled.
        """
        return super().is_enabled()

    def is_enabled_for(self, key: str) -> bool:
        """
        判断配置是否启用且给定的key不在skips列表中。
        Whether the configuration is enabled for the given key.

        Args:
            key (`str`):
                The key.

        Returns:
            `bool`:
                Whether the configuration is enabled for the given key.
        """
        return self.is_enabled() and key not in self.skips

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """
        生成目录名称列表。
        Generate the directory names of the configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix of the directory names.

        Returns:
            `list[str]`:
                The directory names of the configuration.
        """
        # 调用父类的generate_dirnames方法并添加skip.[skips]目录名
        names = [*super().generate_dirnames(**kwargs), "skip.[{}]".format("+".join(self.skips))]  # type: ignore
        # 如果prefix存在，给每个目录名添加前缀
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names

    @classmethod
    def update_get_arguments(
        cls: type["SkipBasedConfig"],
        *,
        overwrites: dict[str, tp.Callable[[omniconfig.Arguments], None] | None] | None = None,
        defaults: dict[str, tp.Any] | None = None,
    ) -> tuple[dict[str, tp.Callable[[omniconfig.Arguments], None] | None], dict[str, tp.Any]]:
        """
        更新并获取配置的参数。
        Get the arguments for the quantization configuration.
        """
        # 调用父类的update_get_arguments方法
        update_get_arguments = getattr(super(), "update_get_arguments", None)
        # 初始化overwrites和defaults
        if update_get_arguments:
            overwrites, defaults = update_get_arguments(overwrites=overwrites, defaults=defaults)
        overwrites = overwrites or {}
        defaults = defaults or {}

        # 定义collect_fn函数，用于添加带有前缀的布尔字段
        collect_fn = omniconfig.ADD_PREFIX_BOOL_FIELDS("skip", **defaults)

        def add_skips_argument(parser):
            """
            内部函数，向解析器添加--skips参数。
            """
            collect_fn(parser)
            parser.add_argument("--skips", nargs="+", default=[], help="The keys of the modules to skip.")

        # 设置默认的skips参数处理函数
        overwrites.setdefault("skips", add_skips_argument)
        return overwrites, defaults

    @classmethod
    def update_from_dict(
        cls: type["SkipBasedConfig"], *, parsed_args: dict[str, tp.Any], overwrites: dict[str, tp.Any]
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """
        根据字典更新配置参数。
        Update the arguments settings for the quantization configuration.
        """
        # 调用父类的update_from_dict方法（如果存在）
        update_from_dict = getattr(super(), "update_from_dict", None)
        # 如果存在，调用update_from_dict方法
        if update_from_dict:
            parsed_args, overwrites = update_from_dict(parsed_args=parsed_args, overwrites=overwrites)

        # 从parsed_args中收集以"skip"为前缀的布尔字段并添加到skips列表中
        parsed_args.setdefault("skips", []).extend(omniconfig.COLLECT_PREFIX_BOOL_FIELDS(parsed_args, "skip"))
        return parsed_args, overwrites


# 处理配置逻辑并自动生成方法
@configclass
@dataclass
class IncludeBasedConfig(KeyEnableConfig, EnableConfig):
    """
    用于基于包含（includes）的配置管理。
    Include-based configuration.

    Args:
        includes (`list[str]`, *optional*, defaults to `[]`):
            The keys of the modules to include.
    """

    # 存储需要包含的模块键
    includes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        初始化后处理。
        """
        # 调用父类的__post_init__方法
        __post_init__ = getattr(super(), "__post_init__", None)
        if __post_init__:
            __post_init__()
        # 对includes列表进行去重并排序，确保唯一且有序
        self.includes = sorted(set(self.includes or []))

    def is_enabled(self) -> bool:
        """
        判断整体配置是否启用。
        Whether the kernel is enabled.
        """
        # 配置整体启用，且includes列表不为空
        return super().is_enabled() and bool(self.includes)

    def is_enabled_for(self, key: str) -> bool:
        """
        判断特定key是否启用配置。
        Whether the config is enabled for the module key.

        Args:
            key (`str`):
                The key.

        Returns:
            `bool`:
                Whether the config is needed.
        """
        # 配置启用，且key在includes列表中
        return self.is_enabled() and key in self.includes

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """
        生成配置相关的目录名称列表。
        Generate the directory names of the configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix of the directory names.

        Returns:
            `list[str]`:
                The directory names. The last directory name is the modules to include.
        """
        # 初始化names列表
        names = []
        if self.includes:
            # 调用父类的generate_dirnames方法并将结果赋给names
            names = super().generate_dirnames(**kwargs)  # type: ignore
            # 添加"include.[{includes}]"格式的目录名
            names.append("include.[{}]".format("+".join(self.includes)))
        # 如果prefix存在，给每个目录名添加前缀
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names

    @classmethod
    def update_get_arguments(
        cls: type["IncludeBasedConfig"],
        *,
        overwrites: dict[str, tp.Callable[[omniconfig.Arguments], None] | None] | None = None,
        defaults: dict[str, tp.Any] | None = None,
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """
        更新并获取配置的参数。
        Update the arguments settings for the quantization configuration.
        """
        # 调用父类的update_get_arguments方法（如果存在）
        update_get_arguments = getattr(super(), "update_get_arguments", None)
        if update_get_arguments:
            overwrites, defaults = update_get_arguments(overwrites=overwrites, defaults=defaults)
        # 初始化overwrites和defaults
        overwrites = overwrites or {}
        defaults = defaults or {}

        # 定义collect_fn，用于添加带有前缀的布尔字段
        collect_fn = omniconfig.ADD_PREFIX_BOOL_FIELDS("include", **defaults)

        def add_includes_argument(parser):
            """
            内部函数，向解析器添加--includes参数。
            """
            collect_fn(parser)
            parser.add_argument("--includes", nargs="+", default=[], help="The keys of the modules to include.")

        # 设置默认的includes参数处理函数
        overwrites.setdefault("includes", add_includes_argument)
        return overwrites, defaults

    @classmethod
    def update_from_dict(
        cls: type["IncludeBasedConfig"], *, parsed_args: dict[str, tp.Any], overwrites: dict[str, tp.Any]
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """
        根据字典更新配置参数。
        Update the arguments settings for the quantization configuration.
        """
        # 调用父类的update_from_dict方法（如果存在）
        update_from_dict = getattr(super(), "update_from_dict", None)
        if update_from_dict:
            parsed_args, overwrites = update_from_dict(parsed_args=parsed_args, overwrites=overwrites)

        # 从parsed_args中收集以"include"为前缀的布尔字段并添加到includes列表中
        parsed_args.setdefault("includes", []).extend(omniconfig.COLLECT_PREFIX_BOOL_FIELDS(parsed_args, "include"))
        return parsed_args, overwrites
