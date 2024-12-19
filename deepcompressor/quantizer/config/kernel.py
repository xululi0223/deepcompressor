# -*- coding: utf-8 -*-
"""Quantizatizer kernel configurations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields

import torch
from omniconfig import configclass

from ...data.dtype import QuantDataType
from ...data.range import QuantRange
from ...data.zero import ZeroPointDomain
from ...utils.config import EnableConfig, IncludeBasedConfig, KeyEnableConfig

__all__ = ["BaseQuantKernel", "BaseQuantKernelConfig", "BaseKeyEnableQuantKernelConfig"]


class BaseQuantKernel(ABC):
    """
    抽象基类（Abstract Base Class，ABC），用于定义量化内核的基本接口。
    它规定了所有具体量化内核必须实现的 quantize 方法，以确保不同的量化内核具有统一的接口和行为。
    Quantization kernel."""

    @abstractmethod
    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        view_shape: torch.Size,
        quant_dtype: QuantDataType,
        zero_domain: ZeroPointDomain | None,
        scale: torch.Tensor,
        zero: torch.Tensor,
        quant_range: QuantRange | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        量化张量。
        Quantize the tensor.

        Args:
            tensor (`torch.Tensor`):
                The tensor to quantize.
            view_shape (`torch.Size`):
                The view shape when quantizing the tensor.
            quant_dtype (`QuantDataType`):
                The quantization data type.
            zero_domain (`ZeroPointDomain` or `None`):
                The zero point domain.
            scale (`torch.Tensor`):
                The scale tensor.
            zero (`torch.Tensor`):
                The zero point tensor.
            quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                The quantization range.
            **kwargs: Other keyword arguments for the quantization kernel.

        Returns:
            `torch.Tensor`:
                The quantized tensor in the shape of ``view_shape``.
        """
        ...


class BaseQuantKernelConfig(ABC):
    """
    抽象基类，定义了量化内核配置的基本接口。
    它要求子类实现 name 属性、build 方法和 generate_dirnames 方法，以确保不同的量化内核配置具有统一的接口和行为。
    Base quantization kernel configuration."""

    @property
    @abstractmethod
    def name(self) -> str:
        """
        量化内核的名称。
        The name of the quantization kernel.
        """
        ...

    @abstractmethod
    def build(self) -> BaseQuantKernel:
        """
        构建量化内核。
        Build the quantization kernel.
        """
        ...

    @abstractmethod
    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """
        生成配置的目录名称。
        Generate the directory names of the configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix for the directory names.

        Returns:
            `list[str]`:
                The directory names.
        """
        ...


@configclass
@dataclass
class BaseKeyEnableQuantKernelConfig(KeyEnableConfig, EnableConfig):
    """
    数据类，用于配置量化内核。
    它继承自 KeyEnableConfig 和 EnableConfig，结合了基于键的启用配置和整体启用配置的功能。
    该类管理多个量化内核配置，通过关键字（键）进行启用和专门化配置，并提供生成配置目录名称的方法。
    Configuration for quantization kernel.
    """

    # 存储配置的名称列表
    _names: list[str] = field(init=False, repr=False, compare=False, default_factory=list)
    # 将键（字符串）映射到量化内核配置的字典
    _kernels: dict[str, BaseQuantKernelConfig | None] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        """
        在数据类的初始化方法之后调用，用于组织和整理配置。
        """
        self.organize()

    def is_enabled(self) -> bool:
        """
        检查是否有任何量化内核配置被启用。
        """
        return bool(self._kernels)

    def is_enabled_for(self, key: str) -> bool:
        """
        检查特定键是否有对应的量化内核配置被启用。
        
        Args:
            key: 要检查的键。
        """
        return key in self._kernels

    def specialize_for(self, key: str) -> BaseQuantKernelConfig | None:
        """
        获取特定键对应的量化内核配置。
        Get the kernel configuration for the module key.

        Args:
            key (`str`):
                The key.

        Returns:
            `QuantKernelConfig` or `None`:
                The kernel configuration for the key.
        """
        # 返回键对应的量化内核配置，如果不存在则返回 None
        return self._kernels.get(key, None)

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """
        生成配置的目录名称列表。
        Generate the directory names of the configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix for the directory names.

        Returns:
            `list[str]`:
                The directory names.
        """
        # 初始化存储目录名称的列表
        names = []
        # 检查是否有任何量化内核配置被启用
        if self.is_enabled():
            # 遍历_names列表中的每个配置名称
            for name in self._names:
                # 获取对应的量化内核配置
                config: IncludeBasedConfig = getattr(self, name)
                # 如果配置对象不为None且已启用，则生成目录名称，并添加到列表中
                if config is not None and config.is_enabled():
                    names.extend(config.generate_dirnames(prefix=prefix, **kwargs))
        return names

    def organize(self) -> None:
        """
        组织和整理配置，将配置对象按键映射到 _kernels 字典中。
        Organize the configuration.
        """
        # 清空_kernels字典，确保重新组织前没有旧的数据
        self._kernels.clear()
        # 遍历数据类的所有字段
        for _field in fields(self):
            # 获取字段名称
            name = _field.name
            # 如果字段名称以下划线开头，则跳过，即不处理私有字段
            if name.startswith("_"):
                continue
            # 将字段名称添加到_names列表中
            self._names.append(name)
            # 获取字段对应的配置对象
            config = getattr(self, name)
            if config is not None:
                # 确保配置对象是IncludeBasedConfig和BaseQuantKernelConfig的实例
                assert isinstance(
                    config, IncludeBasedConfig
                ), f"Field '{name}' must be an instance of IncludeBasedConfig."
                assert isinstance(
                    config, BaseQuantKernelConfig
                ), f"Field '{name}' must be an instance of BaseQuantKernelConfig."
                # 如果配置对象已启用，则将其include列表中的键映射到_kernels字典中
                if config.is_enabled():
                    for key in config.includes:
                        assert (
                            key not in self._kernels
                        ), f"Key '{key}' is already included in other kernel configurations."
                        self._kernels[key] = config
                # 如果配置对象未启用，则将该字段设置为None
                else:
                    setattr(self, name, None)
                    continue
