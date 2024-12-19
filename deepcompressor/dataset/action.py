# -*- coding: utf-8 -*-
"""Actions for caching inputs and outputs."""

import typing as tp
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..data.cache import IOTensorsCache, TensorsCache
from ..utils.hooks import BaseInputPackager, BaseOutputPackager, Hook, IOHook, KeyedInputPackager, KeyedOutputPackager

__all__ = ["CacheAction", "ConcatCacheAction"]


class CacheHook(IOHook):
    """
    继承自 IOHook，用于在 PyTorch 模型的前向传播过程中，通过注册钩子（hook）来缓存模块的输入和输出张量。
    它结合了 CacheAction，实现了对激活（activations）的缓存操作，包括信息更新和实际缓存。
    """
    def __init__(
        self, name: str, module: nn.Module, action: "CacheAction", cache: TensorsCache, info_mode: bool, is_output: bool
    ):
        """Initialize the hook.

        Args:
            name (``str``):
                Module name.
            module (``nn.Module``):
                Module.
            action (``CacheAction``):
                Cache action.
            cache (``TensorsCache``):
                Cache.
            info_mode (``bool``):
                Whether to update cache information.
            is_output (``bool``):
                Whether the hook is an output hook.
        """
        # 调用父类的构造函数
        super().__init__(
            pre=not is_output,
            post=is_output,
            input_packager=None if is_output else action.get_input_packager(name, module, cache),
            output_packager=action.get_output_packager(name, module, cache) if is_output else None,
        )
        
        # 属性赋值
        self.name = name
        self.action = action
        self.cache = cache
        self.info_mode = info_mode

    def pre_forward(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
    ) -> None:
        """
        在模块前向传播之前执行，用于处理和缓存输入张量。
        
        Args:
            module: 当前执行前向传播的模块。
            input_args: 模块的输入张量，作为位置参数传入。
            input_kwargs: 模块的输入张量，作为关键字参数传入。
        """
        # 解包输入张量
        tensors = self.input_packager.unpack(module, input_args, input_kwargs)
        # 信息模式处理：调用action的info方法，仅更新缓存的相关信息，而不执行实际的缓存操作
        if self.info_mode:
            self.action.info(self.name, module, tensors, self.cache)
        # 确保解包后的张量数量与缓存预期的数量一致
        assert len(tensors) == self.cache.num_tensors, f"Expected {self.cache.num_tensors} args, but got {len(tensors)}"
        # 非信息模式处理：调用action的apply方法，执行实际的缓存操作
        if not self.info_mode:
            self.action.apply(self.name, module, tensors, self.cache)

    def post_forward(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        """
        在模块前向传播之后执行，用于处理和缓存输出张量。
        
        Args:
            module: 当前执行前向传播的模块。
            input_args: 模块的输入张量，作为位置参数传入。
            input_kwargs: 模块的输入张量，作为关键字参数传入。
            output: 模块的输出张量。
        """
        # 解包输出张量
        tensors = self.output_packager.unpack(module, input_args, input_kwargs, output)
        # 信息模式处理：调用action的info方法，仅更新缓存的相关信息，而不执行实际的缓存操作
        if self.info_mode:
            self.action.info(self.name, module, tensors, self.cache)
        # 确保解包后的张量数量与缓存预期的数量一致
        assert len(tensors) == self.cache.num_tensors, f"Expected {self.cache.num_tensors} args, but got {len(tensors)}"
        # 非信息模式处理：调用action的apply方法，执行实际的缓存操作
        if not self.info_mode:
            self.action.apply(self.name, module, tensors, self.cache)


class CacheAction(ABC):
    """
    抽象基类（ABC），定义了用于缓存模块激活（输入和输出）的抽象方法。
    它为具体的缓存策略（如 ConcatCacheAction）提供了接口和一些通用的实现方法，包括获取输入输出数据的打包器和注册钩子的方法。
    Actions for caching activations.
    """

    # 指定缓存数据存储的设备
    device: torch.device | None = None

    def __init__(self, device: torch.device | str | None = None) -> None:
        """Initialize the action.

        Args:
            device (`torch.device or `str` or `None, *optional*, defaults to `None`):
                Device for caching.
        """
        self.device = device

    @abstractmethod
    def apply(
        self,
        name: str,
        module: nn.Module,
        tensors: dict[int | str, torch.Tensor],
        cache: TensorsCache,
    ) -> None:
        """
        用于在前向传播期间执行具体的缓存操作。具体实现由子类提供。
        Cache activations.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.
            tensors (`dict[int or str, torch.Tensor]`):
                Tensors to cache.
            cache (`TensorsCache`):
                Cache.
        """
        ...

    @abstractmethod
    def info(
        self,
        name: str,
        module: nn.Module,
        tensors: dict[int | str, torch.Tensor],
        cache: TensorsCache,
    ) -> None:
        """
        用于在信息模式下更新缓存的相关信息，而不执行实际的缓存操作。具体实现由子类提供。
        Update cache information.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.
            tensors (`dict[int or str, torch.Tensor]`):
                Tensors to cache.
            cache (`TensorsCache`):
                Cache.
        """
        ...

    def get_input_packager(self, name: str, module: nn.Module, cache: TensorsCache) -> BaseInputPackager:
        """
        返回一个用于打包输入数据的包装器（BaseInputPackager 实例）。
        Get input packager.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.
            cache (`TensorsCache`):
                Cache.

        Returns:
            `BaseInputPackager`:
                Input packager.
        """
        return KeyedInputPackager(module=module, index_or_keys=list(cache.keys()))

    def get_output_packager(self, name: str, module: nn.Module, cache: TensorsCache) -> BaseOutputPackager:
        """
        返回一个用于打包输出数据的包装器（BaseOutputPackager 实例）。
        Get output packager.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.
            cache (`TensorsCache`):
                Cache.

        Returns:
            `BaseOutputPackager`:
                Output packager.
        """
        return KeyedOutputPackager(index_or_keys=list(cache.keys()))

    def register(
        self,
        name: str,
        module: nn.Module,
        cache: IOTensorsCache,
        info_mode: bool,
        needs_inputs: bool,
        needs_outputs: bool,
    ) -> list[Hook]:
        """
        根据需求注册输入和/或输出钩子，用于在模块前向传播过程中缓存相应的张量。
        Register hooks for caching activations.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.
            cache (`IOTensorsCache`):
                Cache.
            info_mode (`bool`):
                Whether to update cache information.
            needs_inputs (`bool`):
                Whether to cache inputs.
            needs_outputs (`bool`):
                Whether to cache outputs.

        Returns:
            `list[Hook]`:
                Cache hooks.
        """
        # 初始化钩子列表
        hooks = []

        # 如果需要缓存输入数据，则注册输入钩子
        if needs_inputs:
            assert cache.inputs is not None
            hooks.append(CacheHook(name, module, self, cache.inputs, info_mode, is_output=False).register(module))
        # 如果需要缓存输出数据，则注册输出钩子
        if needs_outputs:
            assert cache.outputs is not None
            hooks.append(CacheHook(name, module, self, cache.outputs, info_mode, is_output=True).register(module))
        return hooks


class ConcatCacheAction(CacheAction):
    """
    继承自 CacheAction，实现了将缓存的激活张量沿样本维度（通常是第0维）进行拼接的具体缓存策略。
    这种策略适用于校准（calibration）过程中的批量数据处理，通过连续存储样本的激活数据，有助于后续的量化或分析操作。
    Action for concatenating cached activations for calibration.
    """

    def apply(
        self,
        name: str,
        module: nn.Module,
        tensors: dict[int | str, torch.Tensor],
        cache: TensorsCache,
    ) -> None:
        """
        将每个需要缓存的张量沿样本维度拼接到相应的缓存对象中，以便后续的批量处理或分析。
        Concatenate cached activations along the sample dimension.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.
            tensors (`dict[int or str, torch.Tensor]`):
                Tensors to cache.
            cache (`TensorsCache`):
                Cache.
        """
        # 遍历缓存中的张量
        for k, c in cache.tensors.items():
            x = tensors[k]                                          # 获取当前张量
            shape, device = x.shape, self.device or x.device        # 获取张量的形状和设备
            num_prev_cached = c.num_cached                          # 获取之前已缓存的样本数量
            c.num_cached += shape[0]                                # 更新当前已缓存的样本数量
            # 初始化缓存数据存储
            if num_prev_cached == 0:
                assert len(c.data) == 0
                c.data.append(torch.empty((c.num_total, *shape[1:]), dtype=x.dtype, device=device)) # 初始化缓存张量
            c.data[0][num_prev_cached : c.num_cached].copy_(x)      # 将当前张量拼接到缓存张量中

    def info(
        self,
        name: str,
        module: nn.Module,
        tensors: dict[int | str, torch.Tensor],
        cache: TensorsCache,
    ) -> None:
        """
        更新缓存的基本信息，如总样本数量和原始设备位置，而不实际缓存张量数据。
        Update cache information.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.
            tensors (`dict[int or str, torch.Tensor]`):
                Tensors to cache.
            cache (`TensorsCache`):
                Cache.
        """
        # 遍历缓存中的张量
        for k, c in cache.tensors.items():
            x = tensors[k]                  # 获取当前张量
            c.num_total += x.shape[0]       # 更新总样本数量
            c.orig_device = x.device        # 更新原始设备位置
