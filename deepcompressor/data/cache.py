# -*- coding: utf-8 -*-
"""Activation cache module."""

import math
import typing as tp
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from ..utils.common import tree_map
from .utils.reshape import ReshapeFn

__all__ = ["TensorCache", "TensorsCache", "IOTensorsCache"]


@dataclass
class ModuleForwardInput:
    """
    用于封装模块的前向输入，包括位置参数 (args) 和关键字参数 (kwargs)。
    该类提供了将输入移动到指定设备的方法以及更新输入值的方法，便于在模型推理或训练过程中灵活管理输入数据。
    Module forward input.
    """

    # 位置参数列表
    args: list[tp.Any] = field(default_factory=list)
    # 关键字参数字典
    kwargs: dict[str, tp.Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "ModuleForwardInput":
        """
        将输入数据移动到指定的设备。
        Move input to device.

        Args:
            device (`torch.device` or `str`):
                Device.

        Returns:
            `ModuleForwardInput`:
                Module forward input.
        """
        # 生成并返回一个新的 ModuleForwardInput 对象，其中的数据已经移动到指定设备
        return ModuleForwardInput(
            # 使用tree_map函数将args和kwargs中的所有数据移动到指定设备
            args=tree_map(lambda x: x.to(device=device), self.args),
            kwargs=tree_map(lambda x: x.to(device=device), self.kwargs),
        )

    def update(self, x: dict[str | int, tp.Any] | None = None) -> "ModuleForwardInput":
        """
        更新args和kwargs中的值，返回一个新的 ModuleForwardInput 对象。
        Return a new ModuleForwardInput with updated values.

        Args:
            x (`dict[str | int, tp.Any]` or `None`, *optional*, defaults to `None`):
                Values to update.

        Returns:
            `ModuleForwardInput`:
                Module forward input.
        """
        # 使用tree_map函数将args和kwargs中的所有数据复制到新的列表和字典中
        args, kwargs = tree_map(lambda x: x, self.args), tree_map(lambda x: x, self.kwargs)
        # 遍历x中的所有键值对，将其添加到args或kwargs中
        if x is not None:
            for k, v in x.items():
                if isinstance(k, int):
                    args[k] = v
                else:
                    kwargs[k] = v
        # 生成并返回一个新的 ModuleForwardInput 对象，其中的数据已经更新
        return ModuleForwardInput(args=args, kwargs=kwargs)


@dataclass
class TensorCache:
    """
    用于缓存和管理一组张量数据。
    它支持对张量进行清理、标准化以及重新分区，以适应不同的批处理大小和总大小限制。
    此外，TensorCache 还记录了缓存中张量的数量、总数及样本数，并保持了张量的原始设备信息。
    Tensor cache.

    Args:
        data (`list[torch.Tensor]`):
            Cached tensors.
        channels_dim (`int`, *optional*, defaults to `1`):
            Channels dimension.
        reshape (`ReshapeFn`, *optional*, defaults to `ReshapeFn()`):
            Function for reshaping inputs to 2-dimension used for GEMM.

        num_cached (`int`, *optional*, defaults to `0`):
            Number of cached tensors.
        num_total (`int`, *optional*, defaults to `0`):
            Number of total tensors.
        num_samples (`int`, *optional*, defaults to `0`):
            Number of samples.
        orig_device (`torch.device` or `str`, *optional*, defaults to `torch.device("cpu")`):
            Original device.
    """

    # 存储缓存的张量列表
    data: list[torch.Tensor] = field(default_factory=list)
    # 指定通道维度
    channels_dim: int = 1
    # 用于将输入重塑为二维的函数
    reshape: ReshapeFn = ReshapeFn()

    # 缓存中张量的数量
    num_cached: int = 0
    # 总张量数量
    num_total: int = 0
    # 样本数量
    num_samples: int = 0
    # 张量原始所在设备
    orig_device: torch.device | str = torch.device("cpu")

    def clear(self):
        """
        用于清空缓存中的张量，并重置 num_cached 计数。
        Clear cached tensors.
        """
        self.data.clear()
        self.num_cached = 0

    def get_factory_kwargs(self, **kwargs) -> dict[str, tp.Any]:
        """
        用于获取创建 TensorCache 实例时所需的关键字参数。
        Get factory kwargs.
        """
        # 使用setdefault方法设置关键字参数的默认值
        kwargs.setdefault("channels_dim", self.channels_dim)
        kwargs.setdefault("reshape", self.reshape)
        kwargs.setdefault("num_cached", self.num_cached)
        kwargs.setdefault("num_total", self.num_total)
        kwargs.setdefault("orig_device", self.orig_device)
        return kwargs

    def get_standardized_data(self, reshape: bool = False) -> list[torch.Tensor]:
        """
        用于获取标准化后的张量数据，即在 channels_dim 之前的维度被展平。
        Get standardized data, i.e., flatten dimensions before `channels_dim`.

        Args:
            reshape (`bool`, *optional*, defaults to `False`):
                Whether to apply reshape function.

        Returns:
            `list[torch.Tensor]`:
                Standardized data.
        """
        # 如果 reshape 为 True，则对每个张量先将channels_dim之前的维度展平，然后再使用 reshape 函数进行重塑
        if reshape:
            return [self.reshape(x.view(-1, *x.shape[self.channels_dim :])) for x in self.data]
        # 否则，只对每个张量将channels_dim之前的维度展平
        else:
            return [x.view(-1, *x.shape[self.channels_dim :]) for x in self.data]

    def repartition(self, max_batch_size: int, max_size: int, standardize: bool, reshape: bool) -> "TensorCache":
        """
        根据指定的最大批量大小和总大小，对缓存中的数据进行重新分区和整理。
        Relocate data based on the maximum batch size and size.

        Args:
            max_batch_size (`int`):
                Maximum batch size.
            max_size (`int`):
                Maximum size.
            standardize (`bool`):
                Whether to standardize data, i.e., flatten dimensions before `channels_dim`.
            reshape (`bool`):
                Whether to apply reshape function.

        Returns:
            `TensorCache`:
                Tensor cache.
        """
        # 验证 max_batch_size 和 max_size 的值是否合法，验证缓存数据是否为空，验证所有张量的维度和形状是否一致
        assert len(self.data) > 0, "No data to relocate."
        assert max_batch_size != 0, "max_batch_size must be non-zero."
        assert max_size != 0, "max_size must be non-zero."
        assert all(x.ndim == self.data[0].ndim for x in self.data), "All tensors must have the same #dims."
        assert all(x.shape == self.data[0].shape for x in self.data), "All tensors must have the same shape."

        # 获取缓存中的数据、通道维度、reshape函数
        data, dim, fn = self.data, self.channels_dim, self.reshape
        # 如果 standardize 为 True
        if standardize:
            # 对每个张量先将channels_dim之前的维度展平
            data = [x.view(-1, *x.shape[dim:]) for x in self.data]
            dim = 1
            if reshape:
                # 如果 reshape 为 True，则对每个张量再使用 reshape 函数进行重塑
                data = [fn(x) for x in data]
                dim = -1
                fn = ReshapeFn()
        # 确保dim不超过张量的维度数
        dim = dim % data[0].ndim
        # 计算原始数据的总大小
        orig_total = data[0].shape[0] * len(data)
        if max_batch_size > 0:
            # 获取当前批量大小
            batch_size = data[0].shape[0]
            if batch_size > max_batch_size:
                # 如果当前批量大小大于最大批量大小，则对每个张量进行切片，将其拆分为多个批次，每个批次大小为max_batch_size
                data = [
                    x[i * max_batch_size : (i + 1) * max_batch_size]
                    for x in data
                    for i in range(int(batch_size // max_batch_size))
                ]
            # 更新batch_size为切片后的新批量大小
            batch_size = data[0].shape[0]
            # 如果当前总大小超过max_size，则以步长为int(len(data) // (max_size // batch_size))对数据进行抽样
            if max_size > 0 and batch_size * len(data) > max_size:
                assert max_size >= batch_size, "max_size must be greater than or equal to batch_size."
                data = data[:: int(len(data) // (max_size // batch_size))]
        else:
            assert max_size < 0, "max_size must be negative if max_batch_size is negative."
        # 计算重新分区后的数据总大小
        used_total = data[0].shape[0] * len(data)
        # 计算数据使用率
        ratio = used_total / orig_total
        # 创建并返回一个新的 TensorCache 对象，其中包含重新分区后的数据及更新后的统计信息
        return TensorCache(
            data,
            channels_dim=dim,
            reshape=fn,
            orig_device=self.orig_device,
            num_cached=int(math.ceil(ratio * self.num_cached)),
            num_total=int(math.ceil(ratio * self.num_total)),
            num_samples=int(math.ceil(ratio * self.num_samples)),
        )


class TensorsCache:
    """
    用于管理多个 TensorCache 实例，支持通过键（字符串或整数）访问各个缓存。
    它提供了基本的容器操作方法，如获取数量、遍历、索引访问、清理缓存以及设置样本数量。
    此外，TensorsCache 还提供了 extract 方法，用于根据索引和关键字参数提取数据，生成 ModuleForwardInput 实例，以便将数据绑定到模块的前向传递中。
    Tensors cache.
    """

    # 存储多个 TensorCache 实例的有序字典
    tensors: OrderedDict[str | int, TensorCache]

    def __init__(self, tensors: OrderedDict[str | int, TensorCache] | TensorCache) -> None:
        """
        初始化方法。
        Post initialization.
        
        Args:
            tensors: 一个 TensorCache 实例或多个 TensorCache 实例的有序字典。
        """
        # 如果tensors是TensorCache实例，则将其存储在有序字典中
        self.tensors = OrderedDict({0: tensors}) if isinstance(tensors, TensorCache) else tensors

    @property
    def num_tensors(self) -> int:
        """
        返回当前管理的 TensorCache 实例的数量。
        Get the number of tensor caches.
        """
        return len(self.tensors)

    def front(self) -> TensorCache:
        """
        返回 tensors 中第一个 TensorCache 实例。
        Get the first tensor cache.
        """
        return next(iter(self.tensors.values()))

    def items(self) -> tp.ItemsView[str | int, TensorCache]:
        """
        返回 tensors 的所有键值对视图，便于迭代。
        Iterate over tensor caches.
        """
        return self.tensors.items()

    def keys(self) -> tp.KeysView[str | int]:
        """
        返回 tensors 中所有的键视图。
        Get tensor cache keys.
        """
        return self.tensors.keys()

    def values(self) -> tp.ValuesView[TensorCache]:
        """
        返回 tensors 中所有的值视图。
        Get tensor caches.
        """
        return self.tensors.values()

    def __getitem__(self, key: str | int) -> TensorCache:
        """
        允许通过索引方式（如 cache[key]）访问对应的 TensorCache 实例。
        Get tensor cache.
        """
        return self.tensors[key]

    def __iter__(self) -> tp.Iterator[TensorCache]:
        """
        使 TensorsCache 实例可迭代，遍历过程中返回 TensorCache 实例。
        Iterate over tensor caches.
        """
        return iter(self.tensors.values())

    def __len__(self) -> int:
        """
        返回 TensorsCache 中 TensorCache 实例的数量，支持 len(cache) 语法。
        Get the number of tensor caches.
        """
        return len(self.tensors)

    def clear(self):
        """
        清空所有管理的 TensorCache 实例中的缓存数据。
        Clear cached tensors.
        """
        for tensor in self.tensors.values():
            tensor.clear()

    def set_num_samples(self, num_samples: int):
        """
        设置所有 TensorCache 实例的 num_samples 属性。
        Set the number of samples.
        """
        for tensor in self.tensors.values():
            tensor.num_samples = num_samples

    def extract(self, index: int, kwargs: dict[str, tp.Any]) -> ModuleForwardInput:
        """
        根据给定的索引和关键字参数，从缓存中提取数据，并生成一个 ModuleForwardInput 实例，以便将数据绑定到模块的前向传递中。
        Extract data for binding to module forward.

        Args:
            index (`int`):
                Index.
            kwargs (`dict[str, tp.Any]`):
                Keyword arguments.

        Returns:
            `ModuleForwardInput`:
                Module forward input.
        """
        # 初始化位置参数列表和关键字参数字典
        _args, _kwargs = [], {}
        # 将传入的关键字参数添加到_kwargs中
        _kwargs.update(kwargs)
        # 遍历所有TensorCache实例
        for key, tensor in self.tensors.items():
            # 如果key是整数，则将对应位置的张量数据添加到_args中
            if isinstance(key, int):
                assert len(_args) == key, f"Expected {key} args, but got {len(_args)}"
                _args.append(tensor.data[index].to(tensor.orig_device, non_blocking=True))
            # 否则，将对应键的张量数据添加到_kwargs中
            else:
                _kwargs[key] = tensor.data[index].to(tensor.orig_device, non_blocking=True)
        return ModuleForwardInput(args=_args, kwargs=_kwargs)


class IOTensorsCache:
    """
    用于同时管理输入和输出的张量缓存。
    它包含两个属性：inputs 和 outputs，分别对应输入数据和输出数据的缓存。
    该类提供了清理缓存和设置样本数量的方法，便于在处理模型输入输出时统一管理缓存数据。
    Input and output cache.
    """

    # 缓存输入张量
    inputs: TensorsCache | None
    # 缓存输出张量
    outputs: TensorsCache | None

    def __init__(
        self, inputs: TensorCache | TensorsCache | None = None, outputs: TensorCache | TensorsCache | None = None
    ):
        """
        Args:
            inputs: 输入数据的缓存。
            outputs: 输出数据的缓存。
        """
        # 如果inputs和outputs是TensorCache实例，则将其包装为TensorsCache实例
        self.inputs = TensorsCache(inputs) if isinstance(inputs, TensorCache) else inputs
        self.outputs = TensorsCache(outputs) if isinstance(outputs, TensorCache) else outputs

    def clear(self):
        """
        清空 inputs 和 outputs 中的所有缓存数据。
        Clear cached tensors.
        """
        if self.inputs is not None:
            self.inputs.clear()
        if self.outputs is not None:
            self.outputs.clear()

    def set_num_samples(self, num_samples: int):
        """
        设置 inputs 和 outputs 中所有 TensorCache 实例的 num_samples 属性。
        Set the number of samples.
        """
        if self.inputs is not None:
            self.inputs.set_num_samples(num_samples)
        if self.outputs is not None:
            self.outputs.set_num_samples(num_samples)
