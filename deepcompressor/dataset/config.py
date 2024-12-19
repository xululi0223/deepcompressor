# -*- coding: utf-8 -*-
"""Configuration for collecting calibration dataset for quantization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from omniconfig import configclass
from torch.utils.data import DataLoader, Dataset

from .cache import BaseCalibCacheLoader

__all__ = ["BaseDataLoaderConfig"]


@configclass
@dataclass(kw_only=True)
class BaseDataLoaderConfig(ABC):
    """
    数据集加载器的配置。
    Configuration for dataset loader.

    Args:
        data (`str`):
            Dataset name.
        num_samples (`int`):
            Number of dataset samples.
        batch_size (`int`):
            Batch size when loading dataset.
    """

    # 数据集的名称，用于指定要加载的特定数据集
    data: str
    # 数据集中的样本数量，控制加载的数据规模
    num_samples: int
    # 每个批次加载的数据样本数量，影响训练或推理的效率和内存使用
    batch_size: int

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """
        生成配置字段的名称。
        Get the names of the configuration fields.

        Args:
            prefix (`str`, *optional*):
                Prefix for the names.

        Returns:
            `list[str]`:
                Names of the configuration.
        """
        # 构建目录名称，格式为“数据集名称.样本数量”
        name = f"{self.data}.{self.num_samples}"
        # 如果有前缀，则添加前缀
        return [f"{prefix}.{name}" if prefix else name]

    @abstractmethod
    def build_dataset(self, *args, **kwargs) -> Dataset:
        """
        构建数据集。
        Build dataset.
        """
        ...

    @abstractmethod
    def build_loader(self, *args, **kwargs) -> DataLoader | BaseCalibCacheLoader:
        """
        构建数据加载器。
        Build data loader.
        """
        ...
