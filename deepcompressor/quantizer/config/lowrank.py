# -*- coding: utf-8 -*-

from dataclasses import dataclass

from omniconfig import configclass

from ...utils.common import num2str
from ...utils.config import EnableConfig

__all__ = ["QuantLowRankConfig"]


@configclass
@dataclass
class QuantLowRankConfig(EnableConfig):
    """
    数据类，用于配置量化中的低秩分支（Low-Rank Branch）。
    该类继承自 EnableConfig，结合了启用配置的功能，允许用户通过配置参数控制低秩分支的启用与特性。
    主要配置参数包括 rank、exclusive 和 compensate，分别用于定义低秩分支的秩、是否为每个权重使用独立的低秩分支，以及是否补偿量化误差。
    Quantization low-rank branch configuration.

    Args:
        rank (`int`, *optional*, defaults to `32`):
            The rank of the low-rank branch.
        exclusive (`bool`, *optional*, defaults to `False`):
            Whether to use exclusive low-rank branch for each weight sharing the inputs.
        compensate (`bool`, *optional*, defaults to `False`):
            Whether the low-rank branch compensates the quantization error.
    """

    # 低秩分支的秩
    rank: int = 32
    # 是否为每个权重使用独立的低秩分支
    exclusive: bool = False
    # 是否补偿量化误差
    compensate: bool = False

    def is_enabled(self) -> bool:
        """
        判断低秩分支配置是否启用。
        """
        return self.rank != 0

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """
        生成当前配置对应的目录名称列表，用于配置文件或日志的组织。
        Generate the directory names of the configuration.

        Returns:
            list[str]: The directory names.
        """
        # 如果低秩分支配置未启用，则返回空列表
        if not self.is_enabled():
            return []

        # 生成目录名称
        # 添加低秩分支的秩
        name = f"r{num2str(self.rank)}"
        # 添加是否为每个权重使用独立的低秩分支
        if self.exclusive:
            name += ".exclusive"
        # 添加是否补偿量化误差
        if self.compensate:
            name += ".compensate"
        # 添加前缀，返回目录名称列表
        return [f"{prefix}.{name}" if prefix else name]
