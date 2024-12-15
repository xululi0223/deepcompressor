# -*- coding: utf-8 -*-
"""System tools."""

import psutil
import torch

__all__ = ["get_max_memory_map"]


def _get_visible_gpu_capacity_list() -> list[int]:
    """
    用于获取当前系统中所有可见 GPU 的显存容量列表。
    返回值是一个整数列表，每个元素代表对应 GPU 的总显存容量（单位为 GiB）。
    Get visible GPU capacity list.

    Returns:
        `list[int]`: Visible GPU capacity list.
    """
    return [torch.cuda.get_device_properties(i).total_memory // 1024**3 for i in range(torch.cuda.device_count())]


def _get_ram_capacity() -> int:
    """
    用于获取系统的总内存容量（RAM），单位为 GiB。
    返回值是一个整数，表示总内存的容量。
    Get RAM capacity.

    Returns:
        `int`: RAM capacity in GiB.
    """
    return psutil.virtual_memory().total // 1024**3  # in GiB


def get_max_memory_map(ratio: float = 0.9) -> dict[str, str]:
    """
    用于获取当前系统中所有可见 GPU 和 CPU 的最大可用内存容量映射。
    函数根据输入的 ratio 参数，计算出每个 GPU 和 CPU 的可用内存容量，并将其存储在一个字典中返回。
    默认情况下，ratio 为 0.9，即使用 90% 的总内存。
    Get maximum memory map.

    Args:
        ratio (`float`, *optional*, defaults to `0.9`): The ratio of the maximum memory to use.

    Returns:
        `dict[str, str]`: Maximum memory map.
    """
    # 获取GPU显存容量列表
    gpu_capacity_list = _get_visible_gpu_capacity_list()
    # 获取RAM容量
    ram_capacity = _get_ram_capacity()
    # 计算GPU可用显存容量
    gpu_capacity_list = [str(int(c * ratio)) + "GiB" for c in gpu_capacity_list]
    # 计算CPU可用内存容量
    ram_capacity = str(int(ram_capacity * ratio)) + "GiB"
    # 构建GPU内存映射字典
    ret_dict = {str(idx): gpu_capacity_list[idx] for idx in range(len(gpu_capacity_list))}
    # 添加CPU内存映射
    ret_dict["cpu"] = ram_capacity
    return ret_dict
