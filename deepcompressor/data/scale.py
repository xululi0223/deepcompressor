# -*- coding: utf-8 -*-
"""Quantization scale module."""

import typing as tp

import torch

__all__ = ["QuantScale"]


class QuantScale:
    """
    用于量化过程中管理和操作缩放（scale）张量的模块。
    QuantScale 类在此过程中负责管理缩放因子，确保量化操作的准确性和效率。
    """
    # 存储当前量化缩放的张量
    data: torch.Tensor
    # 一个包含子QuantScale的列表，用于构建层级化的量化缩放结构
    _children: list["QuantScale"]
    # 一个包含张量的列表，用于存储叶子节点的量化缩放因子
    _leaves: list[torch.Tensor]

    def __init__(self):
        self.data, self._children, self._leaves = None, [], []  # type: ignore

    @property
    def num_children(self) -> int:
        """
        返回当前QuantScale的子节点数量。
        Get the number of children.
        """
        return len(self._children)

    @property
    def num_leaves(self) -> int:
        """
        返回当前QuantScale的叶子节点数量。
        Get the number of leaves.
        """
        return len(self._leaves)

    def is_quantized(self) -> bool:
        """
        检查当前QuantScale是否已经量化。
        Check if the scale is quantized.
        """
        return self.data is not None and bool(self._leaves or all(child.is_quantized() for child in self._children))

    def get_child(self, index: int) -> "QuantScale":
        """
        获取制定索引的子节点。
        Get a child scale.
        """
        return self._children[index]

    def append(self, scale: tp.Union[torch.Tensor, "QuantScale"]) -> "QuantScale":
        """
        向当前QuantScale添加一个新的缩放因子。
        Append a scale.
        """
        # 如果scale是张量
        if isinstance(scale, torch.Tensor):
            # 确保当前QuantScale是叶子节点
            assert not self._children, "Cannot append a tensor scale to a non-leaf QuantScale."
            # 将现有的data与新的scale进行合并
            self.data = _join_scale_tensor(self.data, scale)
            # 添加到叶子节点列表
            self._leaves.append(scale)
        # 如果scale是QuantScale
        elif isinstance(scale, QuantScale):
            # 确保当前QuantScale是非叶子节点
            assert not self._leaves, "Cannot append a non-leaf QuantScale to a leaf QuantScale."
            # 将现有的data与新的scale进行合并
            self.data = _join_scale_tensor(self.data, scale.data)
            # 添加到子节点列表
            self._children.append(scale)
        else:
            raise TypeError(f"Unsupported scale type: {type(scale)}")
        return self

    def extend(self, scale: "QuantScale") -> "QuantScale":
        """
        将另一个QuantScale对象合并到当前QuantScale中。
        Extend with another QuantScale.
        """
        # 合并缩放因子
        self.data = _join_scale_tensor(self.data, scale.data)
        # 处理子对象与叶子节点
        if scale._children:
            assert not self._leaves, "Cannot extend a leaf QuantScale with a non-leaf QuantScale."
            self._children.extend(scale._children)
        elif scale._leaves:
            assert not scale._children, "Cannot extend a non-leaf QuantScale with a leaf QuantScale."
            self._leaves.extend(scale._leaves)
        return self

    def join(self, scale: "QuantScale") -> "QuantScale":
        """
        创建一个新的 QuantScale 对象，并将当前对象与另一个 QuantScale 对象连接起来。
        Return a new QuantScale by joining with another QuantScale.
        """
        return QuantScale().append(self).append(scale)

    def remove_zero(self) -> "QuantScale":
        """
        将缩放因子中所有等于零的元素替换为1，避免除以零的情况。
        Remove zero scales.
        """
        self.data[self.data == 0] = 1
        return self

    def state_dict(
        self,
        param_name: str,
        device: torch.device | str = "cpu",
        flatten: bool = True,
        level_base: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        获取当前 QuantScale 对象的状态字典（state_dict），包含所有叶子节点的缩放因子
        Get the state dictionary.

        Args:
            param_name: 参数名称的前缀，用于在状态字典中标识不同的缩放因子。
            device: 用于将缩放因子移动到指定设备。
            flatten: 是否降层级化结构扁平化为单一字典。
            level_base: 用于生成参数名称的基数。
        """
        # 处理子节点
        if self._children:
            # 初始化状态字典
            state_dict = {}
            # 遍历子节点
            for i, child in enumerate(self._children):
                # 生成子节点的参数名称
                child_param_name = param_name if flatten else f"{param_name}.{i}"
                # 计算子节点的等级基数
                child_level_base = len(state_dict) if flatten else 0
                # 递归获取子节点的状态字典
                child_state_dict = child.state_dict(child_param_name, device, flatten, child_level_base)
                # 更新主状态字典
                state_dict.update(child_state_dict)
            return state_dict
        else:
            # 返回叶子节点的状态字典
            return {f"{param_name}.{level_base + i}": leaf.to(device) for i, leaf in enumerate(self._leaves)}


def _join_scale_tensor(global_scale: torch.Tensor | None, local_scale: torch.Tensor) -> torch.Tensor:
    """
    将全局缩放张量（global_scale）与局部缩放张量（local_scale）相乘，生成复合的缩放张量。
    Multiply the local scale tensor by the global scale tensor.

    Args:
        global_scale (`torch.Tensor` or `None`):
            Global scale tensor.
        local_scale (`torch.Tensor`):
            Local scale tensor.

    Returns:
        `torch.Tensor`:
            The compounded scale tensor.
    """
    # global_scale: (#gs_g0, 1, #gs_g1, 1, #gs_g2, 1, ...)
    # local_scale:  (#ss_g0, 1, #ss_g1, 1, #ss_g2, 1, ...) -> (#gs_g0, rs0, #gs_g1, rs1, #gs_g2, rs2, ...)
    # 保存原始形状
    shape = local_scale.shape
    return (
        local_scale                     # 如果global_scale为None，则直接返回local_scale
        if global_scale is None
        else local_scale.view(
            tuple(
                global_scale.shape[i] if j == 0 else local_scale.shape[i] // global_scale.shape[i]
                for i in range(0, len(global_scale.shape), 2)
                for j in range(2)
            )
        ).mul(global_scale)             # 否则，将local_scale重塑为与global_scale相同的形状，并与global_scale相乘
    ).view(shape)                       # 最后将结果重塑为原始形状
