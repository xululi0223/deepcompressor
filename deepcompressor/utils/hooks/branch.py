# -*- coding: utf-8 -*-
"""Branch hook module."""

import typing as tp

import torch
import torch.nn as nn

from .hook import IOHook
from .packager import BaseInputPackager, BaseOutputPackager

__all__ = ["AccumBranchHook"]


class AccumBranchHook(IOHook):
    """
    继承自 IOHook，用于在模型的前向传播过程中对特定模块的输入和输出进行处理。
    具体来说，它在前向传播前记录输入张量，并在前向传播后通过可选的 branch 模块对输出张量进行累加操作。
    该钩子主要用于实现分支处理逻辑，例如在模型中添加辅助路径以增强特定层的表示能力。
    """
    # 可选的分支模块，用于处理和累加输出张量
    branch: nn.Module | None

    def __init__(
        self,
        branch: nn.Module | None,
        input_packager: BaseInputPackager | None = None,
        output_packager: BaseOutputPackager | None = None,
    ):
        """
        Args:
            branch: 用于处理输出张量的分支模块。
            input_packager: 输入包装器。
            output_packager: 输出包装器。
        """
        # 调用父类构造函数，设置 pre 和 post 为 True
        super().__init__(pre=True, post=True, input_packager=input_packager, output_packager=output_packager)
        self.branch = branch
        # 用于存储前向传播前的输入张量
        self.tensor = None

    def pre_forward(
        self, module: nn.Module, input_args: tuple[torch.Tensor, ...], input_kwargs: dict[str, tp.Any]
    ) -> None:
        """
        在前向传播前调用，用于记录当前模块的输入张量。
        Pre-forward function.

        Args:
            module (nn.Module): Module.
            input_args (tuple[torch.Tensor, ...]): Input arguments.
            input_kwargs (dict[str, tp.Any]): Input keyword arguments.
        """
        # 调用输入包装器的 unpack 方法，解包输入张量
        tensors = self.input_packager.unpack(module, input_args, input_kwargs)
        # 确保输入张量只有一个
        assert len(tensors) == 1, "BranchHook only supports single input tensor"
        # 从tensors中提取唯一的输入张量
        self.tensor = next(iter(tensors.values()))
        return None

    def post_forward(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tuple[torch.Tensor, ...],
    ) -> tp.Any:
        """
        在前向传播后调用，用于对输出张量进行处理和累加操作。
        Post-forward function.

        Args:
            module (nn.Module): Module.
            input_args (tuple[torch.Tensor, ...]): Input arguments.
            input_kwargs (dict[str, tp.Any]): Input keyword arguments.
            output (tuple[torch.Tensor, ...]): Output.
        """
        # 调用输出包装器的 unpack 方法，解包输出张量
        output_tensors = self.output_packager.unpack(module, input_args, input_kwargs, output)
        # 确保输出张量只有一个
        assert len(output_tensors) == 1, "LoRAHook only supports single output tensor"
        # 从output_tensors中提取唯一的输出张量
        output_key, output_tensor = next(iter(output_tensors.items()))
        # 如果分支模块不为空，则对输出张量进行累加操作
        if self.branch is not None:
            output_tensor = output_tensor + self.branch(self.tensor)
        # 重置 self.tensor，以清楚之前记录的输入张量
        self.tensor = None
        # 调用输出包装器的 repack 方法，对处理后的输出张量进行打包
        return self.output_packager.repack({output_key: output_tensor}, module, input_args, input_kwargs, output)
