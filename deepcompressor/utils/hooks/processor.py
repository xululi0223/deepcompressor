# -*- coding: utf-8 -*-
"""Tensor processor."""

import abc
import typing as tp

import torch
import torch.ao.quantization
import torch.nn as nn
import torch.utils.hooks

from .hook import IOHook
from .packager import BaseInputPackager, BaseOutputPackager

__all__ = ["BaseTensorProcessor", "ProcessHook"]


class BaseTensorProcessor(abc.ABC):
    """
    抽象基类（Abstract Base Class），定义了处理张量的基本接口。
    它包含了一系列抽象方法，强制子类实现具体的处理逻辑。
    此外，提供了将处理器转换为钩子（hook）的方法 as_hook，以便在 PyTorch 模型的前向传播过程中应用自定义的张量处理逻辑。
    """
    @abc.abstractmethod
    def is_enabled(self) -> bool:
        """
        判断处理器是否可用。
        """
        ...

    @abc.abstractmethod
    def get_input_packager(self) -> BaseInputPackager | None:
        """
        获取用于打包输入的包装器。
        """
        ...

    @abc.abstractmethod
    def get_output_packager(self) -> BaseOutputPackager | None:
        """
        获取用于打包输出的包装器。
        """
        ...

    @abc.abstractmethod
    def process(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        对输入的张量进行处理，返回处理后的张量。
        """
        ...

    def as_hook(
        self, func: tp.Callable[[torch.Tensor], torch.Tensor] | None = None, *, is_output: bool = False
    ) -> "ProcessHook":
        """
        将处理器转换为ProcessHook实例，以便在模型的前向传播过程中应用处理逻辑。
        Convert the processor to a hook.

        Args:
            func (`Callable[[torch.Tensor], torch.Tensor]` or `None`, *optional*, defaults to `None`):
                Function to process the tensors.
            is_output (`bool`, *optional*, defaults to `False`):
                Whether to process the output tensors.

        Returns:
            `ProcessHook`:
                The hook for processing the tensor.
        """
        return ProcessHook(self, func, is_output=is_output)


class ProcessHook(IOHook):
    """
    用于在模型的前向传播过程中处理输入或输出的张量。
    它结合了一个 BaseTensorProcessor 实例和一个可选的自定义处理函数 func，通过调用处理器的 process 方法或 func 函数，对指定的张量进行处理。
    该钩子可以配置为在前向传播前处理输入张量，或在后处理输出张量，具体取决于 is_output 参数。
    """
    def __init__(
        self,
        processor: BaseTensorProcessor,
        func: tp.Callable[[torch.Tensor], torch.Tensor] | None = None,
        is_output: bool = False,
    ):
        """
        Args:
            processor: 处理张量的处理器实例。
            func: 自定义的处理函数。
            is_output: 指示钩子是用于处理输出张量还是输入张量。
        """
        super().__init__(
            pre=not is_output,
            post=is_output,
            input_packager=processor.get_input_packager(),
            output_packager=processor.get_output_packager(),
        )
        self.processor = processor
        self.func = func

    def process(self, tensors: dict[int | str, torch.Tensor]) -> dict[int | str, torch.Tensor]:
        """
        对传入的张量字典进行处理，返回处理后的张量字典。
        
        Args:
            tensors: 需要处理的张量字典，键可以是整数或字符串，值是张量。
        """
        # 遍历张量字典，对每个张量进行处理
        for k, x in tensors.items():
            assert isinstance(x, torch.Tensor)
            # 如果self.func不为空，则调用self.func函数，否则调用self.processor.process方法
            if self.func is not None:
                tensors[k] = self.func(x)
            else:
                tensors[k] = self.processor.process(x)
        return tensors

    def pre_forward(
        self, module: nn.Module, input_args: tuple[torch.Tensor, ...], input_kwargs: dict[str, tp.Any]
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, tp.Any]]:
        """
        在前向传播之前调用的钩子方法，用于处理输入张量。

        Args:
            module: 当前处理的模型模块。
            input_args: 输入的张量参数。
            input_kwargs: 输入的关键字参数。
        """
        # 如果处理器不可用，则直接返回输入参数
        if not self.processor.is_enabled():
            return input_args, input_kwargs
        # 否则，调用处理器的输入包装器，对输入张量进行解包，然后调用process方法进行处理，最后调用输出包装器，对处理后的张量进行打包
        return self.input_packager.repack(
            self.process(self.input_packager.unpack(module, input_args, input_kwargs)), module, input_args, input_kwargs
        )

    def post_forward(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tuple[torch.Tensor, ...],
    ) -> tp.Any:
        """
        在前向传播之后调用的钩子方法，用于处理输出张量。
        
        Args:
            module: 当前处理的模型模块。
            input_args: 输入的张量参数。
            input_kwargs: 输入的关键字参数。
            output: 模块的输出。
        """
        # 如果处理器不可用，则直接返回输出参数
        if not self.processor.is_enabled():
            return output
        # 否则，调用处理器的输出包装器，对输出张量进行解包，然后调用process方法进行处理，最后调用输出包装器，对处理后的张量进行打包
        return self.output_packager.repack(
            self.process(self.output_packager.unpack(module, input_args, input_kwargs, output)),
            module,
            input_args,
            input_kwargs,
            output,
        )
