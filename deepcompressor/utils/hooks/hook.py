# -*- coding: utf-8 -*-
"""nn.Module Hook."""

import typing as tp
from collections import defaultdict

import torch
import torch.ao.quantization
import torch.nn as nn
import torch.utils.hooks

from .packager import BaseInputPackager, BaseOutputPackager, SimpleInputPackager, SimpleOutputPackager

__all__ = ["Hook", "EarlyStopException", "EarlyStopHook", "IOHook"]


class Hook:
    """
    所有钩子（hook）类的基类，提供了注册、激活、停用以及调用钩子的基本功能。
    钩子用于在 PyTorch 模型的前向传播过程中插入自定义的逻辑，既可以在前向传播之前（pre-hook），也可以在之后（post-hook）执行。
    Base class for hook.
    """

    handles: dict[nn.Module, list[torch.utils.hooks.RemovableHandle]]   # 字典，用于管理已注册的钩子句柄
    pre: bool                                                           # 表示是否为前向传播前的钩子
    post: bool                                                          # 表示是否为前向传播后的钩子
    activated: bool                                                     # 表示钩子是否处于激活状态

    def __init__(self, *, pre: bool, post: bool) -> None:
        """Initialize the hook.

        Args:
            pre (`bool`):
                Whether the hook should be called before the forward pass.
            post (`bool`):
                Whether the hook should be called after the forward pass.

        Raises:
            AssertionError:
                If both `pre` and `post` are `False`.
        """
        # 初始化handles
        self.handles = defaultdict(list)
        self.pre = pre
        self.post = post
        # 钩子默认处于激活状态
        self.activated = True
        assert self.pre or self.post, "At least one of pre and post must be True."

    def is_in_hook(self) -> bool:
        """
        判断钩子是否仅为前向传播前的钩子。
        Whether the hook is an in-hook.
        """
        return self.pre and not self.post

    def is_out_hook(self) -> bool:
        """
        判断钩子是否仅为前向传播后的钩子。
        Whether the hook is an out-hook.
        """
        return not self.pre and self.post

    def is_inout_hook(self) -> bool:
        """
        判断钩子是否同时为前向传播前后的钩子。
        Whether the hook is an in-out-hook.
        """
        return self.pre and self.post

    def activate(self) -> tp.Self:
        """
        激活钩子，使其能够被调用。
        Activate the hook.
        """
        self.activated = True
        return self

    def deactivate(self) -> tp.Self:
        """
        停用钩子，使其不再被调用。
        Deactivate the hook.
        """
        self.activated = False
        return self

    def pre_forward(
        self, module: nn.Module, input_args: tuple[torch.Tensor, ...], input_kwargs: dict[str, tp.Any]
    ) -> tp.Any:
        """
        在前向传播之前调用的钩子方法。
        子类可重写此方法以实现自定义的逻辑。
        Pre-forward function.

        Args:
            module (`nn.Module`):
                Module to process.
            input_args (`tuple[torch.Tensor, ...]`):
                Input arguments.
            input_kwargs (`dict[str, tp.Any]`):
                Input keyword arguments.
        """
        return None

    def post_forward(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tuple[torch.Tensor, ...],
    ) -> tp.Any:
        """
        在前向传播之后调用的钩子方法。
        子类可重写此方法以实现自定义的逻辑。
        Post-forward function.

        Args:
            module (`nn.Module`):
                Module to process.
            input_args (`tuple[torch.Tensor, ...]`):
                Input arguments.
            input_kwargs (`dict[str, tp.Any]`):
                Input keyword arguments.
            output (`tuple[torch.Tensor, ...]`):
                Output.
        """
        return None

    def __call__(self, *args, **kwargs) -> tp.Any:
        """
        调用钩子。根据传入的参数数量，调用前向传播前或后的钩子方法。
        """
        # 如果钩子未激活，则返回None
        if not self.activated:
            return None
        # 计算传入的位置参数和关键字参数的总数量
        n = len(args) + len(kwargs)
        # 根据参数数量调用前向传播前或后的钩子方法
        if n == 3:
            return self.pre_forward(*args, **kwargs)
        elif n == 4:
            return self.post_forward(*args, **kwargs)
        else:
            raise ValueError(f"Invalid number of arguments: {n}")

    def register(
        self,
        module: nn.Module | tp.Iterable[nn.Module],
        prepend: bool | tuple[bool, bool] = False,
        always_call: bool = False,
    ) -> tp.Self:
        """
        将钩子注册到指定的模块或模块集合中，以便在前向传播过程中被调用。
        Register the hook to the module(s).

        Args:
            module (`nn.Module` or `Iterable[nn.Module]`):
                The module(s).
            prepend (`bool` or `tuple[bool, bool]`, *optional*, defaults to `False`):
                Whether to prepend the hook.
                If a tuple, the first element is for pre-hook and the second element is for post-hook.
            always_call (`bool`, *optional*, defaults to `False`):
                Whether to always call the hook. This is only used for post-hooks.
        """
        # 如果module是单个模块，则转换为列表
        if isinstance(module, nn.Module):
            module = [module]
        # 如果prepend是布尔值，则转换为元组
        prepends = (prepend, prepend) if isinstance(prepend, bool) else prepend
        # 如果pre为True，为每个模块注册前向传播前的钩子，并将句柄添加到handles中
        if self.pre:
            for mod in module:
                self.handles[mod].append(mod.register_forward_pre_hook(self, prepend=prepends[0], with_kwargs=True))
        # 如果post为True，为每个模块注册前向传播后的钩子，并将句柄添加到handles中
        if self.post:
            for mod in module:
                self.handles[mod].append(
                    mod.register_forward_hook(self, prepend=prepends[1], with_kwargs=True, always_call=always_call)
                )
        return self

    def remove(self, module: nn.Module | tp.Iterable[nn.Module] | None = None) -> tp.Self:
        """
        从指定的模块或所有模块中移除已注册的钩子。
        Remove the hook from the module(s).

        Args:
            module (`nn.Module` or `Iterable[nn.Module]`, *optional*, defaults to `None`):
                The module(s) to remove the hook from. If `None`, remove the hook from all modules.
        """
        # 如果module为None，则从所有模块中移除钩子
        if module is None:
            for handles in self.handles.values():
                for handle in handles:
                    handle.remove()
                handles.clear()
            self.handles.clear()
            return self
        # 如果module是单个模块，则转换为列表
        if isinstance(module, nn.Module):
            module = [module]
        # 遍历指定的模块，从每个模块中移除钩子
        for mod in module:
            handles = self.handles.pop(mod, [])
            for handle in handles:
                handle.remove()
            handles.clear()
        return self


class EarlyStopException(Exception):
    """
    自定义异常类，用于在钩子执行过程中实现早期停止机制。
    当某个条件满足时，可以通过抛出此异常来中断前向传播过程。
    Early stop exception.
    """

    pass


class EarlyStopHook(Hook):
    """
    用于在前向传播过程中立即停止执行。
    通常用于调试或特定控制流程的场景，当钩子被调用时，会抛出 EarlyStopException 异常，从而中断前向传播。
    """
    def __init__(self):
        super().__init__(pre=False, post=True)  # 只注册前向传播后的钩子

    def pre_forward(self, *args, **kwargs) -> None:
        """
        重写 pre_forward 方法，抛出 EarlyStopException 异常。
        """
        raise EarlyStopException()


class IOHook(Hook):
    """
    抽象基类，专门用于处理模型的输入和输出。
    它结合了输入包装器（BaseInputPackager）和输出包装器（BaseOutputPackager），提供了一种机制来解包和重新打包前向传播的输入参数和输出结果。
    通过这种方式，可以在前向传播的不同阶段插入自定义逻辑，如数据监控、日志记录或数据修改。
    Base class for IO hooks.
    """

    # 输入的包装器，用于解包和重新打包输入参数
    input_packager: BaseInputPackager
    """Input packager, used to unpack and repack the input arguments."""
    # 输出的包装器，用于解包和重新打包输出结果
    output_packager: BaseOutputPackager
    """Output packager, used to unpack and repack the output."""

    def __init__(
        self,
        *,
        pre: bool,
        post: bool,
        input_packager: BaseInputPackager | None = None,
        output_packager: BaseOutputPackager | None = None,
    ):
        """Initialize the IO hook.

        Args:
            pre (`bool`):
                Whether the hook should be called before the forward pass.
            post (`bool`):
                Whether the hook should be called after the forward pass.
            input_packager (`BaseInputPackager`, *optional*, defaults to `None`):
                Input packager, used to unpack and repack the input arguments.
            output_packager (`BaseOutputPackager`, *optional*, defaults to `None`):
                Output packager, used to unpack and repack the output.
        """
        # 调用父类的构造函数，传递 pre 和 post 参数
        super().__init__(pre=pre, post=post)
        # 如果 pre 为 True，设置 input_packager
        if pre:
            # 如果 input_packager 不为 None，则使用传入的 input_packager，否则使用默认的 SimpleInputPackager
            self.input_packager = input_packager or SimpleInputPackager()
            assert isinstance(self.input_packager, BaseInputPackager)
        # 如果 pre 为 False，则 input_packager 为 None
        else:
            self.input_packager = None
        # 如果 post 为 True，设置 output_packager
        if post:
            # 如果 output_packager 不为 None，则使用传入的 output_packager，否则使用默认的 SimpleOutputPackager
            self.output_packager = output_packager or SimpleOutputPackager()
            assert isinstance(self.output_packager, BaseOutputPackager)
        # 如果 post 为 False，则 output_packager 为 None
        else:
            self.output_packager = None
