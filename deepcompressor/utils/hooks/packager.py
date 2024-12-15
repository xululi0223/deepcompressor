# -*- coding: utf-8 -*-
"""Packagers for input and output tensors in hooks."""

import functools
import inspect
import typing as tp
from abc import ABC, abstractmethod

import torch
import torch.ao.quantization
import torch.nn as nn
import torch.utils.hooks

__all__ = [
    "BaseInputPackager",
    "SimpleInputPackager",
    "KeyedInputPackager",
    "BaseOutputPackager",
    "SimpleOutputPackager",
    "KeyedOutputPackager",
]


class BaseInputPackager(ABC):
    """
    抽象基类，定义了输入包装器（Input Packager）的基本接口。
    它通过抽象方法 unpack 和 repack 强制子类实现特定的输入解包和重新打包逻辑。
    该类使用了 Python 的 ABC（抽象基类）模块，确保任何继承自它的子类必须实现其定义的所有抽象方法。
    Base class for input packagers.
    """

    @abstractmethod
    def unpack(
        self, module: nn.Module, input_args: tuple[torch.Tensor, ...], input_kwargs: dict[str, tp.Any]
    ) -> dict[int | str, torch.Tensor]:
        """
        解包输入参数，将输入参数组织成字典形式，方便后续处理。
        Unpack inputs in inputs packager.

        Args:
            module (`nn.Module`):
                Module.
            input_args (`tuple[torch.Tensor, ...]`):
                Input arguments.
            input_kwargs (`dict[str, tp.Any]`):
                Input keyword arguments.

        Returns:
            `dict[int | str, torch.Tensor]`:
                The unpacked input tensors.
        """
        ...

    @abstractmethod
    def repack(
        self,
        tensors: dict[int | str, torch.Tensor],
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, tp.Any]]:
        """
        将解包后的张量重新组织成原始的输入参数格式。
        Repack inputs in inputs packager.

        Args:
            tensors (`dict[int | str, torch.Tensor]`):
                The input tensors.
            module (`nn.Module`):
                Module.
            input_args (`tuple[torch.Tensor, ...]`):
                Input arguments.
            input_kwargs (`dict[str, tp.Any]`):
                Input keyword arguments.

        Returns:
            `tuple[tuple[torch.Tensor, ...], dict[str, tp.Any]]`:
                The repacked input arguments and keyword arguments.
        """
        ...


class SimpleInputPackager(BaseInputPackager):
    """
    提供了简单的输入解包和重新打包逻辑。
    该类假设输入参数中第一个张量是主要关注的对象，其他参数保持不变。
    """
    def unpack(
        self, module: nn.Module, input_args: tuple[torch.Tensor, ...], input_kwargs: dict[str, tp.Any]
    ) -> dict[int | str, torch.Tensor]:
        """
        解包输入参数。简单地将第一个输入张量作为解包结果，忽略其他输入参数和关键字参数。
        """
        return {0: input_args[0]}

    def repack(
        self,
        tensors: dict[int | str, torch.Tensor],
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, tp.Any]]:
        """
        重新打包输入参数。将解包后的张量放回原始输入参数的第一个位置，保持其他参数不变。
        """
        return (tensors[0], *input_args[1:]), input_kwargs


class KeyedInputPackager(BaseInputPackager):
    """
    允许通过索引或关键字来指定需要解包和重新打包的输入张量。
    这种包装器适用于输入参数包含多个张量，且需要根据特定的索引或关键字进行选择和操作的场景。
    """
    def __init__(self, module: nn.Module, index_or_keys: list[int | str]):
        """
        Args:
            module: Pytorch 模块。
            index_or_keys: 指定要解包和重新打包的输入参数的索引或关键字列表。
        """
        # 初始化forward_name
        forward_name = "forward"
        # 处理functools.partial包装的前向方法，避免在方法被包装或修改后无法正确映射
        if isinstance(module.forward, functools.partial):
            # 如果由_decompressor_orig_forward属性，则使用该属性作为forward_name
            if hasattr(module, "_deepcompressor_orig_forward"):
                forward_name = "_deepcompressor_orig_forward"
            # 否则，假设模块被accelerate包装，使用_old_forward属性作为forward_name
            else:
                # this module has been wrapped in `accelerate` package
                assert hasattr(module, "_old_forward")
                assert module._old_forward is module.forward.__wrapped__  # type: ignore
                forward_name = "_old_forward"
        # 获取前向方法的签名，便于后续解析参数
        signature = inspect.signature(getattr(module, forward_name))
        # 解析前向方法的参数
        args, kwargs = [], []
        # 遍历前向方法的所有参数，根据参数类型将其分类到args或kwargs列表中
        for key, param in signature.parameters.items():
            # 仅限位置传参
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                args.append(key)
            # 位置或关键字传参
            elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                args.append(key)
                kwargs.append(key)
            # 仅限关键字传参
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                kwargs.append(key)
        # 初始化index_key_pairs列表，用于存储每个指定的index_or_key对应的索引和关键字对
        self.index_key_pairs: list[tuple[int | None, str | None]] = []
        # 处理index_or_keys列表
        for index_or_key in index_or_keys:
            # 整数索引类型
            if isinstance(index_or_key, int):
                index = index_or_key
                # 如果索引超出了参数范围或不在kwargs中
                if index >= len(args) or args[index] not in kwargs:
                    self.index_key_pairs.append((index, None))
                # 否则，将索引和对应的关键字对存入index_key_pairs列表
                else:
                    self.index_key_pairs.append((index, args[index]))
            # 字符串类型
            else:
                key = index_or_key
                # 如果关键字在args中，直接存入index_key_pairs列表
                if key in args:
                    self.index_key_pairs.append((args.index(key), key))
                # 否则，将关键字存入index_key_pairs列表，索引设为None
                else:
                    self.index_key_pairs.append((None, key))
        self.index_or_keys = index_or_keys

    def unpack(
        self, module: nn.Module, input_args: tuple[torch.Tensor, ...], input_kwargs: dict[str, tp.Any]
    ) -> dict[int | str, torch.Tensor]:
        """
        解包输入参数。根据指定的索引或关键字，将输入参数解包成字典形式，方便后续处理。
        
        Args:
            module: Pytorch 模块。
            input_args: 输入的张量参数。
            input_kwargs: 输入的关键字参数。
        """
        # 初始化tensors字典，用于存储解包后的张量
        tensors = {}
        # 遍历index_or_keys和其对应的(index, key)对
        for index_or_key, (index, key) in zip(self.index_or_keys, self.index_key_pairs, strict=True):
            # 根据索引或关键字提取张量
            # 索引存在且有效
            if index is not None and index < len(input_args):
                tensors[index_or_key] = input_args[index]
            # 仅关键字存在
            else:
                assert key is not None
                tensors[index_or_key] = input_kwargs.get(key, None)
        return tensors

    def repack(
        self,
        tensors: dict[int | str, torch.Tensor],
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, tp.Any]]:
        """
        重新打包输入参数。将解包后的张量重新组织成原始的输入参数格式。
        
        Args:
            tensors: 输入的张量字典。
            module: Pytorch 模块。
            input_args: 原始输入张量参数。
            input_kwargs: 原始输入关键字参数。
        """
        # 复制输入参数，避免直接修改原始参数
        _args, _kwargs = list(input_args), dict(input_kwargs)
        # 遍历index_or_keys和其对应的(index, key)对
        for index_or_key, (index, key) in zip(self.index_or_keys, self.index_key_pairs, strict=True):
            # 根据索引或关键字替换张量
            # 索引存在且有效
            if index is not None and index < len(_args):
                _args[index] = tensors[index_or_key]
            # 仅关键字存在
            else:
                assert key is not None
                _kwargs[key] = tensors[index_or_key]
        return tuple(_args), _kwargs


class BaseOutputPackager(ABC):
    """
    抽象基类，定义了输出包装器（Output Packager）的基本接口。
    它通过抽象方法 unpack 和 repack 强制子类实现特定的输出解包和重新打包逻辑。
    该类使用了 Python 的 ABC（抽象基类）模块，确保任何继承自它的子类必须实现其定义的所有抽象方法。
    Base class for output packagers.
    """

    @abstractmethod
    def unpack(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tp.Any,
    ) -> dict[int | str, torch.Tensor]:
        """
        解包输出参数，将输出参数组织成字典形式，方便后续处理。
        Unpack outputs in outputs packager.

        Args:
            module (`nn.Module`):
                Module.
            input_args (`tuple[torch.Tensor, ...]`):
                Input arguments.
            input_kwargs (`dict[str, tp.Any]`):
                Input keyword arguments.
            output (`Any`):
                Output.

        Returns:
            `dict[int | str, torch.Tensor]`:
                The unpacked output tensors.
        """
        ...

    @abstractmethod
    def repack(
        self,
        tensors: dict[int | str, torch.Tensor],
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tp.Any,
    ) -> tp.Any:
        """
        将解包后的张量重新组织成原始的输出参数格式。
        Repack outputs in outputs packager.

        Args:
            tensors (`dict[int | str, torch.Tensor]`):
                The output tensors.
            module (`nn.Module`):
                Module.
            input_args (`tuple[torch.Tensor, ...]`):
                Input arguments.
            input_kwargs (`dict[str, tp.Any]`):
                Input keyword arguments.
            output (`Any`):
                Output.

        Returns:
            `Any`:
                The repacked output.
        """
        ...


class SimpleOutputPackager(BaseOutputPackager):
    """
    提供了简单的输出解包和重新打包逻辑。
    该类假设输出参数中第一个张量是主要关注的对象，其他部分保持不变。
    """
    def unpack(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tp.Any,
    ) -> dict[int | str, torch.Tensor]:
        """
        解包输出参数。简单地将第一个输出张量作为解包结果，忽略其他输出参数。
        """
        if not isinstance(output, torch.Tensor):
            output = output[0]
        return {0: output}

    def repack(
        self,
        tensors: dict[int | str, torch.Tensor],
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tp.Any,
    ) -> tp.Any:
        """
        重新打包输出参数。将解包后的张量放回原始输出参数的第一个位置，保持其他参数不变。
        """
        if isinstance(output, torch.Tensor):
            return tensors[0]
        else:
            return (tensors[0], *output[1:])


class KeyedOutputPackager(BaseOutputPackager):
    """
    允许通过索引或关键字来指定需要解包和重新打包的输出张量。
    这种包装器适用于输出参数包含多个张量，且需要根据特定的索引或关键字进行选择和操作的场景。
    """
    def __init__(self, index_or_keys: list[int | str]):
        """
        Args:
            index_or_keys: 指定要解包和重新打包的输出参数的索引或关键字列表。
        """
        self.index_or_keys = index_or_keys

    def unpack(
        self,
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tp.Any,
    ) -> dict[int | str, torch.Tensor]:
        """
        解包输出参数。根据指定的索引或关键字，将输出参数解包成字典形式，方便后续处理。
        
        Args:
            module: Pytorch 模块。
            input_args: 输入的张量参数。
            input_kwargs: 输入的关键字参数。
            output: 模块的输出。
        """
        # 初始化tensors字典，用于存储解包后的张量
        tensors = {}
        # 对于tuple和list类型的输出参数
        if isinstance(output, (tuple, list)):
            # 遍历index_or_keys列表
            for index_or_key in self.index_or_keys:
                # 确保index_or_key是整数且在输出参数范围内
                assert isinstance(index_or_key, int) and index_or_key < len(output)
                # 将输出参数中对应索引的张量存入tensors字典
                tensors[index_or_key] = output[index_or_key]
        # 对于dict类型的输出参数
        elif isinstance(output, dict):
            # 遍历index_or_keys列表
            for index_or_key in self.index_or_keys:
                # 确保index_or_key是字符串且在输出参数中
                assert isinstance(index_or_key, str) and index_or_key in output
                # 将输出参数中对应关键字的张量存入tensors字典
                tensors[index_or_key] = output[index_or_key]
        # 对于单个张量的输出参数
        else:
            # 确保output是张量
            assert isinstance(output, torch.Tensor)
            # 确保index_or_keys列表只有一个元素且为0
            assert len(self.index_or_keys) == 1
            assert self.index_or_keys[0] == 0
            # 将output存入tensors字典
            tensors[0] = output
        return tensors

    def repack(
        self,
        tensors: dict[int | str, torch.Tensor],
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, tp.Any],
        output: tp.Any,
    ) -> tp.Any:
        """
        重新打包输出参数。将解包后的张量重新组织成原始的输出参数格式。
        
        Args:
            tensors: 输入的张量字典。
            module: Pytorch 模块。
            input_args: 原始输入张量参数。
            input_kwargs: 原始输入关键字参数。
            output: 模块的输出。
        """
        # 对于tuple和list类型的输出参数
        if isinstance(output, (tuple, list)):
            # 复制输出参数，避免直接修改原始参数
            _output = list(output)
            # 遍历index_or_keys列表
            for index_or_key in self.index_or_keys:
                # 确保index_or_key是整数且在输出参数范围内
                assert isinstance(index_or_key, int) and index_or_key < len(_output)
                # 替换输出参数中对应索引的张量
                _output[index_or_key] = tensors[index_or_key]
            return tuple(_output)
        # 对于dict类型的输出参数
        elif isinstance(output, dict):
            # 复制输出参数，避免直接修改原始参数
            _output = dict(output)
            # 遍历index_or_keys列表
            for index_or_key in self.index_or_keys:
                # 确保index_or_key是字符串且在输出参数中
                assert isinstance(index_or_key, str) and index_or_key in _output
                # 替换输出参数中对应关键字的张量
                _output[index_or_key] = tensors[index_or_key]
            return _output
        # 对于单个张量的输出参数
        else:
            # 确保output是张量
            assert isinstance(output, torch.Tensor)
            # 确保index_or_keys列表只有一个元素且为0
            assert len(self.index_or_keys) == 1
            assert self.index_or_keys[0] == 0
            # 直接返回解包后的张量
            return tensors[0]
