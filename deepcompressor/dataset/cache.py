# -*- coding: utf-8 -*-
"""Caching calibration dataset."""

import functools
import gc
import typing as tp
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import MISSING

import psutil
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.hooks
from tqdm import tqdm

from ..data.cache import IOTensorsCache, ModuleForwardInput, TensorCache
from ..data.utils.reshape import ConvInputReshapeFn, ConvOutputReshapedFn, LinearReshapeFn
from ..utils import tools
from ..utils.common import tree_copy_with_ref, tree_map
from ..utils.hooks import EarlyStopException, EarlyStopHook, Hook
from .action import CacheAction

__all__ = ["BaseCalibCacheLoader"]


class BaseCalibCacheLoader(ABC):
    """
    抽象基类，用于缓存校准数据集（calibration dataset）的激活信息。
    该类定义了基本的接口和辅助方法，以支持在模型的不同层级收集和管理激活数据。
    具体的实现需要继承该类并实现其抽象方法。
    Base class for caching calibration dataset.
    """

    # 存储要进行校准的数据集
    dataset: torch.utils.data.Dataset
    # 每个批次的大小
    batch_size: int

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int):
        """Initialize the dataset.

        Args:
            dataset (`torch.utils.data.Dataset`):
                Calibration dataset.
            batch_size (`int`):
                Batch size.
        """
        self.dataset = dataset
        self.batch_size = batch_size

    @property
    def num_samples(self) -> int:
        """
        返回数据集中的样本总数量。
        Number of samples in the dataset.
        """
        return len(self.dataset)

    @abstractmethod
    def iter_samples(self, *args, **kwargs) -> tp.Generator[ModuleForwardInput, None, None]:
        """
        迭代模型的输入样本。
        Iterate over model input samples.
        """
        ...

    def _init_cache(self, name: str, module: nn.Module) -> IOTensorsCache:
        """
        根据不同类型的神经网络模块（如线性层、卷积层），初始化相应的输入和输出缓存。
        Initialize activation cache.

        Args:
            name (`str`):
                Module name.
            module (`nn.Module`):
                Module.

        Returns:
            `IOTensorsCache`:
                Tensors cache for inputs and outputs.
        """
        # 线性层：对输入和输出进行缓存
        if isinstance(module, (nn.Linear,)):
            return IOTensorsCache(
                inputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
                outputs=TensorCache(channels_dim=-1, reshape=LinearReshapeFn()),
            )
        # 卷积层：对输入和输出进行缓存
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            assert module.padding_mode == "zeros", f"Padding mode {module.padding_mode} is not supported"   # 确保填充模式为“zeros”
            # 根据填充方式，计算实际的填充值
            if isinstance(module.padding, str):
                if module.padding == "valid":
                    padding = (0,) * len(module.kernel_size)
                elif module.padding == "same":
                    padding = tuple(reversed(tuple(t for t in module._reversed_padding_repeated_twice[::2])))
            else:
                padding = tuple(module.padding)
            return IOTensorsCache(
                inputs=TensorCache(
                    channels_dim=1,
                    reshape=ConvInputReshapeFn(module.kernel_size, padding, module.stride, module.dilation),
                ),
                outputs=TensorCache(channels_dim=1, reshape=ConvOutputReshapedFn()),
            )
        else:
            raise NotImplementedError(f"Module {module.__class__.__name__} is not supported")

    def _convert_layer_inputs(
        self, m: nn.Module, args: tuple[tp.Any, ...], kwargs: dict[str, tp.Any], save_all: bool = False
    ) -> ModuleForwardInput:
        """
        将层的输入转换为 ModuleForwardInput 对象，以便后续处理。
        Convert layer inputs to module forward input.

        Args:
            m (`nn.Module`):
                Layer.
            args (`tuple[Any, ...]`):
                Layer input arguments.
            kwargs (`dict[str, Any]`):
                Layer input keyword arguments.
            save_all (`bool`, *optional*, defaults to `False`):
                Whether to save all inputs.

        Returns:
            `ModuleForwardInput`:
                Module forward input.
        """
        # 如果 save_all 为 True，则提取第一个输入张量
        x = args[0].detach().cpu() if save_all else MISSING
        # 创建并返回一个 ModuleForwardInput 实例，包含处理后的args和原始kwargs
        return ModuleForwardInput(args=[x, *args[1:]], kwargs=kwargs)

    def _convert_layer_outputs(self, m: nn.Module, outputs: tp.Any) -> dict[str | int, tp.Any]:
        """
        将层的输出转换为字典格式，以便用作下一个层的输入。
        Convert layer outputs to dictionary for updating the next layer inputs.

        Args:
            m (`nn.Module`):
                Layer.
            outputs (`Any`):
                Layer outputs.

        Returns:
            `dict[str | int, Any]`:
                Dictionary for updating the next layer inputs.
        """
        # 如果输出不是张量，则取第一个张量
        if not isinstance(outputs, torch.Tensor):
            outputs = outputs[0]
        # 确保输出是张量
        assert isinstance(outputs, torch.Tensor), f"Invalid outputs type: {type(outputs)}"
        # 返回一个字典，包含输出张量
        return {0: outputs.detach().cpu()}

    def _layer_forward_pre_hook(
        self,
        m: nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, tp.Any],
        cache: list[ModuleForwardInput],
        save_all: bool = False,
    ) -> None:
        """
        作为前向传播的预钩子，用于处理和缓存层的输入。
        
        Args:
            m: 当前层的模块实例
            args: 当前层的输入参数
            kwargs: 当前层的关键字参数
            cache: 用于存储输入的列表
            save_all: 是否保存所有输入
        """
        # 将层的输入转换为 ModuleForwardInput 对象
        inputs = self._convert_layer_inputs(m, args, kwargs, save_all=save_all)

        # 如果 cache 中有数据，则将当前输入的 args 和 kwargs 与 cache 中的第一个输入的 args 和 kwargs 进行对齐
        if len(cache) > 0:
            inputs.args = tree_copy_with_ref(inputs.args, cache[0].args)
            inputs.kwargs = tree_copy_with_ref(inputs.kwargs, cache[0].kwargs)
        # 否则，对输入的 args 和 kwargs 进行映射处理
        else:
            inputs.args = tree_map(lambda x: x, inputs.args)
            inputs.kwargs = tree_map(lambda x: x, inputs.kwargs)
        
        # 将当前输入添加到 cache 中
        cache.append(inputs)

    @torch.inference_mode()
    def _iter_layer_activations(  # noqa: C901
        self,
        model: nn.Module,
        *args,
        action: CacheAction,
        layers: tp.Sequence[nn.Module] | None = None,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = True,
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = None,
        recomputes: list[bool] | None = None,
        use_prev_layer_outputs: list[bool] | None = None,
        early_stop_module: nn.Module | None = None,
        clear_after_yield: bool = True,
        **kwargs,
    ) -> tp.Generator[
        tuple[
            str,
            tuple[
                nn.Module,
                dict[str, IOTensorsCache],
                list[ModuleForwardInput],
            ],
        ],
        None,
        None,
    ]:
        """
        迭代模型的各个层，收集和缓存层的激活信息。
        Iterate over model activations in layers.

        Args:
            model (`nn.Module`):
                Model.
            action (`CacheAction`):
                Action for caching activations.
            layers (`Sequence[nn.Module]` or `None`, *optional*, defaults to `None`):
                Layers to cache activations. If `None`, cache all layers.
            needs_inputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `True`):
                Function for determining whether to cache inputs for a module given its name and itself.
            needs_outputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Function for determining whether to cache outputs for a module given its name and itself.
            recomputes (`list[bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Whether to recompute the activations for each layer.
            use_prev_layer_outputs (`list[bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Whether to use the previous layer outputs as inputs for the current layer.
            early_stop_module (`nn.Module` or `None`, *optional*, defaults to `None`):
                Module for early stopping.
            clear_after_yield (`bool`, *optional*, defaults to `True`):
                Whether to clear the cache after yielding the activations.
            *args: Arguments for ``iter_samples``.
            **kwargs: Keyword arguments for ``iter_samples``.

        Yields:
            Generator[
                tuple[str, tuple[nn.Module, dict[str, IOTensorsCache], list[ModuleForwardInput]]],
                None,
                None
            ]:
                Generator of tuple of
                    - layer name
                    - a tuple of
                        - layer itself
                        - inputs and outputs cache of each module in the layer
                        - layer input arguments
        """
        # 将 needs_inputs_fn 和 needs_outputs_fn 参数规范化为可调用的函数
        if needs_outputs_fn is None:
            needs_outputs_fn = lambda name, module: False  # noqa: E731
        elif isinstance(needs_outputs_fn, bool):
            if needs_outputs_fn:
                needs_outputs_fn = lambda name, module: True  # noqa: E731
            else:
                needs_outputs_fn = lambda name, module: False  # noqa: E731
        if needs_inputs_fn is None:
            needs_inputs_fn = lambda name, module: False  # noqa: E731
        elif isinstance(needs_inputs_fn, bool):
            if needs_inputs_fn:
                needs_inputs_fn = lambda name, module: True  # noqa: E731
            else:
                needs_inputs_fn = lambda name, module: False  # noqa: E731

        # 根据是否指定特定层，初始化或规范化 recomputes 和 use_prev_layer_outputs 参数
        if layers is None:
            recomputes = [True]
            use_prev_layer_outputs = [False]
        else:
            assert isinstance(layers, (nn.Sequential, nn.ModuleList, list, tuple))
            if recomputes is None:
                recomputes = [False] * len(layers)
            elif isinstance(recomputes, bool):
                recomputes = [recomputes] * len(layers)
            if use_prev_layer_outputs is None:
                use_prev_layer_outputs = [True] * len(layers)
            elif isinstance(use_prev_layer_outputs, bool):
                use_prev_layer_outputs = [use_prev_layer_outputs] * len(layers)
            use_prev_layer_outputs[0] = False
            assert len(recomputes) == len(use_prev_layer_outputs) == len(layers)

        # 初始化缓存和相关数据结构
        cache: dict[str, dict[str, IOTensorsCache]] = {}            # 用于存储每个层和模块的激活缓存
        module_names: dict[str, list[str]] = {"": []}               # 记录每个层下的模块名称列表
        named_layers: OrderedDict[str, nn.Module] = {"": model}     # 记录每个层的名称和对应的模块实例
        
        # region we first collect infomations for yield modules
        # 收集需要进行缓存的模块信息
        # 遍历模型的所有模块，确定哪些模块需要缓存其输入/输出，并初始化相应的缓存结构和钩子。
        forward_cache: dict[str, list[ModuleForwardInput]] = {}                 # 用于存储每个层的前向输入
        info_hooks: list[Hook] = []                                             # 存储信息模式下的钩子对象
        forward_hooks: list[torch.utils.hooks.RemovableHandle] = []             # 存储前向模式下的钩子对象
        hook_args: dict[str, list[tuple[str, nn.Module, bool, bool]]] = {}      # 存储每个层的钩子参数
        layer_name = ""                                                         # 记录当前处理的层名称
        for module_name, module in model.named_modules():
            if layers is not None and module_name and module in layers:
                layer_name = module_name
                assert layer_name not in module_names
                named_layers[layer_name] = module                               # 记录当前层的模块实例
                module_names[layer_name] = []                                   # 初始化当前层的模块名称列表
                forward_cache[layer_name] = []                                  # 初始化当前层的前向输入
            if layers is None or (layer_name and module_name.startswith(layer_name)):
                # we only cache modules in the layer
                needs_inputs = needs_inputs_fn(module_name, module)             # 判断当前模块是否需要缓存输入
                needs_outputs = needs_outputs_fn(module_name, module)           # 判断当前模块是否需要缓存输出
                if needs_inputs or needs_outputs:
                    module_names[layer_name].append(module_name)                # 将当前模块名称添加到当前层的模块名称列表中
                    cache.setdefault(layer_name, {})[module_name] = self._init_cache(module_name, module)               # 初始化当前模块的缓存
                    hook_args.setdefault(layer_name, []).append((module_name, module, needs_inputs, needs_outputs))     # 记录当前模块的钩子参数
                    info_hooks.extend(                                          # 注册信息模式下的钩子，并将其添加到 info_hooks 列表中
                        action.register(
                            name=module_name,
                            module=module,
                            cache=cache[layer_name][module_name],
                            info_mode=True,
                            needs_inputs=needs_inputs,
                            needs_outputs=needs_outputs,
                        )
                    )

        # 检查是否有模块需要缓存
        if len(cache) == 0:
            return

        # 处理指定层的情况。如果指定了特定的层，则重新组织 named_layers，并为每个层注册前向预钩子。
        if layers is not None:
            # 从 module_names 和 named_layers 中删除空字符串键
            module_names.pop("")
            named_layers.pop("")
            assert layer_name, "No layer in the given layers is found in the model"
            assert "" not in cache, "The model should not have empty layer name"
            ordered_named_layers: OrderedDict[str, nn.Module] = OrderedDict()       # 用于存储按顺序排列的层
            # 遍历layers，按顺序排列模块
            for layer in layers:
                for name, module in named_layers.items():
                    if module is layer:
                        ordered_named_layers[name] = module
                        break
            assert len(ordered_named_layers) == len(named_layers)
            assert len(ordered_named_layers) == len(layers)
            named_layers = ordered_named_layers
            del ordered_named_layers
            # 为每个层注册前向预钩子
            for layer_idx, (layer_name, layer) in enumerate(named_layers.items()):
                forward_hooks.append(
                    layer.register_forward_pre_hook(
                        functools.partial(
                            self._layer_forward_pre_hook,
                            cache=forward_cache[layer_name],
                            save_all=not recomputes[layer_idx] and not use_prev_layer_outputs[layer_idx],
                        ),
                        with_kwargs=True,
                    )
                )
        else:
            assert len(named_layers) == 1 and "" in named_layers
            assert len(module_names) == 1 and "" in module_names
            assert len(cache) == 1 and "" in cache
        # endregion

        # 收集缓存信息。通过遍历所有样本，运行模型以收集缓存的激活信息。
        with tools.logging.redirect_tqdm():         # 重定向日志
            # region we then collect cache information by running the model with all samples
            # 注册提前停止钩子
            if early_stop_module is not None:
                forward_hooks.append(early_stop_module.register_forward_hook(EarlyStopHook()))
            with torch.inference_mode():
                device = "cuda" if torch.cuda.is_available() else "cpu"     # 设备设置
                tbar = tqdm(                                                # 初始化进度条
                    desc="collecting acts info",
                    leave=False,
                    total=self.num_samples,
                    unit="samples",
                    dynamic_ncols=True,
                )
                num_samples = 0
                for sample in self.iter_samples(*args, **kwargs):           # 遍历所有样本
                    num_samples += self.batch_size                          # 更新样本数量
                    sample = sample.to(device=device)
                    try:
                        model(*sample.args, **sample.kwargs)                # 运行模型的前向传播，捕捉激活信息
                    except EarlyStopException:
                        pass
                    tbar.update(self.batch_size)                            # 更新进度条
                    tbar.set_postfix({"ram usage": psutil.virtual_memory().percent})    # 显示当前RAM使用情况
                    if psutil.virtual_memory().percent > 90:                # 如果RAM使用率超过90%，则抛出异常
                        raise RuntimeError("memory usage > 90%%, aborting")
            for layer_cache in cache.values():
                for module_cache in layer_cache.values():
                    module_cache.set_num_samples(num_samples)               # 为每个模块的缓存设置总样本数量
            for hook in forward_hooks:
                hook.remove()
            for hook in info_hooks:
                hook.remove()
            del info_hooks, forward_hooks
            # endregion

            # 迭代每一层并收集激活信息。针对每一层，注册缓存激活的钩子，运行模型以收集激活数据，并在生成器中 yield 收集到的激活信息。
            for layer_idx, (layer_name, layer) in enumerate(named_layers.items()):
                # region we first register hooks for caching activations
                # 注册缓存激活的钩子，将其添加到 layer_hooks 列表中
                layer_hooks: list[Hook] = []
                for module_name, module, needs_inputs, needs_outputs in hook_args[layer_name]:
                    layer_hooks.extend(
                        action.register(
                            name=module_name,
                            module=module,
                            cache=cache[layer_name][module_name],
                            info_mode=False,
                            needs_inputs=needs_inputs,
                            needs_outputs=needs_outputs,
                        )
                    )
                hook_args.pop(layer_name)
                # endregion
                # 处理是否需要重新计算激活
                if recomputes[layer_idx]:
                    # 如果需要重新计算激活
                    if layers is None:
                        if early_stop_module is not None:
                            layer_hooks.append(EarlyStopHook().register(early_stop_module))     # 注册提前停止钩子
                    else:
                        layer_hooks.append(EarlyStopHook().register(layer))                     # 注册提前停止钩子
                    tbar = tqdm(                                                                # 初始化进度条
                        desc=f"collecting acts in {layer_name}",
                        leave=False,
                        total=self.num_samples,
                        unit="samples",
                        dynamic_ncols=True,
                    )
                    # 遍历所有样本，运行模型以收集激活信息
                    for sample in self.iter_samples(*args, **kwargs):                           
                        sample = sample.to(device=device)
                        try:
                            model(*sample.args, **sample.kwargs)
                        except EarlyStopException:
                            pass
                        tbar.update(self.batch_size)                                            # 更新进度条
                        tbar.set_postfix({"ram usage": psutil.virtual_memory().percent})        # 显示当前RAM使用情况
                        if psutil.virtual_memory().percent > 90:
                            raise RuntimeError("memory usage > 90%%, aborting")
                        gc.collect()
                else:
                    # 如果不需要重新计算激活
                    # region we then forward the layer to collect activations
                    device = next(layer.parameters()).device
                    layer_outputs: list[tp.Any] = []                                            # 存储当前层的输出
                    tbar = tqdm(                                                                # 初始化进度条
                        forward_cache[layer_name],
                        desc=f"collecting acts in {layer_name}",
                        leave=False,
                        unit="batches",
                        dynamic_ncols=True,
                    )
                    if not use_prev_layer_outputs[layer_idx]:                                   # 如果不使用上一层的输出，则初始化 prev_layer_outputs
                        prev_layer_outputs: list[dict[str | int, tp.Any]] = [None] * len(tbar)
                    for i, inputs in enumerate(tbar):
                        inputs = inputs.update(prev_layer_outputs[i]).to(device=device)         # 更新输入
                        outputs = layer(*inputs.args, **inputs.kwargs)                          # 运行当前层的前向传播
                        layer_outputs.append(self._convert_layer_outputs(layer, outputs))       # 将输出转换为字典格式并添加到 layer_outputs 中
                        tbar.set_postfix({"ram usage": psutil.virtual_memory().percent})        # 显示当前RAM使用情况
                        if psutil.virtual_memory().percent > 90:
                            raise RuntimeError("memory usage > 90%%, aborting")
                    prev_layer_outputs = layer_outputs                                          # 更新 prev_layer_outputs
                    del inputs, outputs, layer_outputs
                    if (layer_idx == len(named_layers) - 1) or not use_prev_layer_outputs[layer_idx + 1]:
                        del prev_layer_outputs
                    # endregion
                for hook in layer_hooks:
                    hook.remove()
                del layer_hooks
                # 获取并处理层的输入
                layer_inputs = forward_cache.pop(layer_name, [])
                if not recomputes[layer_idx] and not use_prev_layer_outputs[layer_idx]:
                    layer_inputs = [
                        self._convert_layer_inputs(layer, inputs.args, inputs.kwargs) for inputs in layer_inputs
                    ]
                gc.collect()
                torch.cuda.empty_cache()
                # 生成当前层的名称、模块实例、缓存字典以及层的输入参数
                yield layer_name, (layer, cache[layer_name], layer_inputs)
                # region clear layer cache
                if clear_after_yield:
                    for module_cache in cache[layer_name].values():
                        module_cache.clear()
                cache.pop(layer_name)
                del layer_inputs
                gc.collect()
                torch.cuda.empty_cache()
                # endregion

    @abstractmethod
    def iter_layer_activations(  # noqa: C901
        self,
        model: nn.Module,
        *args,
        action: CacheAction,
        needs_inputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = True,
        needs_outputs_fn: tp.Callable[[str, nn.Module], bool] | bool | None = None,
        **kwargs,
    ) -> tp.Generator[
        tuple[
            str,
            tuple[
                nn.Module,
                dict[str, IOTensorsCache],
                list[ModuleForwardInput],
            ],
        ],
        None,
        None,
    ]:
        """
        定义一个抽象方法，用于迭代模型的各层激活信息。
        Iterate over model activations in layers.

        Args:
            model (`nn.Module`):
                Model.
            action (`CacheAction`):
                Action for caching activations.
            needs_inputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `True`):
                Function for determining whether to cache inputs for a module given its name and itself.
            needs_outputs_fn (`Callable[[str, nn.Module], bool]` or `bool` or `None`, *optional*, defaults to `None`):
                Function for determining whether to cache outputs for a module given its name and itself.
            *args: Arguments for ``iter_samples``.
            **kwargs: Keyword arguments for ``iter_samples``.

        Yields:
            Generator[
                tuple[str, tuple[nn.Module, dict[str, IOTensorsCache], list[ModuleForwardInput]]],
                None,
                None
            ]:
                Generator of tuple of
                    - layer name
                    - a tuple of
                        - layer itself
                        - inputs and outputs cache of each module in the layer
                        - layer input arguments
        """
        ...
