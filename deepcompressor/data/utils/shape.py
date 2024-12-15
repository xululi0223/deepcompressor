# -*- coding: utf-8 -*-
"""Utility functions for shape calulation in quantization."""

import typing as tp

import torch

from ..dtype import QuantDataType
from .dtype import eval_dtype

__all__ = ["infer_group_shape_name", "format_group_configs", "infer_group_shapes", "infer_view_shape", "infer_shape"]


def infer_group_shape_name(group_shape: tp.Sequence[int]) -> str:
    """
    用于根据给定的组形状（group_shape）生成对应的组名字符串。
    组名通常用于标识或区分不同的量化组配置。
    Get the name of the group shape.

    Args:
        group_shape (`Sequence[int]`):
            The group shape.

    Returns:
        `str`:
            The name of the group shape.
    """
    # 检查group_shape的后续纬度是否都小于等于0
    if all(gs <= 0 for gs in group_shape[2:]):
        # 根据前两维的值生成组名
        if group_shape[1] <= 0:
            if group_shape[0] <= 0:
                return "tsnr"
            elif group_shape[0] == 1:
                return "gchn"
            else:
                return f"t{group_shape[0]}gchn"
        else:
            if group_shape[0] <= 0:
                return f"tsg{group_shape[1]}"
            elif group_shape[0] == 1:
                return f"g{group_shape[1]}"
            else:
                return f"t{group_shape[0]}g{group_shape[1]}"
    # 处理后续维度不全部小于等于0的情况，检查后续维度是否全部为1
    elif all(gs == 1 for gs in group_shape[2:]):
        if group_shape[1] <= 0:
            if group_shape[0] <= 0:
                return "tspx"
            elif group_shape[0] == 1:
                return "vchn"
            else:
                return f"t{group_shape[0]}vchn"
        else:
            if group_shape[0] <= 0:
                return f"tsv{group_shape[1]}"
            elif group_shape[0] == 1:
                return f"v{group_shape[1]}"
            else:
                return f"t{group_shape[0]}v{group_shape[1]}"
    # 默认情况
    return f"{'x'.join(str(gs) if gs >= 1 else '_' for gs in group_shape)}"


def format_group_configs(
    *,
    group_shapes: tp.Sequence[tp.Sequence[int]],
    scale_dtypes: tp.Sequence[torch.dtype | QuantDataType | None] | torch.dtype | QuantDataType | None,
) -> tuple[tuple[tuple[int, ...], ...], tuple[torch.dtype | QuantDataType | None, ...]]:
    """
    用于格式化组形状（group_shapes）和缩放数据类型（scale_dtypes），确保它们符合预期的格式和约束条件。
    Format the group shape and scale dtype.

    Args:
        group_shapes (`Sequence[tp.Sequence[int]]`):
            The group shapes.
        scale_dtypes (`Sequence[torch.dtype | QuantDataType | None]` or `torch.dtype` or `QuantDataType` or `None`):
            The scale dtypes.

    Returns:
        `tuple[tuple[tuple[int, ...], ...], tuple[torch.dtype | QuantDataType | None, ...]]`:
            The formatted group shapes and scale dtypes.
    """
    # 验证group_shapes是否为列表或元组
    assert isinstance(group_shapes, (list, tuple)), "group_shapes must be a list or tuple"

    # 格式化group_shapes，遍历每个group_shape，进行格式化和验证
    _group_shapes = []
    for group_shape in group_shapes:
        if isinstance(group_shape, tp.Sequence):
            n = len(group_shape)
            group_shape = tuple(map(int, group_shape))
            assert n >= 2, "the group shape must have at least two dimensions"
            assert all(gs >= -1 for gs in group_shape), "the group shape must be larger than -1"
            # 处理维度数不足的情况
            _group_shapes.append(tuple(group_shape) if n >= 3 else (*group_shape, -1))

    # 格式化scale_dtypes
    # 转换为元组
    _scale_dtypes = tuple(scale_dtypes) if isinstance(scale_dtypes, tp.Sequence) else (scale_dtypes,)
    # 处理每个dtype
    _scale_dtypes = tuple(
        dtype if isinstance(dtype, (torch.dtype, QuantDataType, type(None))) else eval_dtype(dtype)
        for dtype in _scale_dtypes
    )

    # 验证格式化后的列表
    assert len(_group_shapes) > 0, "group_sizes must be a non-empty list"
    assert len(_group_shapes) == len(_scale_dtypes), (
        f"group_shapes and scale_dtypes must have the same length, "
        f"got {_group_shapes}(len={len(_group_shapes)}) and {_scale_dtypes}(len={len(_scale_dtypes)})"
    )

    # 验证指数缩放的顺序，确保所有具有指数特性的量化数据类型都在线性量化数据类型之后
    exp_scale = True        # 用于跟踪是否仍然允许出现指数缩放
    # 逆序遍历_scale_dtypes
    for dtype in reversed(_scale_dtypes):
        if isinstance(dtype, QuantDataType) and dtype.is_exponent:
            if not exp_scale:
                raise ValueError("The exponential scale must be after linear scale")
        else:
            exp_scale = False
    
    # 进一步验证scale_dtypes
    # 确保出了第一个量化数据类型外，其他量化数据类型都是QuantDataType
    assert all(isinstance(dtype, QuantDataType) for dtype in _scale_dtypes[1:])
    return tuple(_group_shapes), _scale_dtypes


def infer_group_shapes(group_shapes: tuple[tuple[int, ...], ...], shape: torch.Size) -> list[torch.Size]:
    """
    用于根据提供的组形状配置（group_shapes）和输入张量的形状（shape），推断每个量化层级的具体组形状。
    Infer the group shapes using group shape config on the given tensor shape.

    Args:
        group_shapes (`tuple[tuple[int, ...], ...]`):
            The group shapes.
        shape (`torch.Size`):
            The shape of the tensor.

    Returns:
        `list[torch.Size]`:
            The inferred group shapes.
    """
    # 类型和维度验证
    assert isinstance(shape, torch.Size), f"shape must be torch.Size, got {shape} ({type(shape)})"
    assert len(shape) >= 2, f"shape must have at least 2 dimensions, got {shape} ({len(shape)} < 2)"
    # 初始化变量
    _group_shapes: list[torch.Size] = []    # 用于存储推断后的组形状
    _prev_group_shape = shape               # 用于后续层级的计算参考

    # 遍历每个量化层级的组形状配置，推断具体的组形状，并确保每个组形状的维度大小符合前一层级的限制
    for level, group_shape in enumerate(group_shapes):
        # 计算索引限制
        m = len(group_shape) - 1
        # 初始化当前层级的组形状列表
        _group_shape = []
        # 逐维度推断组形状
        for i, ts in enumerate(shape):
            # 获取当前维度的组大小
            gs = group_shape[min(i, m)]
            if gs <= 0:
                gs = ts
            # 获取前一层级的组大小
            ps = _prev_group_shape[i]
            # 确保当前层级的组大小小于等于前一层级的组大小
            if gs > ps:
                gs = ps  # the group shape must be less than or equal to the previous group shape
            # 验证组大小能被前一层级整除
            assert ps % gs == 0, (
                f"the level {level} group size ({gs}) must be divisible by "
                f"the previous group size ({_prev_group_shape}[{i}])"
            )
            # 添加到当前层级的组形状列表
            _group_shape.append(gs)
        # 更新推断后的组形状列表
        _group_shapes.append(torch.Size(_group_shape))
        # 更新前一层级的组形状参考
        _prev_group_shape = _group_shape
    return _group_shapes


def infer_view_shape(
    tensor_shape: torch.Size,
    /,
    group_shape: tp.Sequence[int],
    skip_first_dim: bool = False,
) -> torch.Size:
    """
    用于根据输入张量的形状（tensor_shape）和组形状（group_shape），推断视图形状（view_shape），用于调整张量的维度，以便进行分组操作。
    Infer the view shape from the tensor shape and the group shape.

    Args:
        tensor_shape (`torch.Size`):
            The tensor shape.
        group_shape (`Sequence[int]`):
            The group shape.
        skip_first_dim (`bool`, *optional*, defaults to `False`):
            Whether to skip the first dimension.

    Returns:
        `torch.Size`:
            The view shape of (#g0, gs0, #g1, gs1, #g2, gs2, ...)
    """
    # 初始化变量
    m, view_shape = len(group_shape) - 1, []
    # 遍历张量的每个维度，计算视图形状的对应部分
    for i, ts in enumerate(tensor_shape):
        # 获取当前维度的组大小
        gs = group_shape[min(i, m)]
        gs = ts if gs <= 0 else gs
        # 计算视图形状的分块数和组大小
        view_shape.append(ts // gs)
        view_shape.append(gs)
    # 如果skip_first_dim为True，则将第一个分块数设置为1，组大小设置为张量的第一个维度大小
    if skip_first_dim:
        view_shape[0], view_shape[1] = 1, tensor_shape[0]
    return torch.Size(view_shape)


def infer_scale_view_shapes(
    group_shapes: tp.Sequence[tp.Sequence[int]] | tp.Sequence[torch.Size], shape: torch.Size
) -> list[torch.Size]:
    """
    用于根据组形状配置和输入张量的形状，推断量化缩放张量的视图形状。
    Infer the view shapes of quantization scale for the given tensor shape.

    Args:
        group_shapes (`Sequence[tp.Sequence[int]]` or `list[torch.Size]`):
            The group shapes.
        shape (`torch.Size`):
            The shape of the tensor to be quantized.

    Returns:
        `list[torch.Size]`:
            list of view shapes of the scale tensor for each quantization group.
    """
    # 判断并推断group_shapes的类型
    if not isinstance(group_shapes[0], torch.Size):
        group_shapes = infer_group_shapes(group_shapes=group_shapes, shape=shape)  # type: ignore
    # 验证group_shapes的类型
    assert all(isinstance(gs, torch.Size) for gs in group_shapes), "group_shapes must be a list of torch.Size"
    # 获取最小的组形状
    min_group_shape = group_shapes[-1]

    # 推断每个量化组的缩放视图形状
    s_view_shapes = []          # 用于存储每个量化组的缩放视图形状
    for group_shape in group_shapes:
        s_view_shape = []       # 初始化当前组的缩放视图形状
        # 遍历对应维度
        for ts, gs, mgs in zip(shape, group_shape, min_group_shape, strict=True):
            # 计算分块数和缩放减少数
            num_groups, num_reduct = ts // gs, gs // mgs
            s_view_shape.append(num_groups)
            s_view_shape.append(num_reduct)
        # 添加到s_view_shapes
        s_view_shapes.append(torch.Size(s_view_shape))
    return s_view_shapes


def infer_shape(view_shape: torch.Size) -> torch.Size:
    """
    用于根据视图形状（view_shape）推断原始张量的形状。
    视图形状通常是通过 infer_view_shape() 函数生成的，用于调整张量的维度以进行分组操作。
    Infer the shape from the view shape.

    Args:
        view_shape (`torch.Size`):
            The view shape.

    Returns:
        `torch.Size`:
            The shape of the tensor.
    """
    # 即分块数#g乘以组大小gs
    return torch.Size(view_shape[i] * view_shape[i + 1] for i in range(0, len(view_shape), 2))
