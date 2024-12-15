# -*- coding: utf-8 -*-
"""Dataclass utilities."""

from dataclasses import _FIELD, _FIELD_CLASSVAR, _FIELD_INITVAR, _FIELDS, Field

__all__ = ["get_fields"]


def get_fields(class_or_instance, *, init_vars: bool = False, class_vars: bool = False) -> tuple[Field, ...]:
    """
    用于获取数据类的字段。
    Get the fields of the dataclass.

    Args:
        class_or_instance:
            The dataclass type or instance.
        init_vars (`bool`, *optional*, defaults to `False`):
            Whether to include the init vars.
        class_vars (`bool`, *optional*, defaults to `False`):
            Whether to include the class vars.

    Returns:
        tuple[Field, ...]: The fields.
    """
    # 尝试获取字段
    try:
        fields = getattr(class_or_instance, _FIELDS)
    except AttributeError:
        raise TypeError("must be called with a dataclass type or instance") from None
    # 返回值构建
    return tuple(
        v
        for v in fields.values()        # 包含常规字段
        if v._field_type is _FIELD
        or (init_vars and v._field_type is _FIELD_INITVAR)      # 包含初始化变量字段
        or (class_vars and v._field_type is _FIELD_CLASSVAR)    # 包含类变量字段
    )
