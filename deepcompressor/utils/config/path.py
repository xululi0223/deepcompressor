# -*- coding: utf-8 -*-
"""Path configuration."""

import os
import typing as tp

from ..dataclass import get_fields

__all__ = ["BasePathConfig"]


class BasePathConfig:
    """
    用于路径配置的基类，提供了一系列方法来管理和操作路径属性。
    Base path configuration."""

    def is_all_set(self) -> bool:
        """
        检查配置中的所有路径属性是否都已设置（即非空）。
        Check if the path configuration is all set.

        Returns:
            `bool`:
                Whether the path configuration is all set.
        """
        # 获取当前实例的所有字段
        fields = get_fields(self)
        # 遍历所有字段
        for f in fields:
            if not getattr(self, f.name):
                return False
        return True

    def is_all_empty(self) -> bool:
        """
        检查配置中的所有路径属性是否都为空。
        Check if the path configuration is all empty.

        Returns:
            `bool`:
                Whether the path configuration is all empty.
        """
        # 获取当前实例的所有字段
        fields = get_fields(self)
        # 遍历所有字段
        for f in fields:
            if getattr(self, f.name):
                return False
        return True

    def clone(self) -> tp.Self:
        """
        克隆当前路径配置实例，生成一个具有相同属性值的新实例。
        Clone the path configuration.

        Returns:
            `Self`:
                The cloned path configuration.
        """
        # 获取当前实例的所有字段
        fields = get_fields(self)
        # 创建一个新的实例，传入当前实例的所有属性值，实现克隆
        return self.__class__(**{f.name: getattr(self, f.name) for f in fields})

    def add_parent_dirs(self, *parent_dirs: str) -> tp.Self:
        """
        为配置中的每个路径添加指定的父目录。
        Add the parent directories to the paths.

        Args:
            parent_dirs (`str`):
                The parent directories.
        """
        # 获取当前实例的所有字段
        fields = get_fields(self)
        # 遍历所有字段，为每个字段的路径（存在的话）添加父目录
        for f in fields:
            path = getattr(self, f.name)
            if path:
                setattr(self, f.name, os.path.join(*parent_dirs, path))
        return self

    def add_children(self, *children: str) -> tp.Self:
        """
        为配置中的每个路径添加指定的子目录。
        Add the children to the paths.

        Args:
            children (`str`):
                The children paths.
        """
        # 获取当前实例的所有字段
        fields = get_fields(self)
        # 遍历所有字段，为每个字段的路径（存在的话）添加子目录
        for f in fields:
            path = getattr(self, f.name)
            if path:
                setattr(self, f.name, os.path.join(path, *children))
        return self

    def to_dirpath(self) -> tp.Self:
        """
        将配置中的每个路径转换为其所在的目录路径，即去除文件名部分。
        Convert the paths to directory paths.
        """
        # 获取当前实例的所有字段
        fields = get_fields(self)
        # 遍历所有字段，将每个字段的路径（存在的话）转换为其所在的目录路径
        for f in fields:
            path = getattr(self, f.name)
            if path:
                setattr(self, f.name, os.path.dirname(path))
        return self

    def apply(self, fn: tp.Callable) -> tp.Self:
        """
        对配置中的每个路径应用指定的函数，允许对路径进行批量处理或转换。
        Apply the function to the paths.

        Args:
            fn (`Callable`):
                The function to apply.
        """
        # 获取当前实例的所有字段
        fields = get_fields(self)
        # 遍历所有字段，对每个字段的路径（存在的话）应用指定的函数
        for f in fields:
            path = getattr(self, f.name)
            if path:
                setattr(self, f.name, fn(path))
        return self
