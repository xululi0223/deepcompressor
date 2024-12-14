# -*- coding: utf-8 -*-
"""Net configurations."""

import os
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass

from omniconfig import configclass

__all__ = ["BaseModelConfig"]


@configclass
@dataclass
class BaseModelConfig(ABC):
    """
    抽象基类，作为所有模型配置的基类。
    Base class for all model configs.

    Args:
        name (`str`):
            Name of the model.
        family (`str`, *optional*, defaults to `""`):
            Family of the model. If not specified, it will be inferred from the name.
        path (`str`, *optional*, defaults to `""`):
            Path of the model.
        root (`str`, *optional*, defaults to `""`):
            Root directory path for models.
        local_path (`str`, *optional*, defaults to `""`):
            Local path of the model.
        local_root (`str`, *optional*, defaults to `""`):
            Local root directory path for models.
    """

    # 模型名称
    name: str
    # 模型的类别
    family: str = ""
    # 模型的路径
    path: str = ""
    # 模型的根目录路径
    root: str = ""
    # 模型的本地路径
    local_path: str = ""
    # 模型的本地根目录路径
    local_root: str = ""

    def __post_init__(self):
        """
        进一步初始化或处理属性。
        """
        # 推断family
        if not self.family:
            self.family = self.name.split("-")[0]
        # 扩展本地根路径
        self.local_root = os.path.expanduser(self.local_root)
        # 设置local_path
        if not self.local_path:
            self.local_path = os.path.join(self.local_root, self.family, self.name)
        # 设置path
        if not self.path:
            self.path = os.path.join(self.root, self.family, self.name)
        # 优先使用本地路径
        if os.path.exists(self.local_path):
            self.path = self.local_path

    @abstractmethod
    def build(self, *args, **kwargs) -> tp.Any:
        """
        抽象方法，根据配置构建模型。
        Build model from config.
        """
        ...
