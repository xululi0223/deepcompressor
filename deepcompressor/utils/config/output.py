# -*- coding: utf-8 -*-
"""Output configuration."""

import os
from dataclasses import dataclass, field
from datetime import datetime as DateTime

from omniconfig import configclass

__all__ = ["OutputConfig"]


@configclass
@dataclass
class OutputConfig:
    """
    管理输出配置，包含了输出目录的路径管理、锁定机制以及时间戳生成等功能。
    Output configuration.

    Args:
        root (`str`, *optional*, defaults to `"runs"`):
            The output root directory.
        dirname (`str`, *optional*, defaults to `"default"`):
            The output directory name.
        job (`str`, *optional*, defaults to `"run"`):
            The job name.

    Attributes:
        dirpath (`str`):
            The output directory path.
        timestamp (`str`):
            The timestamp.
    """
    # 类属性定义
    # 输出根目录
    root: str = "runs"
    # 输出目录名称
    dirname: str = "default"
    # 作业名称
    job: str = "run"
    # 输出目录的完整路径
    dirpath: str = field(init=False)
    # 时间戳，用于标识输出时间
    timestamp: str = field(init=False)

    def __post_init__(self):
        """
        进一步初始化或处理属性
        """
        # 生成时间戳
        self.timestamp = self.generate_timestamp()
        # 设置输出目录路径
        self.dirpath = os.path.join(self.root, self.dirname)

    @property
    def running_dirpath(self) -> str:
        """
        获取运行目录路径。
        Get the running directory path.
        """
        return f"{self.dirpath}.RUNNING"

    @property
    def error_dirpath(self) -> str:
        """
        获取错误目录路径。
        Get the error directory path.
        """
        return f"{self.dirpath}.ERROR"

    @property
    def job_dirname(self) -> str:
        """
        获取作业目录名称。
        Get the job directory name.
        """
        return f"{self.job}-{self.timestamp}"

    @property
    def job_dirpath(self) -> str:
        """
        获取作业目录路径。
        Get the job directory path.
        """
        return os.path.join(self.dirpath, self.job_dirname)

    @property
    def running_job_dirname(self) -> str:
        """
        获取运行作业目录名称。
        Get the running job directory name.
        """
        return f"{self.job_dirname}.RUNNING"

    @property
    def error_job_dirname(self) -> str:
        """
        获取错误作业目录名称。
        Get the error job directory name.
        """
        return f"{self.job_dirname}.ERROR"

    @property
    def running_job_dirpath(self) -> str:
        """
        获取运行作业目录路径。
        Get the running job directory path.
        """
        return os.path.join(self.running_dirpath, self.running_job_dirname)

    def lock(self) -> None:
        """
        锁定运行中的作业目录，防止其他进程访问或修改。
        Lock the running (job) directory.
        """
        # 尝试重命名目录
        try:
            # 首先尝试重命名dirpath
            if os.path.exists(self.dirpath):
                os.rename(self.dirpath, self.running_dirpath)
            # 其次尝试重命名error_dirpath
            elif os.path.exists(self.error_dirpath):
                os.rename(self.error_dirpath, self.running_dirpath)
        except Exception:
            pass
        # 创建运行中的作业目录running_job_dirpath
        os.makedirs(self.running_job_dirpath, exist_ok=True)

    def unlock(self, error: bool = False) -> None:
        """
        解锁运行中的作业目录，根据是否发生错误，将目录重命名回原始状态或错误状态。
        Unlock the running (job) directory.
        """
        # 确定目标目录路径
        job_dirpath = os.path.join(self.running_dirpath, self.error_job_dirname if error else self.job_dirname)
        # 重命名运行中的作业目录running_job_dirpath
        os.rename(self.running_job_dirpath, job_dirpath)
        # 检查是否有其他锁
        if not self.is_locked_by_others():
            # 根据error参数，重命名目录
            os.rename(self.running_dirpath, self.error_dirpath if error else self.dirpath)

    def is_locked_by_others(self) -> bool:
        """
        检查运行目录是否被其他进程锁定。
        Check if the running directory is locked by others.
        """
        # 获取当前运行的作业目录名称
        running_job_dirname = self.running_job_dirname
        # 遍历running_dirpath目录下的所有目录
        for dirname in os.listdir(self.running_dirpath):
            # 如果目录以.RUNNING结尾且不是当前运行的作业目录
            if dirname.endswith(".RUNNING") and dirname != running_job_dirname:     # TODO: correct?
                return True
        return False

    def get_running_path(self, filename: str) -> str:
        """
        生成在运行目录中的文件路径，附加时间戳以确保文件唯一性。
        Get the file path in the running directory.
        """
        # 分割文件名和扩展名
        name, ext = os.path.splitext(filename)
        # 生成新文件名并组合成完整路径
        return os.path.join(self.running_dirpath, f"{name}-{self.timestamp}{ext}")

    def get_running_job_path(self, filename: str) -> str:
        """
        生成在运行中的作业目录中的文件路径，附加时间戳以确保文件唯一性。
        Get the file path in the running job directory.
        """
        # 分割文件名和扩展名
        name, ext = os.path.splitext(filename)
        # 生成新文件名并组合成完整路径
        return os.path.join(self.running_job_dirpath, f"{name}-{self.timestamp}{ext}")

    @staticmethod
    def generate_timestamp() -> str:
        """
        生成当前时间的时间戳。
        Generate a timestamp."""
        return DateTime.now().strftime("%y%m%d.%H%M%S")
