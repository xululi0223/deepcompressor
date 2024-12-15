# -*- coding: utf-8 -*-
"""Logging tools."""

import logging
import sys
import typing as tp

from tqdm.contrib.logging import logging_redirect_tqdm

__all__ = [
    "CRITICAL",
    "FATAL",
    "ERROR",
    "WARNING",
    "WARN",
    "INFO",
    "DEBUG",
    "NOTSET",
    "log",
    "info",
    "debug",
    "warning",
    "error",
    "critical",
    "Formatter",
    "basicConfig",
    "setup",
    "getLogger",
    "redirect_tqdm",
]


CRITICAL = logging.CRITICAL
FATAL = logging.FATAL
ERROR = logging.ERROR
WARNING = logging.WARNING
WARN = logging.WARN
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET


redirect_tqdm = logging_redirect_tqdm
shutdown = logging.shutdown
Logger = logging.Logger


def getLogger(name: str | None = None) -> logging.Logger:
    """
    获取一个指定名称的日志记录器（Logger）。
    如果未提供名称，则返回根日志记录器。
    Get a logger with the given name.

    Args:
        name (`str` or `None`, *optional*, defaults to `None`): The name of the logger.

    Returns:
        logging.Logger: The logger.
    """
    return logging.getLogger(name)


def log(level: int, msg: str, logger: logging.Logger | None = None) -> None:
    """
    使用指定的日志级别记录一条消息。
    如果消息包含多行，会逐行记录。
    Log a message with the given level.

    Args:
        level (`int`): The logging level.
        msg (`str`): The message to log.
        logger (`logging.Logger` or `None`, *optional*, defaults to `None`):
            The logger to use. If `None`, the root logger is used.
    """
    # 如果未提供日志记录器，则使用根日志记录器
    if logger is None:
        logger = logging.getLogger()
    # 如果日志级别未启用，则直接返回
    if not logger.isEnabledFor(level):
        return
    # 将消息转换为字符串
    msg = str(msg)
    # 如果消息包含换行符，则逐行记录
    if "\n" in msg:
        for line in msg.split("\n"):
            log(level, line, logger)
    else:
        logger.log(level, msg)


def info(msg: str, logger: logging.Logger | None = None):
    """
    使用 INFO 级别记录一条消息。
    Log a message with the INFO level.

    Args:
        msg (`str`): The message to log.
        logger (`logging.Logger` or `None`, *optional*, defaults to `None`):
            The logger to use. If `None`, the root logger is used.
    """
    log(logging.INFO, msg, logger)


def debug(msg: str, logger: logging.Logger | None = None):
    """
    使用 DEBUG 级别记录一条消息。
    Log a message with the DEBUG level.

    Args:
        msg (`str`): The message to log.
        logger (`logging.Logger` or `None`, *optional*, defaults to `None`):
            The logger to use. If `None`, the root logger is used.
    """
    log(logging.DEBUG, msg, logger)


def warning(msg: str, logger: logging.Logger | None = None):
    """
    使用 WARNING 级别记录一条消息。
    Log a message with the WARNING level.

    Args:
        msg (`str`): The message to log.
        logger (`logging.Logger` or `None`, *optional*, defaults to `None`):
            The logger to use. If `None`, the root logger is used.
    """
    log(logging.WARNING, msg, logger)


def error(msg: str, logger: logging.Logger | None = None):
    """
    使用 ERROR 级别记录一条消息。
    Log a message with the ERROR level.

    Args:
        msg (`str`): The message to log.
        logger (`logging.Logger` or `None`, *optional*, defaults to `None`):
            The logger to use. If `None`, the root logger is used.
    """
    log(logging.ERROR, msg, logger)


def critical(msg: str, logger: logging.Logger | None = None):
    """
    使用 CRITICAL 级别记录一条消息。
    Log a message with the CRITICAL level.

    Args:
        msg (`str`): The message to log.
        logger (`logging.Logger` or `None`, *optional*, defaults to `None`):
            The logger to use. If `None`, the root logger is used.
    """
    log(logging.CRITICAL, msg, logger)


class Formatter(logging.Formatter):
    """
    自定义日志格式化器，支持缩进功能。
    A custom formatter for logging.
    """

    # 缩进量，用于控制消息前的空格数量
    indent = 0

    def __init__(self, fmt: str | None = None, datefmt: str | None = None, style: tp.Literal["%", "{", "$"] = "%"):
        """Initialize the formatter.

        Args:
            fmt (`str` or `None`, *optional*, defaults to `None`): The format string.
            datefmt (`str` or `None`, *optional*, defaults to `None`): The date format string.
            style (`str`, *optional*, defaults to `"%"`): The format style.
        """
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录。
        Format the record.

        Args:
            record (`logging.LogRecord`): The log record.

        Returns:
            str: The formatted record.
        """
        # 如果缩进量大于 0，则在消息前添加相应数量的空格
        record.message = " " * self.indent + record.getMessage()
        # 格式化时间（如果启用）
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        # 格式化消息
        s = self.formatMessage(record)
        # 如果启用异常信息，则格式化异常信息
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        # 如果启用堆栈信息，则格式化堆栈信息
        if record.stack_info:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + self.formatStack(record.stack_info)
        return s

    @staticmethod
    def indent_inc(delta: int = 2):
        """
        增加缩进量。
        Increase the indent.
        """
        Formatter.indent += delta

    @staticmethod
    def indent_dec(delta: int = 2):
        """
        减少缩进量。
        Decrease the indent.
        """
        Formatter.indent -= delta

    @staticmethod
    def indent_reset(indent: int = 0):
        """
        重置缩进量。
        Reset the indent.
        """
        Formatter.indent = indent


def basicConfig(**kwargs) -> None:
    """
    配置根日志记录器，设置格式化器。
    Configure the root logger.
    """
    # 获取格式化器参数
    fmt = kwargs.pop("format", None)
    # 获取日期格式化参数
    datefmt = kwargs.pop("datefmt", None)
    # 获取格式风格参数
    style = kwargs.pop("style", "%")
    # 配置根日志记录器
    logging.basicConfig(**kwargs)
    # 遍历根日志记录器的所有处理器，将自定义的格式化器应用到每个处理器
    for h in logging.root.handlers[:]:
        h.setFormatter(Formatter(fmt, datefmt, style))


def setup(
    path: str | None = None,
    level: int = logging.DEBUG,
    format: str = "%(asctime)s | %(levelname).1s | %(message)s",
    datefmt: str = "%y-%m-%d %H:%M:%S",
    **kwargs,
) -> None:
    """
    设置默认的日志配置，支持控制台和文件输出。
    Setup the default logging configuration.

    Args:
        path (`str` | `None`, *optional*, defaults to `None`):
            The path to the log file. If `None`, only the console is used.
        level (`int`, *optional*, defaults to `logging.DEBUG`): The logging level.
        format (`str`, *optional*, defaults to `"%(asctime)s | %(levelname).1s | %(message)s"`):
            The format string.
        datefmt (`str`, *optional*, defaults to `"%y-%m-%d %H:%M:%S"`): The date format string.
        **kwargs: Additional keyword arguments.
    """
    # 获取处理器列表
    handlers = kwargs.pop("handlers", None)
    # 获取是否强制覆盖的标志
    force = kwargs.pop("force", True)
    # 如果未提供handlers，则默认使用StreamHandler输出到控制台
    if handlers is None:
        handlers = [logging.StreamHandler(sys.stdout)]
        # 如果提供了path，则添加FileHandler输出到文件
        if path is not None:
            handlers.append(logging.FileHandler(path, mode="w"))
    # 配置根日志记录器
    basicConfig(
        level=level,
        format=format,
        datefmt=datefmt,
        handlers=handlers,
        force=force,
    )
