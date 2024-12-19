# -*- coding: utf-8 -*-
"""Quantization SVD calibration module."""

from dataclasses import _MISSING_TYPE, MISSING

import torch
import torch.nn as nn

from ..data.common import TensorType
from ..nn.patch.lowrank import LowRankBranch
from ..quantizer.processor import Quantizer
from ..utils import math, tools
from ..utils.config import KeyEnableConfig
from .config import QuantLowRankCalibConfig, SearchBasedCalibObjective
from .search import SearchBasedCalibrator

__all__ = ["QuantLowRankCalibrator"]


class QuantLowRankCalibrator(SearchBasedCalibrator[QuantLowRankCalibConfig, LowRankBranch]):
    """
    用于量化低秩分支校准的校准器类。
    它继承自 SearchBasedCalibrator 类，专门用于在量化过程中优化和调整模型权重的低秩分解，以提高量化模型的性能和效率。
    The quantization low-rank branch calibrator.
    """

    def __init__(
        self,
        config: QuantLowRankCalibConfig,
        w_quantizer: Quantizer,
        x_quantizer: Quantizer | None,
        develop_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the calibrator.

        Args:
            config (`QuantLowRankCalibConfig`):
                The configuration of the quantization low-rank branch calibrator.
            w_quantizer (`Quantizer`):
                The quantizer for weights.
            x_quantizer (`Quantizer` or `None`):
                The quantizer for inputs.
            develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                The development data type.
        """
        # 检查校准器是否对给定的量化器启用
        if isinstance(config, KeyEnableConfig):
            assert config.is_enabled_for(w_quantizer.key), "The calibrator should be enabled for the quantizer."
        else:
            assert config.is_enabled(), "The calibrator should be enabled."
        # 调用父类 SearchBasedCalibrator 的构造函数，初始化必要的属性
        super().__init__(
            tensor_type=TensorType.Weights,
            config=config,
            w_quantizer=w_quantizer,
            x_quantizer=x_quantizer,
            y_quantizer=None,
            develop_dtype=develop_dtype,
        )
        assert self.needs_quant, "The tensor should be quantized."
        # 设置校准迭代次数
        self.num_iters = config.num_iters

    @property
    def population_size(self) -> int:
        """
        返回当前迭代的种群大小。
        Return the population size of the current iteration.
        """
        return 1

    @property
    def allows_x_quant_for_wgts(self) -> bool:
        """
        指示当 tensor_type 为 Weights 时，校准器是否允许对输入进行量化。
        Whether the calibrator allows input quantization when tensor_type is Weights.
        """
        return True

    @property
    def allows_w_quant_for_wgts(self) -> bool:
        """
        指示当 tensor_type 为 Weights 时，校准器是否需要对权重进行量化。
        hether the calibrator needs weight quantization when tensor_type is Weights.
        """
        return True

    def is_done(self) -> bool:
        """
        检查校准过程是否完成。
        Check if the calibration is done.
        """
        return self.iter >= self.num_iters or self.early_stopped

    def is_last_iter(self) -> bool:
        """
        检查当前迭代是否为最后一次迭代。
        Check if the current iteration is the last one.
        """
        return self.iter == self.num_iters - 1

    def _reset(self, x_wgts: list[torch.Tensor | nn.Parameter], **kwargs) -> None:  # noqa: C901
        """
        重置校准器的内部状态，为新的校准过程做准备。
        Reset the calibrator.

        Args:
            x_wgts (`list[torch.Tensor | nn.Parameter]`):
                The weights in x-w computation.
        """
        # 初始化 best_branch、best_error、error_history 和 early_stopped 等属性
        self.best_branch: LowRankBranch = None
        self.best_error: torch.Tensor = None
        self.error_history: list[tuple[float, float]] = []
        self.early_stopped = False

        # 决定是否串联多个权重张量
        if len(x_wgts) > 1 and not self.config.exclusive:
            self.w = torch.cat([wgt.data for wgt in x_wgts], dim=0)
        else:
            assert len(x_wgts) == 1
            self.w = x_wgts[0].data

        # 如果需要量化补偿，则量化每个权重张量并串联为qw
        if self.config.compensate:
            self.qw = torch.cat(
                [
                    self.w_quantizer.quantize(wgt.data, kernel=None, develop_dtype=self.develop_dtype).data
                    for wgt in x_wgts
                ],
                dim=0,
            )
        else:
            self.qw = 0
        # 初始化 hat_ws 列表，用于存储量化后的权重
        self.hat_ws: list[torch.Tensor] = [None] * len(x_wgts)
        # 使用 ocs 列表记录每个权重张量的第一维度大小
        self.ocs: list[int] = [wgt.shape[0] for wgt in x_wgts]

    def get_best(self) -> LowRankBranch:
        """
        获取当前校准过程中表现最好的低秩分支候选。
        Get the best candidate.

        Returns:
            `LowRankBranch`:
                The best candidate.
        """
        return self.best_branch

    def _ask(self) -> LowRankBranch:
        """
        生成下一个低秩分支候选。
        Ask for the next candidate.

        Returns:
            `LowRankBranch`:
                The next candidate.
        """
        # 创建新的 LowRankBranch 对象，基于当前权重 self.w 和 量化权重 self.qw，指定秩 rank
        branch = LowRankBranch(
            self.w.shape[1],
            self.w.shape[0],
            rank=self.config.rank,
            weight=self.w - self.qw,
        )
        # 初始化权重索引 wgt_idx 为 0
        self.wgt_idx = 0

        # 如果存在多个量化权重 hat_ws：
        # 1. 获取有效权重 lw，计算剩余权重 rw
        # 2. 遍历每个输出通道 oc， 量化对应的剩余权重片段，并更新 hat_ws
        # 3. 重新计算量化权重 qw
        # 4. 根据目标函数，如果不是 OutputsError，则将 lw 加回到量化权重 hat_ws 中
        if len(self.hat_ws) > 1:
            lw = branch.get_effective_weight()
            rw = self.w - lw
            oc_idx = 0
            for idx, oc in enumerate(self.ocs):
                self.hat_ws[idx] = self.w_quantizer.quantize(
                    rw[oc_idx : oc_idx + oc], kernel=None, develop_dtype=self.develop_dtype
                ).data
                oc_idx += oc
            self.qw = torch.cat(self.hat_ws, dim=0)
            if self.objective != SearchBasedCalibObjective.OutputsError:
                oc_idx = 0
                for idx, oc in enumerate(self.ocs):
                    self.hat_ws[idx].add_(lw[oc_idx : oc_idx + oc])
                    oc_idx += oc
        # 如果只有一个量化权重：
        # 1. 量化剩余权重并更新 qw
        # 2. 根据目标函数，如果不是 OutputsError，则将 lw 加回到量化权重 hat_ws 中
        else:
            lw = branch.get_effective_weight()
            self.qw = self.w_quantizer.quantize(self.w - lw, kernel=None, develop_dtype=self.develop_dtype).data
            if self.objective != SearchBasedCalibObjective.OutputsError:
                self.hat_ws = [self.qw + lw]
            else:
                self.hat_ws = [self.qw]
        return branch

    def _tell(self, error: list[torch.Tensor]) -> None:  # noqa: C901
        """
        接收当前候选分支的误差，更新最佳候选分支，并记录误差历史。
        Tell the error of the last candidate and update the best candidate.

        Args:
            errors (list[torch.Tensor]): The error of the last candidate.
        """
        # 如果误差长度大于1，则将所有误差相加
        if len(error) > 1:
            error = [sum(error)]
        # 确保误差是一个单一的 torch.Tensor 值
        error = error[0]
        assert isinstance(error, torch.Tensor)
        assert error.numel() == 1, "The error should only have one value."
        # 如果当前误差优于或等于最佳误差，则更新 best_error 和 best_branch
        if self.best_error is None or error <= self.best_error:
            self.best_error = error
            self.best_branch = self.candidate
        # 如果误差没有改善且配置中起用了提前停止，则设置 early_stopped 为 True
        elif self.config.early_stop:
            self.early_stopped = True
        # 如果日志级别为 DEBUG，则记录误差历史
        if self.logger.level <= tools.logging.DEBUG:
            self.error_history.append(
                (
                    math.root_(error.to(torch.float64), self.config.degree).item(),
                    math.root_(self.best_error.to(torch.float64), self.config.degree).item(),
                )
            )
            # 每10次迭代、最后一次迭代或提前停止时，打印最近10次迭代的误差信息
            if self.iter % 10 == 9 or self.is_last_iter() or self.early_stopped:
                iter_end = ((self.iter + 10) // 10) * 10
                iter_start = iter_end - 10
                iter_end = min(iter_end, self.iter + 1)
                history = self.error_history[iter_start:iter_end]
                self.logger.debug("  -      iter  = [%s]", ", ".join(f"{i:10d}" for i in range(iter_start, iter_end)))
                self.logger.debug("  -      error = [%s]", ", ".join(f"{e[0]:10.4f}" for e in history))
                self.logger.debug("  - best error = [%s]", ", ".join(f"{e[1]:10.4f}" for e in history))

    def _process_x_in_xw(self, x: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        """
        处理输入张量 x，根据需要进行量化。

        Args:
            x: 输入张量。
            channels_dim: 通道维度。
        """
        # 如果不需要对权重进行输入量化，则直接返回原始输入。
        if not self.needs_x_quant_for_wgts:
            return x
        # 否则，使用 x_quantizer 对输入进行量化，并返回量化后的数据。
        return self.x_quantizer.quantize(x, channels_dim=channels_dim, develop_dtype=self.develop_dtype).data

    def _process_w_in_xw(self, w: torch.Tensor) -> torch.Tensor:
        """
        处理权重张量 w，根据需要使用量化后的权重 hat_w。

        Args:
            w: 权重张量。
        """
        # 获取当前索引 wgt_idx 对应的量化权重 hat_w
        hat_w = self.hat_ws[self.wgt_idx]
        # 将对应位置的 hat_w 设置为 None，表示1️已使用
        self.hat_ws[self.wgt_idx] = None
        # 更新权重索引 wgt_idx
        self.wgt_idx += 1
        # 如果需要对权重进行量化，则返回 hat_w，否则返回原始权重 w
        return hat_w if self.needs_w_quant_for_wgts else w

    # 下面的方法不应该被调用，调用时会抛出运行时错误
    def _process_y_in_yx(self, y: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        raise RuntimeError("_process_y_in_yx should not be called in QuantSVDCalibrator.")

    def _process_x_in_yx(self, x: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        raise RuntimeError("_process_x_in_yx should not be called in QuantSVDCalibrator.")

    def _process_xw_in_yx(self, w: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("_process_xw_in_yx should not be called in QuantSVDCalibrator.")

    def _process_yw_in_yx(self, w: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("_process_yw_in_yx should not be called in QuantSVDCalibrator.")

    def _process_wgts_centric_mod(
        self, wgts: list[nn.Parameter], mods: list[nn.Module], update_state_dict: bool = True, **kwargs
    ) -> None:
        """
        处理权重中心化的模块，应用低秩分支到模块权重，更新权重数据，并注册钩子以跟踪权重变化。

        Args:
            wgts: 权重参数列表。
            mods: 模块列表。
            update_state_dict: 是否更新模块的状态字典。
        """
        # 断言 hat_ws、wgts 和 mods 的长度相等
        assert len(self.hat_ws) == len(wgts) == len(mods)
        # 获取当前候选 shared 分支
        shared = self.candidate
        # 如果存在多个量化权重 hat_ws：
        # 遍历每个模块、权重和量化权重
            # 1. 如果需要更新状态字典，则记录原始权重数据
            # 2. 更新权重数据为量化后的权重 hat_w
            # 3. 创建新的 LowRankBranch 对象，设置其参数 a 和 b，并复制权重
            # 4. 注册钩子以跟踪权重变化
        if len(self.hat_ws) > 1:
            oc_idx = 0
            for mod, wgt, hat_w in zip(mods, wgts, self.hat_ws, strict=True):
                if update_state_dict:
                    self._state_dict.append((wgt, wgt.data))
                wgt.data = hat_w
                branch = LowRankBranch(wgt.shape[1], wgt.shape[0], rank=self.config.rank)
                branch.a = shared.a
                branch.b.to(dtype=wgt.dtype, device=wgt.device)
                branch.b.weight.copy_(shared.b.weight[oc_idx : oc_idx + wgt.data.shape[0]])
                oc_idx += wgt.data.shape[0]
                self._hooks.append(branch.as_hook().register(mod))
        # 如果只有一个量化权重：
        # 1. 如果需要更新状态字典，则记录原始权重数据
        # 2. 更新权重数据为量化后的权重
        # 3. 注册 shared 分支的钩子
        else:
            if update_state_dict:
                self._state_dict.append((wgts[0], wgts[0].data))
            wgts[0].data = self.hat_ws[0]
            self._hooks.append(shared.as_hook().register(mods))

        # 如果需要对权重进行输入量化，则注册 x_quantizer 的钩子
        if self.needs_x_quant_for_wgts:
            self._hooks.append(self.x_quantizer.as_hook().register(mods))
        # 重置 hat_ws 列表为全 None
        self.hat_ws = [None] * len(self.hat_ws)
