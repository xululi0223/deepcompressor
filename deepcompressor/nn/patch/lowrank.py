# -*- coding: utf-8 -*-

import torch
import torch.linalg
import torch.nn as nn

from ...utils.hooks import AccumBranchHook, BaseInputPackager, BaseOutputPackager

__all__ = ["LowRankBranch"]


class LowRankBranch(nn.Module):
    """
    用于实现低秩（Low-Rank）分支的神经网络模块。
    该模块通过分解权重矩阵为两个低秩矩阵的乘积，以减少参数数量和计算复杂度，从而在保持模型性能的同时提高效率。
    该类支持不同的秩（rank）设定，并提供方法来初始化参数、获取有效权重、执行前向传播以及将模块封装为钩子（Hook）。
    """
    def __init__(
        self, in_features: int, out_features: int, rank: int, alpha: float = 1.0, weight: torch.Tensor | None = None
    ):
        """
        Args:
            in_features: 输入特征的维度。
            out_features: 输出特征的维度。
            rank: 分解的秩。决定了低秩矩阵的尺寸和参数数量。
            alpha: 分支输出的缩放因子。
            weight: 可选的初始权重张量，用于权重初始化。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        # 根据秩的不同，初始化不同的分支模块
        if rank == 0:
            # 如果秩为 0，则不使用低秩分支
            self.a, self.b = None, None
        elif rank < 0:
            # 如果秩小于 0，则self.a 为全连接层，self.b 为恒等映射
            self.a, self.b = nn.Linear(in_features, out_features, bias=False), nn.Identity()
        else:
            # 如果秩大于 0，则self.a 将输入特征映射到秩维度，self.b 将秩维度映射到输出特征
            self.a, self.b = nn.Linear(in_features, rank, bias=False), nn.Linear(rank, out_features, bias=False)
        # 初始化权重
        self.reset_parameters(weight)

    @torch.no_grad()
    def reset_parameters(self, weight: torch.Tensor | None = None) -> None:
        """
        初始化或重置分支模块的权重参数。
        
        Args:
            weight: 可选的初始权重张量，用于权重初始化。
        """
        # 如果权重为 None，则随机初始化权重
        if weight is None:
            if self.rank < 0:
                nn.init.zeros_(self.a.weight)
            elif self.rank > 0:
                nn.init.kaiming_uniform_(self.a.weight)
                nn.init.zeros_(self.b.weight)
            return
        assert weight.ndim == 2, "LinearLoRAHook only supports 2D input tensor"
        device, dtype = weight.device, weight.dtype
        self.to(device=device, dtype=dtype)
        out_features, in_features = weight.shape
        assert self.in_features == in_features, "Input features size mismatch"
        assert self.out_features == out_features, "Output features size mismatch"
        
        # 根据rank进行权重复制或分解
        if self.rank < 0:
            self.a.weight.data.copy_(weight)                # 直接复制权重
        elif self.rank > 0:
            u, s, vh = torch.linalg.svd(weight.double())    # 对权重进行奇异值分解
            # tensor: [oc, ic], u: [oc, oc], s: [oc], vh: [ic, ic]
            # us: [oc, rank], vh: [rank, ic]
            us = u[:, : self.rank] * s[: self.rank]         # 计算低秩分支中下采样矩阵
            vh = vh[: self.rank]                            # 计算低秩分支中上采样矩阵
            assert not us.isnan().any(), "NaN in U * S"
            assert not vh.isnan().any(), "NaN in V^T"
            assert not us.isinf().any(), "Inf in U * S"
            assert not vh.isinf().any(), "Inf in V^T"
            self.a.weight.data.copy_(vh.to(dtype))
            self.b.weight.data.copy_(us.to(dtype))

    def get_effective_weight(self) -> torch.Tensor | None:
        """
        获取模块的有效权重矩阵，根据不同的 rank 设置返回不同的权重形式。
        """
        if self.rank == 0:
            # 如果秩为 0，则不使用低秩分支
            return None
        elif self.rank < 0:
            # 如果秩小于 0，则直接返回全连接层的权重
            return self.a.weight
        else:
            # 如果秩大于 0，则返回两个低秩矩阵的乘积，重构出近似的权重矩阵
            return self.b.weight @ self.a.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor | None:
        """
        前向传播方法。
        """
        if self.a is None:
            return None
        else:
            # 通过self.a进行映射，再通过self.b进行映射，最后乘以缩放因子
            return self.alpha * self.b(self.a(input))

    def as_hook(
        self,
        input_packager: BaseInputPackager | None = None,
        output_packager: BaseOutputPackager | None = None,
    ) -> AccumBranchHook:
        """
        将 LowRankBranch 模块封装为一个累积分支钩子（AccumBranchHook），用于在神经网络中插入钩子函数，以便在特定位置累积分支输出。
        Wrap the module as a branch hook.

        Args:
            input_packager (`BaseInputPackager` or `None`, *optional*, defaults to `None`):
                Input packager.
            output_packager (`BaseOutputPackager` or `None`, *optional*, defaults to `None`):
                Output packager.
        Returns:
            `AccumBranchHook`:
                The branch hook.
        """
        return AccumBranchHook(self, input_packager=input_packager, output_packager=output_packager)
