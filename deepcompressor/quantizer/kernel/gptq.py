# -*- coding: utf-8 -*-
"""GPTQ Quantization kernel."""

import gc
import math
from dataclasses import dataclass

import torch
from omniconfig import configclass

from ...data.cache import TensorCache
from ...data.dtype import QuantDataType
from ...data.range import QuantRange, RangeBound
from ...data.zero import ZeroPointDomain
from ...utils import tools
from ...utils.common import num2str
from ..config.kernel import BaseQuantKernel, BaseQuantKernelConfig
from ..impl.simple import simple_quantize

__all__ = ["gptq_quantize"]


@configclass
@dataclass
class QuantGptqConfig(BaseQuantKernelConfig):
    """
    用于配置 GPTQ量化过程中的参数。
    该类继承自 BaseQuantKernelConfig，通过数据类（@dataclass）和配置类装饰器（@configclass）实现，支持自动生成初始化方法及其他便利功能。
    Configuration for GPTQ quantization.

    Args:
        damp_percentage (`float`, *optional*, defaults to `0.01`):
            The percentage of damping.
        block_size (`int`, *optional*, defaults to `128`):
            The block size of the GPTQ quantization.
        num_inv_tries (`int`, *optional*, defaults to `200`):
            The number of tries for the inverse.
        hessian_block_size (`int`, *optional*, defaults to `-1`):
            The block size when calculing the Hessian.
    """

    # 阻尼百分比，用于避免数值不稳定
    damp_percentage: float = 0.01
    # GPTQ量化的块大小，控制量化过程中的分块操作
    block_size: int = 128
    # 逆矩阵计算的尝试次数，用于处理可能的数值不稳定
    num_inv_tries: int = 200
    # 计算Hessian矩阵时的块大小
    hessian_block_size: int = -1

    @property
    def name(self) -> str:
        """
        返回该配置的名称 "GPTQ"。
        """
        return "GPTQ"

    def build(self) -> "QuantGptqKernel":
        """
        根据当前配置实例构建相应的量化内核实例。
        """
        return QuantGptqKernel(self)

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """
        根据配置参数生成对应的目录名称，用于文件系统或日志的组织和管理。
        Generate the directory names of the configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix for the directory names.

        Returns:
            `list[str]`:
                The directory names.
        """
        # 添加阻尼百分比和块大小
        name = f"gptq.d{num2str(self.damp_percentage)}.b{num2str(self.block_size)}"
        # 添加前缀，返回目录名称列表
        return [f"{prefix}.{name}" if prefix else name]


class QuantGptqKernel(BaseQuantKernel):
    """
    GPTQ 量化内核的具体实现，继承自 BaseQuantKernel。
    该类负责根据配置参数对输入张量进行 GPTQ 量化操作。
    核心方法为 quantize，它调用了 gptq_quantize 函数，执行具体的量化逻辑。
    """
    def __init__(self, config: "QuantGptqConfig"):
        """
        Args:
            config: GPTQ 量化内核的配置实例。
        """
        self.config = config

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        view_shape: torch.Size,
        quant_dtype: QuantDataType,
        zero_domain: ZeroPointDomain | None,
        scale: torch.Tensor,
        zero: torch.Tensor,
        inputs: TensorCache,
        quant_range: QuantRange | None = None,
        range_bound: RangeBound | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        量化张量。
        Quantize the tensor.

        Args:
            tensor (`torch.Tensor`):
                The tensor to quantize.
            view_shape (`torch.Size`):
                The view shape when quantizing the tensor.
            quant_dtype (`QuantDataType`):
                The quantization data type.
            zero_domain (`ZeroPointDomain` or `None`):
                The zero point domain.
            scale (`torch.Tensor`):
                The scale tensor.
            zero (`torch.Tensor`):
                The zero point tensor.
            inputs (`TensorCache`):
                The input activations.
            quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                The quantization range.
            range_bound (`RangeBound` or `None`, *optional*, defaults to `None`):
                The range bound.
            **kwargs: Other keyword arguments.

        Returns:
            `torch.Tensor`:
                The quantized tensor in the shape of ``view_shape``.
        """
        # 确保输入张量、缩放因子和零点张量不需要梯度
        assert not tensor.requires_grad, "tensor must not require gradient."
        assert not scale.data.requires_grad, "scale must not require gradient."
        assert not zero.data.requires_grad, "zero must not require gradient."
        
        # 调用 gptq_quantize 函数，执行 GPTQ 量化操作
        return gptq_quantize(
            tensor,
            view_shape=view_shape,
            quant_dtype=quant_dtype,
            zero_domain=zero_domain,
            scale=scale,
            zero=zero,
            gptq_config=self.config,
            inputs=inputs,
            quant_range=quant_range,
            range_bound=range_bound,
        )


@torch.no_grad()
def gptq_quantize(  # noqa: C901
    tensor: torch.Tensor,
    *,
    view_shape: torch.Size,
    quant_dtype: QuantDataType,
    zero_domain: ZeroPointDomain | None,
    scale: torch.Tensor,
    zero: torch.Tensor,
    gptq_config: QuantGptqConfig,
    inputs: TensorCache,
    quant_range: QuantRange | None = None,
    range_bound: RangeBound | None = None,
) -> torch.Tensor:
    """
    实现了 GPTQ 量化算法，用于对输入张量进行高效且精确的量化操作。
    Quantize the tensor using the GPTQ quantization kernel.

    Args:
        tensor (`torch.Tensor`):
            The tensor to quantize.
        view_shape (`torch.Size`):
            The view shape when quantizing the tensor.
        quant_dtype (`QuantDataType`):
            The quantization data type.
        zero_domain (`ZeroPointDomain` or `None`):
            The zero point domain.
        scale (`torch.Tensor`):
            The scale tensor.
        zero (`torch.Tensor`):
            The zero point tensor.
        inputs (`TensorCache`):
            The input activations.
        quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
            The quantization range.
        range_bound (`RangeBound` or `None`, *optional*, defaults to `None`):
            The range bound.

    Returns:
        `torch.Tensor`:
            The quantized tensor in the shape of ``view_shape``.
    """
    # 重塑和重新排列张量，以便进行批量量化
    # 将输入张量重塑为view_shape指定的形状
    view_tensor = tensor.view(view_shape)
    view_shape = view_tensor.shape  # remove any -1 in the view_shape
    # region step 1: reshape the tensor to (#g0 * gs0, #g1 * #g2 * ... * gs1 * gs2, ...)
    # 计算视图形状长度，决定后续的重排列顺序
    len_view_shape = len(view_shape)
    # view_tensor: (#g0, gs0, #g1, gs1, #g2, gs2, ...) -> (#g0, gs0, #g1, #g2, ..., gs1, gs2, ...)
    reshaped_tensor = view_tensor.permute(0, 1, *range(2, len_view_shape, 2), *range(3, len_view_shape, 2)) # 便于后续的重塑操作，将张量分块为行和列组
    # reshaped_tensor: (#g0 * gs0, #g1 * #g2 * ... * gs1 * gs2 * ...)
    reshaped_tensor = reshaped_tensor.reshape(view_shape[0] * view_shape[1], -1)        # 展平成二维，以便进行矩阵操作
    # 计算行组数、列组数
    num_row_groups, num_column_groups = view_shape[0], view_shape[2::2].numel()
    # 计算行组大小、列组大小
    row_group_size, column_group_size = view_shape[1], view_shape[3::2].numel()
    # 获取张量的形状
    num_rows, num_columns = reshaped_tensor.shape
    # 重塑缩放张量，以便与重塑后的张量对应
    reshaped_scale = scale.view(num_row_groups, 1, num_column_groups)
    # 检查零点张量是否为标量，如果不是，则重塑零点张量，以便与重塑后的张量对应
    zero_is_number = isinstance(zero, (int, float)) or zero.numel() == 1
    reshaped_zero = zero if zero_is_number else zero.view(num_row_groups, 1, num_column_groups)
    # endregion
    
    # 计算Hessian矩阵，利用输入激活计算Hessian矩阵的逼近，用于后续的量化优化
    # region step 2: get Hessian matrix
    hessian = torch.zeros((num_columns, num_columns), device=view_tensor.device, dtype=view_tensor.dtype)   # 初始化Hessian矩阵
    # 遍历输入激活，计算Hessian矩阵
    for x in inputs.data:
        x: torch.Tensor = inputs.reshape(x.view(-1, *x.shape[inputs.channels_dim :]))
        # 如果hessian_block_size大于0且x的样本数超过该块大小，则分块计算Hessian矩阵
        if gptq_config.hessian_block_size > 0 and x.shape[0] > gptq_config.hessian_block_size:
            for b in range(0, x.shape[0], gptq_config.hessian_block_size):
                # 获取当前块的子张量
                _x = x[b : min(b + gptq_config.hessian_block_size, x.shape[0])]
                # 根据样本数量缩放子张量
                _x = math.sqrt(2 / inputs.num_samples) * _x.to(device=view_tensor.device, dtype=view_tensor.dtype)
                # 累积Hessian矩阵，即计算子张量的内积
                hessian += torch.matmul(_x.t(), _x)
        # 否则，直接计算Hessian矩阵整体
        else:
            # 缩放张量
            x = math.sqrt(2 / inputs.num_samples) * x.to(device=view_tensor.device, dtype=view_tensor.dtype)
            # 累积Hessian矩阵，即计算张量的内积
            hessian += torch.matmul(x.t(), x)
    # 处理零对角线元素
    dead = hessian.diagonal() == 0
    # 将零对角线元素设置为1，避免数值不稳定
    hessian[dead, dead] = 1
    # 将对应的reshaped_tensor的列设置为0
    reshaped_tensor[:, dead] = 0
    del x, inputs, dead
    gc.collect()
    torch.cuda.empty_cache()
    # endregion
    
    # Hessian矩阵的置换
    # region step 3: permute the Hessian matrix
    # 提取Hessian矩阵的对角线元素，作为重要性指标
    importance = torch.diag(hessian)  # (#g1 * #g2 * ... * gs1 * gs2 * ..., )
    # 根据重要性指标进行排序，获取置换索引
    permute = torch.argsort(importance, descending=True)
    # 按照置换索引重新排列Heissian矩阵和重塑后的张量
    hessian = hessian[permute][:, permute]
    reshaped_tensor = reshaped_tensor[:, permute]
    # 生成将列恢复到原始顺序的逆置换索引
    inverse_permute = torch.argsort(permute)
    del importance
    # endregion
    
    # 应用阻尼处理，避免数值不稳定
    # region step 4: apply dampening to avoid numerical instability
    hessian_diag = hessian.diagonal()               # 获取Hessian矩阵的对角线元素
    hessian_diag_mean = hessian_diag.mean()         # 计算对角线元素的均值
    hessian_diag += gptq_config.damp_percentage * hessian_diag_mean     # 对对角线元素应用阻尼处理
    # endregion
    
    # 计算Hessian矩阵的逆矩阵
    # region step 5: get the inverse of the Hessian matrix
    stable_inv, num_inv_tries = False, 0            # 初始化变量。stable_inv表示Hessian逆矩阵是否稳定，num_inv_tries表示尝试次数
    # 循环尝试计算逆矩阵
    while (not stable_inv) and num_inv_tries < gptq_config.num_inv_tries:
        num_inv_tries += 1          # 更新尝试次数
        # 尝试Cholesky分解计算逆矩阵
        try:
            hessian_inv = torch.linalg.cholesky(hessian)                    # 对Hessian矩阵进行Cholesky分解，得到下三角矩阵
            hessian_inv = torch.cholesky_inverse(hessian_inv)               # 计算逆矩阵
            hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)    # 对逆矩阵进行Cholesky分解，得到上三角矩阵
        # 如果计算失败，则增加阻尼处理
        except RuntimeError:
            hessian_diag += (gptq_config.damp_percentage * 0.1) * hessian_diag_mean     
            continue
        stable_inv = True
    if num_inv_tries > 1:
        logger = tools.logging.getLogger(f"{__name__}.GPTQ")
        logger.debug("        - Hessian is not stable %s %d tries.", "until" if stable_inv else "after", num_inv_tries)
    # 确保逆矩阵不包含NaN和Inf
    assert not hessian_inv.isinf().any(), "Inverse of Hessian matrix contains Inf."
    assert not hessian_inv.isnan().any(), "Inverse of Hessian matrix contains NaN."
    del hessian, hessian_diag, hessian_diag_mean, num_inv_tries
    # endregion
    
    # 量化张量，对重塑后的张量进行分块量化，并应用误差补偿
    # region step 6: quantize the tensor
    qtensor = torch.zeros_like(reshaped_tensor)                     # 初始化量化张量
    # 按块遍历列组，对每个列组进行量化
    for c_start in range(0, num_columns, gptq_config.block_size):
        c_end = min(c_start + gptq_config.block_size, num_columns)      # 计算块的结束索引
        block_tensor = reshaped_tensor[:, c_start:c_end].clone()        # 获取当前块的子张量
        block_qtensor = qtensor[:, c_start:c_end]                       # 获取当前块的量化子张量
        block_hessian_inv = hessian_inv[c_start:c_end, c_start:c_end]   # 获取当前块的Hessian逆矩阵
        block_error = torch.zeros_like(block_tensor)                    # 初始化当前块的误差张量
        # 遍历列组，对每个列组进行量化
        for _c in range(c_end - c_start):
            c = c_start + _c                                            # 获取当前列索引
            column = block_tensor[:, _c]  # (#g0 * gs0, )               # 获取当前列
            pos_diag = block_hessian_inv[_c, _c]                        # 获取Hessian逆矩阵对角线元素
            column_group_index = permute[c] // column_group_size        # 获取列组索引
            column_scale = reshaped_scale[:, :, column_group_index]  # (#g0, 1)                             # 获取缩放因子
            column_zero = reshaped_zero if zero_is_number else reshaped_zero[:, :, column_group_index]      # 获取零点张量
            qcolumn = column.view(num_row_groups, row_group_size).clone()  # (#g0, gs0)                     # 重塑并复制当前列
            # 应用范围裁剪
            if range_bound is not None and range_bound.is_set():
                qcolumn = qcolumn.clamp_(min=range_bound.min, max=range_bound.max)
            # 应用缩放和零点偏移
            if zero_domain == ZeroPointDomain.PostScale:
                qcolumn = qcolumn.add_(column_zero)
            qcolumn = qcolumn.div_(column_scale)
            if zero_domain == ZeroPointDomain.PreScale:
                qcolumn = qcolumn.add_(column_zero)
            # 量化当前列
            qcolumn = simple_quantize(
                qcolumn, quant_dtype=quant_dtype, has_zero_point=zero_domain is not None, quant_range=quant_range
            )
            # 存储量化结果
            block_qtensor[:, _c] = qcolumn.view(-1)  # ! copy the quantized column
            # 反量化，反向应用缩放和零点偏移
            if zero_domain == ZeroPointDomain.PreScale:
                qcolumn = qcolumn.sub_(column_zero)
            qcolumn = qcolumn.mul_(column_scale)
            if zero_domain == ZeroPointDomain.PostScale:
                qcolumn = qcolumn.sub_(column_zero)
            # 计算量化误差
            column_error = column.sub_(qcolumn.view(column.shape)).div_(pos_diag)
            # 将当前列的误差添加到误差张量
            block_error[:, _c] = column_error.view(-1)
            # 应用误差补偿
            block_tensor[:, _c:] -= column_error.view(-1, 1).matmul(block_hessian_inv[_c, _c:].view(1, -1))
        # 应用误差补偿到剩余列
        reshaped_tensor[:, c_end:] -= block_error.matmul(hessian_inv[c_start:c_end, c_end:])
    # 恢复量化张量原始的列顺序
    qtensor = qtensor[:, inverse_permute]
    # endregion
    
    # 重塑张量，恢复原始形状
    # region step 7: reshape the tensor back to (#g0, gs0, #g1, gs1, #g2, gs2, ...)
    _view_shape = view_shape[:2] + view_shape[2::2] + view_shape[3::2]          # 重新组织视图形状，便于后续的重塑操作
    # qtensor: (#g0 * gs0, #g1 * #g2 * ... * gs1 * gs2, ...) -> (#g0, gs0, #g1, #g2, ..., gs1, gs2, ...)
    qtensor = qtensor.reshape(_view_shape)                                      # 重塑量化张量
    # qtensor: (#g0, gs0, #g1, #g2, ..., gs1, gs2, ...) -> (#g0, gs0, #g1, gs1, #g2, gs2, ...)
    # 构建维度排列的索引序列，用于调整张量的维度顺序
    permute_dims = [0, 1]                                           # 保持前两个维度不变
    for i in range(1, len_view_shape // 2):
        # 交替排列后续维度
        permute_dims.append(1 + i)
        permute_dims.append(len_view_shape // 2 + i)
    qtensor = qtensor.permute(*permute_dims).reshape(view_shape)    # 重新排列与重塑张量
    # endregion
    
    # 检查量化张量是否包含NaN和Inf
    assert not qtensor.isnan().any(), "GPTQ Quantized tensor contains NaN."
    assert not qtensor.isinf().any(), "GPTQ Quantized tensor contains Inf."
    return qtensor
