# -*- coding: utf-8 -*-
"""Codebook for quantization."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import repeat

import bitsandbytes.functional as bnb
import torch

__all__ = ["Codebook"]


@dataclass
class Codebook:
    """
    用于实现量化过程中的码本（codebook），它通过映射浮点数值到二进制代码，实现对张量的量化与反量化。
    A codebook for quantization.

    Attributes:
        size (`int`):
            Number of values in the codebook.
        norm_value (`float` or `None`):
            Normalization value.
        value_bits (`int`):
            Number of bits for the value.
        code_bits (`int`):
            Number of bits for the binary code.
        values (`torch.FloatTensor`):
            A value book in ascending order.
        codes (`torch.ByteTensor`):
            A binary book containing the binary representation of the value.
    """

    # 码本中值的数量
    size: int
    # 归一化值，用于量化前对输入进行归一化
    norm_value: float | None
    # 表示值的位数
    value_bits: int
    # 表示二进制代码的位数
    code_bits: int
    # 一个按升序排列的值张量
    values: torch.Tensor
    # 包含值二进制表示的二进制代码张量
    codes: torch.Tensor

    def __post_init__(self):
        """
        在数据类初始化后进行验证，确保码本的属性符合预期。
        """
        # 确保码本大小不超过值的数量
        assert self.size <= self.values.numel(), "Codebook size is larger than the values size"
        # 确保值张量和代码张量形状相同
        assert self.values.shape == self.codes.shape, "Values and Codes must have the same shape"
        # 确保代码张量的元素数量等于2的code_bits次方
        assert self.codes.numel() == 2**self.code_bits, "Codebook size must be 2**code_bits"
        # 确保归一化值为正数
        if self.norm_value is not None:
            assert self.norm_value > 0, "Normalization value must be positive"

    @property
    def normalized(self) -> bool:
        """
        检查码本是否被归一化。
        Check if the codebook is normalized.

        Returns:
            bool:
                `True` if the codebook is normalized, `False` otherwise.
        """
        return self.norm_value is not None

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        使用码本对输入张量进行量化，并返回量化后的张量。
        Quantize a tensor with a codebook.

        Args:
            tensor (`torch.Tensor`):
                A tensor to quantize.

        Returns:
            `torch.Tensor`:
                A quantized tensor.
        """
        # 保存原始属性
        dtype, shape, numel = tensor.dtype, tensor.shape, tensor.numel()
        # 将张量转换为连续的浮点张量
        tensor = tensor.contiguous().to(torch.float32)
        # 归一化张量
        if self.norm_value is not None:
            tensor = tensor.div(self.norm_value)
        # 定义块大小，用于分块处理大张量以节省内存
        block_size = 128 * 512 * 4096
        # 如果张量元素数量大于块大小
        if numel > block_size:
            # 将张量重塑为一维张量
            tensor = tensor.view(-1)
            # 创建一个与输入张量相同形状的输出张量
            out = torch.empty_like(tensor)
            # 对张量进行分块处理
            for i in range(0, numel, block_size):
                # 计算块的起始和结束索引
                start, end = i, min(i + block_size, numel)
                # 按块对张量进行量化和反量化
                bnb.dequantize_no_absmax(
                    bnb.quantize_no_absmax(tensor[start:end], code=self.values),
                    code=self.values,
                    out=out[start:end],
                )
            # 将输出张量重塑为原始形状
            out = out.view(shape)
        # 如果张量元素数量小于等于块大小，则直接对整个张量进行量化和反量化
        else:
            out = bnb.dequantize_no_absmax(bnb.quantize_no_absmax(tensor, code=self.values), code=self.values)
        # 反归一化张量
        if self.norm_value is not None:
            out = out.mul_(self.norm_value)
        # 恢复原始数据类型
        return out.to(dtype=dtype)

    @staticmethod
    def construct(
        maps: list[tuple[float, int]],
        *,
        value_bits: int,
        code_bits: int,
        normalize: bool | float | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "Codebook":
        """
        构建一个新的 Codebook 实例，根据提供的值与代码映射关系。
        Create a map of values to a code of `code_bits` bits.

        Args:
            maps (`list[tuple[float, int]]`):
                A list of tuples of (value, binary code).
            value_bits (`int`):
                Number of bits for the value.
            code_bits (`int`):
                Number of bits for the binary code.
            normalize (`bool` or `float` or `None`, *optional*, defaults to `None`):
                Normalization value. If `True`, normalize the values based on the maximum value.
            device (`torch.device` or str, *optional*, defaults to `"cpu"`):
                Device to put the codebook and binarybook on.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Dtype of the codebook.

        Returns:
            `Codebook`:
                A codebook.

        Raises:
            `AssertionError`:
                If the number of values is greater than 2**code_bits,
                or if normalize value is smaller than codebook max absolute value.
        """
        # 如果代码位数大于32，则抛出异常
        if code_bits > 32:
            raise NotImplementedError("Codebook with more than 32 bits is not supported")
        # 如果码本的值数量大于2的code_bits次方，则抛出异常，确保每个代码唯一
        assert len(maps) <= 2**code_bits, "Too many (value, code) maps for the code bits"
        # 确定码本的大小
        size = len(maps)
        # 按值的绝对值对映射进行排序
        maps.sort(key=lambda x: abs(x[0]))
        # 如果码本的大小小于2的code_bits次方，则用最小幅度的值填充空白
        maps.extend(repeat(maps[0], 2**code_bits - size))  # fill the gap with the value of the smallest magnitude
        # 按值对映射进行排序
        maps.sort(key=lambda x: x[0])
        # 创建值张量和代码张量
        values = torch.tensor([v[0] for v in maps], device=device, dtype=dtype) # 提取所有值并转换为Tensor
        codes = torch.tensor(           # 提取所有代码并转换为Tensor，根据代码位数确定数据类型
            [v[1] for v in maps],
            dtype=torch.uint8 if code_bits <= 8 else (torch.int16 if code_bits < 16 else torch.int32),
            device=device,
        )
        # 归一化处理
        if normalize:
            # 如果归一化值为布尔值，则将最大值归一化
            if isinstance(normalize, bool):
                normalize = values.abs().max().item()
            # 确保归一化值为浮点数或整数
            assert isinstance(normalize, (float, int)), "Normalization value must be a float or an int"
            # 确保归一化值不小于码本的最大绝对值
            assert values.abs().max() <= normalize, "The maximum value is larger than the given normalization value"
            # 确保归一化值为正数
            assert normalize > 0, "Normalization value must be positive"
            # 归一化值
            values.div_(normalize)
        else:
            normalize = None
        # 返回码本实例
        return Codebook(
            size=size,
            norm_value=normalize,
            value_bits=value_bits,
            code_bits=code_bits,
            values=values,
            codes=codes,
        )

    @staticmethod
    def build_with_splits(
        maps: list[tuple[float, int]],
        *,
        value_bits: int,
        code_bits: int,
        normalize: bool,
        split_mask: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> list["Codebook"]:
        """
        根据提供的掩码将值-代码映射分割成多个子码本，并构建相应的 Codebook 列表。
        Create a map of values to a code of `code_bits` bits.

        Args:
            maps (`list[tuple[float, int]]`): A list of tuples of (value, binary code).
            value_bits (`int`): Number of bits for the value.
            code_bits (`int`): Number of bits for the binary code.
            normalize (`bool`): Whether to normalize the values based on the maximum value.
            split_mask (`int` or `None`, *optional*, defaults to `None`):
                A mask to split the values into multiple codebooks.
            device (`torch.device` or str, *optional*, defaults to `"cpu"`):
                Device to put the codebook and binarybook on.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Dtype of the codebook.

        Returns:
            `list[Codebook]`: A list of codebooks.
        """
        # 无分割掩码
        # 如果split_mask为None，则所有映射都集中在一个子码本中
        if split_mask is None:
            split_maps = [maps]
            # 计算最大绝对值
            max_value = max(abs(v) for v, _ in maps)
        # 有分割掩码
        else:
            # 创建一个默认字典，用于存储分割后的映射
            _split_maps: dict[int, list[tuple[float, int]]] = defaultdict(list)
            # 初始化最大绝对值
            max_value = -float("inf")
            # 遍历每个映射
            for value, code in maps:
                # 根据掩码分割映射
                split = code & split_mask
                # 将映射添加到相应的子码本中
                _split_maps[split].append((value, code))
                # 更新最大绝对值
                max_value = max(max_value, abs(value))
            # 按分组顺序创建split_maps列表
            split_maps = [_split_maps[split] for split in sorted(_split_maps)]
        # 对每个子码本创建一个码本实例
        return [
            Codebook.construct(
                maps=split,
                value_bits=value_bits,
                code_bits=code_bits,
                normalize=max_value if normalize else None,
                device=device,
                dtype=dtype,
            )
            for split in split_maps
        ]

    @staticmethod
    def build_fp_with_splits(
        *,
        total_bits: int,
        exponent_bits: int,
        signed: bool = True,
        has_subnormal: bool = True,
        has_inf: bool = False,
        has_nan: bool = False,
        code_bits: int = 8,
        normalize: bool = False,
        split_mask: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> list["Codebook"]:
        """
        根据浮点数格式参数生成浮点数值与二进制代码的映射，并构建相应的码本列表。
        Create a map of floating point values to a code of `code_bits` bits.

        Args:
            total_bits (`int`):
                Number of bits for the floating point value.
            exponent_bits (`int`):
                Number of bits for the exponent.
            signed (`bool`, *optional*, defaults to `True`):
                Whether to use signed code.
            has_inf (`bool`, *optional*, defaults to `False`):
                Whether to include infinity.
            has_nan (`bool`, *optional*, defaults to `False`):
                Whether to include NaN.
            code_bits (`int`, *optional*, defaults to `8`):
                Number of bits for the code.
            normalize (`bool`, *optional*, defaults to `False`):
                Whether to normalize the values based on the maximum value.
            split_mask (`int`, *optional*, defaults to `None`):
                A mask to split the values into multiple codebooks.
            device (`torch.device` or str, *optional*, defaults to `"cpu"`):
                Device to put the codebook on.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Dtype of the codebook.

        Returns:
            `list[Codebook]`:
                A list of codebooks.
        """
        # 计算尾数位数
        mantissa_bits = total_bits - exponent_bits - int(signed)
        # 确保指数位数为正数
        assert exponent_bits > 0, "Exponent bits must be positive"
        # 确保尾数位数为非负数
        assert mantissa_bits >= 0, "Mantissa bits must be non-negative"
        # 确保总位数不超过代码位数
        assert (
            total_bits <= code_bits
        ), f"Too many bits ({exponent_bits} + {mantissa_bits} + {int(signed)} = {total_bits}) for {code_bits}-bit code"
        # 处理NaN，如果包含无穷大，则必须包含NaN
        has_nan = has_inf or has_nan

        # 定义符号掩码，用于提取符号位
        sign_mask = 1 << (total_bits - 1)
        # 确定最大指数值
        if mantissa_bits > 0:
            end_evalue = 2**exponent_bits - int(has_inf)
        else:
            end_evalue = 2**exponent_bits - int(has_nan)
        # 确定最大尾数值
        end_mvalue = 2**mantissa_bits
        # 计算偏置值
        bias = 2 ** (exponent_bits - 1) - 1
        # 创建值-代码映射列表
        maps, code = [], 0
        # 遍历所有可能的指数和尾数值
        for evalue in range(end_evalue):
            for mvalue in range(end_mvalue):
                # 根据是否为非规格化数计算value
                if evalue == 0 and has_subnormal:
                    value = (mvalue / end_mvalue) * (2 ** (1 - bias))
                else:
                    value = (1 + mvalue / end_mvalue) * (2 ** (evalue - bias))
                # 将value与当前代码添加到映射列表中
                maps.append((value, code))
                # 如果为有符号数，则添加负值及其对应的代码
                if signed:
                    maps.append((-value, code | sign_mask))
                # code自增
                code += 1
        # 如果有尾数位，且不包含无穷大，但包含NaN，则调整maps以排除多余的映射
        if mantissa_bits > 0 and not has_inf and has_nan:
            maps = maps[: -(1 + int(signed))]
        # 创建码本列表
        return Codebook.build_with_splits(
            maps,
            value_bits=total_bits,
            code_bits=code_bits,
            normalize=normalize,
            split_mask=split_mask,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def build_int_with_splits(
        *,
        total_bits: int,
        signed: bool = True,
        magnitude: bool = False,
        code_bits: int = 8,
        normalize: bool = False,
        split_mask: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> list["Codebook"]:
        """
        根据整数格式参数生成整数值与二进制代码的映射，并构建相应的码本列表。
        Create a map of integer values to a code of `code_bits` bits.

        Args:
            total_bits (`int`):
                Number of bits for the integer value.
            signed (`bool`, *optional*, defaults to `True`):
                Whether to use signed code.
            magnitude (`bool`, *optional*, defaults to `False`):
                Whether to use magnitude-based integer.
            code_bits (`int`, *optional*, defaults to `8`):
                Number of bits for the code.
            normalize (`bool`, *optional*, defaults to `False`):
                Whether to normalize the values based on the maximum value.
            split_mask (`int`, *optional*, defaults to `None`):
                A mask to split the values into multiple codebooks.
            device (`torch.device` or `str`, *optional*, defaults to `"cpu"`):
                Device to put the codebook on.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Dtype of the codebook.

        Returns:
            `list[Codebook]`:
                A list of codebooks.
        """
        # 计算范围
        if signed:
            end_value = 2 ** (total_bits - 1)           # 正整数的上限
            min_value = -end_value + int(magnitude)     # 负整数的下限，取决于是否使用幅度表示
        else:
            end_value = 2**total_bits                   # 无符号整数的上限
            min_value = 0                               # 无符号整数的下限
        # 初始化映射列表
        maps = []
        # 生成映射
        # 遍历所有可能的值
        for value in range(min_value, end_value):
            # 如果值大于等于0，则直接使用值作为代码
            if value >= 0:
                code = value
            # 如果值小于0，则根据是否使用幅度表示计算代码
            elif magnitude:
                code = end_value - value
            else:
                code = end_value + end_value + value
            # 将值与代码添加到映射列表中
            maps.append((value, code))
        # 创建码本列表
        return Codebook.build_with_splits(
            maps,
            value_bits=total_bits,
            code_bits=code_bits,
            normalize=normalize,
            split_mask=split_mask,
            device=device,
            dtype=dtype,
        )

    def split(
        self,
        split_mask: int | None,
        normalize: bool | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> list["Codebook"]:
        """
        根据提供的掩码将现有码本分割成多个子码本。
        Split a codebook into multiple codebooks.

        Args:
            split_mask (`int` or `None`):
                A mask to split the values into multiple codebooks.
            normalize (`bool`, *optional*, defaults to `None`):
                Whether to normalize the values based on the maximum value.
            device (`torch.device` or str, *optional*, defaults to `"cpu"`):
                Device to put the codebook on.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Dtype of the codebook.

        Returns:
            `list[Codebook]`:
                A list of codebooks.
        """
        # 重塑值和代码张量，展平为一维张量
        values = self.values.view(-1)
        codes = self.codes.view(-1)
        # 反归一化值
        if self.norm_value is not None:
            values = values.mul(self.norm_value)
        # 构建分割后的码本列表
        return Codebook.build_with_splits(
            [(float(value.item()), int(code.item())) for value, code in zip(values, codes, strict=True)],
            value_bits=self.value_bits,
            code_bits=self.code_bits,
            normalize=self.normalized if normalize is None else normalize,
            split_mask=split_mask,
            device=device,
            dtype=dtype,
        )
