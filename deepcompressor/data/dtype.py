# -*- coding: utf-8 -*-
"""Quantization data type."""

import typing as tp

import torch

from .codebook import Codebook

__all__ = ["QuantDataType", "QDType"]


class QuantDataType:
    """
    用于定义和管理量化数据类型的属性和行为。
    它支持整数和浮点数数据类型，并允许配置不同的位宽、符号、指数、子正常数等参数。
    此外，类还支持与 Codebook 的集成，用于量化过程中的码本管理。
    Quantization data type.
    """

    def __init__(
        self,
        total_bits: int,
        *,
        signed: bool = True,
        exponent_bits: int = 0,
        has_subnormal: bool = True,
        has_nan: bool = False,
        has_inf: bool = False,
        magnitude: bool = False,
        codebook: Codebook | None = None,
        codebook_name: str = "",
    ):
        """Initialize the quantization data type.

        Args:
            total_bits (`int`):
                Total number of bits. Must be greater than 0.
            signed (`bool`, *optional*, defaults to `True`):
                Whether the data type is signed.
            exponent_bits (`int`, *optional*, defaults to `0`):
                Number of bits for the exponent.
            has_subnormal (`bool`, *optional*, defaults to `True`):
                Whether the data type has subnormal.
            has_nan (`bool`, *optional*, defaults to `False`):
                Whether the data type has NaN if it is float-point.
            has_inf (`bool`, *optional*, defaults to `False`):
                Whether the data type has Inf if it is float-point.
            magnitude (`bool`, *optional*, defaults to `False`):
                Whether the data type is magnitude-based if it is integer.
            codebook (`Codebook` or `None`, *optional*, defaults to `None`):
                Codebook for the data type.
            codebook_name (`str`, *optional*, defaults to `""`):
                Name of the codebook. Must be specified if `codebook` is not `None`.
        """
        # 设置符号属性
        self.__signed = signed
        
        # 设置位宽
        # region set bit widths
        self.__total_bits = total_bits
        self.__exponent_bits = exponent_bits
        # 确保总位数大于 0
        assert self.__total_bits > 0, "Total bits must be greater than 0."
        # 确保指数位数大于等于 0
        assert self.__exponent_bits >= 0, "Exponent bits must be non-negative."
        # 计算尾数位数
        self.__mantissa_bits = self.__total_bits - self.__exponent_bits - int(self.__signed)
        # endregion
        
        # 设置数据类型属性
        # region set data type properties
        if self.__exponent_bits > 0:            # 判断数据类型是浮点数还是整数
            # for floating-point data type
            self.__has_subnormal = has_subnormal
            self.__has_inf = has_inf
            self.__has_nan = has_inf or has_nan
            self.__magnitude = True
            if self.__mantissa_bits == 0:       # 如果尾数位数为 0，则不支持无穷大
                assert not self.__has_inf, "Inf is not supported for exponent-only floating-point data type."
                if self.__exponent_bits == 1:   # 如果指数位数为 1，则不支持 NaN
                    assert not self.__has_nan, "NaN is not supported for 1-bit exponent-only floating-point data type."
        else:
            # for integer data type
            self.__has_subnormal = False
            self.__has_inf = False
            self.__has_nan = False
            self.__magnitude = magnitude
        # endregion
        
        # 设置码本
        # region set codebook
        if codebook is not None:            # 提供了码本
            # 确保数据类型是浮点数
            assert self.is_float_point, "Codebook is only supported for floating-point data type."
            self.__codebook = codebook
            # 确保码本名称不为空
            assert codebook_name, "Codebook name must be specified."
            self.__codebook_name = codebook_name
            # 确保码本的最大值为非负数
            assert self.max_value >= 0, "Max value must be non-negative."
        else:                               # 未提供码本
            self.__codebook = None
            self.__codebook_name = ""
        # endregion
        
        # 设置分割码本
        # region set split codebooks
        self.__split_codebooks: dict[tuple[int, bool, int | None, torch.device, torch.dtype], list[Codebook]] = {}
        # endregion
        self.__name = self.to_str()

    # region properties
    @property
    def name(self) -> str:
        """
        数据类型名称。
        Name of the data type.
        """
        return self.__name

    @property
    def codebook_name(self) -> str:
        """
        码本名称。
        Name of the codebook.
        """
        return self.__codebook_name

    @property
    def signed(self) -> bool:
        """
        数据类型是否有符号。
        Whether the data type is signed.
        """
        return self.__signed

    @property
    def unsigned(self) -> bool:
        """
        数据类型是否无符号。
        Whether the data type is unsigned.
        """
        return not self.__signed

    @property
    def total_bits(self) -> int:
        """
        总位数。
        Total number of bits.
        """
        return self.__total_bits

    @property
    def exponent_bits(self) -> int:
        """
        指数位数。
        Number of bits for the exponent.
        """
        return self.__exponent_bits

    @property
    def mantissa_bits(self) -> int:
        """
        尾数位数。
        Number of bits for the mantissa.
        """
        return self.__mantissa_bits

    @property
    def has_subnormal(self) -> bool:
        """
        数据类型是否有子正常数。
        Whether the data type has subnormal.
        """
        return self.__has_subnormal

    @property
    def has_inf(self) -> bool:
        """
        数据类型是否有无穷大。
        Whether the data type has Inf.
        """
        return self.__has_inf

    @property
    def has_nan(self) -> bool:
        """
        数据类型是否有 NaN。
        Whether the data type has NaN.
        """
        return self.__has_nan

    @property
    def magnitude(self) -> bool:
        """
        数据类型是否基于幅度表示。
        Whether the data type is magnitude-based.
        """
        return self.__magnitude

    @property
    def is_float_point(self) -> bool:
        """
        数据类型是否浮点数。
        Whether the data type is floating-point.
        """
        return self.exponent_bits > 0

    @property
    def is_integer(self) -> bool:
        """
        数据类型是否整数。
        Whether the data type is integer.
        """
        return self.exponent_bits == 0

    @property
    def is_exponent(self) -> bool:
        """
        数据类型是否仅指数。
        Whether the data type is exponent-only floating-point.
        """
        return self.exponent_bits > 0 and self.mantissa_bits == 0 and not self.has_subnormal

    @property
    def exponent_mask(self) -> int:
        """
        指数的位掩码。
        Bit mask for the exponent.
        """
        return ((1 << self.exponent_bits) - 1) << self.mantissa_bits

    @property
    def mantissa_mask(self) -> int:
        """
        尾数的位掩码。
        Bit mask for the mantissa.
        """
        return (1 << self.mantissa_bits) - 1

    @property
    def _end_mantissa(self) -> int:
        """
        尾数的结束值。
        """
        return 2**self.mantissa_bits

    @property
    def _end_exponent(self) -> int:
        """
        指数的结束值。
        """
        if self.mantissa_bits > 0:
            return 2**self.exponent_bits - int(self.has_inf)
        else:
            return 2**self.exponent_bits - int(self.has_nan)

    @property
    def exponent_bias(self) -> int:
        """
        指数偏移量。
        Exponent bias.
        """
        if self.is_float_point:
            return 2 ** (self.exponent_bits - 1) - 1
        else:
            return 0

    @property
    def max_exponent_value(self) -> int:
        """
        指数值的最大值。
        Maximum exponent value.
        """
        if self.is_float_point:
            return self._end_exponent - 1 - self.exponent_bias
        else:
            return self.total_bits - 1 - int(self.signed)

    @property
    def min_exponent_value(self) -> int:
        """
        指数值的最小值。
        Minimum exponent value.
        """
        if self.is_float_point:
            return int(self.has_subnormal) - self.exponent_bias
        else:
            return 0

    @property
    def max_positive_normal_value(self) -> float:
        """
        正常值的最大正值。
        Maximum positive normal value.
        """
        if self.is_float_point:
            if self.mantissa_bits > 0 and not self.has_inf and self.has_nan:
                base_value = 2 - 2 / self._end_mantissa
            else:
                base_value = 2 - 1 / self._end_mantissa
            return base_value * 2**self.max_exponent_value
        else:
            return self._end_mantissa - 1

    @property
    def min_positive_normal_value(self) -> float:
        """
        正常值的最小正值。
        Minimum positive normal value.
        """
        return 2**self.min_exponent_value

    @property
    def max_positive_subnormal(self) -> float:
        """
        最大正子正常值。
        Maximum positive subnormal value.
        """
        if self.is_float_point and self.has_subnormal and self.mantissa_bits > 0:
            b = 1 - 1 / self._end_mantissa
            e = 1 - self.exponent_bias
            return b * 2**e
        else:
            return 0

    @property
    def min_positive_subnormal(self) -> float:
        """
        最小正子正常值。
        Minimum non-negative subnormal value.
        """
        if self.is_float_point and self.has_subnormal and self.mantissa_bits > 0:
            b = 1 / self._end_mantissa
            e = 1 - self.exponent_bias
            return b * 2**e
        else:
            return 0

    @property
    def max_value(self) -> float:
        """
        数据类型的最大值。
        Maximum value.
        """
        return self.max_positive_normal_value if self.__codebook is None else self.__codebook.values[-1].item()

    @property
    def min_value(self) -> float:
        """
        数据类型的最小值。
        Minimum value.
        """
        if self.__codebook is not None:
            return self.__codebook.values[0].item()
        if self.signed:
            if self.magnitude:
                return -self.max_value
            else:
                return -self.max_value - 1
        else:
            return 0

    # endregion

    def to_unsigned(self) -> "QuantDataType":
        """
        返回当前数据类型的无符号版本。
        Get an unsigned version of the data type.

        Returns:
            `QuantDataType`:
                The unsigned version of the data type.
        """
        return QuantDataType(
            total_bits=self.total_bits,
            signed=False,
            exponent_bits=self.exponent_bits,
            has_subnormal=self.has_subnormal,
            has_nan=self.has_nan,
            has_inf=self.has_inf,
            magnitude=self.magnitude,
        )

    def __build_split_codebooks(
        self,
        *,
        code_bits: int = 8,
        normalize: bool = False,
        split_mask: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> list[Codebook]:
        """
        根据数据类型是否为浮点数，构建分割后的码本列表。
        """
        if self.is_float_point:
            return Codebook.build_fp_with_splits(
                total_bits=self.total_bits,
                exponent_bits=self.exponent_bits,
                signed=self.signed,
                has_subnormal=self.has_subnormal,
                has_inf=self.has_inf,
                has_nan=self.has_nan,
                code_bits=code_bits,
                normalize=normalize,
                split_mask=split_mask,
                device=device,
                dtype=dtype,
            )
        else:
            return Codebook.build_int_with_splits(
                total_bits=self.total_bits,
                signed=self.signed,
                magnitude=self.magnitude,
                code_bits=code_bits,
                normalize=normalize,
                split_mask=split_mask,
                device=device,
                dtype=dtype,
            )

    def get_split_codebooks(
        self,
        *,
        code_bits: int = 8,
        normalize: bool = False,
        split_mask: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> list[Codebook]:
        """
        获取分割后的码本列表，支持缓存以避免重复构建。
        Get a get_codebook of `code_bits` bits for the quantization.

        Args:
            code_bits (`int`, *optional*, defaults to `8`):
                Number of bits for the codebook.
            normalize (`bool`, *optional*, defaults to `False`):
                Whether to normalize the codebook values based on the maximum value.
            split_mask (`int` or `None`, *optional*, defaults to `None`):
                Bit mask to split the codebook into parts.
            device (`torch.device` or `str`, *optional*, defaults to `"cpu"`):
                Device to create the codebook on.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type to create the codebook with.

        Returns:
            `list[Codebook]`:
                A list of codebooks.
        """
        device = torch.device("cpu") if device is None else torch.device(device)
        # 生成键，包含所有参数
        key = (code_bits, normalize, split_mask, device, dtype)
        # 如果key不在缓存中
        if key not in self.__split_codebooks:
            # 如果已有码本，则根据码本分割
            if self.__codebook is not None:
                self.__split_codebooks[key] = self.__codebook.split(
                    split_mask=split_mask, normalize=normalize, device=device, dtype=dtype
                )
            # 否则，根据数据类型构建分割码本
            else:
                self.__split_codebooks[key] = self.__build_split_codebooks(
                    code_bits=code_bits,
                    normalize=normalize,
                    split_mask=split_mask,
                    device=device,
                    dtype=dtype,
                )
        # 返回分割码本
        return self.__split_codebooks[key]

    def get_codebook(
        self,
        *,
        code_bits: int = 8,
        normalize: bool = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Codebook:
        """
        获取单个码本，默认返回分割后的第一个码本。
        Get a get_codebook of `code_bits` bits for the quantization.

        Args:
            code_bits (`int`, *optional*, defaults to `8`):
                Number of bits for the codebook.
            normalize (`bool`, *optional*, defaults to `False`):
                Whether to normalize the codebook values based on the maximum value.
            device (`torch.device` or `str`, *optional*, defaults to `"cpu"`):
                Device to create the codebook on.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type to create the codebook with.

        Returns:
            `Codebook`:
                The codebook.
        """
        return self.get_split_codebooks(code_bits=code_bits, normalize=normalize, device=device, dtype=dtype)[0]

    def __str__(self) -> str:
        """
        获取数据类型的字符串表示。
        """
        return self.__name

    def __repr__(self) -> str:
        """
        获取数据类型的字符串表示。
        """
        return self.__name

    @staticmethod
    def from_str(s: str, /) -> "QuantDataType":
        """
        根据字符串创建对应的 QuantDataType 实例。

        Args:
            s: 字符串表示的数据类型。
        """
        # 将字符串转换为小写并去除首尾空格
        s = s.strip().lower()
        # 判断数据类型是否有符号
        signed = s[0] == "s"
        # 去除符号
        s = s[1:]
        # 是否为整数类型
        if s.startswith("int"):
            return QuantDataType(int(s[3:]), signed=signed)
        # 是否为幅度类型
        elif s.startswith("mag"):
            return QuantDataType(int(s[3:]), signed=signed, magnitude=True)
        # 是否为指数类型
        elif s.startswith("exp"):
            ss = s.split("_")
            # 获取指数位数
            total_bits = int(ss[0][3:])
            # 是否有 NaN
            if len(ss) >= 2:
                has_nan = ss[1] == "nan"
            else:
                has_nan = False
            # 返回指数类型数据
            return QuantDataType(
                total_bits=total_bits,
                signed=signed,
                exponent_bits=total_bits - int(signed),
                has_subnormal=False,
                has_nan=has_nan,
            )
        # 是否为浮点类型
        elif s.startswith("f"):
            # 分割字符串
            ss = s.split("_")
            # 是否有子正常数
            has_subnormal = s[1] == "p"
            # 获取总位数
            total_bits = int(ss[0][2:])
            # 获取指数位数
            exponent_bits = int(ss[1][1 : ss[1].find("m")])
            # 是否有无穷大和 NaN
            if len(ss) >= 3:
                has_inf = ss[2] == "inf"
                has_nan = has_inf or (ss[2] == "nan")
            else:
                has_inf, has_nan = False, False
            # 返回浮点类型数据
            return QuantDataType(
                total_bits=total_bits,
                signed=signed,
                exponent_bits=exponent_bits,
                has_subnormal=has_subnormal,
                has_inf=has_inf,
                has_nan=has_nan,
            )
        else:
            raise ValueError(f"Unknown QuantDataType {s}")

    def to_str(self) -> str:
        """
        生成数据类型的字符串表示。
        Get the string representation of the QuantDataType.

        Returns:
            str: The string representation.
        """
        # 符号位字符
        s = "s" if self.signed else "u"
        # 如果有码本名称，则返回码本名称和总位数
        if self.__codebook_name:
            return f"{s}{self.__codebook_name}{self.total_bits}"
        # 如果是浮点数，根据是否有子正常数和尾数位数，添加fp或fn，添加总位数、指数位数和尾数位数，根据是否有无穷大或 NaN，添加inf或nan
        if self.is_float_point:
            if self.has_subnormal or self.mantissa_bits > 0:
                s += "fp" if self.has_subnormal else "fn"
                s += f"{self.total_bits}_e{self.exponent_bits}m{self.mantissa_bits}"
                s += "_inf" if self.has_inf else ("_nan" if self.has_nan else "_all")
            else:
                assert not self.has_subnormal, "Subnormal is not supported for exponent-only floating-point data type."
                assert not self.has_inf, "Inf is not supported for exponent-only floating-point data type."
                s += f"exp{self.exponent_bits}"
                s += "_nan" if self.has_nan else "_all"
        # 如果是整数，根据是否基于幅度表示，添加mag或int，添加总位数
        else:
            s += "mag" if self.magnitude else "int"
            s += f"{self.total_bits}"
        return s

    def __eq__(self, value: object) -> bool:
        """
        定义等于操作的行为。
        """
        if not isinstance(value, QuantDataType):
            return False
        return self.name == value.name

    def __hash__(self) -> int:
        """
        定义哈希值，以便在集合或字典中使用。
        """
        return hash(self.name)


class _QDTypeMeta(type):
    """
    元类，用于动态生成 QuantDataType 实例。
    当访问 QDType 类的未定义属性时，元类会尝试将属性名解析为对应的 QuantDataType。
    """
    def __getattr__(cls, __name: str) -> tp.Any:
        """
        拦截对未定义属性的访问。
        """
        # 如果属性名以下划线开头，则调用父类的方法
        if __name.startswith("_"):
            return getattr(super(), __name)
        # 尝试根据属性名创建 QuantDataType 实例
        else:
            return QuantDataType.from_str(__name)


class QDType(metaclass=_QDTypeMeta):
    """
    使用 _QDTypeMeta 作为元类，允许通过属性访问的方式快速获取特定名称的 QuantDataType 实例。
    QuantDataType class for easy access to QuantDataType by name.
    """

    pass
