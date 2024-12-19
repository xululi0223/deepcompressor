# -*- coding: utf-8 -*-
"""Dynamic range calculation for quantization."""

import math
import typing as tp
from dataclasses import dataclass

import torch

from .dtype import QuantDataType
from .zero import ZeroPointDomain

__all__ = ["RangeBound", "QuantRange", "LogQuantRange", "ProtectiveQuantRange", "DynamicRange"]


@dataclass
class RangeBound:
    """
    数据类，用于表示一个数值范围的下限 (min) 和上限 (max)。
    该类提供了检查范围是否被设置的方法，以及将范围转换为字典和从字典创建范围的方法。
    Range bound data class.
    """

    # 范围的下限
    min: float | None = None
    # 范围的上限
    max: float | None = None

    def is_set(self) -> bool:
        """
        用于检查范围是否至少设置了下限或上限。
        Return whether the range bound is set.
        """
        return self.min is not None or self.max is not None

    def to_dict(self) -> dict[str, tp.Any]:
        """
        将 RangeBound 对象转换为字典。
        Return the dictionary representation of the range bound.
        """
        return {"min": self.min, "max": self.max}

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any] | None) -> tp.Optional[tp.Self]:
        """
        从给定的字典创建一个 RangeBound 实例。
        Return the range bound from the given dictionary.
        
        Args:
            data: 包含 min 和 max 的字典。
        """
        return cls(min=data["min"], max=data["max"]) if data is not None else None


class QuantRange(RangeBound):
    """
    用于表示量化过程中使用的数值范围。
    它提供了将范围转换为对数尺度 (log2)、与特定量化数据类型的范围交集，以及构造量化范围的方法。
    Quantization range data class.
    """

    def log2(self) -> "LogQuantRange":
        """
        将当前量化范围转换为对数尺度的 LogQuantRange。
        Return the log-scale of the current quantization range.
        """
        # 计算min和max绝对值中的最小值的对数，并取整
        log2_abs_min = int(math.log2(min(abs(self.min or 0), abs(self.max or 0))))
        # 创建一个 LogQuantRange 对象，min 为 None，max 为 log2_abs_min（如果 self.max 不为 None）
        return LogQuantRange(
            min=None,
            max=None if self.max is None else log2_abs_min,
        )

    def intersect(self, quant_dtype: QuantDataType, *, has_zero_point: bool) -> "QuantRange":
        """
        计算当前量化范围与给定量化数据类型 (QuantDataType) 的交集。
        Return the intersection of the current quantization range and the given data type.

        Args:
            quant_dtype (`QuantDataType`):
                The quantization data type.
            has_zero_point (`bool`):
                Whether the quantization range has zero-point.

        Returns:
            `QuantRange`:
                The intersection of the current quantization range and the given data type.
        """
        # 取当前量化范围与给定量化数据类型的范围交集
        max_value = quant_dtype.max_value if self.max is None else min(self.max, quant_dtype.max_value)
        min_value = quant_dtype.min_value if self.min is None else max(self.min, quant_dtype.min_value)
        # 如果量化数据类型是有符号的，且没有零点，则将 min_value 和 max_value 取绝对值后取最小值作为 max_value，取相反数作为 min_value
        if quant_dtype.signed and not has_zero_point:
            max_value = min(abs(min_value), abs(max_value))
            min_value = -max_value
        # 返回一个 QuantRange 对象，min 为 min_value，max 为 max_value
        return QuantRange(min=min_value, max=max_value)

    def intersect_log2(self, quant_dtype: QuantDataType) -> "LogQuantRange":
        """
        计算当前量化范围与给定量化数据类型在对数尺度下的交集。
        Return the intersection of the current quantization range and the given data type in log2 space.

        Args:
            quant_dtype (`QuantDataType`):
                The quantization data type.

        Returns:
            `LogQuantRange`:
                The intersection of the current quantization range and the given data type in log2 space.
        """
        # 将当前范围转换为对数尺度的 LogQuantRange，然后计算与给定量化数据类型的交集
        return self.log2().intersect_log2(quant_dtype)

    @staticmethod
    def construct(
        dtype: QuantDataType, *, has_zero_point: bool, quant_range: tp.Optional["QuantRange"] = None
    ) -> "QuantRange":
        """
        用于构造一个新的 QuantRange，通过与给定的量化数据类型的交集。
        Return the intersection of the given quantization range and the given data type.

        Args:
            dtype (`QuantDataType`):
                The quantization data type.
            has_zero_point (`bool`):
                Whether the quantization range has zero-point.
            quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                The extra quantization range.

        Returns:
            `QuantRange`:
                The intersection of the given quantization range and the given data type.
        """
        # 如果 quant_range 为 None，则创建一个新的 QuantRange 对象，否则取 quant_range 本身
        # 然后计算与给定量化数据类型的交集
        return (quant_range or QuantRange()).intersect(dtype, has_zero_point=has_zero_point)


class LogQuantRange(QuantRange):
    """
    用于表示以对数尺度（基于2）表示的量化范围。
    该类主要用于对数尺度下的量化范围交集计算，并重载了一些方法以适应对数尺度的需求。
    Log-scale quantization range data class."""

    def log2(self) -> "LogQuantRange":
        """
        返回自身，因为已经处于对数尺度。
        Return the log-scale of the quantization range.
        """
        return self

    def intersect(self, quant_dtype: QuantDataType, *, has_zero_point: bool) -> "QuantRange":
        """
        不被支持。
        Return the intersection of the current quantization range and the given data type.

        Args:
            quant_dtype (`QuantDataType`):
                The quantization data type.
            has_zero_point (`bool`):
                Whether the quantization range has zero-point.

        Returns:
            `QuantRange`:
                The intersection of the current quantization range and the given data type.
        """
        raise NotImplementedError("LogQuantRange does not support intersect method")

    def intersect_log2(self, quant_dtype: QuantDataType) -> "LogQuantRange":
        """
        计算当前对数尺度量化范围与给定量化数据类型在对数尺度下的交集。
        Return the intersection of the current quantization range and the given data type in log2 space.

        Args:
            quant_dtype (`QuantDataType`):
                The quantization data type.

        Returns:
            `LogQuantRange`:
                The intersection of the current quantization range and the given data type in log2 space.
        """
        # 取当前量化范围与给定量化数据类型的范围交集
        max_value = (
            quant_dtype.max_exponent_value if self.max is None else min(self.max, quant_dtype.max_exponent_value)
        )
        min_value = (
            quant_dtype.min_exponent_value if self.min is None else max(self.min, quant_dtype.min_exponent_value)
        )
        return LogQuantRange(min=min_value, max=max_value)

    @staticmethod
    def construct(
        dtype: QuantDataType, quant_range: tp.Optional[tp.Union["LogQuantRange", QuantRange]] = None
    ) -> "LogQuantRange":
        """
        用于构造一个新的 LogQuantRange，通过与给定的量化数据类型在对数尺度下的交集。
        Return the intersection of the given quantization range and the given data type in log2 space.

        Args:
            dtype (`QuantDataType`):
                The quantization data type.
            quant_range (`LogQuantRange` or `QuantRange` or `None`, *optional*, defaults to `None`):
                The extra quantization range.

        Returns:
            `LogQuantRange`:
                The intersection of the given quantization range and the given data type in log2 space.
        """
        # 如果 quant_range 为 None，则创建一个新的 LogQuantRange 对象，否则取 quant_range 本身
        # 然后计算与给定量化数据类型的交集
        return (quant_range or LogQuantRange()).intersect_log2(dtype)


class ProtectiveQuantRange(QuantRange):
    """
    用于构建保护性的量化范围，确保在多级量化过程中上层和下层量化范围的一致性和有效性。
    该类通过缓存实例以避免重复计算。
    """
    # 用于缓存已经创建的 ProtectiveQuantRange 实例
    # 避免重复创建相同参数的实例，提高效率
    _instances: tp.ClassVar[
        dict[tuple[QuantDataType, QuantDataType, tuple[float, float], ZeroPointDomain], "ProtectiveQuantRange"]
    ] = {}

    @staticmethod
    def construct(
        outer_dtype: QuantDataType,
        inner_dtype: QuantDataType,
        zero_domain: ZeroPointDomain | None,
        inner_quant_range: QuantRange | None = None,
    ) -> QuantRange:
        """
        用于构建一个保护性的量化范围 (ProtectiveQuantRange)。
        Return the protective quantization range.

        Args:
            outer_dtype (`QuantDataType`):
                The data type of the outer level in the quantization hierarchy.
            inner_dtype (`QuantDataType`):
                The data type of the inner level in the quantization hierarchy.
            zero_domain (`ZeroPointDomain` or `None`):
                The zero-point domain.
            inner_quant_range (`QuantRange` or `None`, *optional*, defaults to `None`):
                The inner quantization range.

        Returns:
            `QuantRange`:
                The protective quantization range.
        """
        # 类型和条件验证
        assert outer_dtype.is_integer, "outer_dtype must be integer data type"
        assert inner_dtype.is_integer, "inner_dtype must be integer data type"
        assert zero_domain is not None or outer_dtype.signed == inner_dtype.signed
        # 处理无零点域的情况，调用QuantRange.construct方法构建外层量化范围
        if zero_domain is None:
            return QuantRange.construct(outer_dtype, has_zero_point=False)
        # 构建内层量化范围
        inner_quant_range = QuantRange.construct(inner_dtype, has_zero_point=True, quant_range=inner_quant_range)
        # 获取内层量化范围的最大值和最小值
        qmax, qmin = int(inner_quant_range.max), int(inner_quant_range.min)  # type: ignore
        # 生成缓存键
        key = (outer_dtype, inner_dtype, (qmin, qmax), zero_domain)
        # 检查缓存
        if key not in ProtectiveQuantRange._instances:
            # 构建外层量化范围
            outer_quant_range = QuantRange.construct(outer_dtype, has_zero_point=False)
            # 获取外层量化数据类型的最大值和最小值
            vrmax, vrmin = int(outer_quant_range.max), int(outer_quant_range.min)  # type: ignore
            # 获取内层量化数据类型的最大值和最小值
            qrmax, qrmin = int(inner_dtype.max_value), int(inner_dtype.min_value)
            # 计算有效的值范围集合
            vranges: set[tuple[int, int]] = set()       # 用于存储有效的值范围组合
            # 遍历所有可能的vrmax和vrmin的组合
            for vmax in range(0, vrmax + 1):
                for vmin in range(vrmin, vmax + 1):
                    # 计算缩放因子
                    s = round((vmax - vmin) / (qmax - qmin))
                    assert s >= 0, "s must be non-negative"
                    # 如果s为0，则设置为1
                    s = 1 if s == 0 else s
                    # 确保s不超过vrmax
                    s = min(s, vrmax)
                    # 根据零点域计算零点z、最大值m和最小值n
                    if zero_domain == ZeroPointDomain.PreScale:
                        z = max(min(round(qmin - vmin / s), qrmax), qrmin)
                        m = (max(min(round(vmax / s + z), qmax), qmin) - z) * s
                        n = (max(min(round(vmin / s + z), qmax), qmin) - z) * s
                    elif zero_domain == ZeroPointDomain.PostScale:
                        z = max(min(round(qmin * s - vmin), vrmax), vrmin)          # TODO: correct?
                        m = max(min(round((vmax + z) / s), qmax), qmin) * s - z
                        n = max(min(round((vmin + z) / s), qmax), qmin) * s - z
                    else:
                        raise ValueError(f"unsupported zero-point domain {zero_domain}")
                    # 检查m和n是否在有效范围内，如果是，则添加到有效值范围集合中
                    if vrmin <= m <= vrmax and vrmin <= n <= vrmax:
                        vranges.add((vmin, vmax))
            # 查找合适的found_pmax
            found_pmax = None
            for pmax in range(vrmax, 0, -1):
                pmin = -pmax
                valid = True
                # 寻找一个pmax使得所有（vmin，vmax）都在有效值范围集合中
                for vmax in range(0, pmax + 1):
                    for vmin in range(pmin, vmax + 1):
                        if (vmin, vmax) not in vranges:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    found_pmax = pmax
                    break
            assert found_pmax is not None, "failed to find the protective quantization range"
            # 创建一个新的 ProtectiveQuantRange 对象，min 为 -found_pmax，max 为 found_pmax，存入缓存中
            ProtectiveQuantRange._instances[key] = ProtectiveQuantRange(min=-found_pmax, max=found_pmax)
        return ProtectiveQuantRange._instances[key]


@dataclass
class DynamicRange:
    """
    数据类，用于动态计算和管理量化过程中的数值范围。
    它处理张量的最小值、最大值以及缩放比例 (ratio)，并提供了交集计算、范围测量、范围缩放以及与字典的转换方法。
    Dynamic range data class.
    """

    # 动态范围的最小值
    min: torch.Tensor | None = None
    # 动态范围的最大值
    max: torch.Tensor | None = None
    # 用于缩放动态范围的比例
    ratio: float | torch.Tensor | None = None

    def __post_init__(self) -> None:
        """
        在初始化后的后初始化方法。
        """
        # 如果 max 为 None，则 min 也必须为 None
        if self.max is None:
            assert self.min is None, "min must be None if max is None"

    def is_set(self) -> bool:
        """
        用于检查动态范围是否已经设置。
        Return whether the dynamic range is set.
        """
        return self.min is not None or self.max is not None or self.ratio is not None

    def intersect(self, range_bound: RangeBound | None) -> "DynamicRange":
        """
        计算当前动态范围与给定的 RangeBound 实例的交集。
        Return the intersection of the current dynamic range and the given range bound.

        Args:
            range_bound (`RangeBound` or `None`):
                The range bound.

        Returns:
            `DynamicRange`:
                The intersection of the current dynamic range and the given range bound.
        """
        # 确保当前实例的 max 不为 None
        assert self.max is not None, "max must be specified"
        # 初始化 vmax 和 vmin 为当前实例的 max 和 min
        vmax, vmin = self.max, self.min
        # 如果 range_bound 不为 None，则根据其 min 和 max 对 vmax 和 vmin 进行裁剪
        if range_bound is not None:
            if range_bound.max is not None:
                vmax = vmax.clamp(max=range_bound.max)
            if vmin is not None and range_bound.min is not None:
                vmin = vmin.clamp(min=range_bound.min)
        # 返回一个新的 DynamicRange 对象，包含交集后的 vmax 和 vmin
        return DynamicRange(min=vmin, max=vmax)

    def measure(  # noqa: C901
        self,
        tensors: torch.Tensor | list[torch.Tensor],
        /,
        *,
        zero_domain: ZeroPointDomain | None,
        is_float_point: bool,
    ) -> "DynamicRange":
        """
        计算给定张量或张量列表的动态范围。
        Return a dynamic range of the given tensor.

        Args:
            tensors (`torch.Tensor` or `list[torch.Tensor]`):
                The tensor in the shape of (#g0, gs0, #g1, gs1, ..., #gn, gsn).
            zero_domain (`ZeroPointDomain` or `None`):
                The zero-point domain.
            is_float_point (`bool`):
                Whether the data type is floating-point.

        Returns:
            `DynamicRange`:
                The dynamic range. If the max value is already specified, return the current object.
        """
        # 如果 tensors 为单个张量，则转换为张量列表
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        # 静态范围处理
        if self.ratio is None and self.max is not None:  # static range
            # 选取第一个张量
            tensor = tensors[0]
            # 如果张量的维度为偶数，则 shape 为张量 shape 的每个偶数索引位置的值，否则为 1
            shape = torch.Size([s if i % 2 == 0 else 1 for i, s in enumerate(tensor.shape)])
            # 格式化 min 和 max，并扩展到指定的 shape
            vmax = self._format_m_(self.max, shape=shape, dtype=tensor.dtype, device=tensor.device)
            vmin = self._format_m_(self.min, shape=shape, dtype=tensor.dtype, device=tensor.device)
        # 动态范围处理
        else:
            
            # 确定值范围（vmax 和 vmin）
            if self.max is None:
                assert self.min is None, "min must be None if max is None"
            reduced = list(range(1, tensors[0].ndim, 2))        # 选取张量的奇数索引位置的维度
            # region step 1: determine the value range (i.e., vmax and vmin)
            if zero_domain is None:
                # 如果 zero_domain 为 None，则计算所有张量的最大值vmax，最小值vmin为 None
                vmin = None
                vmax = tensors[0].abs().amax(dim=reduced, keepdim=True)
                for tensor in tensors[1:]:
                    vmax = torch.maximum(vmax, tensor.abs().amax(dim=reduced, keepdim=True).to(vmax.device))
            else:
                # 如果 zero_domain 不为 None，则计算所有张量的最大值、最小值和平均值（浮点数）
                vmax = tensors[0].amax(dim=reduced, keepdim=True)
                for tensor in tensors[1:]:
                    vmax = torch.maximum(vmax, tensor.amax(dim=reduced, keepdim=True).to(vmax.device))
                vmin = tensors[0].amin(dim=reduced, keepdim=True)
                for tensor in tensors[1:]:
                    vmin = torch.minimum(vmin, tensor.amin(dim=reduced, keepdim=True).to(vmin.device))
                if is_float_point:  # ! we adapt the zero-point to be the mean of the data
                    vavg = tensors[0].mean(dim=reduced, keepdim=True)
                    if len(tensors) > 1:
                        for tensor in tensors[1:]:
                            vavg = vavg + tensor.mean(dim=reduced, keepdim=True).to(vavg.device)
                        vavg = vavg / len(tensors)
            # endregion
            
            # region step 2: scale the value range by self.ratio
            # 按ratio缩放值范围
            if zero_domain is None:
                if self.ratio is not None:
                    vmax = vmax * self.ratio        # 缩放最大值
            else:
                assert vmin is not None, "vmin must be specified"
                if is_float_point:
                    vmag = torch.maximum(vmax - vavg, vavg - vmin)  # 计算最大值和最小值的差值
                    if self.ratio is not None:
                        vmag = vmag * self.ratio                    # 缩放差值
                    vmax = vavg + vmag                              # 调整最大值
                    vmin = vavg - vmag                              # 调整最小值
                else:
                    if self.ratio is not None:
                        vmin = vmin * self.ratio                    # 缩放最小值
                        vmax = vmax * self.ratio                    # 缩放最大值
                if zero_domain == ZeroPointDomain.PreScale:
                    # 如果零点域为 PreScale，则限制vmax不小于0，vmin不大于0
                    vmax = vmax.clamp(min=0)
                    vmin = vmin.clamp(max=0)
            # endregion
            
            # 将值范围限制在（self.min, self.max）之间
            # region step 3: clamp the value range by (self.min, self.max)
            if self.max is not None:
                vmax = vmax.clamp(max=self.max.to(vmax.device))
                if vmin is not None and self.min is not None:
                    vmin = vmin.clamp(min=self.min.to(vmin.device))
            # endregion
        # 返回一个新的 DynamicRange 对象，包含计算后的 vmin 和 vmax
        return DynamicRange(min=vmin, max=vmax)

    def scale(
        self, ratio: float | torch.Tensor, zero_domain: ZeroPointDomain | None, is_float_point: bool
    ) -> "DynamicRange":
        """
        根据给定的 ratio 缩放当前动态范围，返回一个新的 DynamicRange 实例。
        Return new dynamic range by scaling the current range.

        Args:
            ratio (`float` or `torch.Tensor`):
                The scaling ratio.
            zero_domain (`ZeroPointDomain` or `None`):
                The zero-point domain.
            is_float_point (`bool`):
                Whether the data type is floating-point.

        Returns:
            `DynamicRange`:
                The new dynamic range.
        """
        # 确保 ratio 不为 None
        assert ratio is not None, "ratio must be specified"

        # 无零点域处理
        if zero_domain is None:
            # 确保 self.max 不为 None，self.min 必须为 None
            assert self.max is not None, "self.max must be specified"
            assert self.min is None, "self.min must be None for data type without zero-point"
            # 计算max_value为self.max * ratio，min_value为None
            max_value = self.max * ratio
            min_value = None
        # 有零点域处理
        else:
            # 确保 self.max 和 self.min 不为 None
            assert self.min is not None, "self.min must be specified"
            assert self.max is not None, "self.max must be specified"
            # 根据是否为浮点数，计算min_value和max_value
            if is_float_point:
                centroid_value = (self.min + self.max) / 2      # 计算质心值
                vmag = (self.max - centroid_value) * ratio      # 计算幅度vmag
                max_value = centroid_value + vmag               # 调整最大值
                min_value = centroid_value - vmag               # 调整最小值
            else:
                min_value = self.min * ratio                    # 直接缩放最小值
                max_value = self.max * ratio                    # 直接缩放最大值
            if zero_domain == ZeroPointDomain.PreScale:
                # 如果零点域为 PreScale，则限制max_value不小于0，min_value不大于0
                max_value = max_value.clamp(min=0)
                min_value = min_value.clamp(max=0)
        # 返回一个新的 DynamicRange 对象，包含缩放后的 min_value 和 max_value
        return DynamicRange(min=min_value, max=max_value)

    @staticmethod
    def construct(
        tensors: torch.Tensor | list[torch.Tensor],
        /,
        *,
        zero_domain: ZeroPointDomain | None,
        is_float_point: bool,
    ) -> "DynamicRange":
        """
        提供了一种简便的方式来创建 DynamicRange 实例，通过调用 measure 方法。
        """
        # 调用 measure 方法计算动态范围，返回一个新的 DynamicRange 对象
        return DynamicRange().measure(tensors, zero_domain=zero_domain, is_float_point=is_float_point)

    @staticmethod
    def _format_m_(
        value: torch.Tensor | float | None,
        *,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        """
        用于格式化给定的 min 或 max 值，使其与指定的形状、数据类型和设备相匹配。
        """
        # 如果 value 为 None，则返回 None
        if value is None:
            return None
        # 如果 value 是torch.Tensor，则根据 shape、dtype 和 device 进行格式化
        elif isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.view(-1).to(dtype=dtype, device=device).expand(shape)
            elif value.numel() == shape.numel():
                return value.view(shape).to(dtype=dtype, device=device)
            elif value.shape[1:] == shape[1:] and value.shape[0] == 1:
                return value.to(dtype=dtype, device=device).expand(shape)
            else:
                raise ValueError(f"Invalid value shape: {value.shape}")
        # 如果 value 是 float，则返回一个全为 value 的张量
        else:
            return torch.full(shape, value, dtype=dtype, device=device)

    def to_dict(self) -> dict[str, tp.Any]:
        """
        将 DynamicRange 实例转换为字典。
        Return the dictionary representation of the dynamic range.
        """
        return {"min": self.min, "max": self.max, "ratio": self.ratio}

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any] | None) -> tp.Optional[tp.Self]:
        """
        如果 data 不为 None，则返回一个新的 DynamicRange 实例；否则返回 None。
        Return the dynamic range from the given dictionary.
        
        Args:
            data: 包含 min、max 和 ratio 的字典。
        """
        return cls(min=data["min"], max=data["max"], ratio=data["ratio"]) if data is not None else None
