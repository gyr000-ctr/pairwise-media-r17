"""等功率（equal-power）交叉淡化核心 API。

设计要点
========
- 等功率增益曲线：淡入增益 ``g_b(i) = sin(theta_i)``、淡出增益
  ``g_a(i) = cos(theta_i)``，其中 ``theta_i = pi/2 * i/(n-1)``，
  逐点恒有 ``g_a^2 + g_b^2 = 1``（中点处二者均为 sqrt(2)/2，功率和恰为 1）。
- 端点精确：``i=0`` 输出严格等于 ``a[0]``，``i=n-1`` 输出严格等于 ``b[-1]``，
  淡化段与前后序列在拼接处能量连续。
- 增益全程为正，相位对齐的同相信号叠加不会产生抵消。
- ``Crossfade`` 不保存任何实例状态、模块内也无可变全局状态，
  同一个实例可被多线程并发调用。
- 仅依赖标准库 ``math``；有限性校验与淡化计算融合在单次循环中，
  100 万样本耗时在百毫秒量级。
"""

import math
import operator

__all__ = [
    "Crossfade",
    "CrossfadeError",
    "LengthMismatchError",
    "InvalidLengthError",
    "NonFiniteError",
]


class CrossfadeError(ValueError):
    """本模块所有参数类错误的基类（本身仍是 ValueError）。"""


class LengthMismatchError(CrossfadeError):
    """两条输入序列长度不一致时抛出。"""


class InvalidLengthError(CrossfadeError):
    """增益长度参数非法（小于 1）时抛出。"""


class NonFiniteError(CrossfadeError):
    """输入序列包含 NaN 或无穷大等非有限值时抛出。"""


def _classify_nonfinite(a, b):
    """在已知存在非有限值的前提下，构造可区分 nan / inf 的异常。

    先报 NaN 再报 inf，序列 a 优先于序列 b，保证错误信息确定且可分类。
    """
    for name, seq in (("a", a), ("b", b)):
        for i, v in enumerate(seq):
            if math.isnan(v):
                return NonFiniteError(
                    f"输入序列 {name} 索引 {i} 包含非有限值: nan"
                )
    for name, seq in (("a", a), ("b", b)):
        for i, v in enumerate(seq):
            if math.isinf(v):
                return NonFiniteError(
                    f"输入序列 {name} 索引 {i} 包含非有限值: inf"
                )
    return NonFiniteError("输入包含非有限值")


class Crossfade:
    """无状态等功率交叉淡化器，实例可在多线程间安全共享。"""

    def gains(self, n):
        """返回长度为 ``n`` 的淡入增益序列（sin 曲线，单调递增）。

        与 :meth:`fade` 内部对序列 ``b`` 使用的增益完全一致；
        对应淡出增益为 ``cos``，二者平方和恒为 1。

        :raises InvalidLengthError: ``n < 1`` 时抛出。
        :raises TypeError: ``n`` 不是整数时抛出。
        """
        n = operator.index(n)
        if n < 1:
            raise InvalidLengthError(f"增益长度必须 >= 1，实际为 {n}")
        if n == 1:
            return [1.0]
        last = n - 1
        step = math.pi / (2.0 * last)
        return [math.sin(step * i) for i in range(n)]

    def fade(self, a, b):
        """对两条等长序列做等功率交叉淡化。

        :param a: 淡出序列，淡化起点处增益为 1。
        :param b: 淡入序列，淡化终点处增益为 1。
        :return: 与输入等长的新 ``list``，不修改任何输入序列。
        :raises LengthMismatchError: 两序列长度不一致时抛出。
        :raises NonFiniteError: 任一元素为 NaN 或无穷大时抛出，
            错误信息可区分 ``nan`` 与 ``inf``。
        """
        n = len(a)
        if n != len(b):
            raise LengthMismatchError(
                f"两条序列长度不一致: len(a)={n}, len(b)={len(b)}"
            )
        if n == 0:
            return []

        # 单元素时起点与终点重合，按淡入侧增益 1 取 b[0]。
        if n == 1:
            x, y = a[0], b[0]
            if not (math.isfinite(x) and math.isfinite(y)):
                raise _classify_nonfinite(a, b)
            return [float(y)]

        last = n - 1
        step = math.pi / (2.0 * last)
        out = [0.0] * n
        sin = math.sin
        cos = math.cos
        isfinite = math.isfinite
        for i in range(n):
            x = a[i]
            y = b[i]
            # 有限性校验与计算融合，正常路径只扫描一遍。
            if not (isfinite(x) and isfinite(y)):
                raise _classify_nonfinite(a, b)
            theta = step * i
            out[i] = cos(theta) * x + sin(theta) * y

        # 端点强制取精确值，消除浮点误差，保证拼接处能量连续。
        out[0] = float(a[0])
        out[last] = float(b[last])
        return out
