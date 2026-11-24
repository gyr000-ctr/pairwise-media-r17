"""蓝噪声抖动矩阵生成（void-and-cluster）。

实现思路
--------
采用 Ulichney 的 void-and-cluster（虚空-聚集）算法生成阈值/抖动矩阵：

1. 从随机稀疏二值图案出发，交替"移除最紧聚集点"与"填补最大虚空点"，
   直到两者位置重合，得到稳定的原型图案；
2. 以原型为起点，逐步移除聚集点，按序分配低秩（rank）；
3. 以原型为起点，逐步填补虚空点，按序分配高秩。

最终每个像素拥有唯一的 rank，矩阵值为 ``rank / N``，因此恰好铺满
``[0, 1)`` 且分布均匀；同时任意阈值切分出来的二值图案都具有蓝噪声
（高频占优）的频谱特性。

所有随机性只来自 ``numpy.random.default_rng(seed)``，因此同一 seed
在不同调用/实例间完全可复现。生成过程不修改任何共享可变状态，可安全
地并发调用。
"""

from __future__ import annotations

import functools
import math
import numbers

import numpy as np

# 高斯核宽度（像素）。Ulichney 的经典取值。
_SIGMA = 1.5

# 原型初始"点"所占比例。
_PROTOTYPE_FRACTION = 0.1

# 低频能量统计的截止半径（cycles / pixel）。
_LOW_FREQ_CUTOFF = 0.25


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------

def _as_int(value, name):
    """把整数输入归一化为 int，并对错误分类。

    - 布尔值：TypeError（bool 是 int 的子类，但不应当作尺寸/种子）
    - 非有限浮点（nan/inf）：ValueError（数值非法，而非类型错误）
    - 其他浮点：TypeError（即使是整数值的浮点，也要求显式整数）
    - 非数值类型：TypeError
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v!r}")
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    raise TypeError(f"{name} must be an integer, got {type(value).__name__}")


def _check_seed(seed):
    seed = _as_int(seed, "seed")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return seed


def _check_size(size):
    size = _as_int(size, "size")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if size & (size - 1):
        raise ValueError(f"size must be a power of two, got {size}")
    return size


# ---------------------------------------------------------------------------
# 生成算法
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def _kernel(size):
    """环面（周期边界）上的高斯能量核。"""
    d = np.arange(size)
    d = np.minimum(d, size - d)
    d2 = d[:, None] ** 2 + d[None, :] ** 2
    k = np.exp(-d2 / (2.0 * _SIGMA * _SIGMA))
    k.setflags(write=False)
    return k


def _shifted(k, pos):
    """把核平移到 pos=(row, col)（环面平移）。"""
    return np.roll(k, (int(pos[0]), int(pos[1])), axis=(0, 1))


def _energy(pattern, k):
    """图案在环面上的能量场（循环卷积）。"""
    return np.fft.irfft2(
        np.fft.rfft2(pattern.astype(np.float64)) * np.fft.rfft2(k),
        s=pattern.shape,
    )


def _argmax_where(energy, mask):
    """mask 为 True 的位置中能量最大者（扁平索引）。"""
    flat = np.where(mask.ravel(), energy.ravel(), -np.inf)
    return int(np.argmax(flat))


def _argmin_where(energy, mask):
    """mask 为 True 的位置中能量最小者（扁平索引）。"""
    flat = np.where(mask.ravel(), energy.ravel(), np.inf)
    return int(np.argmin(flat))


def _seed_offset(size, rng):
    """随机平移到 (0, 0) 以外的归一化位置，以区分不同种子的结果。

    void-and-cluster 生成的图案在环面平移（以及翻转）下是等价的，因此
    固定从 (0, 0) 开始编号会让不同种子在极小尺寸下给出同一矩阵。
    这里把随机起点平移到环面上的随机位置，保持蓝噪声性质不变。
    """
    if size <= 1:
        return 0, 0
    return int(rng.integers(0, size)), int(rng.integers(0, size))


def _make_prototype(size, rng):
    """随机初始化并稳定化，返回 (原型图案, 原型能量场, 原型点数)。"""
    n = size * size
    count = max(1, int(round(_PROTOTYPE_FRACTION * n)))
    count = min(count, n)

    pattern = np.zeros(n, dtype=bool)
    idx = rng.choice(n, size=count, replace=False)
    pattern[idx] = True
    pattern = pattern.reshape(size, size)

    k = _kernel(size)
    energy = _energy(pattern, k)

    # 交替移除最紧聚集、填补最大虚空，直到稳定：
    # 当移除的聚集点又成为下一个最大虚空点时，说明图案已收敛。
    max_iters = 20 * n + 100
    for _ in range(max_iters):
        cluster = _argmax_where(energy, pattern)
        cy, cx = divmod(cluster, size)
        pattern[cy, cx] = False
        energy -= _shifted(k, (cy, cx))

        void = _argmin_where(energy, ~pattern)
        if void == cluster:
            # 已稳定：放回刚移除的点，本轮不产生交换。
            pattern[cy, cx] = True
            energy += _shifted(k, (cy, cx))
            break

        vy, vx = divmod(void, size)
        pattern[vy, vx] = True
        energy += _shifted(k, (vy, vx))

    return pattern, energy, int(pattern.sum())


def _rank_matrix(size, seed):
    """返回 shape=(size, size) 的整数 rank 矩阵，rank 为 0..N-1 的排列。"""
    n = size * size
    rng = np.random.default_rng(seed)
    k = _kernel(size)

    prototype, proto_energy, n0 = _make_prototype(size, rng)

    rank = np.full(n, -1, dtype=np.int64)

    # 阶段一：从原型出发，不断移除最紧聚集点，倒序分配低秩。
    # （本阶段结束时 n0 个点被全部移除，图案为空，无需再维护图案。）
    energy = proto_energy.copy()
    pattern = prototype.copy()
    for r in range(n0 - 1, -1, -1):
        pos = _argmax_where(energy, pattern)
        rank[pos] = r
        py, px = divmod(pos, size)
        pattern[py, px] = False
        energy -= _shifted(k, (py, px))

    # 阶段二：从原型出发，不断填补最大虚空点，顺序分配高秩。
    energy = proto_energy.copy()
    pattern = prototype.copy()
    for r in range(n0, n):
        pos = _argmin_where(energy, ~pattern)
        rank[pos] = r
        py, px = divmod(pos, size)
        pattern[py, px] = True
        energy += _shifted(k, (py, px))

    if np.any(rank < 0):  # 防御性检查：不应发生
        raise RuntimeError("internal error: incomplete rank assignment")

    # 平移到随机原点，使不同种子在极小尺寸下也给出可区分的矩阵。
    oy, ox = _seed_offset(size, rng)
    if oy or ox:
        rank = np.roll(rank.reshape(size, size), (oy, ox), axis=(0, 1))

    return rank.reshape(size, size)


@functools.lru_cache(maxsize=32)
def _cached_matrix(seed, size):
    """缓存为不可变 tuple，避免调用方修改影响后续调用。"""
    rank = _rank_matrix(size, seed)
    n = size * size
    values = rank.astype(np.float64) / n
    return tuple(tuple(float(v) for v in row) for row in values)


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

class BlueNoise:
    """蓝噪声抖动矩阵生成器。

    Parameters
    ----------
    seed:
        非负整数种子。相同 seed 生成相同矩阵；非法种子按类别抛出
        ``TypeError``（类型错误）或 ``ValueError``（数值非法，含 nan/inf）。
    """

    def __init__(self, seed):
        self._seed = _check_seed(seed)

    @property
    def seed(self):
        return self._seed

    def matrix(self, size):
        """生成 ``size x size`` 的抖动矩阵。

        返回值为 list[list[float]]，取值在 ``[0, 1)`` 内，近似均匀分布。
        ``size`` 必须是 2 的幂；否则抛出 ``ValueError``。
        """
        size = _check_size(size)
        cached = _cached_matrix(self._seed, size)
        # 每次返回新的 list 结构，调用方修改不会污染缓存。
        return [list(row) for row in cached]

    def low_energy_ratio(self, size):
        """低频能量占总交流能量的比例。

        把矩阵去均值后做二维 FFT，统计径向频率不超过
        ``_LOW_FREQ_CUTOFF``（cycles / pixel）的频段能量占比。蓝噪声
        矩阵该比例应很小；该返回值恒为 float。
        """
        size = _check_size(size)
        cached = _cached_matrix(self._seed, size)
        if size == 1:
            return 0.0

        a = np.asarray(cached, dtype=np.float64)
        a = a - a.mean()

        power = np.abs(np.fft.fft2(a)) ** 2
        power[0, 0] = 0.0  # 直流分量不计入交流能量

        total = float(power.sum())
        if total == 0.0:
            return 0.0

        freq = np.fft.fftfreq(size)
        radius = np.sqrt(freq[:, None] ** 2 + freq[None, :] ** 2)
        low = float(power[radius <= _LOW_FREQ_CUTOFF].sum())
        return float(low / total)
