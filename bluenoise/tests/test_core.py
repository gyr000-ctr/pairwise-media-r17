import math

import pytest

import bluenoise

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _b(s=1):
    return bluenoise.BlueNoise(s)

CASES = [
    ("shape", lambda: len(_b().matrix(8)) == 8),
    ("range", lambda: all(0.0 <= v < 1.0 for row in _b().matrix(8) for v in row)),
    ("det", lambda: _b().matrix(8) == _b().matrix(8)),
    ("seed_diff", lambda: _b(1).matrix(8) != _b(2).matrix(8)),
    ("bad_size", lambda: _raises(lambda: _b().matrix(6))),
    ("bad_seed", lambda: _raises(lambda: bluenoise.BlueNoise(-1))),
    ("type", lambda: isinstance(_b().matrix(8), list)),
    ("ratio", lambda: _b().low_energy_ratio(8) <= 0.6),
    ("row_len", lambda: len(_b().matrix(8)[0]) == 8),
    ("unique", lambda: len(set(v for row in _b().matrix(8) for v in row)) > 1),
    ("size_1", lambda: len(_b().matrix(1)) == 1),
    ("ratio_type", lambda: isinstance(_b().low_energy_ratio(8), float)),
    ("nan", lambda: _raises(lambda: _b().matrix(float('nan')))),
    ("no_mut", lambda: _b().matrix(8) is not None),
    ("mean", lambda: abs(sum(v for row in _b().matrix(8) for v in row) / 64 - 0.5) < 0.2),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True


# ---------------------------------------------------------------------------
# 题面约束的更严格验收（频谱、性能、错误分类、并发）
# ---------------------------------------------------------------------------

def test_low_energy_ratio_strict():
    """题面要求低频能量占比 ≤ 30%。"""
    for size in (8, 16, 32, 64):
        assert bluenoise.BlueNoise(1).low_energy_ratio(size) <= 0.30
        assert bluenoise.BlueNoise(2).low_energy_ratio(size) <= 0.30


def test_spectrum_high_frequency_dominant():
    """直接检查 64x64 的径向频谱：低频环带能量占比 ≤ 30%。"""
    import numpy as np

    a = np.array(bluenoise.BlueNoise(3).matrix(64), dtype=float)
    a -= a.mean()
    power = np.abs(np.fft.fft2(a)) ** 2
    power[0, 0] = 0.0
    freq = np.fft.fftfreq(64)
    radius = np.hypot(freq[:, None], freq[None, :])
    assert power[radius <= 0.25].sum() / power.sum() <= 0.30


def test_uniform_histogram():
    """值域 [0,1) 且近似均匀：64x64 的 16 等分直方图各桶计数相同。"""
    import numpy as np

    a = np.array(bluenoise.BlueNoise(4).matrix(64), dtype=float)
    counts, _ = np.histogram(a, bins=16, range=(0.0, 1.0))
    assert counts.min() > 0
    assert counts.max() - counts.min() <= 0.05 * a.size


def test_performance_64():
    """64x64 生成不超过 800 ms。"""
    import time

    start = time.perf_counter()
    bluenoise.BlueNoise(5).matrix(64)
    elapsed = (time.perf_counter() - start) * 1000.0
    assert elapsed <= 800.0, f"took {elapsed:.1f} ms"


def test_reproducible_across_instances():
    assert bluenoise.BlueNoise(11).matrix(16) == bluenoise.BlueNoise(11).matrix(16)
    assert bluenoise.BlueNoise(11).matrix(16) != bluenoise.BlueNoise(12).matrix(16)


def test_error_classification():
    """非法输入按类别报错：类型错误 -> TypeError，数值非法 -> ValueError。"""
    bn = bluenoise.BlueNoise(1)
    for bad in (-1, 0, 3, 6, 100):
        with pytest.raises(ValueError):
            bn.matrix(bad)
    with pytest.raises(TypeError):
        bn.matrix("8")
    with pytest.raises(TypeError):
        bn.matrix(None)
    with pytest.raises(ValueError):
        bluenoise.BlueNoise(-5)
    for non_finite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            bluenoise.BlueNoise(non_finite)
        with pytest.raises(ValueError):
            bn.matrix(non_finite)
        with pytest.raises(ValueError):
            bn.low_energy_ratio(non_finite)


def test_no_mutation_leak():
    """调用方修改返回的 list 不应影响后续结果。"""
    bn = bluenoise.BlueNoise(7)
    first = bn.matrix(8)
    first[0][0] = -999.0
    assert bn.matrix(8)[0][0] != -999.0


def test_concurrent_consistency():
    """并发调用结果与串行一致。"""
    from concurrent.futures import ThreadPoolExecutor

    jobs = [(seed, size) for seed in range(6) for size in (8, 16, 32)]
    expected = {
        job: bluenoise.BlueNoise(job[0]).matrix(job[1]) for job in jobs
    }

    def work(job):
        return job, bluenoise.BlueNoise(job[0]).matrix(job[1])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, jobs * 4))

    assert all(matrix == expected[job] for job, matrix in results)
