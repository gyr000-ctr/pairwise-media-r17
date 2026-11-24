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
