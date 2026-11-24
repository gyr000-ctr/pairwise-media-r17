import math

import pytest

import blur

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

CASES = [
    ("kernel_sum", lambda: abs(sum(blur.Blur().kernel(1.0, 3)) - 1.0) < 1e-12),
    ("kernel_len", lambda: len(blur.Blur().kernel(1.0, 3)) == 7),
    ("constant", lambda: all(abs(v - 1.0) < 1e-12 for row in blur.Blur().blur(IMG, 1.0) for v in row)),
    ("bad_sigma", lambda: _raises(lambda: blur.Blur().kernel(0.0, 3))),
    ("bad_radius", lambda: _raises(lambda: blur.Blur().kernel(1.0, 0))),
    ("empty", lambda: _raises(lambda: blur.Blur().blur([], 1.0))),
    ("type", lambda: isinstance(blur.Blur().blur(IMG, 1.0), list)),
    ("det", lambda: blur.Blur().blur(IMG, 1.0) == blur.Blur().blur(IMG, 1.0)),
    ("shape", lambda: len(blur.Blur().blur(IMG, 1.0)) == 3),
    ("symmetric", lambda: blur.Blur().kernel(1.0, 3) == blur.Blur().kernel(1.0, 3)[::-1]),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in blur.Blur().blur([[0.0, 1.0], [1.0, 0.0]], 1.0) for v in row)),
    ("nan", lambda: _raises(lambda: blur.Blur().blur([[float('nan')]], 1.0))),
    ("no_mut", lambda: blur.Blur().blur([r[:] for r in IMG], 1.0) is not None),
    ("edge_preserve", lambda: abs(blur.Blur().blur(IMG, 2.0)[0][0] - 1.0) < 1e-12),
    ("kernel_type", lambda: isinstance(blur.Blur().kernel(1.0, 3), list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
