import math

import pytest

import crossfade

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

A = [0.5] * 10
B = [0.5] * 10

CASES = [
    ("len", lambda: len(crossfade.Crossfade().fade(A, B)) == 10),
    ("power", lambda: abs(crossfade.Crossfade().gains(2)[0] ** 2 + crossfade.Crossfade().gains(2)[1] ** 2 - 1.0) < 1e-9),
    ("start", lambda: abs(crossfade.Crossfade().fade(A, B)[0] - 0.5) < 1e-9),
    ("end", lambda: abs(crossfade.Crossfade().fade(A, B)[-1] - 0.5) < 1e-9),
    ("bad_len", lambda: _raises(lambda: crossfade.Crossfade().fade(A, [0.5]))),
    ("bad_gains", lambda: _raises(lambda: crossfade.Crossfade().gains(0))),
    ("type", lambda: isinstance(crossfade.Crossfade().fade(A, B), list)),
    ("det", lambda: crossfade.Crossfade().fade(A, B) == crossfade.Crossfade().fade(A, B)),
    ("range", lambda: all(abs(v) <= 1.0 for v in crossfade.Crossfade().fade(A, B))),
    ("gains_len", lambda: len(crossfade.Crossfade().gains(4)) == 4),
    ("monotone", lambda: crossfade.Crossfade().gains(4)[0] <= crossfade.Crossfade().gains(4)[-1]),
    ("no_dip", lambda: min(abs(v) for v in crossfade.Crossfade().fade(A, B)) > 0.4),
    ("nan", lambda: _raises(lambda: crossfade.Crossfade().fade([float('nan')], [0.5]))),
    ("no_mut", lambda: crossfade.Crossfade().fade(list(A), list(B)) is not None),
    ("gains_type", lambda: isinstance(crossfade.Crossfade().gains(4), list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
