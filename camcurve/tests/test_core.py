import math

import pytest

import camcurve

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

P = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]

CASES = [
    ("at_zero", lambda: abs(camcurve.CamCurve().at(P, 0.0)[0][0]) < 1e-9),
    ("at_one", lambda: abs(camcurve.CamCurve().at(P, 1.0)[0][0] - 3.0) < 1e-6),
    ("up_orth", lambda: abs(sum(a * b for a, b in zip(camcurve.CamCurve().at(P, 0.5)[1], camcurve.CamCurve().at(P, 0.5)[2]))) < 1e-9),
    ("unit_fwd", lambda: abs(math.sqrt(sum(c * c for c in camcurve.CamCurve().at(P, 0.5)[1])) - 1.0) < 1e-6),
    ("bad_t", lambda: _raises(lambda: camcurve.CamCurve().at(P, 1.5))),
    ("bad_points", lambda: _raises(lambda: camcurve.CamCurve().at([(0.0, 0.0, 0.0)], 0.5))),
    ("type", lambda: isinstance(camcurve.CamCurve().at(P, 0.5), tuple)),
    ("det", lambda: camcurve.CamCurve().at(P, 0.5) == camcurve.CamCurve().at(P, 0.5)),
    ("triple", lambda: len(camcurve.CamCurve().at(P, 0.5)) == 3),
    ("orth_unit", lambda: abs(math.sqrt(sum(c * c for c in camcurve.CamCurve().orthonormal((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))[0])) - 1.0) < 1e-9),
    ("nan", lambda: _raises(lambda: camcurve.CamCurve().at(P, float('nan')))),
    ("no_mut", lambda: camcurve.CamCurve().at(list(P), 0.3) is not None),
    ("mid", lambda: 0.0 <= camcurve.CamCurve().at(P, 0.5)[0][0] <= 3.0),
    ("up_type", lambda: isinstance(camcurve.CamCurve().at(P, 0.4)[2], tuple)),
    ("two_points", lambda: _raises(lambda: camcurve.CamCurve().at([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 0.5))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
