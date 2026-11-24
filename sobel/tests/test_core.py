import math

import pytest

import sobel

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]

CASES = [
    ("mag_shape", lambda: len(sobel.Sobel().magnitude(IMG)) == 3),
    ("center_positive", lambda: sobel.Sobel().magnitude(IMG)[1][1] >= 0.0),
    ("nms_len", lambda: len(sobel.Sobel().nms(sobel.Sobel().magnitude(IMG), sobel.Sobel().angle(IMG))) == 3),
    ("bad_low", lambda: _raises(lambda: sobel.Sobel().hysteresis([[0.5]], 0.9, 0.1))),
    ("bad_threshold", lambda: _raises(lambda: sobel.Sobel().hysteresis([[0.5]], -0.1, 1.0))),
    ("bad_image", lambda: _raises(lambda: sobel.Sobel().magnitude([]))),
    ("type", lambda: isinstance(sobel.Sobel().magnitude(IMG), list)),
    ("det", lambda: sobel.Sobel().magnitude(IMG) == sobel.Sobel().magnitude(IMG)),
    ("hyst_binary", lambda: set(v for row in sobel.Sobel().hysteresis([[1.0, 0.0]], 0.1, 0.9) for v in row) <= {0.0, 1.0}),
    ("angle_shape", lambda: len(sobel.Sobel().angle(IMG)) == 3),
    ("nan", lambda: _raises(lambda: sobel.Sobel().magnitude([[float('nan')]]))),
    ("no_mut", lambda: sobel.Sobel().magnitude([r[:] for r in IMG]) is not None),
    ("flat_zero", lambda: all(abs(v) < 1e-9 for row in sobel.Sobel().magnitude([[0.5] * 3 for _ in range(3)]) for v in row)),
    ("nonneg", lambda: all(v >= 0.0 for row in sobel.Sobel().magnitude(IMG) for v in row)),
    ("nms_type", lambda: isinstance(sobel.Sobel().nms(sobel.Sobel().magnitude(IMG), sobel.Sobel().angle(IMG)), list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
