"""camcurve 核心 API（初始未实现）。"""


class CamCurve:
    def at(self, points, t):
        """返回 (pos, fwd, up)"""
        raise NotImplementedError

    def orthonormal(self, fwd, up):
        """正交化"""
        raise NotImplementedError
