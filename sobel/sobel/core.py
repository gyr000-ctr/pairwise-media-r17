"""sobel 核心 API（初始未实现）。"""


class Sobel:
    def magnitude(self, image):
        """梯度幅值"""
        raise NotImplementedError

    def nms(self, mag, angle):
        """非极大抑制"""
        raise NotImplementedError

    def hysteresis(self, mag, low, high):
        """双阈值"""
        raise NotImplementedError
