"""blur 核心 API（初始未实现）。"""


class Blur:
    def kernel(self, sigma, radius):
        """一维核"""
        raise NotImplementedError

    def blur(self, image, sigma, radius=3):
        """模糊"""
        raise NotImplementedError
