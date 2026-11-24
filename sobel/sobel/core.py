"""sobel 核心 API。

约定：
- 输入/输出均为 Python 二维 list（元素为 float）。
- 边界采用边缘复制（replicate / clamp）填充：内部像素的梯度与解析
  Sobel 卷积完全一致，常量图像在边界处梯度同样为 0。
- 梯度角 ``angle`` 以弧度表示，取值 [-pi, pi]，由 atan2(gy, gx) 给出。
- ``Sobel`` 实例不持有任何可变状态，所有缓冲均为函数内局部变量，
  同一实例可被多线程并发调用。
"""

import math

import numpy as np

__all__ = ["Sobel"]


def _as_matrix(image, name):
    """把二维序列校验并转换为 float64 的 ndarray。

    结构非法（空、行类型错、行长不一致）抛 ValueError/TypeError；
    含 NaN/Inf 等非有限值抛 ValueError。
    """
    if not isinstance(image, (list, tuple)):
        raise TypeError(f"{name} 必须是二维 list/tuple")
    if len(image) == 0:
        raise ValueError(f"{name} 不能为空")
    width = None
    for row in image:
        if not isinstance(row, (list, tuple)):
            raise TypeError(f"{name} 的每一行必须是 list/tuple")
        if width is None:
            width = len(row)
            if width == 0:
                raise ValueError(f"{name} 不能有空行")
        elif len(row) != width:
            raise ValueError(f"{name} 的各行长度必须一致")
    try:
        arr = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(f"{name} 必须全部由实数组成")
    if arr.ndim != 2:
        raise ValueError(f"{name} 必须是二维矩阵")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} 含有非有限值（NaN/Inf）")
    return arr


def _check_threshold(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是实数")
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限值")
    if value < 0.0:
        raise ValueError(f"{name} 不能为负数")
    return float(value)


def _replicate_pad(arr):
    """边缘复制填充一圈，返回 (h+2, w+2) 的新数组。"""
    h, w = arr.shape
    padded = np.empty((h + 2, w + 2), dtype=np.float64)
    padded[1:-1, 1:-1] = arr
    padded[0, 1:-1] = arr[0, :]
    padded[-1, 1:-1] = arr[-1, :]
    padded[:, 0] = padded[:, 1]
    padded[:, -1] = padded[:, -2]
    return padded


class Sobel:
    # 经典 Sobel 核：
    # gx = [[-1,0,1],[-2,0,2],[-1,0,1]]  沿列方向求水平梯度
    # gy = [[-1,-2,-1],[0,0,0],[1,2,1]]  沿行方向求垂直梯度

    def _gradients(self, image):
        """返回 (gx, gy)，分别为水平、垂直方向的 Sobel 梯度。"""
        arr = _as_matrix(image, "image")
        padded = _replicate_pad(arr)
        top, mid, bot = padded[:-2], padded[1:-1], padded[2:]
        left_col = top[:, :-2] + 2.0 * mid[:, :-2] + bot[:, :-2]
        right_col = top[:, 2:] + 2.0 * mid[:, 2:] + bot[:, 2:]
        top_row = top[:, :-2] + 2.0 * top[:, 1:-1] + top[:, 2:]
        bot_row = bot[:, :-2] + 2.0 * bot[:, 1:-1] + bot[:, 2:]
        gx = right_col - left_col
        gy = bot_row - top_row
        return gx, gy

    def magnitude(self, image):
        """Sobel 梯度幅值 sqrt(gx^2 + gy^2)，返回与输入同形的二维 list。"""
        gx, gy = self._gradients(image)
        return np.hypot(gx, gy).tolist()

    def angle(self, image):
        """梯度方向角（弧度，[-pi, pi]），atan2(gy, gx)。"""
        gx, gy = self._gradients(image)
        return np.arctan2(gy, gx).tolist()

    def nms(self, mag, angle):
        """非极大抑制：沿梯度方向比较两个邻域，非局部极大置 0。

        方向量化为 4 个扇区（水平、垂直、两条对角线），比较采用
        ``>=``：平滑梯度剖面上只有屋脊单个像素通过，保证边缘宽度 1。
        """
        m = _as_matrix(mag, "mag")
        a = _as_matrix(angle, "angle")
        if m.shape != a.shape:
            raise ValueError("mag 与 angle 形状不一致")
        h, w = m.shape

        # 角度折到 [0, pi)，梯度是无方向的（正负方向等价）
        theta = np.mod(a, math.pi)
        s0 = (theta < math.pi / 8.0) | (theta >= 7.0 * math.pi / 8.0)
        s1 = (theta >= math.pi / 8.0) & (theta < 3.0 * math.pi / 8.0)
        s2 = (theta >= 3.0 * math.pi / 8.0) & (theta < 5.0 * math.pi / 8.0)
        # 其余即 s3: [5pi/8, 7pi/8)

        padded = np.empty((h + 2, w + 2), dtype=np.float64)
        padded[1:-1, 1:-1] = m
        padded[0, 1:-1] = m[0, :]
        padded[-1, 1:-1] = m[-1, :]
        padded[:, 0] = padded[:, 1]
        padded[:, -1] = padded[:, -2]

        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        top_left = padded[:-2, :-2]
        top = padded[:-2, 1:-1]
        top_right = padded[:-2, 2:]
        bot_left = padded[2:, :-2]
        bot = padded[2:, 1:-1]
        bot_right = padded[2:, 2:]

        # 扇区 -> 沿梯度方向（强度变化方向）上的两个邻点：
        #   s0 角度≈0       水平向右 -> 左右 (left, right)
        #   s1 角度≈pi/4    指向右下 -> (top_left, bot_right)
        #   s2 角度≈pi/2    垂直向下 -> 上下 (top, bot)
        #   s3 角度≈3pi/4   指向左下 -> (top_right, bot_left)
        n1 = np.select(
            [s0, s1, s2],
            [left, top_left, top],
            default=top_right,
        )
        n2 = np.select(
            [s0, s1, s2],
            [right, bot_right, bot],
            default=bot_left,
        )
        keep = (m >= n1) & (m >= n2)
        return np.where(keep, m, 0.0).tolist()

    def hysteresis(self, mag, low, high):
        """双阈值滞后连接，返回 0.0/1.0 的二维 list。

        - mag >= high：强边，直接保留；
        - low <= mag < high：弱边，仅当其与某个强边 8 连通时保留；
        - mag < low：抑制。
        low > high、阈值为负/非有限/非数值均抛错（ValueError/TypeError）。
        """
        m = _as_matrix(mag, "mag")
        low = _check_threshold(low, "low")
        high = _check_threshold(high, "high")
        if low > high:
            raise ValueError("low 不能大于 high")

        strong = m >= high
        weak = (m >= low) & ~strong
        if not strong.any():
            return np.zeros_like(m).tolist()

        reached = self._connect(strong, weak)
        return reached.astype(np.float64).tolist()

    @staticmethod
    def _dilate8(mask):
        """3x3 八邻域膨胀。"""
        h, w = mask.shape
        padded = np.zeros((h + 2, w + 2), dtype=bool)
        padded[1:-1, 1:-1] = mask
        out = np.zeros((h, w), dtype=bool)
        for dr in (0, 1, 2):
            for dc in (0, 1, 2):
                if dr == 1 and dc == 1:
                    continue
                out |= padded[dr:dr + h, dc:dc + w]
        return out

    @staticmethod
    def _connect(strong, weak):
        """返回所有与强边 8 连通的弱边（含强边本身）。"""
        traversable = strong | weak
        reached = strong.copy()

        # 向量化波前扩张：自然图像通常几十波以内收敛。
        waves = 0
        while waves < 128:
            ring = Sobel._dilate8(reached) & traversable & ~reached
            if not ring.any():
                return reached
            reached |= ring
            waves += 1

        # 病态细长链兜底：纯 Python 迭代 DFS，保证正确性。
        h, w = strong.shape
        trav = traversable.tolist()
        seen = reached.tolist()
        stack = list(map(tuple, np.argwhere(reached).tolist()))
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                nr = r + dr
                if nr < 0 or nr >= h:
                    continue
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nc = c + dc
                    if nc < 0 or nc >= w:
                        continue
                    if trav[nr][nc] and not seen[nr][nc]:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
        return np.asarray(seen, dtype=bool)
