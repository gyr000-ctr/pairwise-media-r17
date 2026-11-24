# bluenoise

## 题面（可直接粘贴）

子目录：bluenoise。蓝噪声抖动矩阵。约束：频谱高频能量占优（低频能量占比 ≤30%）；值域 [0,1) 且近似均匀；同种子可复现；尺寸必须为 2 的幂；64x64 不超过 800 ms；非法种子可区分报错。验收：cd bluenoise && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd bluenoise && python -m pytest tests -q
```

初始 15 failed。

## 实现说明

采用 Ulichney 的 void-and-cluster（虚空-聚集）算法：

1. 用 `numpy.random.default_rng(seed)` 生成随机稀疏点阵，交替"移除最紧聚集点 /
   填补最大虚空点"直至稳定，得到原型图案；
2. 从原型出发逐步移除聚集点，倒序分配低秩；
3. 从原型出发逐步填补虚空点，顺序分配高秩；
4. 矩阵值为 `rank / (size*size)`，恰好铺满 `[0, 1)` 且 16 等分直方图均匀。

能量场用环面（周期边界）高斯核的 FFT 卷积计算，单步 O(N log N)。
实测 64x64 约 40–55 ms（上限 800 ms）。

### 输入校验与错误分类

| 输入 | 行为 |
| --- | --- |
| 尺寸为 2 的幂（含 1） | 正常返回 |
| 尺寸 ≤0 或非 2 的幂 | `ValueError` |
| 尺寸 nan / inf | `ValueError`（非有限输入） |
| 尺寸为浮点、字符串、None、bool | `TypeError` |
| 种子为负整数 | `ValueError` |
| 种子 nan / inf | `ValueError` |
| 种子为浮点、字符串、None、bool | `TypeError` |

### 其他性质

- **可复现**：同一 seed + size 在任意实例、任意线程下结果一致（有 lru_cache，
  只缓存不可变 tuple；`matrix()` 每次返回新的 list，调用方修改不会污染结果）。
- **并发安全**：生成过程不写共享可变状态，缓存由 `functools.lru_cache` 保证线程安全。
- **频谱**：`low_energy_ratio(size)` 返回径向频率 ≤ 0.25 cycles/pixel 的能量占比；
  实测 8/16/32/64 均 ≤ 0.01（题面上限 0.30）。任意阈值切分出的点阵相较白噪声
  低频能量也显著更低。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
