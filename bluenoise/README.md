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

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
