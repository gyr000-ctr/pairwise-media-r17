# blur

## 题面（可直接粘贴）

子目录：blur。可分离高斯模糊。约束：二维卷积结果与一维两遍完全一致（≤1e-12）；核归一化且和为 1；sigma 与半径关系确定；边界 clamp 不引入暗边（常量图不变 ≤1e-12）；4K 图不超过 600 ms；非法 sigma 可区分报错；并发安全。验收：cd blur && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd blur && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
