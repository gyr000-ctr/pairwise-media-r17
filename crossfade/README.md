# crossfade

## 题面（可直接粘贴）

子目录：crossfade。等功率交叉淡化。约束：淡化中点功率和为 1（≤1e-9）；端点能量连续；相位对齐后无抵消（相关 >0.9）；长度不等可区分报错；100 万样本不超过 300 ms；并发安全。验收：cd crossfade && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd crossfade && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
