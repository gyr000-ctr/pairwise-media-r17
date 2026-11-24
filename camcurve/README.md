# camcurve

## 题面（可直接粘贴）

子目录：camcurve。相机路径与朝向稳定。约束：位置 Catmull-Rom 通过所有控制点；上向量必须正交化（≤1e-9）；沿路径速度连续（加速度有界）；穿透控制点处切线连续；非法控制点可区分报错；1 万次采样不超过 300 ms。验收：cd camcurve && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd camcurve && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
