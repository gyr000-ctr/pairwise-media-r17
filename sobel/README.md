# sobel

## 题面（可直接粘贴）

子目录：sobel。Sobel 边缘与 NMS。约束：梯度幅值与解析式一致（≤1e-6）；NMS 后边缘宽度为 1 像素；双阈值滞后连接（弱边连到强边才保留）；非法阈值可区分报错；100 万像素不超过 500 ms；并发安全。验收：cd sobel && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd sobel && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
