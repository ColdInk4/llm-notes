# 第 2 章 §附录 A 反向传播梯度推导

主章节 [第 2 章 §2.4.3 PyTorch 中的梯度计算](../chapter2/chapter2_pytorch与资源核算.md) 已经给出 4 个梯度的结论公式、形状与 FLOPs；本附录按元素级展开每条结论的链式法则路径，验证矩阵乘形式与形状一致。

前向关系：

$$
h_1 = x W_1, \qquad h_2 = h_1 W_2, \qquad L = \mathrm{mean}(h_2^2)
$$

四个待求梯度按反向传播顺序计算：先 `h2.grad`、再 `w2.grad`、再 `h1.grad`、最后 `w1.grad`。下文统一用 $G_z = \partial L / \partial z$ 表示损失对张量 $z$ 的梯度。

## §A.1 `h2.grad`

$L = \mathrm{mean}(h_2^2) = (1 / (BK)) \sum_{i,k} h_2[i,k]^2$ 对 $h_2$ 的每个元素独立求导：

$$
\frac{\partial L}{\partial h_2[i,k]} = \frac{2}{BK}\, h_2[i,k] \quad \Longrightarrow \quad G_{h_2} = \frac{2}{BK} h_2 .
$$

形状 $(B, K)$，element-wise 标量乘法，FLOPs 为 $B K$。

## §A.2 `w2.grad`

从前向关系 $h_2[i, k] = \sum_{j} h_1[i, j] \cdot W_2[j, k]$ 出发，沿 batch 维度 $i$ 求链式法则：

$$
\frac{\partial L}{\partial W_2[j,k]} = \sum_i \frac{\partial L}{\partial h_2[i,k]} \cdot \frac{\partial h_2[i,k]}{\partial W_2[j,k]} .
$$

线性映射下 $h_1$ 不依赖 $W_2$，故 $\partial h_2[i,k] / \partial W_2[j,k] = h_1[i, j]$，代入即得沿 batch 维度的内积：

$$
G_{W_2}[j,k] = \sum_i h_1[i,j] \cdot G_{h_2}[i,k] \quad \Longrightarrow \quad G_{W_2} = h_1^{\mathrm{T}} G_{h_2} .
$$

形状验证：$h_1^{\mathrm{T}}$ 为 $(D, B)$，$G_{h_2}$ 为 $(B, K)$，相乘得 $(D, K)$，与 `w2.grad` 一致。FLOPs $= 2 B D K$。

## §A.3 `h1.grad`

链式法则沿输出维度 $k$ 把局部导数累加：

$$
\frac{\partial L}{\partial h_1[i,j]} = \sum_k \frac{\partial L}{\partial h_2[i,k]} \cdot \frac{\partial h_2[i,k]}{\partial h_1[i,j]} .
$$

线性映射下 $\partial h_2[i,k] / \partial h_1[i,j] = W_2[j, k]$，代入即得沿输出维度的内积：

$$
G_{h_1}[i,j] = \sum_k G_{h_2}[i,k] \cdot W_2[j,k] \quad \Longrightarrow \quad G_{h_1} = G_{h_2} W_2^{\mathrm{T}} .
$$

形状验证：$G_{h_2}$ 为 $(B, K)$，$W_2^{\mathrm{T}}$ 为 $(K, D)$，相乘得 $(B, D)$，与 `h1.grad` 一致。FLOPs $= 2 B D K$，与 `h1 @ w2` 前向相同。

## §A.4 `w1.grad`

沿用同一模式，从 $h_1[i, k] = \sum_j x[i, j] \cdot W_1[j, k]$ 出发沿 batch 维度 $i$ 求链式法则，局部导数 $\partial h_1[i,k] / \partial W_1[j,k] = x[i, j]$，代入后升格为矩阵乘：

$$
G_{W_1} = x^{\mathrm{T}} G_{h_1} .
$$

形状验证：$x^{\mathrm{T}}$ 为 $(D, B)$，$G_{h_1}$ 为 $(B, D)$，相乘得 $(D, D)$，与 `w1.grad` 一致。FLOPs $= 2 B D D$。

## §A.5 统一形式

对线性映射 $h_{\text{out}} = h_{\text{in}} W$：

$$
G_{h_{\text{in}}} = G_{h_{\text{out}}} W^{\mathrm{T}}, \qquad G_W = h_{\text{in}}^{\mathrm{T}} G_{h_{\text{out}}} .
$$

两层都按 $h_{\text{in}}^{\mathrm{T}} G_{h_{\text{out}}}$ 这一矩阵乘算 weight grad，按 $G_{h_{\text{out}}} W^{\mathrm{T}}$ 算 activation grad；两者 FLOPs 均为 $2 \cdot (\text{batch 维度}) \cdot (\text{输入维度}) \cdot (\text{输出维度})$，对应该层前向的 2 倍。
