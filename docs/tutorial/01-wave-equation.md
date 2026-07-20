# wavelab 教程 · 1 — 波动方程与适定性：为什么 c=i 让问题变"坏"

本章推导波动方程、给出它的显式解公式，然后回答整个项目的核心问题的前一半：
**为什么把 $c$ 换成 $i$ 之后，这个初值问题在数学上就"病"了**。

## 1.1 波动方程从哪里来

考虑一根拉紧的弦：线密度 $\rho$，张力 $T$，竖直位移 $u(x,t)$，振幅很小。取弦上
$[x, x+\Delta x]$ 一小段，它受到左右两端张力的竖直分量：

$$T\sin\theta(x+\Delta x) - T\sin\theta(x) \approx T\,[u_x(x+\Delta x) - u_x(x)] \approx T\,u_{xx}\,\Delta x$$

（小振幅时 $\sin\theta \approx \tan\theta = u_x$。）牛顿第二定律 $F=ma$，
这一小段质量 $\rho\,\Delta x$，加速度 $u_{tt}$：

$$\rho\,\Delta x\; u_{tt} = T\,u_{xx}\,\Delta x
\quad\Longrightarrow\quad
u_{tt} = c^2 u_{xx}, \qquad c^2 = T/\rho .$$

$c$ 是波沿弦传播的速度。如果弦还受到一个依赖于位移的回复力/驱动力（比如弹性地基、
非线性材料），右边就多出源项：

$$u_{tt} - c^2 u_{xx} = f(u).$$

wavelab 处理的正是 $f(u)=\sum_k a_k u^k$ 的多项式情形。**初值问题**（Cauchy 问题）
是给定初始位移和初始速度求以后的演化：

$$u(x,0)=\varphi(x), \qquad u_t(x,0)=\psi(x).$$

## 1.2 d'Alembert 公式：线性齐次情形的显式解

先解 $f=0$、全空间的情形。关键观察：算子可以因式分解，

$$\partial_{tt} - c^2\partial_{xx} = (\partial_t - c\,\partial_x)(\partial_t + c\,\partial_x).$$

令 $\xi = x - ct,\ \eta = x + ct$（特征坐标），链式法则给出
$\partial_\xi \partial_\eta u = 0$，所以

$$u = F(x-ct) + G(x+ct)$$

——一列右行波加一列左行波。用初值定 $F, G$：由 $u(x,0)=F+G=\varphi$ 和
$u_t(x,0) = -cF' + cG' = \psi$，第二式积分得 $-F+G = \frac{1}{c}\int_0^x \psi$，
与第一式联立解出 $F, G$，得到 **d'Alembert 公式**：

$$\boxed{\;u(x,t) = \frac{\varphi(x+ct) + \varphi(x-ct)}{2}
   + \frac{1}{2c}\int_{x-ct}^{x+ct} \psi(y)\,dy\;}$$

**请记住这个公式的结构**——第 4 章的蒙特卡洛方法就是把它（连同下面的源项）逐项
"翻译"成期望：

- $\frac{\varphi(x+ct)+\varphi(x-ct)}{2}$ —— 光锥**两端**各取一半；
- $\frac{1}{2c}\int_{x-ct}^{x+ct}\psi$ —— 光锥**内部**的一个积分（即将变成"均匀随机点"）。

有源项时（Duhamel 原理，对每个时刻的源当作新的初速度叠加）：

$$u(x,t) = \underbrace{\frac{\varphi(x+ct)+\varphi(x-ct)}{2}
   + \frac{1}{2c}\int_{x-ct}^{x+ct}\psi(y)\,dy}_{\text{齐次部分}}
   + \underbrace{\frac{1}{2c}\int_0^t\!\!\int_{x-c(t-s)}^{x+c(t-s)} f(u(y,s))\,dy\,ds}_{\text{源项：过去光锥上的二重积分}}. \tag{1.1}$$

这是一个**积分方程**：右边还含有未知的 $u$。它叫做**温和形式**（mild form），
是第 4 章的出发点。注意它对复数 $c$、复数自变量同样逐字成立（只要 $\varphi,\psi$
是解析函数，比如 $\sin$、$\tanh$、多项式的倒数——论文的所有算例都是）。

## 1.3 Fourier 模式分析：方程的"体检报告"

在区间 $[0,1]$ 上加 Dirichlet 边界 $u(0)=u(1)=0$，解可以展开成正弦级数：

$$u(x,t) = \sum_{k=1}^\infty \hat u_k(t)\,\sin(k\pi x).$$

每个模式代入方程（先**线性化**：$f(u)\approx f'(0)\,u$；对我们的
$f=-u+u^3$，$f'(0)=-1$）。因为 $\partial_{xx}\sin(k\pi x) = -k^2\pi^2\sin(k\pi x)$：

$$\hat u_k'' = \big(\,c^2\cdot(-k^2\pi^2) + f'(0)\,\big)\,\hat u_k
  \;=:\; \omega_k\,\hat u_k. \tag{1.2}$$

这个常微分方程的解完全由 $\omega_k$ 的符号决定：

- $\omega_k < 0$：$\hat u_k(t) = A\cos(\sqrt{-\omega_k}\,t)+B\sin(\sqrt{-\omega_k}\,t)$ —— **振荡**，幅度不变；
- $\omega_k > 0$：$\hat u_k(t) = A e^{\sqrt{\omega_k}\,t} + B e^{-\sqrt{\omega_k}\,t}$ —— **指数增长**。

现在对比两种波速：

| | $c=1$（正常波动方程） | $c=i$（我们的问题） |
|---|---|---|
| $c^2$ | $+1$ | $-1$ |
| $\omega_k$ | $-k^2\pi^2 - 1 < 0$ | $+k^2\pi^2 - 1 > 0\ (k\ge 1)$ |
| 模式行为 | 全部振荡 ✓ | 全部增长，速率 $\sqrt{k^2\pi^2-1}\approx k\pi$ |

**$c=i$ 时，频率越高的模式增长越快，且增长率无上界**。$k$ 号模式在时间 $t$ 内
放大约 $e^{k\pi t}$ 倍：$k=10$ 在 $t=0.5$ 放大 $e^{5\pi}\approx 6.6\times10^6$ 倍；
$k=100$ 放大 $e^{50\pi}\approx 10^{68}$ 倍。

从算子角度看：$c=i$ 使 $\partial_{tt} - c^2\partial_{xx} = \partial_{tt}+\partial_{xx}$
—— 这是 **Laplace 算子**。我们是在给一个**椭圆**方程解**初值**问题
（elliptic Cauchy problem）。

## 1.4 Hadamard 适定性：这个问题为什么叫"不适定"

Hadamard（1902）定义一个问题是**适定的**（well-posed），如果：

1. 解**存在**；
2. 解**唯一**；
3. 解**连续依赖**于数据（初值扰动小 ⟹ 解的扰动小）。

$c=i$ 的初值问题违反第 3 条，而且是灾难性地违反。取两组初值，只差一个高频小扰动：

$$\varphi_2 - \varphi_1 = \varepsilon\,\sin(k\pi x).$$

不管 $\varepsilon$ 多小，由式 (1.2)，这个差在时间 $t$ 后放大成约
$\varepsilon\, e^{k\pi t}$。对任意固定的 $t>0$ 和任意大的目标 $M$，总能选够大的
$k$ 让 $\varepsilon e^{k\pi t} > M$。**数据的差可以任意小，解的差可以任意大**——
连续依赖不成立。这正是 Hadamard 当年用来展示不适定性的原始例子（Laplace 方程的
Cauchy 问题）。

**物理直觉**：椭圆方程（如 Laplace）描述的是**平衡态**，它的自然定解方式是给定
整个边界（边值问题）。强行给它"初值 + 时间推进"的剧本，等于要求从一条边的数据
外推平衡场——就像只量了一面墙的温度想推算整个房间——微小误差在外推中指数放大。

## 1.5 那还解个什么？——解析初值与短时间

不适定 ≠ 无解。若初值是**解析函数**（可展开成收敛幂级数——$\sin$、$\tanh$ 都是），
Cauchy–Kovalevskaya 定理保证**短时间内**存在唯一的解析解。我们的论文
（Chan & Privault 2026）给出的正是这个解的**概率表示**，并用它做数值计算。

问题在于**数值方法能不能算到它**。真解是初值中已有的低频成分的温和演化（初值
$\sin(\pi x)$ 只含 $k=1$；非线性 $u^3$ 会逐渐喂出 $k=3,5,\dots$，系数快速衰减），
但任何数值噪声——舍入误差 $10^{-16}$——都含有**全部频率**的成分，而 1.4 节说了，
高频成分会被指数放大。于是：

> **真解存在且光滑；但"把噪声当初值的一部分演化"的任何方法都会把噪声炸上天。**

这句话是第 2、3 章的全部剧情。第 4 章的方法则完全绕开了它。

## 1.6 主角方程

本教程通篇使用论文 §7.2 的算例（代码 `library.SINE_CI_1D`，= C++ Simulation 11–14）：

$$u_{tt} + u_{xx} + u - u^3 = 0,\qquad
\varphi = \sin(\pi x),\quad \psi = -\sin(\pi x),\quad x\in[0,1],$$

即 $c=i,\ f(u) = -u + u^3$（focusing 椭圆情形）。wavelab 里它是一份数据：

```python
import cmath, math
from wavelab import WaveEquation

SINE_CI_1D = WaveEquation(
    dim=1,
    c=1j,                                  # c = i：本章的全部剧情由这一行触发
    f={1: -1, 3: 1},                       # f(u) = -u + u^3，写成 {幂次: 系数}
    phi=lambda z: cmath.sin(math.pi * z),  # 注意 cmath：初值必须能吃复数参数
    psi=lambda z: -cmath.sin(math.pi * z),
    domain=((0.0, 1.0),),                  # FD 需要边界；MC 会忽略它
)
```

几个设计决定的"为什么"：

- **`f` 是 dict 而不是函数**：差分法只需要能计算 $f(u)$（`eq.f_callable()` 会把
  dict 拼成函数），但蒙特卡洛需要**每一项的幂次和系数**来构造分支树（第 4 章）。
  存 dict，两种需求都满足，方程定义永不分叉。
- **`phi` 用 `cmath` 不用 `math`**：$c=i$ 时光锥端点 $x \pm ct = x \pm it$ 是复数,
  初值会在复平面上被求值。`math.sin` 遇到复数直接报错——`WaveEquation` 构造时就
  会做这个检查（fail fast）。
- **`domain` 是可选的**：MC 用的是全空间积分表示 (1.1)，根本没有边界的概念；
  只有差分法需要网格和边界条件。

## 1.7 本章小结（后面各章要反复引用的三个事实）

1. **温和形式 (1.1)**：解 = 光锥端点的 $\varphi$ 平均 + 光锥内的 $\psi$ 积分 +
   过去光锥上的 $f(u)$ 二重积分。→ 第 4 章的出发点。
2. **模式增长律 (1.2)**：$c=i$ 时 $k$ 号模式以 $e^{k\pi t}$ 增长，高频最快、无上界。
   → 第 2、3 章解释一切爆炸现象的钥匙。
3. **不适定但可解**：真解（解析初值、短时间）存在且光滑；被放大的是**噪声**，
   不是真解本身。→ 正则化（第 3 章）与逐点方法（第 4 章）的立足点。

[→ 下一章：显式有限差分](02-finite-differences.md)
