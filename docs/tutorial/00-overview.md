# wavelab 教程 · 0 — 总览：这个仓库在解决什么问题

> 本教程面向**第一次接触 PDE 数值解**的读者。目标不是"会调用这个库"，而是让你能
> 从零开始**重构整个求解过程**：从方程的推导、方法为什么成立（数学证明思路）、到
> 每一行代码为什么这么写。读完后你应该有能力自己重写一遍 wavelab。

## 0.1 一句话说明

wavelab 求解这一族非线性波动方程：

$$\partial_{tt} u - c^2 \Delta u = f(u), \qquad f(u) = \sum_k a_k u^k$$

其中波速 $c$ 可以是**复数**（$c=i$ 时方程变成椭圆型、初值问题不适定——这正是最有
研究价值的情形），初值 $\varphi, \psi$ 可以取复值，维度 $d = 1, 2, 3$。

设计哲学：**方程是数据，方法是可互换的求解器**。同一个方程定义一次，就能用五种
方法求解并互相对比：

```python
from wavelab import library, ExplicitFD, RegularizedFD, BranchingMC, compare

eq = library.SINE_CI_1D                       # 论文 Figure-6 的方程 (c=i)
times = [0.1, 0.2, 0.3, 0.4]
fd  = ExplicitFD(N=51, dt=0.002).solve(eq, times)
reg = RegularizedFD(N=51, dt=0.002, k_max=12).solve(eq, times)
mc  = BranchingMC(lam=0.25, n=40_000, seed=0).solve(eq, times)
compare(fd, reg, mc).plot("compare.png")
```

## 0.2 教程路线图

| 章节 | 内容 | 回答的问题 |
|---|---|---|
| [01 波动方程与适定性](01-wave-equation.md) | 方程推导、d'Alembert 公式、Fourier 模式、Hadamard 不适定性 | 我们在解什么？为什么 $c=i$ 让问题变"坏"？ |
| [02 显式有限差分](02-finite-differences.md) | Taylor 展开 → 差分格式 → von Neumann 稳定性分析 → 代码走读 | 最直接的解法怎么构造？它为什么必然爆炸？ |
| [03 隐式格式与正则化](03-implicit-and-regularization.md) | θ-格式、Newton 迭代、能量守恒的诅咒、谱截断正则化 | "换隐式格式"能救吗？（不能）什么才能救？ |
| [04 分支蒙特卡洛](04-branching-monte-carlo.md) | 积分表示 → 概率化改写（指数钟/光锥/分支）→ 逐行对应代码 | 为什么随机方法完全免疫这个不稳定性？ |
| [05 同一个方程五种解法](05-worked-example.md) | 完整对比实验 + 方法选择指南 | 实践中到底该用哪个？ |

**推荐顺序**：按编号读。02 依赖 01 的模式分析；03 依赖 02 的放大因子；04 依赖
01 的 d'Alembert 公式；05 汇总全部。

## 0.3 环境与运行

```
conda env create -f environment.yml     # 创建 conda 环境 "wavelab"
conda activate wavelab
pip install -e .                        # 可编辑安装
pytest -m "not slow"                    # ~9 秒，应全绿
```

注意 `environment.yml` 把 numpy 钉在 `<2.4`——numpy 2.4.x 在 Windows 上会让
matplotlib 原生崩溃，且 numba 0.65 不支持它。不要"顺手升级"。

## 0.4 仓库地图

```
wavelab/
├── wavelab/
│   ├── equation.py        # WaveEquation：方程 = 一份不可变数据（01 章）
│   ├── solution.py        # Solution：所有求解器的统一输出
│   ├── library.py         # 论文全部 11 个算例（含解析解 → 即测试集）
│   ├── solvers/
│   │   ├── fd_explicit.py         # 显式 leapfrog（02 章）
│   │   ├── fd_implicit.py         # θ-格式 + Newton（03 章）
│   │   ├── fd_implicit_linear.py  # 论文 Fig 7 的 LinearlyImplicitEuler（03 章）
│   │   ├── fd_regularized.py      # 谱截断正则化——真正有效的那个（03 章）
│   │   └── mc/
│   │       ├── reference.py       # 分支 MC 纯 Python 参考实现（04 章）
│   │       └── fast.py            # numba 加速后端（04 章）
│   └── experiments/       # compare / blowup_scan / mode_amplification / variance_profile
├── examples/              # fig6_side_by_side.py, illposedness_report.py
├── tests/                 # 101 个测试；闭式解 = golden tests
└── docs/
    ├── tutorial/          # 本教程（中文）
    ├── agents/            # 给 AI agent 的分层参考（英文）
    └── superpowers/       # 设计 spec 与实施计划（英文）
```

## 0.5 记号约定

| 记号 | 含义 |
|---|---|
| $u(x,t)$ 或 $u(z,t)$ | 未知解；$z$ 强调允许复数自变量 |
| $\varphi(x) = u(x,0)$ | 初始位移 |
| $\psi(x) = \partial_t u(x,0)$ | 初始速度 |
| $f(u)=\sum_k a_k u^k$ | 多项式非线性（代码里是 dict `{k: a_k}`） |
| $N,\ \Delta x,\ \Delta t$ | 网格点数、空间步长、时间步长（代码 `N, dx, dt`） |
| $\lambda,\ q_k$ | MC 的指数钟速率与分支概率——**自由参数**，只影响方差不影响均值 |
| 主角方程 | `SINE_CI_1D`：$u_{tt}+u_{xx}+u-u^3=0$，$\varphi=\sin\pi x$，$\psi=-\sin\pi x$，$x\in[0,1]$，Dirichlet 边界 |

## 0.6 全书一句话剧透

> 在不适定问题上，**任何一致的时间推进格式都必然爆炸**（显式爆得响亮，隐式爆得
> 无声）；有限差分只有靠**正则化**（主动删掉高频模）才能存活，代价是精度；而分支
> 蒙特卡洛把解写成**一个点一个点独立的期望**，根本没有"网格上的模"可以被放大，
> 所以天生稳定——但它有自己的墙：**方差**随时间爆炸。没有免费的午餐，只有不同的
> 死法。
