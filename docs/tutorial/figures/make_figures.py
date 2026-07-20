"""Regenerate every figure used by docs/tutorial/. Deterministic (fixed seeds).

Run:  <anaconda>/envs/wavelab/python.exe docs/tutorial/figures/make_figures.py
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.simplefilter("ignore")

import cmath, math
from wavelab import (WaveEquation, library, ExplicitFD, RegularizedFD,
                     BranchingMC, compare)
from wavelab.experiments import mode_amplification, variance_profile

OUT = os.path.dirname(os.path.abspath(__file__))
EQ = library.SINE_CI_1D
WELL = WaveEquation(dim=1, c=1, f={}, phi=lambda z: cmath.sin(math.pi * z),
                    psi=lambda z: 0j, domain=((0.0, 1.0),), name="wellposed")


def fig_mode_growth():
    ill = mode_amplification(EQ, N=101, dt=0.002)
    well = mode_amplification(WELL, N=101, dt=0.0005)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ill["k"], ill["growth"], "-", lw=2, label="ill-posed  c=i  (dt=0.002)")
    ax.plot(well["k"], well["growth"], "--", lw=2, label="well-posed  c=1  (dt=0.0005)")
    ax.axhline(1.0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("Fourier mode k  (sin(k$\\pi$x))")
    ax.set_ylabel("per-step amplification  max|g|")
    ax.set_title("Leapfrog mode amplification: ill-posed grows, well-posed stays on |g|=1")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mode_growth.png"), dpi=140)
    plt.close(fig)
    print("mode_growth.png")


def fig_three_methods():
    times = [0.1, 0.2, 0.3, 0.4]
    fd = ExplicitFD(N=51, dt=0.002).solve(EQ, times)
    reg = RegularizedFD(N=51, dt=0.002, k_max=12).solve(EQ, times)
    mc = BranchingMC(lam=0.25, n=20_000, seed=0).solve(EQ, times)
    fig = compare(fd, reg, mc).plot(os.path.join(OUT, "three_methods.png"))
    plt.close(fig)
    print("three_methods.png")


def fig_branching_tree():
    """Schematic of ONE sample tree for f = -u + u^3 (children J in {1,3})."""
    fig, ax = plt.subplots(figsize=(7.6, 5))
    # Layout: x horizontal, REMAINING time vertical (t at top, 0 at bottom).
    # Node = (x, remaining). Cones drawn with |slope| = 1 for the schematic.
    root = (0.5, 0.40)

    def cone(node, dt_seg, color="0.75"):
        (x, s) = node
        ax.plot([x, x - dt_seg], [s, s - dt_seg], ":", color=color, lw=1)
        ax.plot([x, x + dt_seg], [s, s - dt_seg], ":", color=color, lw=1)

    def seg(a, b, **kw):
        ax.annotate("", xy=(b[0], b[1]), xytext=(a[0], a[1]),
                    arrowprops=dict(arrowstyle="-|>", lw=kw.get("lw", 1.8),
                                    color=kw.get("color", "C0")))

    # root branches after tau1 = 0.14 into J = 3 children at mark x=0.62
    b1 = (0.62, 0.26)
    cone(root, 0.14)
    seg(root, b1)
    ax.plot(*b1, "o", color="C3", ms=8, zorder=5)
    ax.annotate("clock $\\tau_1<t$: branch, $J=3$\nweight $e^{\\lambda\\tau_1}\\frac{\\tau_1}{\\lambda}\\frac{a_3}{q_3}$",
                xy=b1, xytext=(0.80, 0.31), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.4"))

    # child A: leaf (clock exceeds remaining 0.26)
    a_leaf = (0.47, 0.0)
    cone(b1, 0.26)
    seg(b1, (0.47, 0.02), color="C0")
    ax.plot(a_leaf[0], 0.012, "s", color="C2", ms=8, zorder=5)

    # child B: branches again after tau=0.11 with J=1
    b2 = (0.70, 0.15)
    seg(b1, b2, color="C0")
    ax.plot(*b2, "o", color="C3", ms=7, zorder=5)
    ax.annotate("branch, $J=1$\nweight $e^{\\lambda\\tau}\\frac{\\tau}{\\lambda}\\frac{a_1}{q_1}$",
                xy=b2, xytext=(0.86, 0.17), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.4"))
    b2_leaf = (0.78, 0.0)
    seg(b2, (0.78, 0.02), color="C0")
    ax.plot(b2_leaf[0], 0.012, "s", color="C2", ms=8, zorder=5)

    # child C: immediate leaf
    seg(b1, (0.56, 0.02), color="C0")
    ax.plot(0.56, 0.012, "s", color="C2", ms=8, zorder=5)

    ax.plot(*root, "o", color="k", ms=9, zorder=5)
    ax.annotate("root: estimate $u(x,t)$", xy=root, xytext=(0.13, 0.385), fontsize=10)
    ax.annotate("leaf (clock $\\tau>$ remaining time):\nevaluate initial data,"
                " weight $e^{\\lambda t}\\,(\\frac{\\varphi}{2}+\\frac{\\varphi}{2}+t\\psi)$",
                xy=(0.56, 0.012), xytext=(0.05, 0.10), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.4"))
    ax.annotate("$H$ = product of ALL node weights;   $u(x,t)=\\mathbb{E}[H]$",
                xy=(0.5, -0.055), fontsize=11, ha="center", annotation_clip=False)

    ax.axhline(0, color="k", lw=1)
    ax.text(0.015, 0.005, "remaining time = 0\n(initial data $\\varphi,\\psi$ live here)",
            fontsize=8, va="bottom")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.09, 0.46)
    ax.set_xlabel("space $x$  (marks move on the light cone $x \\pm c\\tau u$)")
    ax.set_ylabel("remaining time")
    ax.set_title("One sample of the branching tree  ($f=-u+u^3$: J ∈ {1,3})")
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "branching_tree.png"), dpi=140)
    plt.close(fig)
    print("branching_tree.png")


def fig_variance_wall():
    rows = variance_profile(EQ, BranchingMC(lam=0.25, n=20_000, seed=1),
                            times=[0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.4], point=0.5)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ts = [r["t"] for r in rows]
    ax.semilogy(ts, [r["stderr"] for r in rows], "-o", ms=5, label="stderr")
    ax.semilogy(ts, [r["rel_stderr"] for r in rows], "-s", ms=5,
                label="relative stderr")
    ax.axhline(1.0, color="r", lw=0.8, ls="--", alpha=0.6)
    ax.text(0.12, 1.25, "rel. stderr = 100%: pure noise", color="r", fontsize=9)
    ax.set_xlabel("t")
    ax.set_ylabel("MC error (log scale)")
    ax.set_title("Branching MC's own wall: variance, not instability (n=20k)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "variance_wall.png"), dpi=140)
    plt.close(fig)
    print("variance_wall.png")


def fig_kmax_tradeoff():
    ks, bts = [], []
    for K in (6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36):
        sol = RegularizedFD(N=101, dt=0.002, k_max=K).solve(EQ, times=[0.7])
        ks.append(K)
        bts.append(sol.meta["blowup_time"] if sol.meta["blowup_time"] else 0.7)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["C2" if b >= 0.7 else "C3" for b in bts]
    ax.bar(ks, bts, width=2.0, color=colors, alpha=0.85)
    ax.axhline(0.7, color="C2", lw=0.8, ls="--", alpha=0.7)
    ax.text(6, 0.71, "survived the whole run (t=0.7)", color="C2", fontsize=9)
    ax.set_xlabel("k_max  (number of sine modes kept)")
    ax.set_ylabel("blow-up time  (0.7 = survived)")
    ax.set_title("RegularizedFD: keep more modes → recover the blow-up")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kmax_tradeoff.png"), dpi=140)
    plt.close(fig)
    print("kmax_tradeoff.png")


if __name__ == "__main__":
    fig_mode_growth()
    fig_three_methods()
    fig_branching_tree()
    fig_variance_wall()
    fig_kmax_tradeoff()
    print("all figures written to", OUT)
