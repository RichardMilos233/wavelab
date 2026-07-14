"""House style: every wavelab figure looks like family (spec §4)."""
import matplotlib.pyplot as plt

DPI = 140

def apply_style(ax):
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)

def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
