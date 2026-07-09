"""
Analysis for the noise-optimal depth frontier study.
Reads results/runs.csv, produces:
  figs/heatmap_<dataset>.pdf     -- accuracy landscape (depth x noise) per encoding
  figs/frontier.pdf              -- d*(lambda) per encoding, averaged over datasets
  figs/selection_rule.csv        -- compact noise-budget -> (encoding, depth) rule

Frontier definition:
  d*(lambda, E) = min{ d : acc(d, lambda, E) >= (1 - eps) * max_d' acc(d', lambda, E) }
with eps = 0.02, accuracies seed-averaged.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 0.02
df = pd.read_csv("results/runs.csv")
os.makedirs("figs", exist_ok=True)

# Seed-averaged accuracy with std
g = (df.groupby(["encoding", "dataset", "depth", "noise_scale"])["test_acc"]
       .agg(["mean", "std"]).reset_index())

encodings = sorted(df.encoding.unique())
datasets = sorted(df.dataset.unique())
depths = sorted(df.depth.unique())
noises = sorted(df.noise_scale.unique())

# ---------------------------------------------------------------- heatmaps
for ds in datasets:
    fig, axes = plt.subplots(1, len(encodings), figsize=(4 * len(encodings), 3.2),
                             sharey=True)
    for ax, enc in zip(np.atleast_1d(axes), encodings):
        sub = g[(g.dataset == ds) & (g.encoding == enc)]
        M = sub.pivot(index="depth", columns="noise_scale", values="mean") \
               .reindex(index=depths, columns=noises)
        im = ax.imshow(M.values, aspect="auto", origin="lower",
                       vmin=0.5, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(noises)), [f"{n:g}" for n in noises], fontsize=8)
        ax.set_yticks(range(len(depths)), depths, fontsize=8)
        ax.set_xlabel(r"noise scale $\lambda$")
        ax.set_title(enc, fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("ansatz depth $L$")
    fig.colorbar(im, ax=axes, label="test accuracy", shrink=0.85)
    fig.suptitle(f"Accuracy landscape -- {ds}", fontsize=11)
    fig.savefig(f"figs/heatmap_{ds}.pdf", bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- frontier
def frontier(sub):
    """d*(lambda) for one (encoding, dataset): sub has columns depth, noise_scale, mean."""
    out = {}
    for lam in noises:
        s = sub[sub.noise_scale == lam].set_index("depth")["mean"].reindex(depths)
        thresh = (1 - EPS) * s.max()
        ok = s[s >= thresh]
        out[lam] = ok.index.min() if len(ok) else np.nan
    return out

fig, ax = plt.subplots(figsize=(5.2, 3.6))
frontier_rows = []
for enc in encodings:
    per_ds = []
    for ds in datasets:
        f = frontier(g[(g.encoding == enc) & (g.dataset == ds)])
        per_ds.append([f[lam] for lam in noises])
        for lam in noises:
            frontier_rows.append(dict(encoding=enc, dataset=ds,
                                      noise_scale=lam, d_star=f[lam]))
    mean_f = np.nanmean(np.array(per_ds, dtype=float), axis=0)
    ax.plot(noises, mean_f, marker="o", label=enc)
ax.set_xscale("symlog", linthresh=0.1)
ax.set_xlabel(r"noise scale $\lambda$ (1.0 = anchored device point)")
ax.set_ylabel(r"noise-optimal depth $d^*$")
ax.legend(fontsize=8)
ax.set_title("Noise-optimal depth frontier (dataset-averaged)", fontsize=10)
fig.savefig("figs/frontier.pdf", bbox_inches="tight")
plt.close(fig)
pd.DataFrame(frontier_rows).to_csv("figs/frontier_data.csv", index=False)

# ---------------------------------------------------------------- selection rule
# For each noise level: best (encoding, d*) by seed-averaged accuracy at the frontier depth.
rule = []
fr = pd.DataFrame(frontier_rows)
for lam in noises:
    best = None
    for enc in encodings:
        dstars = fr[(fr.encoding == enc) & (fr.noise_scale == lam)]["d_star"].dropna()
        if dstars.empty:
            continue
        d = int(np.round(dstars.median()))
        accs = g[(g.encoding == enc) & (g.noise_scale == lam) & (g.depth == d)]["mean"]
        a = accs.mean()
        if best is None or a > best[2]:
            best = (enc, d, a)
    rule.append(dict(noise_scale=lam, encoding=best[0], depth=best[1],
                     mean_acc=round(best[2], 3)))
rule_df = pd.DataFrame(rule)
rule_df.to_csv("figs/selection_rule.csv", index=False)
print("Selection rule (noise budget -> encoding, depth):")
print(rule_df.to_string(index=False))
print("\nFigures in figs/: heatmap_<ds>.pdf, frontier.pdf, frontier_data.csv, selection_rule.csv")
