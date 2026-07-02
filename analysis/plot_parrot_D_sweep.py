"""Regenerate the parrot D-sweep figures (paper Figs. 4, 9, 10 and Fig. 5 data)
from analysis/parrot_D_sweep/ (full-scale sMAPE; replaces the half-scale
figures — see legacy/README.md).

Outputs (written next to this script):
    smape_rolling_parrot_normalized.pdf   Fig. 4, Parrot panel (D=5)
    vpt_vs_embedding_dimension.pdf        Fig. 9
    scaling_laws_vs_D.pdf                 Fig. 10 (two panels)
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, 'parrot_D_sweep')
GRANULARITY = 30
FORECAST_LENGTH = 300
CLS = [2 ** k for k in range(6, 17)]
DS = [5, 10, 15, 20, 25, 30]

plt.rcParams['font.family'] = 'Helvetica'


def load(D, key):
    return np.load(os.path.join(SWEEP, f'D={D}_{key}.npy'), allow_pickle=True).item()


def styled_axes(figsize=(17, 15)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for side in ['bottom', 'left', 'top', 'right']:
        ax.spines[side].set_linewidth(5)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    fig.set_size_inches(*figsize)
    ax.tick_params(axis='y', pad=15, size=23, width=5, labelsize=50)
    ax.tick_params(axis='x', pad=20, size=23, width=5, labelsize=50)
    return fig, ax


# ---------------------------------------------------------------- Fig. 4 left
def fig4_parrot_panel(D=5):
    med = load(D, 'median_smape')
    eqs = sorted(set(k[0] for k in med))
    time = np.arange(1 / GRANULARITY, (FORECAST_LENGTH + 1) / GRANULARITY, 1 / GRANULARITY)
    colors = ["#7400b8", "#6930c3", "#5e60ce", "#5390d9", "#4ea8de", "#48bfe3",
              "#56cfe1", "#64dfdf", "#72efdd", "#80ffdb", "#9cffc9"]
    fig, ax = styled_axes()
    for i, cl in enumerate(CLS):
        curves = [med[(eq, cl)] for eq in eqs if (eq, cl) in med]
        trj = np.median(curves, axis=0)
        err = np.std(curves, axis=0) / np.sqrt(len(curves))
        ax.plot(time, trj, lw=5, color=colors[i], label=cl)
        ax.fill_between(time, trj - err, trj + err, alpha=0.2, color=colors[i])
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim([0.1, 10]); ax.set_ylim([1, 300])
    ax.set_xlabel('Time (Lyapunov times)', fontsize=60)
    ax.set_ylabel('Error (sMAPE)', fontsize=60)
    ax.legend(loc='lower right', frameon=False, prop={'size': 40}, ncol=2)
    fig.set_tight_layout(True)
    fig.savefig(os.path.join(HERE, 'smape_rolling_parrot_normalized.pdf'), bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------- Fig. 9
def fig9_vpt_vs_D(cls_shown=CLS[:-1]):
    fig, ax = styled_axes()
    cmap = plt.get_cmap('cool')
    for i, cl in enumerate(cls_shown):
        means, errs = [], []
        for D in DS:
            avg = load(D, 'average_vpt')
            vals = [v for (eq, c), v in avg.items() if c == cl]
            means.append(np.mean(vals))
            errs.append(np.std(vals) / np.sqrt(len(vals)))
        means, errs = np.array(means), np.array(errs)
        color = cmap(i / max(len(cls_shown) - 1, 1))
        ax.plot(DS, means, lw=5, color=color, label=cl)
        ax.fill_between(DS, means - errs, means + errs, alpha=0.2, color=color)
    ax.set_xlabel('Embedding dimension', fontsize=60)
    ax.set_ylabel('VPT (Lyap. times)', fontsize=60)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False, prop={'size': 35})
    fig.set_tight_layout(True)
    fig.savefig(os.path.join(HERE, 'vpt_vs_embedding_dimension.pdf'), bbox_inches='tight')
    plt.close(fig)


# -------------------------------------------------------------------- Fig. 10
def fig10_scaling_laws():
    fig, axes = plt.subplots(1, 2, figsize=(30, 13))
    colors = ["#7400b8", "#6930c3", "#5e60ce", "#5390d9", "#4ea8de", "#48bfe3"]
    for ax in axes:
        for side in ['bottom', 'left', 'top', 'right']:
            ax.spines[side].set_linewidth(5)
        ax.tick_params(axis='both', pad=15, size=23, width=5, labelsize=45)
    for j, (key, ylabel, agg_idx) in enumerate([
            ('median_smape', 'One-step error (sMAPE)', 0),
            ('median_l2', 'Matching motif distance', None)]):
        ax = axes[j]
        for i, D in enumerate(DS):
            d = load(D, key)
            means, errs = [], []
            for cl in CLS:
                vals = [(v[0] if agg_idx == 0 else v) for (eq, c), v in d.items() if c == cl]
                means.append(np.median(vals))
                errs.append(np.std(vals) / np.sqrt(len(vals)))
            means, errs = np.array(means), np.array(errs)
            ax.plot(CLS, means, lw=5, color=colors[i], label=f'D = {D}')
            ax.fill_between(CLS, means - errs, means + errs, alpha=0.2, color=colors[i])
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Context length', fontsize=55)
        ax.set_ylabel(ylabel, fontsize=55)
        ax.legend(frameon=False, prop={'size': 35})
    fig.set_tight_layout(True)
    fig.savefig(os.path.join(HERE, 'scaling_laws_vs_D.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig4_parrot_panel()
    print('Fig. 4 Parrot panel done')
    fig9_vpt_vs_D()
    print('Fig. 9 done')
    fig10_scaling_laws()
    print('Fig. 10 done')
