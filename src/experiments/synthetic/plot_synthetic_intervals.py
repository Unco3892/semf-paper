import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import seaborn as sns
from synthetic_data_generator import generate_synthetic_data, compute_true_values
from matplotlib.lines import Line2D

# Use Seaborn's "seaborn-v0_8-paper" style for a clean, publication-ready theme
plt.style.use("seaborn-v0_8-paper")
mpl.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 300,
    "text.usetex": True,       # Enable if you have LaTeX installed; otherwise, set to False
    "font.family": "serif",
})

# Function to plot data with prediction intervals
def plot_with_intervals(ax, x, y, y_true, noise_type, interval_width=1.96):
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_true_sorted = y_true[idx]
    
    if noise_type == 'normal':
        std = np.std(y - y_true)
        lower = y_true - interval_width * std
        upper = y_true + interval_width * std
        ax.set_title("(a) Normal Noise")
    elif noise_type == 'uniform':
        bound = np.max(np.abs(y - y_true)) * 1.05
        lower = y_true - bound
        upper = y_true + bound
        y_min, y_max = ax.get_ylim()
        ax.hlines(y=[np.min(lower), np.max(upper)], xmin=-3, xmax=3, 
                  colors='darkblue', linestyles='--', alpha=0.3, label='_nolegend_')
        ax.set_title("(b) Uniform Noise")
    elif noise_type == 'lognormal':
        residuals = y - y_true
        lower = y_true + np.percentile(residuals, 2.5)
        upper = y_true + np.percentile(residuals, 97.5)
        ax.set_title("(c) Log-normal Noise")
    elif noise_type == 'gumbel':
        residuals = y - y_true
        lower = y_true + np.percentile(residuals, 2.5)
        upper = y_true + np.percentile(residuals, 97.5)
        ax.set_title("(d) Gumbel Noise")
    
    ax.scatter(x, y, alpha=0.5, s=15, color='blue', label='Observations')
    ax.plot(x_sorted, y_true_sorted, color='red', linewidth=2, label='$f(x)$')
    ax.fill_between(x_sorted, lower[idx], upper[idx], alpha=0.3, color='blue', label='95\\% Prediction interval')
    
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

if __name__ == "__main__":
    seed = 42
    n = 500
    # Generate synthetic data for each noise type using the unified generator with method "quadratic"
    data_normal = generate_synthetic_data(n, 1, seed, noise_distribution="normal", method="quadratic")
    data_uniform = generate_synthetic_data(n, 1, seed, noise_distribution="uniform", method="quadratic")
    data_lognormal = generate_synthetic_data(n, 1, seed, noise_distribution="lognormal", method="quadratic")
    data_gumbel = generate_synthetic_data(n, 1, seed, noise_distribution="gumbel", method="quadratic")
    
    # Extract and sort the x-values (from column "x1") and corresponding noisy outputs;
    # compute the true (deterministic) function for each (for a 1D quadratic-plus-periodic process):
    x_normal = np.sort(data_normal["x1"].values)
    y_normal = data_normal.loc[np.argsort(data_normal["x1"].values), "Output"].values
    y_true_normal = compute_true_values(x_normal, method="quadratic")
    
    x_uniform = np.sort(data_uniform["x1"].values)
    y_uniform = data_uniform.loc[np.argsort(data_uniform["x1"].values), "Output"].values
    y_true_uniform = compute_true_values(x_uniform, method="quadratic")
    
    x_lognormal = np.sort(data_lognormal["x1"].values)
    y_lognormal = data_lognormal.loc[np.argsort(data_lognormal["x1"].values), "Output"].values
    y_true_lognormal = compute_true_values(x_lognormal, method="quadratic")
    
    x_gumbel = np.sort(data_gumbel["x1"].values)
    y_gumbel = data_gumbel.loc[np.argsort(data_gumbel["x1"].values), "Output"].values
    y_true_gumbel = compute_true_values(x_gumbel, method="quadratic")
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    
    # Set common y-limits using data from all noise types
    all_y = np.concatenate((y_normal, y_uniform, y_lognormal, y_gumbel))
    y_min = np.min(all_y) - 1
    y_max = np.max(all_y) + 1
    y_min = max(y_min, -5)
    y_max = min(y_max, 15)
    for ax in axes.flatten():
        ax.set_ylim(y_min, y_max)
    
    plot_with_intervals(axes[0, 0], x_normal, y_normal, y_true_normal, 'normal')
    plot_with_intervals(axes[0, 1], x_uniform, y_uniform, y_true_uniform, 'uniform')
    plot_with_intervals(axes[1, 0], x_lognormal, y_lognormal, y_true_lognormal, 'lognormal')
    plot_with_intervals(axes[1, 1], x_gumbel, y_gumbel, y_true_gumbel, 'gumbel')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Get legend handles from one of the subplots...
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Create a dummy handle for the uniform noise dashed line.
    uniform_line = Line2D([], [], color='darkblue', linestyle='--', alpha=0.3, label='Uniform noise bounds')
    handles.append(uniform_line)
    labels.append('Uniform noise bounds')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=4, fontsize=12)
    
    os.makedirs('../../../results/', exist_ok=True)
    fig.savefig('../../../results/synthetic_prediction_intervals.pdf', dpi=300, bbox_inches='tight')

    if os.path.exists('../../../paper/semf_2024_arxiv/assets/'):
        fig.savefig('../../../paper/semf_2024_arxiv/assets/synthetic_prediction_intervals.pdf', dpi=300, bbox_inches='tight')

    plt.show() 