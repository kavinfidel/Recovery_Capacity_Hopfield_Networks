import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_all_overlaps(all_overlaps_by_setting, act='leaky', n_value=50, k_values=None, size=32, n_trials=50):
    """
    Plot histogram of overlaps for each k value in the stored results.
    
    Parameters:
        all_overlaps_by_setting : dict
            Dictionary of overlaps saved from experiment.
        act : str
            Activation function name ('leaky', etc.)
        n_value : int
            Pattern count key in the dictionary.
        k_values : list[float]
            List of k (threshold) values to plot.
        size : int
            Pattern size (for normalization).
        n_trials : int
            Number of trials (for y-axis scaling).
    """
    
    if k_values is None:
        # automatically extract available k values for this activation & n_value
        k_values = sorted(k for (n, a, k) in all_overlaps_by_setting.keys() if n == n_value and a == act)
    
    n_cols = len(k_values)
    plt.figure(figsize=(4 * n_cols, 4))
    
    for i, k in enumerate(k_values):
        # Retrieve and normalize overlaps
        overlaps = np.array(all_overlaps_by_setting[(n_value, act, k)]) / size

        plt.subplot(1, n_cols, i + 1)
        plt.hist(overlaps, bins=15, range=(-1, 2), color='skyblue', edgecolor='k', alpha=0.8)
        plt.title(f'k = {k:.2f}', fontsize=12)
        plt.xlabel('Normalized Overlap')
        plt.ylabel('Frequency')
        plt.ylim(0, n_trials)
        plt.grid(True, linestyle='--', linewidth=0.5)
    
    plt.suptitle(f'Overlap Distributions — Activation: {act}, Patterns: {n_value}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{n_value}_Patterns_{act}_overlap_hist_dumkopf.png")
    plt.show()

with open("all_overlaps_leaky_2000.pkl", "rb") as f:
    all_overlaps_by_setting = pickle.load(f)


plot_all_overlaps(all_overlaps_by_setting, act='leaky', n_value=50, k_values=[0.44, 0.34, 0.0], size=32, n_trials=50)


# python /Users/kavinfidel/Documents/Fidel/IMSC/Project/Recovery_Capacity_Hopfield_Networks/plot_overlaps.py