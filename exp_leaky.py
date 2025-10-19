import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid',font_scale = 1.2)
plt.style.use('default')

from hopfield_gpu import Hopfield_Network_gpu


def generate_patterns(n, size):


    return np.random.choice([-1, 1], size=(n, size))

k_values = [0.44, 0.34, 0.0]
n_trials = 2000
size = 100
epochs = 1000
temp = 0
pattern_counts = [2000]



all_overlaps_by_setting = {}  # for histogram use later

for n_value in pattern_counts:
    print(f"\n--- Pattern count = {n_value} ---")
    patterns = generate_patterns(n_value, size)

    plt.figure(figsize=(8, 6))  # << bigger, separate figure for each pattern count

    for act, style in [('leaky', '^-')]:
        print(f"  Activation: {act}")
        curve = []
         #max_overlaps = []

        for k in k_values:
            print(f"{k}th value")
            max_overlaps = []
            successes = 0
            model = Hopfield_Network_gpu(size, patterns, k, activation=act, device = 'cuda')
            for _ in range(n_trials):
                
                success, max_overlap = model.mc_simulation_recovery(epochs, temp)
                successes += success
                max_overlaps.append(max_overlap)

            curve.append(successes / n_trials)

        
            all_overlaps_by_setting[(n_value, act,k)] = max_overlaps
        plt.plot(k_values, curve, style, label=act, linewidth=2, markersize=6)

    plt.title(f'{n_value} Patterns', fontsize=14)
    plt.xlabel('Threshold k', fontsize=12)
    plt.ylabel('Recovery Fraction', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(title='Activation', fontsize=10)
    plt.tight_layout()
    plt.savefig("leaky_large_k.png")
    plt.show()
