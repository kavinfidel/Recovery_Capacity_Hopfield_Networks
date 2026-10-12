import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from hopfield_batched import Hopfield_Network_gpu


def generate_patterns(n, size):
    return np.random.choice([-1, 1], size=(n, size))


k_values = [0.44, 0.34, 0.0] # figure out these k values
n_trials = 10000
size = 100
epochs = 1000
temp = 0
pattern_counts = [2000]
batch_size = 128  

all_overlaps_by_setting = {}

for n_value in pattern_counts:
    print(f"\n--- Pattern count = {n_value} ---")
    patterns = generate_patterns(n_value, size)

    plt.figure(figsize=(8, 6))

    for act, style in [('relu', '^-')]:
        print(f"  Activation: {act}")
        curve = []

        for k in k_values:
            print(f"    Threshold k = {k}")
            model = Hopfield_Network_gpu(size, patterns, k, activation=act, device='auto')

            successes = 0
            max_overlaps = []
            n_done = 0

            while n_done < n_trials:
                bs = min(batch_size, n_trials - n_done)
                perfect_recovery, max_ov = model.mc_simulation_recovery_batch(bs, epochs, temp)
                successes += int(perfect_recovery.sum().item())
                max_overlaps.extend(max_ov.cpu().tolist())
                n_done += bs
                print(n_done)

            curve.append(successes / n_trials)
            all_overlaps_by_setting[(n_value, act, k)] = max_overlaps

        plt.plot(k_values, curve, style, label=act, linewidth=2, markersize=6)

    plt.title(f'{n_value} Patterns', fontsize=14)
    plt.xlabel('Threshold k', fontsize=12)
    plt.ylabel('Recovery Fraction', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(title='Activation', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"relu_large_k_{n_value}.png")
    plt.show()

    with open(f"all_overlaps_relu_{n_value}.pkl", "wb") as f:
        pickle.dump(all_overlaps_by_setting, f)
