import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np

from hopfield_batched import Hopfield_Network_gpu

def generate_patterns(n, size):
    return np.random.choice([-1, 1], size=(n, size))

def evaluate_grid(acts, k_values, n_trials, size, epochs, temp, n_value, batch_size=128, device='cuda'):
    """
    Evaluate recovery performance across activation types and thresholds k.
    Returns metrics dictionary: (act,k) -> {'recovery_fraction', 'max_overlap'}
    """
    metrics = {}
    patterns = generate_patterns(n_value, size)

    for act in acts:
        print(f"\nEvaluating activation: {act}")
        for k in k_values:
            print(f"  k = {k:.3f}")
            model = Hopfield_Network_gpu(size, patterns, k, activation=act, device=device)

            successes = 0
            #max_overlaps = []
            n_done = 0

            while n_done < n_trials:
                bs = min(batch_size, n_trials - n_done)
                # Run batch simulation
                perfect_recovery, _ = model.mc_simulation_recovery_batch(bs, epochs, temp)
                successes += int(perfect_recovery.sum().item())
                #max_overlaps.extend(max_ov.cpu().tolist())
                n_done += bs  
                print(f" Trials done: {n_done}/{n_trials}", end='\r')

            metrics[(act, k)] = {
                'recovery_fraction': successes / n_trials
                #'max_overlap': np.array(max_overlaps)
            }
    return metrics


def summarize_for_act(metrics, act, k_values):
    """
    Given metrics for one activation, return:
        - best_k: highest recovery fraction
        - trans_k: approximate transition (largest discrete gradient)
        - recovery fractions for all k_values
    """
    recs = np.array([metrics[(act, k)]['recovery_fraction'] for k in k_values])
    best_idx = recs.argmax()
    best_k = k_values[best_idx]

    # approximate transition: largest discrete slope
    grads = np.abs(np.diff(recs))
    trans_idx = int(np.argmax(grads))
    trans_k = k_values[min(trans_idx + 1, len(k_values) - 1)]

    return best_k, trans_k, recs


# -------------------------
# Experiment settings
# -------------------------
acts = ['step', 'relu', 'leaky','elu']
coarse_k_values = np.linspace(0, 0.6, 31)  # 0 → 0.6
n_trials = 40
size = 100
epochs = 1000
temp = 0
pattern_counts = [20]
n_value = pattern_counts[0]
batch_size = 32
device = 'mps'  # or 'mps'

# -------------------------
# Run evaluation
# -------------------------
metrics = evaluate_grid(acts, coarse_k_values, n_trials, size, epochs, temp, n_value, batch_size, device)

# -------------------------
# Summarize results
# -------------------------
with open("selected_k_values_2.txt", "w") as f:
    for act in acts:
        print(act)
        best_k, trans_k, recs = summarize_for_act(metrics, act, coarse_k_values)
        k_low = coarse_k_values[np.where(recs < 0.2)[0][0]] if np.any(recs < 0.2) else coarse_k_values[0]
        f.write(f"{act}: best_k={best_k:.3f}, trans_k={trans_k:.3f}, low_k={k_low:.3f}\n")


# python /Users/kavinfidel/Documents/Fidel/IMSC/Project/Recovery_Capacity_Hopfield_Networks/batched/best_k.py