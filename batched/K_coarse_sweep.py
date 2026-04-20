import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

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


# def summarize_for_act(metrics, act, k_values):
#     """
#     Given metrics for one activation, return:
#         - best_k: highest recovery fraction
#         - trans_k: approximate transition (largest discrete gradient)
#         - recovery fractions for all k_values
#     """
#     recs = np.array([metrics[(act, k)]['recovery_fraction'] for k in k_values])
#     best_idx = recs.argmax()
#     best_k = k_values[best_idx]

#     # approximate transition: largest discrete slope
#     grads = np.abs(np.diff(recs))
#     trans_idx = int(np.argmax(grads))
#     trans_k = k_values[min(trans_idx + 1, len(k_values) - 1)]

#     return best_k, trans_k, recs

def summarize_for_act(metrics, act, k_values):
    recs = np.array([metrics[(act, k)]['recovery_fraction'] for k in k_values])
    
    # 1. Best K: Peak performance
    best_idx = recs.argmax()
    best_k = k_values[best_idx]

    # 2. Trans K (The Transition): Where the drastic "jump" happens?
    # We find the largest change in recovery between two adjacent k steps
    grads = np.diff(recs)
    trans_idx = int(np.argmax(grads))
    trans_k = k_values[trans_idx + 1] # The k where it finished the jump

    # 3. Low K (The Breaking Point): 
    # The LARGEST k-value where the model is still failing (< 50% success)
    # This represents the "floor" of usability.
    failing_indices = np.where(recs >= 0.5)[0]
    if len(failing_indices) > 0:
        k_low = k_values[failing_indices[0]] # The last k that was still "bad"
    else:
        k_low = k_values[-1]

    return best_k, trans_k, k_low, recs




# -------------------------
# Experiment settings
# -------------------------
acts = ['step', 'relu', 'leaky','elu']
coarse_k_values = np.linspace(0.15, 0.45, 51)  # 0 → 0.6
n_trials = 500
size = 100
epochs = 1000
temp = 0
pattern_counts = [2000]
n_value = pattern_counts[0]
batch_size = 128
device = 'mps'  # or 'mps'

# -------------------------
# Run evaluation
# -------------------------
metrics = evaluate_grid(acts, coarse_k_values, n_trials, size, epochs, temp, n_value, batch_size, device)

# -------------------------
# Summarize results
# -------------------------
# with open("selected_k_values_2.txt", "w") as f:
#     for act in acts:
#         print(act)
#         best_k, trans_k, recs = summarize_for_act(metrics, act, coarse_k_values)
#         k_low = coarse_k_values[np.where(recs < 0.2)[0][0]] if np.any(recs < 0.2) else coarse_k_values[0]
#         f.write(f"{act}: best_k={best_k:.3f}, trans_k={trans_k:.3f}, low_k={k_low:.3f}\n")

with open("K_values_iteration_4.txt", "w") as f:
    for act in acts:
        best_k, trans_k, k_low, recs = summarize_for_act(metrics, act, coarse_k_values)
        
        # Special check for traditional Hopfield
        f.write(f"--- {act.upper()} ---\n")
        if act == 'step':
            try:
                traditional_acc = metrics[('step', 0.0)]['recovery_fraction']
                #f.write(f"--- {act.upper()} ---\n")
                f.write(f"Traditional Hopfield (k=0) Accuracy: {traditional_acc:.3f}\n")
            except:
                f.write("Traditional Hopfield (k=0): Not tested in this range\n")
            
        f.write(f"Best K (Peak): {best_k:.3f} (Acc: {recs.max():.3f})\n")
        f.write(f"Transition K (The Jump): {trans_k:.3f}\n")
        f.write(f"Low K (Breaking Point): {k_low:.3f}\n\n")



results_table = {"k_value": coarse_k_values}
for act in acts:
    results_table[act] = [metrics[(act, k)]['recovery_fraction'] for k in coarse_k_values]

df = pd.DataFrame(results_table)
df.to_csv("full_recovery_curves_P2000.csv", index=False)

# Optional: Save the whole metrics object just in case
with open("metrics_dump_P2000.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("\nSaved summary to Selected_k_values_summary.txt")
print("Saved full raw data to full_recovery_curves_P2000.csv")




# python /Users/kavinfidel/Documents/Fidel/IMSC/Project/Recovery_Capacity_Hopfield_Networks/batched/K_coarse_sweep.py