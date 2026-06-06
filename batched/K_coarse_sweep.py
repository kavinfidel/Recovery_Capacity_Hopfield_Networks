import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import argparse

from hopfield_batched import Hopfield_Network_gpu

def generate_patterns(n, size):
    return np.random.choice([-1, 1], size=(n, size))

def evaluate_grid(acts, k_values, n_trials, size, epochs, temp, n_value, batch_size=128, device='cuda', corruption_rate=0.10):
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
                idx = torch.randint(0, n_value, (bs,), device=model.device)
                target_patterns = model.patterns[idx]
                x_corrupted = target_patterns.clone()
                noise_mask = torch.rand(x_corrupted.shape, device=model.device) < corruption_rate
                x_corrupted[noise_mask] *= -1.0

                perfect_recovery, max_ov = model.mc_simulation_recovery_corrupted_batch(
                    x_corrupted,
                    epochs,
                    temp,
                    target_indices=idx,
                )
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
    passing_indices = np.where(recs >= 0.5)[0]
    if len(passing_indices) > 0:
        k_low = k_values[passing_indices[0]]
    else:
        k_low = k_values[-1]

    return best_k, trans_k, k_low, recs


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep k values for batched Hopfield recovery.")
    parser.add_argument("--acts", nargs="+", default=["elu"])
    parser.add_argument("--k-min", type=float, default=0.15)
    parser.add_argument("--k-max", type=float, default=0.5)
    parser.add_argument("--k-steps", type=int, default=51)
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--patterns", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--corruption-rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def choose_device(requested_device):
    if requested_device is not None:
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    acts = args.acts
    coarse_k_values = np.linspace(args.k_min, args.k_max, args.k_steps)
    device = choose_device(args.device)

    metrics = evaluate_grid(
        acts,
        coarse_k_values,
        args.n_trials,
        args.size,
        args.epochs,
        args.temp,
        args.patterns,
        args.batch_size,
        device,
        args.corruption_rate,
    )

    summary_path = f"K_values_N{args.size}_P{args.patterns}.txt"
    curves_path = f"full_recovery_curves_N{args.size}_P{args.patterns}.csv"
    metrics_path = f"metrics_dump_N{args.size}_P{args.patterns}.pkl"

    with open(summary_path, "w") as f:
        f.write(f"N={args.size}, patterns={args.patterns}, corruption={args.corruption_rate:.0%}, seed={args.seed}\n")
        f.write("Protocol: corrupted-memory retrieval; success requires exact recovery of the target memory.\n\n")
        for act in acts:
            best_k, trans_k, k_low, recs = summarize_for_act(metrics, act, coarse_k_values)

            f.write(f"--- {act.upper()} ---\n")
            if act == 'step':
                try:
                    traditional_acc = metrics[('step', 0.0)]['recovery_fraction']
                    f.write(f"Traditional Hopfield (k=0) Accuracy: {traditional_acc:.3f}\n")
                except KeyError:
                    f.write("Traditional Hopfield (k=0): Not tested in this range\n")

            f.write(f"Best K (Peak): {best_k:.3f} (Acc: {recs.max():.3f})\n")
            f.write(f"Transition K (The Jump): {trans_k:.3f}\n")
            f.write(f"Low K (Breaking Point): {k_low:.3f}\n\n")

    results_table = {"k_value": coarse_k_values}
    for act in acts:
        results_table[act] = [metrics[(act, k)]['recovery_fraction'] for k in coarse_k_values]

    df = pd.DataFrame(results_table)
    df.to_csv(curves_path, index=False)

    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)

    print(f"\nSaved summary to {summary_path}")
    print(f"Saved full raw data to {curves_path}")
    print(f"Saved metrics dump to {metrics_path}")


if __name__ == "__main__":
    main()


# python /Users/kavinfidel/Documents/Fidel/IMSC/Project/Recovery_Capacity_Hopfield_Networks/batched/K_coarse_sweep.py --size 200 --patterns 2000
