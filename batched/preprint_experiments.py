import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hopfield_batched import Hopfield_Network_gpu


DEFAULT_ACTIVATIONS = ("step", "relu", "elu")


def choose_device(requested_device):
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_patterns(num_patterns, size):
    return np.random.choice([-1, 1], size=(num_patterns, size))


def corrupted_target_batch(model, num_patterns, batch_size, corruption_rate):
    idx = torch.randint(0, num_patterns, (batch_size,), device=model.device)
    x = model.patterns[idx].clone()
    noise_mask = torch.rand(x.shape, device=model.device) < corruption_rate
    x[noise_mask] *= -1.0
    return idx, x


def recovery_fraction(patterns, activation, k_value, n_trials, epochs, temp, batch_size, device, corruption_rate):
    model = Hopfield_Network_gpu(
        size=patterns.shape[1],
        patterns=patterns,
        k=k_value,
        activation=activation,
        device=device,
    )

    successes = 0
    max_overlaps = []
    n_done = 0
    while n_done < n_trials:
        bs = min(batch_size, n_trials - n_done)
        target_idx, corrupted_x = corrupted_target_batch(model, patterns.shape[0], bs, corruption_rate)
        recovered, max_overlap = model.mc_simulation_recovery_corrupted_batch(
            corrupted_x,
            epochs,
            temp,
            target_indices=target_idx,
        )
        successes += int(recovered.sum().item())
        max_overlaps.extend((max_overlap / patterns.shape[1]).detach().cpu().tolist())
        n_done += bs

    return successes / n_trials, np.array(max_overlaps)


def random_initial_recovery_fraction(
    patterns,
    activation,
    k_value,
    n_samples,
    max_sweeps,
    temp,
    batch_size,
    device,
    stable_sweeps_required,
):
    model = Hopfield_Network_gpu(
        size=patterns.shape[1],
        patterns=patterns,
        k=k_value,
        activation=activation,
        device=device,
    )

    successes = 0
    max_overlaps = []
    n_done = 0
    while n_done < n_samples:
        bs = min(batch_size, n_samples - n_done)
        recovered, max_overlap = model.mc_simulation_recovery_batch_until_convergence(
            bs,
            max_sweeps,
            temp,
            stable_sweeps_required=stable_sweeps_required,
        )
        successes += int(recovered.sum().item())
        max_overlaps.extend((max_overlap / patterns.shape[1]).detach().cpu().tolist())
        n_done += bs

    return successes / n_samples, np.array(max_overlaps)


def binomial_se(fraction, n_trials):
    return float(np.sqrt(max(fraction * (1.0 - fraction), 0.0) / n_trials))


def crossing_k(k_values, recoveries, threshold):
    passing = np.where(recoveries >= threshold)[0]
    if len(passing) == 0:
        return np.nan
    return float(k_values[passing[0]])


def interpolate_threshold_crossing(x_values, y_values, threshold):
    order = np.argsort(x_values)
    xs = np.asarray(x_values, dtype=float)[order]
    ys = np.asarray(y_values, dtype=float)[order]

    crossing_indices = np.where(ys >= threshold)[0]
    if len(crossing_indices) == 0:
        return np.nan

    idx = int(crossing_indices[0])
    if idx == 0:
        return float(xs[0])

    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    if y1 == y0:
        return float(x1)
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def run_k_sweep(args):
    device = choose_device(args.device)
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_values = np.linspace(args.k_min, args.k_max, args.k_steps)
    patterns = generate_patterns(args.patterns, args.size)
    rows = []
    overlap_rows = []

    for activation in args.activations:
        print(f"\nActivation: {activation}")
        for k_value in k_values:
            fraction, overlaps = recovery_fraction(
                patterns,
                activation,
                float(k_value),
                args.n_trials,
                args.epochs,
                args.temp,
                args.batch_size,
                device,
                args.corruption_rate,
            )
            rows.append(
                {
                    "N": args.size,
                    "P": args.patterns,
                    "activation": activation,
                    "k": float(k_value),
                    "recovery_fraction": fraction,
                    "standard_error": binomial_se(fraction, args.n_trials),
                    "n_trials": args.n_trials,
                    "epochs": args.epochs,
                    "corruption_rate": args.corruption_rate,
                    "seed": args.seed,
                    "protocol": "corrupted_target_retrieval",
                }
            )
            for overlap in overlaps:
                overlap_rows.append(
                    {
                        "activation": activation,
                        "k": float(k_value),
                        "normalized_max_overlap": float(overlap),
                    }
                )
            print(f"  k={k_value:.4f} recovery={fraction:.3f}")

    df = pd.DataFrame(rows)
    curve_path = out_dir / f"k_sweep_N{args.size}_P{args.patterns}.csv"
    df.to_csv(curve_path, index=False)

    overlap_path = out_dir / f"overlaps_N{args.size}_P{args.patterns}.csv"
    pd.DataFrame(overlap_rows).to_csv(overlap_path, index=False)

    summary_rows = []
    for activation, group in df.groupby("activation"):
        ordered = group.sort_values("k")
        recs = ordered["recovery_fraction"].to_numpy()
        ks = ordered["k"].to_numpy()
        best_idx = int(np.argmax(recs))
        summary_rows.append(
            {
                "activation": activation,
                "best_k": float(ks[best_idx]),
                "best_recovery_fraction": float(recs[best_idx]),
                "first_k_above_threshold": crossing_k(ks, recs, args.capacity_threshold),
                "threshold": args.capacity_threshold,
            }
        )
    summary_path = out_dir / f"k_sweep_summary_N{args.size}_P{args.patterns}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    plot_k_sweep(df, out_dir / f"k_sweep_N{args.size}_P{args.patterns}.png", args.capacity_threshold)
    print(f"\nSaved {curve_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {overlap_path}")


def run_capacity_curve(args):
    device = choose_device(args.device)
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for activation in args.activations:
        k_value = args.k_by_activation.get(activation, args.k)
        print(f"\nActivation: {activation}, k={k_value:.4f}")
        for p_value in args.pattern_values:
            patterns = generate_patterns(p_value, args.size)
            fraction, _ = recovery_fraction(
                patterns,
                activation,
                k_value,
                args.n_trials,
                args.epochs,
                args.temp,
                args.batch_size,
                device,
                args.corruption_rate,
            )
            rows.append(
                {
                    "N": args.size,
                    "P": p_value,
                    "activation": activation,
                    "k": k_value,
                    "recovery_fraction": fraction,
                    "standard_error": binomial_se(fraction, args.n_trials),
                    "n_trials": args.n_trials,
                    "epochs": args.epochs,
                    "corruption_rate": args.corruption_rate,
                    "seed": args.seed,
                    "protocol": "corrupted_target_retrieval",
                }
            )
            print(f"  P={p_value} recovery={fraction:.3f}")

    df = pd.DataFrame(rows)
    curve_path = out_dir / f"capacity_curve_N{args.size}.csv"
    df.to_csv(curve_path, index=False)
    plot_capacity_curve(df, out_dir / f"capacity_curve_N{args.size}.png", args.capacity_threshold)
    print(f"\nSaved {curve_path}")


def run_random_k_half(args):
    device = choose_device(args.device)
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for activation in args.activations:
        k_value = args.k_by_activation.get(activation, args.k)
        print(f"\nActivation: {activation}, activation threshold k={k_value:.4f}")
        for n_value in args.n_values:
            for num_memories in args.memory_values:
                patterns = generate_patterns(num_memories, n_value)
                fraction, _ = random_initial_recovery_fraction(
                    patterns,
                    activation,
                    k_value,
                    args.n_samples,
                    args.max_sweeps,
                    args.temp,
                    args.batch_size,
                    device,
                    args.stable_sweeps,
                )
                rows.append(
                    {
                        "N": n_value,
                        "K": num_memories,
                        "activation": activation,
                        "activation_threshold_k": k_value,
                        "recovery_fraction": fraction,
                        "standard_error": binomial_se(fraction, args.n_samples),
                        "n_samples": args.n_samples,
                        "max_sweeps": args.max_sweeps,
                        "stable_sweeps": args.stable_sweeps,
                        "seed": args.seed,
                        "protocol": "random_initial_any_memory",
                    }
                )
                print(f"  N={n_value} K={num_memories} recovery={fraction:.3f}")

    df = pd.DataFrame(rows)
    grid_path = out_dir / "random_initial_grid.csv"
    df.to_csv(grid_path, index=False)

    summary_rows = []
    for (activation, n_value), group in df.groupby(["activation", "N"]):
        ordered = group.sort_values("K")
        k_half = interpolate_threshold_crossing(
            ordered["K"].to_numpy(),
            ordered["recovery_fraction"].to_numpy(),
            args.half_threshold,
        )
        summary_rows.append(
            {
                "N": n_value,
                "activation": activation,
                "activation_threshold_k": float(ordered["activation_threshold_k"].iloc[0]),
                "K_half": k_half,
                "threshold": args.half_threshold,
                "max_observed_recovery_fraction": float(ordered["recovery_fraction"].max()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "random_initial_K_half.csv"
    summary.to_csv(summary_path, index=False)

    plot_random_initial_grid(df, out_dir / "random_initial_recovery_grid.png", args.half_threshold)
    plot_k_half(summary, out_dir / "K_half_vs_N.png")
    print(f"\nSaved {grid_path}")
    print(f"Saved {summary_path}")


def plot_k_sweep(df, output_path, threshold):
    fig, ax = plt.subplots(figsize=(8, 5))
    for activation, group in df.groupby("activation"):
        group = group.sort_values("k")
        ax.errorbar(
            group["k"],
            group["recovery_fraction"],
            yerr=group["standard_error"],
            marker="o",
            linewidth=2,
            capsize=2,
            label=activation.upper(),
        )
    ax.axhline(threshold, color="0.35", linestyle=":", linewidth=1.5, label=f"{threshold:.0%} threshold")
    ax.set_xlabel("Threshold k")
    ax.set_ylabel("Target recovery fraction")
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_capacity_curve(df, output_path, threshold):
    fig, ax = plt.subplots(figsize=(8, 5))
    for activation, group in df.groupby("activation"):
        group = group.sort_values("P")
        ax.errorbar(
            group["P"],
            group["recovery_fraction"],
            yerr=group["standard_error"],
            marker="o",
            linewidth=2,
            capsize=2,
            label=f"{activation.upper()} (k={group['k'].iloc[0]:.3f})",
        )
    ax.axhline(threshold, color="0.35", linestyle=":", linewidth=1.5, label=f"{threshold:.0%} threshold")
    ax.set_xlabel("Number of stored memories P")
    ax.set_ylabel("Target recovery fraction")
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_random_initial_grid(df, output_path, threshold):
    fig, axes = plt.subplots(
        len(df["activation"].unique()),
        1,
        figsize=(8, 4 * len(df["activation"].unique())),
        squeeze=False,
    )

    for ax, (activation, act_group) in zip(axes[:, 0], df.groupby("activation")):
        for n_value, group in act_group.groupby("N"):
            group = group.sort_values("K")
            ax.errorbar(
                group["K"],
                group["recovery_fraction"],
                yerr=group["standard_error"],
                marker="o",
                linewidth=2,
                capsize=2,
                label=f"N={n_value}",
            )
        ax.axhline(threshold, color="0.35", linestyle=":", linewidth=1.5)
        ax.set_title(f"{activation.upper()}")
        ax.set_xlabel("Number of stored memories K")
        ax.set_ylabel("Random-initial recovery fraction")
        ax.set_ylim(-0.04, 1.04)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_k_half(summary, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for activation, group in summary.groupby("activation"):
        group = group.sort_values("N")
        ax.plot(group["N"], group["K_half"], marker="o", linewidth=2, label=activation.upper())
    ax.set_xlabel("N")
    ax.set_ylabel("K_half")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_k_by_activation(raw_values):
    parsed = {}
    for item in raw_values or []:
        name, value = item.split("=", 1)
        parsed[name.strip()] = float(value)
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description="Preprint experiment pipeline for gated Hopfield recovery.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser):
        subparser.add_argument("--activations", nargs="+", default=list(DEFAULT_ACTIVATIONS))
        subparser.add_argument("--size", type=int, default=200)
        subparser.add_argument("--n-trials", type=int, default=500)
        subparser.add_argument("--epochs", type=int, default=1000)
        subparser.add_argument("--temp", type=float, default=0.0)
        subparser.add_argument("--batch-size", type=int, default=128)
        subparser.add_argument("--device", default=None)
        subparser.add_argument("--corruption-rate", type=float, default=0.10)
        subparser.add_argument("--capacity-threshold", type=float, default=0.95)
        subparser.add_argument("--seed", type=int, default=12345)
        subparser.add_argument("--out-dir", default="preprint_results")

    sweep = subparsers.add_parser("k-sweep", help="Sweep k for fixed N and P.")
    add_common(sweep)
    sweep.add_argument("--patterns", type=int, default=2000)
    sweep.add_argument("--k-min", type=float, default=0.0)
    sweep.add_argument("--k-max", type=float, default=0.6)
    sweep.add_argument("--k-steps", type=int, default=31)
    sweep.set_defaults(func=run_k_sweep)

    capacity = subparsers.add_parser("capacity-curve", help="Sweep number of memories P for fixed N and k.")
    add_common(capacity)
    capacity.add_argument("--pattern-values", type=int, nargs="+", default=[100, 200, 400, 800, 1200, 1600, 2000])
    capacity.add_argument("--k", type=float, default=0.2)
    capacity.add_argument("--k-by-activation", nargs="*", default=[])
    capacity.set_defaults(func=run_capacity_curve)

    random_half = subparsers.add_parser(
        "random-k-half",
        help="Paper-style K_half experiment from random initial states.",
    )
    random_half.add_argument("--activations", nargs="+", default=list(DEFAULT_ACTIVATIONS))
    random_half.add_argument("--n-values", type=int, nargs="+", default=[50, 75, 100, 125, 150, 175, 200])
    random_half.add_argument(
        "--memory-values",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1500],
    )
    random_half.add_argument("--n-samples", type=int, default=1000)
    random_half.add_argument("--max-sweeps", type=int, default=100)
    random_half.add_argument("--stable-sweeps", type=int, default=1)
    random_half.add_argument("--temp", type=float, default=0.0)
    random_half.add_argument("--batch-size", type=int, default=256)
    random_half.add_argument("--device", default=None)
    random_half.add_argument("--half-threshold", type=float, default=0.5)
    random_half.add_argument("--seed", type=int, default=12345)
    random_half.add_argument("--out-dir", default="preprint_results/random_k_half")
    random_half.add_argument("--k", type=float, default=0.2)
    random_half.add_argument("--k-by-activation", nargs="*", default=[])
    random_half.set_defaults(func=run_random_k_half)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.k_by_activation = parse_k_by_activation(getattr(args, "k_by_activation", []))
    args.func(args)


if __name__ == "__main__":
    main()
