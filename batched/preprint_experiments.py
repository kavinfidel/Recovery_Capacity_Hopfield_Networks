import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError:
    torch = None

Hopfield_Network_gpu = None


DEFAULT_ACTIVATIONS = ("relu", "elu")
EXTRA_ACTIVATIONS = ("dam",)
ALL_ACTIVATIONS = DEFAULT_ACTIVATIONS + EXTRA_ACTIVATIONS
VALID_ACTIVATIONS = set(ALL_ACTIVATIONS)
PROGRESS_INTERVAL_SECONDS = 30.0


def log(message):
    print(message, flush=True)


def write_run_config(args, output_path):
    config = {
        key: value
        for key, value in vars(args).items()
        if key != "func" and isinstance(value, (str, int, float, bool, list, dict, type(None)))
    }
    output_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def require_torch():
    global torch, Hopfield_Network_gpu
    if torch is None:
        import torch as torch_module
        torch = torch_module
    if Hopfield_Network_gpu is None:
        from hopfield_batched import Hopfield_Network_gpu as model_class
        Hopfield_Network_gpu = model_class
    return torch, Hopfield_Network_gpu


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
    if torch is not None:
        torch.manual_seed(seed)


def generate_patterns(num_patterns, size):
    return np.random.choice([-1, 1], size=(num_patterns, size))


def corrupted_target_batch(model, num_patterns, batch_size, corruption_rate):
    idx = torch.randint(0, num_patterns, (batch_size,), device=model.device)
    x = model.patterns[idx].clone()
    noise_mask = torch.rand(x.shape, device=model.device) < corruption_rate
    x[noise_mask] *= -1.0
    return idx, x


def random_binary_initial_batch(model, batch_size):
    return (
        torch.randint(0, 2, (batch_size, model.size), device=model.device, dtype=torch.float32)
        * 2.0
        - 1.0
    )


def recovery_fraction(
    patterns,
    activation,
    k_value,
    n_trials,
    epochs,
    temp,
    batch_size,
    device,
    corruption_rate,
    dam_power,
    label=None,
):
    _, model_class = require_torch()
    model = model_class(
        size=patterns.shape[1],
        patterns=patterns,
        k=k_value,
        activation=activation,
        device=device,
        dam_power=dam_power,
    )

    successes = 0
    max_overlaps = []
    n_done = 0
    start_time = time.time()
    last_log_time = start_time
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
        now = time.time()
        if label and (now - last_log_time >= PROGRESS_INTERVAL_SECONDS or n_done == n_trials):
            log(
                f"    {label}: {n_done}/{n_trials} trials, "
                f"running recovery={successes / n_done:.3f}, elapsed={now - start_time:.1f}s"
            )
            last_log_time = now

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
    dam_power,
    label=None,
):
    _, model_class = require_torch()
    model = model_class(
        size=patterns.shape[1],
        patterns=patterns,
        k=k_value,
        activation=activation,
        device=device,
        dam_power=dam_power,
    )

    successes = 0
    max_overlaps = []
    n_done = 0
    start_time = time.time()
    last_log_time = start_time
    while n_done < n_samples:
        bs = min(batch_size, n_samples - n_done)
        initial_states = random_binary_initial_batch(model, bs)
        recovered, max_overlap = model._mc_simulation_until_convergence_from_state(
            initial_states,
            max_sweeps,
            temp,
            stable_sweeps_required=stable_sweeps_required,
        )
        successes += int(recovered.sum().item())
        max_overlaps.extend((max_overlap / patterns.shape[1]).detach().cpu().tolist())
        n_done += bs
        now = time.time()
        if label and (now - last_log_time >= PROGRESS_INTERVAL_SECONDS or n_done == n_samples):
            log(
                f"    {label}: {n_done}/{n_samples} samples, "
                f"running recovery={successes / n_done:.3f}, elapsed={now - start_time:.1f}s"
            )
            last_log_time = now

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


def interpolate_upper_threshold_crossing(x_values, y_values, threshold):
    order = np.argsort(x_values)
    xs = np.asarray(x_values, dtype=float)[order]
    ys = np.asarray(y_values, dtype=float)[order]

    passing = np.where(ys >= threshold)[0]
    if len(passing) == 0:
        return np.nan

    last = int(passing[-1])
    if last == len(xs) - 1:
        return float(xs[last])

    x0, x1 = xs[last], xs[last + 1]
    y0, y1 = ys[last], ys[last + 1]
    if y1 == y0:
        return float(x0)
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def run_k_sweep(args):
    require_torch()
    device = choose_device(args.device)
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_values = np.linspace(args.k_min, args.k_max, args.k_steps)
    patterns = generate_patterns(args.patterns, args.size)
    rows = []
    overlap_rows = []

    log(
        f"Running k-sweep on {device}: N={args.size}, P={args.patterns}, "
        f"activations={args.activations}, trials={args.n_trials}, epochs={args.epochs}"
    )

    for activation in args.activations:
        log(f"\nActivation: {activation}")
        for k_value in k_values:
            label = f"{activation} k={k_value:.4f}"
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
                args.dam_power,
                label=label,
            )
            rows.append(
                {
                    "N": args.size,
                    "P": args.patterns,
                    "activation": activation,
                    "k": float(k_value),
                    "dam_power": args.dam_power if activation == "dam" else np.nan,
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
            log(f"  done {label}: recovery={fraction:.3f}")

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
    log(f"\nSaved {curve_path}")
    log(f"Saved {summary_path}")
    log(f"Saved {overlap_path}")


def run_capacity_curve(args):
    require_torch()
    device = choose_device(args.device)
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"Running capacity curve on {device}: N={args.size}, "
        f"activations={args.activations}, trials={args.n_trials}, epochs={args.epochs}"
    )

    rows = []
    for activation in args.activations:
        k_value = args.k_by_activation.get(activation, args.k)
        log(f"\nActivation: {activation}, k={k_value:.4f}")
        for p_value in args.pattern_values:
            label = f"{activation} P={p_value} k={k_value:.4f}"
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
                args.dam_power,
                label=label,
            )
            rows.append(
                {
                    "N": args.size,
                    "P": p_value,
                    "activation": activation,
                    "k": k_value,
                    "dam_power": args.dam_power if activation == "dam" else np.nan,
                    "recovery_fraction": fraction,
                    "standard_error": binomial_se(fraction, args.n_trials),
                    "n_trials": args.n_trials,
                    "epochs": args.epochs,
                    "corruption_rate": args.corruption_rate,
                    "seed": args.seed,
                    "protocol": "corrupted_target_retrieval",
                }
            )
            log(f"  done {label}: recovery={fraction:.3f}")

    df = pd.DataFrame(rows)
    curve_path = out_dir / f"capacity_curve_N{args.size}.csv"
    df.to_csv(curve_path, index=False)
    plot_capacity_curve(df, out_dir / f"capacity_curve_N{args.size}.png", args.capacity_threshold)
    log(f"\nSaved {curve_path}")


def run_random_k_half(args):
    require_torch()
    device = choose_device(args.device)
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"Running random-k-half on {device}: N values={args.n_values}, "
        f"K values={args.memory_values}, activations={args.activations}, samples={args.n_samples}, "
        f"max_sweeps={args.max_sweeps}"
    )
    write_run_config(args, out_dir / "run_config.json")

    rows = []
    for activation in args.activations:
        k_value = args.k_by_activation.get(activation, args.k)
        log(f"\nActivation: {activation}, activation threshold k={k_value:.4f}")
        for n_value in args.n_values:
            seen_above_threshold = False
            for num_memories in sorted(args.memory_values):
                label = f"{activation} N={n_value} K={num_memories}"
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
                    args.dam_power,
                    label=label,
                )
                rows.append(
                    {
                        "N": n_value,
                        "K": num_memories,
                        "activation": activation,
                        "activation_threshold_k": k_value,
                        "dam_power": args.dam_power if activation == "dam" else np.nan,
                        "recovery_fraction": fraction,
                        "standard_error": binomial_se(fraction, args.n_samples),
                        "n_samples": args.n_samples,
                        "max_sweeps": args.max_sweeps,
                        "stable_sweeps": args.stable_sweeps,
                        "seed": args.seed,
                        "protocol": "random_initial_any_memory",
                        "initial_condition": "random_binary_configuration",
                    }
                )
                log(f"  done {label}: recovery={fraction:.3f}")
                seen_above_threshold |= fraction >= args.half_threshold
                if args.skip_after_collapse and seen_above_threshold and fraction == 0.0:
                    log(f"  recovery collapsed to 0 at {label}; skipping larger K for N={n_value}")
                    break

    df = pd.DataFrame(rows)
    grid_path = out_dir / "random_initial_grid.csv"
    df.to_csv(grid_path, index=False)

    summary_rows = []
    for (activation, n_value), group in df.groupby(["activation", "N"]):
        ordered = group.sort_values("K")
        recoveries = ordered["recovery_fraction"].to_numpy()
        memory_counts = ordered["K"].to_numpy()
        k_half = interpolate_threshold_crossing(
            memory_counts,
            recoveries,
            args.half_threshold,
        )
        k_half_upper = interpolate_upper_threshold_crossing(
            memory_counts,
            recoveries,
            args.half_threshold,
        )
        is_lower_bound = bool(len(recoveries) > 0 and recoveries[-1] >= args.half_threshold)
        summary_rows.append(
            {
                "N": n_value,
                "activation": activation,
                "activation_threshold_k": float(ordered["activation_threshold_k"].iloc[0]),
                "dam_power": float(ordered["dam_power"].iloc[0]) if "dam_power" in ordered else np.nan,
                "K_half_first_crossing": k_half,
                "K_half_upper_capacity": k_half_upper,
                "K_half_upper_is_lower_bound": is_lower_bound,
                "threshold": args.half_threshold,
                "max_observed_recovery_fraction": float(ordered["recovery_fraction"].max()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "random_initial_K_half.csv"
    summary.to_csv(summary_path, index=False)

    plot_random_initial_grid(df, out_dir / "random_initial_recovery_grid.png", args.half_threshold)
    plot_k_half(summary, out_dir / "K_half_vs_N.png")
    fit_summary = plot_k_half_loglog(summary, out_dir / "K_half_vs_N_loglog.png")
    fit_path = out_dir / "K_half_power_law_fit.csv"
    fit_summary.to_csv(fit_path, index=False)
    log(f"\nSaved {grid_path}")
    log(f"Saved {summary_path}")
    log(f"Saved {fit_path}")


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


def double_factorial(m):
    result = 1.0
    while m > 1:
        result *= m
        m -= 2
    return result


def krotov_theory_k_half(n_values, power):
    """Krotov & Hopfield (2016) Eq. (6): K_max ~ N^(n-1) / (2*(2n-3)!! * ln N)."""
    n_values = np.asarray(n_values, dtype=float)
    omega = double_factorial(2 * power - 3)
    return n_values ** (power - 1) / (2.0 * omega * np.log(n_values))


def _add_krotov_theory_curve(ax, summary, loglog=False):
    if "dam" not in set(summary["activation"]):
        return
    dam_group = summary[summary["activation"] == "dam"]
    power = dam_group["dam_power"].dropna()
    if power.empty:
        return
    power = float(power.iloc[0])
    n_min = float(summary["N"].min())
    n_max = float(summary["N"].max())
    n_fit = np.linspace(n_min, n_max, 200) if not loglog else np.logspace(
        np.log10(n_min), np.log10(n_max), 200
    )
    theory = krotov_theory_k_half(n_fit, power)
    ax.plot(
        n_fit,
        theory,
        linestyle="--",
        color="0.35",
        linewidth=1.5,
        label=f"Krotov theory n={power:.0f} (Eq. 6)",
    )


def plot_k_half(summary, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for activation, group in summary.groupby("activation"):
        group = group.sort_values("N")
        ax.plot(
            group["N"],
            group["K_half_upper_capacity"],
            marker="o",
            linewidth=2,
            label=activation.upper(),
        )
    _add_krotov_theory_curve(ax, summary, loglog=False)
    ax.set_xlabel("N")
    ax.set_ylabel("K_half capacity")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_k_half_loglog(summary, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fit_rows = []

    for activation, group in summary.groupby("activation"):
        group = group.sort_values("N")
        valid = group["K_half_upper_capacity"].notna() & (group["K_half_upper_capacity"] > 0)
        finite_group = group[valid]
        if finite_group.empty:
            continue

        ax.scatter(
            finite_group["N"],
            finite_group["K_half_upper_capacity"],
            marker="o",
            label=f"{activation.upper()} data",
        )

        fit_group = finite_group[~finite_group["K_half_upper_is_lower_bound"]]
        fit_note = "uncensored"
        if len(fit_group) < 2:
            fit_group = finite_group
            fit_note = "all_finite_including_lower_bounds"

        if len(fit_group) >= 2:
            log_n = np.log10(fit_group["N"].to_numpy(dtype=float))
            log_k = np.log10(fit_group["K_half_upper_capacity"].to_numpy(dtype=float))
            slope, intercept = np.polyfit(log_n, log_k, 1)
            predicted = slope * log_n + intercept
            ss_res = float(np.sum((log_k - predicted) ** 2))
            ss_tot = float(np.sum((log_k - np.mean(log_k)) ** 2))
            r_squared = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

            x_fit = np.linspace(fit_group["N"].min(), fit_group["N"].max(), 100)
            y_fit = 10 ** intercept * x_fit ** slope
            ax.plot(x_fit, y_fit, linewidth=2, label=f"{activation.upper()} fit slope={slope:.2f}")
        else:
            slope = np.nan
            intercept = np.nan
            r_squared = np.nan

        fit_rows.append(
            {
                "activation": activation,
                "power_law_slope": slope,
                "power_law_intercept_log10": intercept,
                "r_squared_loglog": r_squared,
                "n_fit_points": len(fit_group),
                "fit_note": fit_note,
            }
        )

    _add_krotov_theory_curve(ax, summary, loglog=True)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Upper K_half capacity")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return pd.DataFrame(fit_rows)


def parse_k_by_activation(raw_values):
    parsed = {}
    for item in raw_values or []:
        name, value = item.split("=", 1)
        parsed[name.strip()] = float(value)
    return parsed


def parse_activation_list(raw_values):
    if raw_values is None:
        return None

    activations = []
    for value in raw_values:
        activations.extend(part.strip() for part in value.split(",") if part.strip())

    invalid = sorted(set(activations) - VALID_ACTIVATIONS)
    if invalid:
        raise ValueError(f"Unknown activation(s): {invalid}. Valid choices: {sorted(VALID_ACTIVATIONS)}")
    return activations


def resolve_activations(args):
    parsed_activations = parse_activation_list(getattr(args, "activations", None))
    if parsed_activations:
        args.activations = parsed_activations
        return

    if getattr(args, "k_by_activation", None):
        args.activations = list(args.k_by_activation.keys())
        return

    args.activations = list(DEFAULT_ACTIVATIONS)


def merge_summary_tables(paths):
    summary = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    if "K_half_upper_is_lower_bound" not in summary:
        summary["K_half_upper_is_lower_bound"] = False

    summary["_is_lower_bound_sort"] = summary["K_half_upper_is_lower_bound"].astype(bool)
    summary["_capacity_sort"] = summary["K_half_upper_capacity"].fillna(-np.inf)
    summary = summary.sort_values(
        ["activation", "N", "_is_lower_bound_sort", "_capacity_sort"],
        ascending=[True, True, True, False],
    )
    summary = summary.drop_duplicates(["activation", "N"], keep="first")
    return summary.drop(columns=["_is_lower_bound_sort", "_capacity_sort"]).sort_values(["activation", "N"])


def run_fit_k_half(args):
    summary = merge_summary_tables(args.summary_csv)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.summary_csv[0]).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    required = {"N", "activation", "K_half_upper_capacity", "K_half_upper_is_lower_bound"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Missing required column(s) in {args.summary_csv}: {sorted(missing)}")

    plot_k_half(summary, out_dir / "K_half_vs_N.png")
    fit_summary = plot_k_half_loglog(summary, out_dir / "K_half_vs_N_loglog.png")
    merged_path = out_dir / "merged_random_initial_K_half.csv"
    fit_path = out_dir / "K_half_power_law_fit.csv"
    summary.to_csv(merged_path, index=False)
    fit_summary.to_csv(fit_path, index=False)

    log(f"Saved {out_dir / 'K_half_vs_N.png'}")
    log(f"Saved {out_dir / 'K_half_vs_N_loglog.png'}")
    log(f"Saved {merged_path}")
    log(f"Saved {fit_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Preprint experiment pipeline for gated Hopfield recovery.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser):
        subparser.add_argument("--activations", nargs="+", default=None)
        subparser.add_argument("--size", type=int, default=200)
        subparser.add_argument("--n-trials", type=int, default=500)
        subparser.add_argument("--epochs", type=int, default=3000)
        subparser.add_argument("--temp", type=float, default=0.0)
        subparser.add_argument("--batch-size", type=int, default=128)
        subparser.add_argument("--device", default=None)
        subparser.add_argument("--corruption-rate", type=float, default=0.10)
        subparser.add_argument("--capacity-threshold", type=float, default=0.95)
        subparser.add_argument("--dam-power", type=float, default=3.0)
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
    random_half.add_argument("--activations", nargs="+", default=None)
    random_half.add_argument("--n-values", type=int, nargs="+", default=[50, 75, 100, 125, 150, 175, 200])
    random_half.add_argument(
        "--memory-values",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1500, ],
    )
    random_half.add_argument("--n-samples", type=int, default=1000)
    random_half.add_argument("--max-sweeps", type=int, default=100)
    random_half.add_argument("--stable-sweeps", type=int, default=1)
    random_half.add_argument("--temp", type=float, default=0.0)
    random_half.add_argument("--batch-size", type=int, default=256)
    random_half.add_argument("--device", default=None)
    random_half.add_argument("--half-threshold", type=float, default=0.5)
    random_half.add_argument("--dam-power", type=float, default=3.0)
    random_half.add_argument("--seed", type=int, default=12345)
    random_half.add_argument("--out-dir", default="preprint_results/random_k_half")
    random_half.add_argument(
        "--skip-after-collapse",
        action="store_true",
        help="Stop the K sweep for an (activation, N) once recovery has crossed the "
        "half threshold and then fallen to exactly 0 (saves the most expensive large-K points).",
    )
    random_half.add_argument("--k", type=float, default=0.2)
    random_half.add_argument("--k-by-activation", nargs="*", default=[])
    random_half.set_defaults(func=run_random_k_half)

    fit_k_half = subparsers.add_parser(
        "fit-k-half",
        help="Replot and fit K_half scaling from a saved random_initial_K_half.csv file.",
    )
    fit_k_half.add_argument("--summary-csv", nargs="+", required=True)
    fit_k_half.add_argument("--out-dir", default=None)
    fit_k_half.set_defaults(func=run_fit_k_half)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.k_by_activation = parse_k_by_activation(getattr(args, "k_by_activation", []))
    if hasattr(args, "activations"):
        resolve_activations(args)
    args.func(args)


if __name__ == "__main__":
    main()
