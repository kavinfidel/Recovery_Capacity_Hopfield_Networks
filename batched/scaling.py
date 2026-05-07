import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import linregress
import time
import pandas as pd

from hopfield_batched import Hopfield_Network_gpu
# K - number of patterns; N- neurons
def generate_patterns(num_patterns, N):
    return np.random.choice([-1, 1], size=(num_patterns, N))

def evaluate_capacity(N, K, k_val, act, n_trials, epochs, temp, base_batch_size, device):
    """Evaluates the recovery fraction for a specific N and K."""
    patterns = generate_patterns(K, N)
    
    # Initialize exactly matching your __init__ signature
    model = Hopfield_Network_gpu(size=N, patterns=patterns, k=k_val, activation=act, device=device)
    
    # Dynamic batch size to prevent OOM errors with large J_stack (n_patterns, N, N)
    # Scales down the batch size as N and K get very large
    memory_factor = (N / 100) ** 2 * (K / 100)
    dynamic_batch_size = max(8, int(base_batch_size / max(1, memory_factor * 0.1)))
    
    successes = 0
    n_done = 0
    
    while n_done < n_trials:
        bs = min(dynamic_batch_size, n_trials - n_done)

        idx = torch.randint(0,K,(bs,), device = device)
        target_patterns = model.patterns[idx]
        x_corrupted = target_patterns.clone()
        noise_mask = torch.rand(x_corrupted.shape,device = device) < 0.10
        x_corrupted[noise_mask] *= -1.0

        # Calling your method exactly as defined
        perfect_recovery, _ = model.mc_simulation_recovery_batch(x_corrupted, epochs, temp)
        successes += int(perfect_recovery.sum().item())
        n_done += bs
        
    # Free up memory explicitly before the next large initialization
    del model

    
    del patterns
    
    if 'cuda' in device:
        torch.cuda.empty_cache()
    elif 'mps' in device:
        torch.mps.empty_cache()
        
    return successes / n_trials

def find_critical_capacity(N, k_val, act, threshold=0.5, n_trials=500, epochs=1000, temp=0, batch_size=128, device='mps'):
    """Uses binary search to efficiently find Kc where accuracy drops below the threshold."""
    print(f"--- Finding Kc for N={N} ---")
    
    k_adaptive = 3.0 / np.sqrt(N)

    low = int(0.1 * N)
    high = int(2.0 * N) 
    best_K = low
    
    # Fast initial expansion
    while evaluate_capacity(N, high, k_adaptive, act, min(100, n_trials), epochs, temp, batch_size, device) >= threshold:
        low = high
        high = int(high * 2)
        print(f"    Expanding search bounds to high={high}")
        if high > 10000: break

    # Binary Search
    while low <= high:
        mid = (low + high) // 2
        acc = evaluate_capacity(N, mid, k_adaptive, act, n_trials, epochs, temp, batch_size, device)
        
        print(f"    Testing K={mid} -> Acc: {acc:.3f}")
        
        if acc >= threshold:
            best_K = mid
            low = mid + 1  
        else:
            high = mid - 1 
            
    return best_K

def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device.upper()}")




   # Hyperparameters
    N_values = [100,200,500,1000]
    k_fixed = 0.3
    activation = 'elu'
    n_trials = 200
    epochs = 1000
    temp = 0
    base_batch_size = 256
    
    results = []
    
    for N in N_values:
        start_time = time.time()
        Kc = find_critical_capacity(N, k_fixed, activation, 0.95, n_trials, epochs, temp, base_batch_size, device)
        elapsed = time.time() - start_time
        
        print(f"=> N={N} | Critical Capacity Kc={Kc} | Time: {elapsed:.1f}s\n")
        results.append({'N': N, 'Kc': Kc})
        
    df = pd.DataFrame(results)
    df.to_csv("scaling_results.csv", index=False)
    
    # Calculate log-log slope
    log_N = np.log10(df['N'])
    log_Kc = np.log10(df['Kc'])
    slope, intercept, r_value, p_value, std_err = linregress(log_N, log_Kc)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(log_N, log_Kc, color='blue', label='Data points', zorder=5)
    plt.plot(log_N, slope * log_N + intercept, color='red', linestyle='--', 
             label=f'Fit (Slope = {slope:.3f})')
    
    plt.title(f'Scaling of Critical Capacity (Activation: {activation.upper()}, k={k_fixed})')
    plt.xlabel('log10(N)')
    plt.ylabel('log10(Kc)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig("capacity_scaling_loglog.png", dpi=300)
    print(f"\nScaling Slope: {slope:.3f}")

if __name__ == "__main__":
    main()