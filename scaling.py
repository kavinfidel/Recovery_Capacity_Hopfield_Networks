import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import linregress
import time
import pandas as pd

# Import your custom network
from hopfield_batched import Hopfield_Network_gpu

def generate_patterns(num_patterns, N):
    return np.random.choice([-1, 1], size=(num_patterns, N))

def evaluate_capacity(N, K, k_val, act, n_trials, epochs, temp, batch_size, device):
    """Evaluates the recovery fraction for a specific N and K."""
    patterns = generate_patterns(K, N)
    model = Hopfield_Network_gpu(N, patterns, k_val, activation=act, device=device)
    
    successes = 0
    n_done = 0
    
    while n_done < n_trials:
        bs = min(batch_size, n_trials - n_done)
        perfect_recovery, _ = model.mc_simulation_recovery_batch(bs, epochs, temp)
        successes += int(perfect_recovery.sum().item())
        n_done += bs
        
    return successes / n_trials

def find_critical_capacity(N, k_val, act, threshold=0.95, n_trials=500, epochs=1000, temp=0, batch_size=128, device='cpu'):
    """Uses binary search to efficiently find Kc where accuracy drops below the threshold."""
    print(f"--- Finding Kc for N={N} ---")
    
    # Define search space based on expected scaling (O(N) to O(N^2))
    low = int(0.1 * N)
    high = int(3.0 * N) # Expand this if super-linearity is extreme
    best_K = low
    
    # Fast initial expansion if 'high' is too low
    while evaluate_capacity(N, high, k_val, act, min(100, n_trials), epochs, temp, batch_size, device) >= threshold:
        low = high
        high = int(high * 1.5)
        print(f"    Expanding search bounds to high={high}")

    # Binary Search for precision
    while low <= high:
        mid = (low + high) // 2
        acc = evaluate_capacity(N, mid, k_val, act, n_trials, epochs, temp, batch_size, device)
        
        print(f"    Testing K={mid} -> Acc: {acc:.3f}")
        
        if acc >= threshold:
            best_K = mid
            low = mid + 1  # Try to push for a higher capacity
        else:
            high = mid - 1 # Capacity too high, model failed
            
    return best_K

def main():
    # Setup Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device.upper()}")

    # Hyperparameters
    N_values =
    k_fixed = 0.3
    activation = 'elu'  # Or 'leaky'
    n_trials = 500
    epochs = 1000
    temp = 0
    batch_size = 256
    
    results = []
    
    for N in N_values:
        start_time = time.time()
        Kc = find_critical_capacity(N, k_fixed, activation, 0.95, n_trials, epochs, temp, batch_size, device)
        elapsed = time.time() - start_time
        
        print(f"=> N={N} | Critical Capacity Kc={Kc} | Time: {elapsed:.1f}s\n")
        results.append({'N': N, 'Kc': Kc})
        
    df = pd.DataFrame(results)
    df.to_csv("scaling_results.csv", index=False)
    
    # Calculate log-log slope
    log_N = np.log10(df['N'])
    log_Kc = np.log10(df['Kc'])
    slope, intercept, r_value, p_value, std_err = linregress(log_N, log_Kc)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(log_N, log_Kc, color='blue', label='Data points', zorder=5)
    plt.plot(log_N, slope * log_N + intercept, color='red', linestyle='--', 
             label=f'Fit (Slope = {slope:.3f})')
    
    plt.title(f'Scaling of Critical Capacity (Activation: {activation.upper()}, k={k_fixed})')
    plt.xlabel('$\log_{10}(N)$')
    plt.ylabel('$\log_{10}(K_c)$')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig("capacity_scaling_loglog.png", dpi=300)
    print(f"\nScaling Slope: {slope:.3f}")
    if slope > 1.05:
        print("Conclusion: SUPER-LINEAR SCALING DETECTED.")
    else:
        print("Conclusion: Linear (or sub-linear) scaling detected.")

if __name__ == "__main__":
    main()