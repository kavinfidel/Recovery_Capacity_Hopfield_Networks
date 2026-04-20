# import numpy as np
# import matplotlib.pyplot as plt
# import pickle

# import numpy as np
# import matplotlib.pyplot as plt
# import pickle

# def plot_dam_style_histograms(overlaps_dict, act='leaky', n_value=2000, k_values=None, size=100, n_trials=500):
#     """
#     Plots the distribution of overlaps to see the transition from 
#     spurious states (overlap ~0) to perfect retrieval (overlap ~1).
#     """
#     # If no k_values provided, try to find interesting ones from the data
#     if k_values is None:
#         # Example: Plot the start, middle, and end of your sweep
#         k_values = [0.15, 0.20, 0.30, 0.45] 

#     n_plots = len(k_values)
#     fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=True)
    
#     # Handle single plot case
#     if n_plots == 1: axes = [axes]

#     for i, k in enumerate(k_values):
#         # The key must match your evaluate_grid storage format
#         # In your case, it was (act, k) inside the metrics dict
#         try:
#             # Note: We divide by 'size' to get normalized overlap [-1, 1]
#             overlaps = np.array(overlaps_dict[(act, k)]['max_overlap']) / size
#         except KeyError:
#             print(f"Warning: Key ({act}, {k}) not found in data.")
#             continue

#         axes[i].hist(overlaps, bins=20, range=(-0.2, 1.2), color='salmon', edgecolor='black', alpha=0.7)
#         axes[i].set_title(f'Threshold k = {k:.3f}', fontsize=14)
#         axes[i].set_xlabel('Normalized Overlap')
#         if i == 0: axes[i].set_ylabel('Frequency')
#         axes[i].grid(axis='y', linestyle='--', alpha=0.6)

#     plt.suptitle(f'DAM-Style Retrieval Quality: {act.upper()} (N={size}, P={n_value})', fontsize=16)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.savefig(f"DAM_histograms_{act}_{n_value}.png")
#     plt.show()


# try:
#     with open("metrics_dump_P2000.pkl", "rb") as f:
#         metrics = pickle.load(f)
    
#     # Target k values based on your previous run's "Goldilocks Zone"
#     plot_dam_style_histograms(metrics, act='elu', target_ks=[0.18, 0.20, 0.30, 0.45], size=100)
# except FileNotFoundError:
#     print("Error: Could not find metrics_dump_P2000.pkl. Run the simulation first!")
# # python /Users/kavinfidel/Documents/Fidel/IMSC/Project/Recovery_Capacity_Hopfield_Networks/plot_overlaps.py



import numpy as np
import matplotlib.pyplot as plt
import pickle

def find_closest_k(metrics_keys, target_act, target_k):
    """Finds the actual key in the dict that is closest to our target k."""
    valid_keys = [k for (act, k) in metrics_keys if act == target_act]
    if not valid_keys:
        return None
    closest_k = min(valid_keys, key=lambda x: abs(x - target_k))
    return (target_act, closest_k)

def plot_dam_style_histograms(metrics_data, act='elu', target_ks=[0.15, 0.20, 0.30, 0.45], size=100):
    n_plots = len(target_ks)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=True)
    
    if n_plots == 1: axes = [axes]

    for i, t_k in enumerate(target_ks):
        # Use fuzzy search to find the closest k actually in the data
        actual_key = find_closest_k(metrics_data.keys(), act, t_k)
        
        if actual_key and 'max_overlap' in metrics_data[actual_key]:
            overlaps = np.array(metrics_data[actual_key]['max_overlap']) / size
            
            axes[i].hist(overlaps, bins=25, range=(-0.2, 1.2), 
                         color='skyblue', edgecolor='black', alpha=0.8)
            axes[i].set_title(f'Actual k ≈ {actual_key[1]:.3f}', fontsize=12)
            axes[i].set_xlabel('Normalized Overlap')
            if i == 0: axes[i].set_ylabel('Frequency')
            axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        else:
            axes[i].text(0.5, 0.5, f"Data missing for\n{act} at k≈{t_k}", 
                         ha='center', va='center')
            print(f"Skipping {act} at k≈{t_k}: No overlap data found.")

    plt.suptitle(f'DAM Retrieval Quality: {act.upper()} (N={size}, P=2000)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"DAM_histograms_{act}_refined.png")
    plt.show()

# --- RUN THIS PART ---
try:
    with open("metrics_dump_P2000.pkl", "rb") as f:
        metrics = pickle.load(f)
    
    # Target k values based on your previous run's "Goldilocks Zone"
    plot_dam_style_histograms(metrics, act='elu', target_ks=[0.18, 0.20, 0.30, 0.45], size=100)
except FileNotFoundError:
    print("Error: Could not find metrics_dump_P2000.pkl. Run the simulation first!")