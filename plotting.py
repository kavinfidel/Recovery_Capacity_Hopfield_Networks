import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv("full_recovery_curves_P2000.csv")

# 2. Setup the plot style
plt.figure(figsize=(10, 6))
plt.grid(True, linestyle='--', alpha=0.7)

# 3. Plot each activation function
# We use different markers and line styles to make them distinct
styles = {
    'step':  {'color': 'black', 'ls': '-', 'label': 'Step Function'},
    'relu':  {'color': 'blue',  'ls': '-',  'label': 'ReLU Function'},
    'leaky': {'color': 'green', 'ls': '-',  'label': 'Leaky ReLU Function'},
    'elu':   {'color': 'red',   'ls': '-',  'label': 'ELU Function'}
}

for act in ['step', 'relu', 'leaky', 'elu']:
    plt.plot(df['k_value'], df[act], 
             label=styles[act]['label'], 
             color=styles[act]['color'], 
             linestyle=styles[act]['ls'], 
             linewidth=2)

# 4. Add the "Breaking Point" indicator
plt.axvline(x=0.192, color='gray', linestyle=':', label='50% recovery(0.192)')

# 5. Labeling
plt.title("Gated Hopfield Network:  K = 2000, N = 100", fontsize=14)
plt.xlabel("Gating Threshold ($k$)", fontsize=12)
plt.ylabel("Recovery Fraction (Success Rate)", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
plt.ylim(-0.05, 1.05) # Give a little room at top/bottom

# 6. Save and show
plt.tight_layout()
plt.savefig("Phase_Transition_P2000.png", dpi=300)
print("Graph saved as Phase_Transition_P2000.png")
plt.show()


# python /Users/kavinfidel/Documents/Fidel/IMSC/Project/Recovery_Capacity_Hopfield_Networks/plotting.py