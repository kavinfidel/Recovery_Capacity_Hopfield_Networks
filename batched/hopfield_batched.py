# hopfield_network_gpu.py (modified)
import torch

class Hopfield_Network_gpu:
    def __init__(self, size, patterns, k , activation, alpha = 0.01, device = 'mps'):
        self.size = size
        self.k = k 
        self.activation = activation
        self.alpha = alpha
        # Use CUDA on T4
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
        print("Device:", self.device)

        # move patterns to device once
        self.patterns = torch.tensor(patterns, dtype=torch.float32, device=self.device)  # (n_patterns, size)

        # Precompute J_stack (n_patterns, size, size) and Pt_stack (n_patterns, size)
        self.J_stack, self.Pt_stack = self._preprocess_patterns()


    def _elu(self, x):
        # shift by k similar to relu
        return torch.where(x >= self.k, x - self.k, self.alpha * (torch.exp(x - self.k) - 1))

    def _step_threshold(self,x):
        return (x >= self.k).float()

    def _relu(self,x):
        return torch.relu(x - self.k)

    def _leaky_relu(self, x):
        # vectorized for any shape
        return torch.where(x < self.k, self.alpha * (x - self.k), x - self.k)

    def _activation_function(self,x):
        if self.activation == 'step':
            return self._step_threshold(x)
        elif self.activation == 'relu':
            return self._relu(x)
        elif self.activation == 'leaky':
            return self._leaky_relu(x)
        elif self.activation == 'elu':
            return self._elu(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _preprocess_patterns(self):
        with torch.no_grad():
            # patterns: (n_patterns, size)
            pattern_col = self.patterns.unsqueeze(2)   # (n_patterns, size, 1)
            pattern_row = self.patterns.unsqueeze(1)   # (n_patterns, 1, size)  but we'll use Pt_stack as (n_patterns, size)
            J_stack = torch.matmul(pattern_col, pattern_row) / self.size  # (n_patterns, size, size)
            Pt_stack = self.patterns  # (n_patterns, size)
            return J_stack.contiguous(), Pt_stack.contiguous()

    def energy_calc_batch(self, x_batch):
        """
        x_batch: (batch, size)
        returns energies: (batch,)
        """
        with torch.no_grad():
            batch = x_batch.shape[0]  # number of trials in batch
            # overlaps: (n_patterns, batch) = Pt_stack (n_patterns, size) @ x_batch.T (size, batch)
            overlaps = torch.matmul(self.Pt_stack, x_batch.t()) / self.size  # (n_patterns, batch)

            thresholds = self._activation_function(overlaps)  # (n_patterns, batch)

            # net_J_batch: (batch, size, size) = sum_p thresholds[p,b] * J_stack[p]
            # einsum: 'pb,pij->bij' will produce (batch, size, size)
            net_J_batch = torch.einsum('pb,pij->bij', thresholds, self.J_stack)  # (batch, size, size)

            # compute energy: energy[b] = -0.5 * x[b]T @ net_J_batch[b] @ x[b]
            x_col = x_batch.unsqueeze(2)         # (batch, size, 1)
            tmp = torch.bmm(net_J_batch, x_col)  # (batch, size, 1)
            energy = -0.5 * torch.bmm(x_batch.unsqueeze(1), tmp).squeeze()  # (batch,)
            return energy  # float tensor (batch,)

    def mc_simulation_recovery_batch(self, n_trials_in_batch, epochs, temp):
        """
        Simulate `n_trials_in_batch` independent trials in parallel and return:
            perfect_recovery: boolean tensor (batch,)
            max_overlap: float tensor (batch,) (max overlaps across patterns)
        """
        ratio = 1.0 / self.size
        batch = n_trials_in_batch

        # initialize x for the batch: random +-1 values
        x = (torch.randint(0,2,(batch, self.size), device=self.device, dtype=torch.float32) * 2.0 - 1.0).contiguous()
        patterns_arr = self.patterns  # (n_patterns, size)

        arange_batch = torch.arange(batch, device=self.device)

        # run epochs: one bit flip per trial per epoch (vectorized)
        for _ in range(epochs):
            # choose per-trial flip indices
            flip_indices = torch.randint(0, self.size, (batch,), device=self.device)

            # prepare x_flipped: flip the chosen bit for each trial
            x_flipped = x.clone()
            x_flipped[arange_batch, flip_indices] *= -1.0

            # compute energies for whole batch
            energy_x = self.energy_calc_batch(x)
            energy_x_flip = self.energy_calc_batch(x_flipped)
            energy_diff = energy_x_flip - energy_x  # (batch,)

            if temp == 0:
                accept_mask = energy_diff < 0
            else:
                # vectorized metropolis acceptance
                probs = torch.exp(-energy_diff / temp)
                rand = torch.rand(batch, device=self.device)
                accept_mask = (energy_diff < 0) | (rand < probs)

            # update only accepted trials
            if accept_mask.any():
                x[accept_mask] = x_flipped[accept_mask]

        # compute overlaps final: (n_patterns, batch) = patterns (n_patterns,size) @ x.T (size,batch)
        overlaps = torch.matmul(patterns_arr, x.t())  # (n_patterns, batch)

        perfect_recovery = (overlaps / self.size >= 1 - ratio).any(dim=0) 

        max_overlap = overlaps.max(dim=0).values
        
  
        return perfect_recovery, max_overlap
