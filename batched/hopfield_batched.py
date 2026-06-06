# hopfield_network_gpu.py (modified)
import torch

class Hopfield_Network_gpu:
    def __init__(self, size, patterns, k , activation, alpha = 0.01, device = 'mps'):
        self.size = size
        self.k = k 
        self.activation = activation
        self.alpha = alpha
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print("Device:", self.device)

        # move patterns to device once
        self.patterns = torch.tensor(patterns, dtype=torch.float32, device=self.device)  # (n_patterns, size)

        # Store pattern rows once. Energies are computed from overlaps directly:
        # x.T @ (p p.T / N) @ x = (p.x)^2 / N.
        self.Pt_stack = self._preprocess_patterns()


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
            return self.patterns.contiguous()

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
            energy = -0.5 * self.size * torch.sum(thresholds * overlaps.square(), dim=0)
            return energy  # float tensor (batch,)

    def _energy_from_normalized_overlaps(self, overlaps):
        thresholds = self._activation_function(overlaps)
        return -0.5 * self.size * torch.sum(thresholds * overlaps.square(), dim=0)

    def _mc_simulation_from_state(self, x, epochs, temp, target_indices=None):
        """
        Simulate independent trials from the supplied initial states and return:
            perfect_recovery: boolean tensor (batch,). If target_indices is
                provided, success means recovery of that exact target pattern.
                Otherwise, success means recovery of any stored pattern.
            max_overlap: float tensor (batch,) (max overlaps across patterns)
        """
        x = x.clone().contiguous()
        ratio = 1.0 / self.size
        batch = x.shape[0]
        patterns_arr = self.patterns  # (n_patterns, size)

        arange_batch = torch.arange(batch, device=self.device)
        overlaps_state = torch.matmul(patterns_arr, x.t()) / self.size

        # run epochs: one bit flip per trial per epoch (vectorized)
        for _ in range(epochs):
            # choose per-trial flip indices
            flip_indices = torch.randint(0, self.size, (batch,), device=self.device)

            current_bits = x[arange_batch, flip_indices]
            pattern_bits = patterns_arr[:, flip_indices]
            overlap_delta = pattern_bits * (-2.0 * current_bits / self.size).unsqueeze(0)
            overlaps_flipped = overlaps_state + overlap_delta

            # compute energies for whole batch
            energy_x = self._energy_from_normalized_overlaps(overlaps_state)
            energy_x_flip = self._energy_from_normalized_overlaps(overlaps_flipped)
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
                accepted_batch = arange_batch[accept_mask]
                x[accepted_batch, flip_indices[accept_mask]] *= -1.0
                overlaps_state[:, accept_mask] = overlaps_flipped[:, accept_mask]

        if target_indices is None:
            perfect_recovery = (overlaps_state >= 1 - ratio).any(dim=0)
        else:
            target_indices = target_indices.to(self.device)
            target_overlaps = overlaps_state[target_indices, arange_batch]
            perfect_recovery = target_overlaps >= 1 - ratio

        max_overlap = overlaps_state.max(dim=0).values * self.size
        

        return perfect_recovery, max_overlap

    def _mc_simulation_until_convergence_from_state(
        self,
        x,
        max_sweeps,
        temp,
        target_indices=None,
        stable_sweeps_required=1,
    ):
        """
        Evolve states until no accepted flips occur for stable_sweeps_required
        sweeps. One sweep is size random single-bit proposals per trial.
        """
        x = x.clone().contiguous()
        ratio = 1.0 / self.size
        batch = x.shape[0]
        patterns_arr = self.patterns

        arange_batch = torch.arange(batch, device=self.device)
        overlaps_state = torch.matmul(patterns_arr, x.t()) / self.size
        stable_sweeps = torch.zeros(batch, dtype=torch.int32, device=self.device)

        for _ in range(max_sweeps):
            accepted_this_sweep = torch.zeros(batch, dtype=torch.bool, device=self.device)

            for _ in range(self.size):
                active_mask = stable_sweeps < stable_sweeps_required
                if not active_mask.any():
                    break

                flip_indices = torch.randint(0, self.size, (batch,), device=self.device)
                current_bits = x[arange_batch, flip_indices]
                pattern_bits = patterns_arr[:, flip_indices]
                overlap_delta = pattern_bits * (-2.0 * current_bits / self.size).unsqueeze(0)
                overlaps_flipped = overlaps_state + overlap_delta

                energy_x = self._energy_from_normalized_overlaps(overlaps_state)
                energy_x_flip = self._energy_from_normalized_overlaps(overlaps_flipped)
                energy_diff = energy_x_flip - energy_x

                if temp == 0:
                    accept_mask = energy_diff < 0
                else:
                    probs = torch.exp(-energy_diff / temp)
                    rand = torch.rand(batch, device=self.device)
                    accept_mask = (energy_diff < 0) | (rand < probs)

                accept_mask &= active_mask
                if accept_mask.any():
                    accepted_batch = arange_batch[accept_mask]
                    x[accepted_batch, flip_indices[accept_mask]] *= -1.0
                    overlaps_state[:, accept_mask] = overlaps_flipped[:, accept_mask]
                    accepted_this_sweep |= accept_mask

            stable_sweeps = torch.where(
                accepted_this_sweep,
                torch.zeros_like(stable_sweeps),
                stable_sweeps + 1,
            )
            if (stable_sweeps >= stable_sweeps_required).all():
                break

        if target_indices is None:
            perfect_recovery = (overlaps_state >= 1 - ratio).any(dim=0)
        else:
            target_indices = target_indices.to(self.device)
            target_overlaps = overlaps_state[target_indices, arange_batch]
            perfect_recovery = target_overlaps >= 1 - ratio

        max_overlap = overlaps_state.max(dim=0).values * self.size
        return perfect_recovery, max_overlap

    def mc_simulation_recovery_batch(self,n_trials_in_batch, epochs, temp):

        """
        Simulate `n_trials_in_batch` independent trials from random states.
        """
        x = (
            torch.randint(0, 2, (n_trials_in_batch, self.size), device=self.device, dtype=torch.float32)
            * 2.0
            - 1.0
        )
        return self._mc_simulation_from_state(x, epochs, temp)


    def mc_simulation_recovery_corrupted_batch(self,corrupted_x, epochs, temp, target_indices=None):
        """
        Simulate independent trials from supplied corrupted pattern states.
        """
        return self._mc_simulation_from_state(corrupted_x, epochs, temp, target_indices)

    def mc_simulation_recovery_batch_until_convergence(
        self,
        n_trials_in_batch,
        max_sweeps,
        temp,
        stable_sweeps_required=1,
    ):
        """
        Simulate random initial states until convergence to a local minimum.
        """
        x = (
            torch.randint(0, 2, (n_trials_in_batch, self.size), device=self.device, dtype=torch.float32)
            * 2.0
            - 1.0
        )
        return self._mc_simulation_until_convergence_from_state(
            x,
            max_sweeps,
            temp,
            stable_sweeps_required=stable_sweeps_required,
        )
