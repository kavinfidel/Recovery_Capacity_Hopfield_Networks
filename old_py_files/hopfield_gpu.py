
import numpy as np

import matplotlib.pyplot as plt
import os
import torch


class Hopfield_Network_gpu:
    def __init__(self, size, patterns, k , activation, alpha = 0.01, device = 'mps'):
        self.size = size
        self.k = k 
        self.activation = activation
        self.alpha = alpha
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
        print(self.device)


        self.patterns = torch.tensor(patterns, dtype = torch.float32, device = self.device)

        self.J_stack, self.Pt_stack = self._preprocess_patterns()

    def _step_threshold(self,x):
        return (x >= self.k).float()

    def _relu(self,x):
        return torch.relu(x-self.k)
    
    def _leaky_relu(self, x):
        return torch.where(x < self.k, self.alpha * (x - self.k), x - self.k)

    
    def _activation_function(self,x):
        if self.activation == 'step':
            return self._step_threshold(x)
        elif self.activation == 'relu':
            return self._relu(x)
        elif self.activation == 'leaky':
            return self._leaky_relu(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


    def _preprocess_patterns(self):
        with torch.no_grad():
            pattern_col = self.patterns.unsqueeze(2)
            pattern_row = self.patterns.unsqueeze(1)
            J_stack = torch.matmul(pattern_col, pattern_row)/self.size
            Pt_stack = pattern_row
            return J_stack, Pt_stack
    
    def energy_calc(self,x):
        with torch.no_grad():
            x_col = x.to(self.device).float().view(-1,1) # size ,1

            # (n_patterns, 1, 1):  overlaps of x with each pattern
            overlaps = torch.matmul(self.Pt_stack, x_col)/self.size

            thresholds = self._activation_function(overlaps)

            net_J = torch.tensordot(thresholds.squeeze(), self.J_stack, dims=([0],[0]))

            energy = -0.5 * torch.matmul(x.T, torch.matmul(net_J, x))

            return energy.squeeze()
        
    def mc_simulation_recovery(self,epochs, temp):
        ratio = 1/ self.size
        x     = torch.randint(0,2, (self.size,),device = self.device, dtype = torch.float32)* 2 -1
        patterns_arr = self.patterns

        for _ in range(epochs):
            flip_index = torch.randint(0,self.size,(1,), device = self.device)
            x_flipped = x.clone()
            x_flipped[flip_index] *= -1

            energy_x = self.energy_calc(x)
            energy_x_flip = self.energy_calc(x_flipped)
            energy_diff = energy_x_flip - energy_x

            if temp == 0:
                if energy_diff < 0:
                    x = x_flipped

            else:
                if energy_diff < 0 or torch.rand(1, device = self.device) < torch.exp(-energy_diff/temp):
                    x = x_flipped

        overlaps = torch.matmul(patterns_arr,x)
        
        perfect_recovery = (overlaps/self.size >= 1-ratio).any(dim=1)
        max_overlap = overlaps.max(dim=1).values

        return perfect_recovery, max_overlap





