
import numpy as np

import matplotlib.pyplot as plt
import os
class Hopfield_Network:
    def __init__(self,size,patterns, k, activation,alpha=0.01):
        self.size = size
        self.patterns = patterns
        self.k = k 
        self.activation = activation
        self.alpha = alpha
        self.J_list, self.Pt_list = self._preprocess_patterns()

    def _step_threshold(self,x):
        return float(x>=self.k)

    def _relu(self,x):
        if x < self.k :
            y = 0
        elif x>= self.k:
            y = x
        return y
    def _leaky_relu(self, x):
        if x < self.k:
            y = self.alpha * (x - self.k)  # leak below threshold
        else:
            y = x - self.k                 # standard positive slope
        return y
    
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
        """
    Precompute J_matrices and dot templates for all patterns.
    Outputs:
        J_list: list of pattern @ pattern.T / size
        pt_list: list of pattern.T 
    """
        J_list = []
        Pt_list = []
        for pattern in self.patterns:
            #pattern = -pattern
            pattern = pattern.reshape(-1,1)
            pt = pattern.T
            J_matrix = (pattern @ pt)/self.size # the weight matrix
            J_list.append(J_matrix)
            Pt_list.append(pt)
        return J_list, Pt_list


    def energy_calc(self,random_vector):

        random_vector = random_vector.reshape(-1,1)
        net_J_matrix = np.zeros((random_vector.shape[0], random_vector.shape[0]))
        
        for J_matrix, pt in zip(self.J_list,self.Pt_list):

            threshold_factor = self._activation_function(float((pt @ random_vector)/self.size))
            net_J_matrix += J_matrix * threshold_factor
        #net_J_matrix /= size

        Energy =  -0.5 * float(random_vector.T @ net_J_matrix @ random_vector)

        return Energy


    def mc_simulation(self,epochs, temp,save_gif = False,gif_shape = (8,8)):
        # gif dimensions = 8 * 8 = 64 --> size of vector
        x = np.random.choice([-1,1], size = (self.size,))
        overlap_history = []
        images = []
        #J_list, Pt_list = self._preprocess_patterns(self.patterns,self.size)
        patterns_arr = np.array(self.patterns)
        for epoch in range(epochs):
            flip_index = np.random.randint(0,self.size)
            x_flipped = x.copy()
            x_flipped[flip_index] *= -1

            energy_x = self.energy_calc(x)
            energy_x_flip = self.energy_calc(x_flipped)

            energy_diff = energy_x_flip - energy_x

            if energy_diff < 0  or np.random.rand() < np.exp(-energy_diff/temp):
                x = x_flipped

            max_overlap = np.max(np.dot(patterns_arr,x)/self.size) 
            #max_overlap = max(np.dot(x, p)/size for p in patterns)
            overlap_history.append(max_overlap)
            
            if save_gif and epoch % 10 == 0:
                if epoch % 500 == 0:
                    print(f"{epoch}")
                    
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(x.reshape(gif_shape), cmap='binary', vmin=-1, vmax=1)
                ax.set_title(f"Epoch {epoch}",fontsize = 8)
                ax.axis('off')

                # Save frame as RGB image
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                images.append(img)
                plt.close(fig)



        if save_gif:
            return overlap_history, images
        else:
            return overlap_history

    def mc_simulation_recovery(self,epochs,temp):

        ratio = 1/self.size
        x = np.random.choice([-1,1], size = (self.size,))
        #overlap_history = []
        #J_list, Pt_list = self._preprocess_patterns(self.patterns,self.size)
        patterns_arr = np.array(self.patterns)
        
        for epoch in range(epochs):
            flip_index = np.random.randint(0,self.size)
            x_flipped = x.copy()
            x_flipped[flip_index] *= -1

            energy_x = self.energy_calc(x)
            energy_x_flip = self.energy_calc(x_flipped)

            energy_diff = energy_x_flip - energy_x

            if temp == 0:
                if energy_diff < 0:
                    x = x_flipped
            else:
                if energy_diff < 0 or np.random.rand() < np.exp(-energy_diff/temp):
                    x = x_flipped



        overlaps = [np.dot(x, p) for p in self.patterns]
        overlaps = np.array(overlaps)

        # ---- Perfect recovery check ----
        perfect_recovery = 0
        for o in overlaps:
            o = o/self.size
            if o >= 1 - ratio:  # success criterion
                perfect_recovery = 1
                break  # no need to check others
        max_overlap = overlaps.max()

        return perfect_recovery, max_overlap
                

    
