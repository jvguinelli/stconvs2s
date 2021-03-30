import torch


class Normalizer():
    def __init__(self, num_channels=0, mean=None, std=None):
        self.num_channels = num_channels
        if mean is None:
            mean = []
        if std is None:
            std = []
        self.mean = mean
        self.std = std

    def observe(self, x):
        self.num_channels = x.shape[1]
        for i in range(self.num_channels):
            self.mean.append(torch.mean(x[:, i]))
            self.std.append(torch.std(x[:, i]))

    def normalize(self, inputs):
        if self.num_channels != inputs.shape[1]:
            raise
        
        y = inputs.clone()
        for i in range(self.num_channels):
            y[:, i] = ( y[:, i] - self.mean[i] ) / self.std[i]
        
        return y
    
    def denormalize(self, inputs):
        if self.num_channels != inputs.shape[1]:
            raise
        
        y = inputs.clone()
        for i in range(self.num_channels):
            y[:, i] = ( y[:, i] * self.std[i] ) + self.mean[i]
        
        return y
    