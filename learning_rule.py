import torch
import torch.nn as nn


# anti-Hebbian learning
class AntiHebbian(nn.Module):
    def __init__(self, args):
        super(AntiHebbian, self).__init__()
        self.lr = args.lr

    def forward(self, x, input):
        # searching median of outputs activity
        # if activity higher than the median the neuron is active, otherwise resting state
        # replacing resting neurons activity value by 0
        y = torch.full((x.size(dim=0),), 0, dtype=torch.float32)
        # replacing active neurons activity value by 1
        y[x > x.median()] = 1.
        # computing delta
        w = y.repeat(x.size(dim=0), 1) * input.reshape((-1, 1))
        # multiplication with the learning rate
        return -self.lr * w


# Hebbian learning
class Hebbian(nn.Module):
    def __init__(self, args):
        super(Hebbian, self).__init__()
        self.lr = args.lr

    def forward(self, _, x):
        xbis = x.reshape(-1)
        y = torch.full((xbis.size(dim=0),), 0, dtype=torch.float32)
        y[xbis > xbis.median()] = 1.0
        w = y.repeat(x.size(dim=0), 1) * x.reshape((-1, 1))
        return self.lr * w


