import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        # TODO YOUR CODE HERE FOR INITIALIZING THE MODEL
        # Guidelines for network size: start with 2 hidden layers and maximum 32 neurons per layer
        # feel free to explore different sizes
        self.layers = nn.Sequential(
            nn.Linear(state_size, 256, True),
            nn.ReLU(),
            nn.Linear(256, 256, True),
            nn.ReLU(),
            nn.Linear(256, action_size, True),
            )
        

    def forward(self, x):
        return self.layers(torch.flatten(x, 1))
        
    def select_action(self, state):
        self.eval()
        x = self.forward(state)
        self.train()
        return x.max(1)[1].view(1, 1).to(torch.long)
