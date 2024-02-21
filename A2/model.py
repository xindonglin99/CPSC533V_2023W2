import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        # TODO YOUR CODE HERE FOR INITIALIZING THE MODEL
        # Guidelines for network size: start with 2 hidden layers and maximum 32 neurons per layer
        # feel free to explore different sizes
        self.layers = nn.Sequential(
            nn.Linear(state_size, 32, True),
            nn.ReLU(),
            nn.Linear(32, 32, True),
            nn.ReLU(),
            nn.Linear(32, action_size, True),
            )

    def forward(self, x):
        return self.layers(x)
        
    def select_action(self, state):
        self.eval()
        x = self.forward(state)
        self.train()
        return x.max(1)[1].view(1, 1).to(torch.long)
