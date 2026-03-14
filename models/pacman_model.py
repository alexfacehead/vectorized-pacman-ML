import torch
import torch.nn as nn
import torch.nn.functional as F


class PacmanDQN(nn.Module):
    """DQN for Pacman agent operating on grid state.

    Input:  (batch, in_channels, 31, 28) — multi-channel grid state
    Output: (batch, num_actions) — Q-values for each action

    Architecture matches the original PacmanML model exactly.
    """

    def __init__(self, in_channels: int = 7, num_actions: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        # After conv3: floor((31-3)/2 + 1) = 15, floor((28-3)/2 + 1) = 13
        self.fc1 = nn.Linear(64 * 15 * 13, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
