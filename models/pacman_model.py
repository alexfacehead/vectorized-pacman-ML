import torch
import torch.nn as nn
import torch.nn.functional as F


class PacmanDRQN(nn.Module):
    """Dueling DRQN for Pacman — LSTM-augmented for temporal reasoning.

    Input:  (batch, C, H, W) single step or (batch, seq_len, C, H, W) sequence
    Output: (q_values, hidden_state)

    Architecture:
      - 3 conv layers: in→32→64→128 (stride 2 on conv3) — spatial features
      - FC projection: 24960→512 — compress to LSTM input
      - LSTM: 512→512 — temporal reasoning (route planning, ghost tracking)
      - Dueling heads: value V(s) + advantage A(s,a)
      - Q(s,a) = V(s) + A(s,a) - mean(A)
    """

    def __init__(self, in_channels: int = 9, num_actions: int = 4,
                 lstm_hidden: int = 512):
        super().__init__()
        self.num_actions = num_actions
        self.lstm_hidden = lstm_hidden

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)

        # After conv3 with stride 2: floor((31-3)/2 + 1) = 15, floor((28-3)/2 + 1) = 13
        flat_size = 128 * 15 * 13  # 24,960

        # Project to LSTM input size
        self.fc_pre_lstm = nn.Linear(flat_size, lstm_hidden)

        # LSTM for temporal reasoning
        self.lstm = nn.LSTM(lstm_hidden, lstm_hidden, batch_first=True)

        # Dueling heads (from LSTM output)
        self.value_fc1 = nn.Linear(lstm_hidden, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.adv_fc1 = nn.Linear(lstm_hidden, 256)
        self.adv_fc2 = nn.Linear(256, num_actions)

    def forward(self, x: torch.Tensor, hidden=None):
        """Forward pass supporting single-step and sequence modes.

        Args:
            x: (batch, C, H, W) for single step, or
               (batch, seq_len, C, H, W) for sequence.
            hidden: optional (h, c) tuple, each (1, batch, lstm_hidden).

        Returns:
            q: (batch, num_actions) for single step, or
               (batch, seq_len, num_actions) for sequence.
            hidden: updated (h, c) tuple.
        """
        single_step = (x.dim() == 4)
        if single_step:
            x = x.unsqueeze(1)  # (batch, 1, C, H, W)

        batch, seq_len, C, H, W = x.shape

        # Conv features for all frames
        x = x.reshape(batch * seq_len, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(batch, seq_len, -1)  # (batch, seq_len, flat_size)

        # Project to LSTM input
        x = F.relu(self.fc_pre_lstm(x))  # (batch, seq_len, lstm_hidden)

        # LSTM
        lstm_out, hidden = self.lstm(x, hidden)  # (batch, seq_len, lstm_hidden)

        # Dueling heads
        if single_step:
            h = lstm_out.squeeze(1)  # (batch, lstm_hidden)
        else:
            h = lstm_out.reshape(batch * seq_len, -1)

        v = F.relu(self.value_fc1(h))
        v = self.value_fc2(v)  # (..., 1)

        a = F.relu(self.adv_fc1(h))
        a = self.adv_fc2(a)  # (..., num_actions)

        q = v + a - a.mean(dim=-1, keepdim=True)

        if not single_step:
            q = q.reshape(batch, seq_len, -1)

        return q, hidden

    def init_hidden(self, batch_size: int, device=None):
        """Initialize LSTM hidden state to zeros."""
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)
