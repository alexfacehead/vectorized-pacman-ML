"""Episode-based sequence replay buffer for DRQN training."""

import numpy as np
import random
from collections import deque
from typing import Tuple


class SequenceReplayBuffer:
    """Episode-based replay buffer that samples fixed-length sequences.

    Stores complete episode trajectories and samples contiguous subsequences
    of length ``seq_len`` for LSTM-based training with BPTT.

    Memory efficiency: does not store redundant next_states — they are derived
    from consecutive states within each episode.
    """

    def __init__(self, max_transitions: int = 200_000, seq_len: int = 16):
        self.max_transitions = max_transitions
        self.seq_len = seq_len
        self.episodes: deque = deque()
        self.total_transitions = 0

    def add_episode(self, states: np.ndarray, actions: np.ndarray,
                    rewards: np.ndarray, dones: np.ndarray) -> None:
        """Add a complete episode trajectory.

        Args:
            states: (T+1, C, H, W) float32 — T+1 states (initial + T post-action).
            actions: (T,) int64 — actions taken.
            rewards: (T,) float32 — clipped rewards.
            dones: (T,) float32 — 1.0 at terminal step.
        """
        T = len(actions)
        if T < 1:
            return

        episode = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'length': T,
        }

        self.episodes.append(episode)
        self.total_transitions += T

        # Evict oldest episodes to stay within capacity
        while self.total_transitions > self.max_transitions and len(self.episodes) > 1:
            removed = self.episodes.popleft()
            self.total_transitions -= removed['length']

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of sequences for DRQN training.

        For each sample, picks a random episode and a random starting position,
        then extracts ``seq_len`` consecutive transitions.  Sequences shorter
        than ``seq_len`` are zero-padded with masks=0.

        Returns:
            states_full: (B, L+1, C, H, W) — L+1 consecutive states so that
                         states[:,t] is the current state and states[:,t+1] is
                         the next state for transition t.
            actions:     (B, L) int64
            rewards:     (B, L) float32
            dones:       (B, L) float32
            masks:       (B, L) float32 — 1.0 for valid (non-padded) steps.
        """
        L = self.seq_len
        state_shape = self.episodes[0]['states'].shape[1:]  # (C, H, W)

        batch_states = np.zeros((batch_size, L + 1, *state_shape), dtype=np.float32)
        batch_actions = np.zeros((batch_size, L), dtype=np.int64)
        batch_rewards = np.zeros((batch_size, L), dtype=np.float32)
        batch_dones = np.zeros((batch_size, L), dtype=np.float32)
        batch_masks = np.zeros((batch_size, L), dtype=np.float32)

        n_episodes = len(self.episodes)

        for i in range(batch_size):
            ep = self.episodes[random.randint(0, n_episodes - 1)]
            T = ep['length']

            # Random start — ensure at least 1 valid transition
            max_start = max(0, T - 1)
            start = random.randint(0, max_start)
            actual_len = min(L, T - start)

            # Copy data (states has T+1 entries, so start+actual_len+1 ≤ T+1)
            batch_states[i, :actual_len + 1] = ep['states'][start:start + actual_len + 1]
            batch_actions[i, :actual_len] = ep['actions'][start:start + actual_len]
            batch_rewards[i, :actual_len] = ep['rewards'][start:start + actual_len]
            batch_dones[i, :actual_len] = ep['dones'][start:start + actual_len]
            batch_masks[i, :actual_len] = 1.0

        return batch_states, batch_actions, batch_rewards, batch_dones, batch_masks

    def __len__(self) -> int:
        return self.total_transitions

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)
