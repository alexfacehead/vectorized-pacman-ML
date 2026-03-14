"""Numpy array-based replay buffer for fast vectorized sampling."""

import numpy as np
from typing import Tuple


class ReplayBuffer:
    """Ring-buffer backed by pre-allocated numpy arrays.

    Stores transitions as (state, action, reward, next_state, done).
    Sampling is fully vectorized — no Python loops.
    """

    def __init__(self, capacity: int, state_shape: Tuple[int, ...] = (7, 31, 28)):
        self.capacity = capacity
        self.size = 0
        self.pos = 0

        # Pre-allocate flat arrays
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                  next_states: np.ndarray, dones: np.ndarray) -> None:
        """Add N transitions at once."""
        n = states.shape[0]
        if n == 0:
            return

        # Check if we wrap around the buffer
        end = self.pos + n
        if end <= self.capacity:
            self.states[self.pos:end] = states
            self.actions[self.pos:end] = actions
            self.rewards[self.pos:end] = rewards
            self.next_states[self.pos:end] = next_states
            self.dones[self.pos:end] = dones
        else:
            # Split into two parts
            first = self.capacity - self.pos
            self.states[self.pos:] = states[:first]
            self.actions[self.pos:] = actions[:first]
            self.rewards[self.pos:] = rewards[:first]
            self.next_states[self.pos:] = next_states[:first]
            self.dones[self.pos:] = dones[:first]

            second = n - first
            self.states[:second] = states[first:]
            self.actions[:second] = actions[first:]
            self.rewards[:second] = rewards[first:]
            self.next_states[:second] = next_states[first:]
            self.dones[:second] = dones[first:]

        self.pos = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size
