"""Phase 8 smoke tests: Training integration."""

import numpy as np
import torch
import pytest

from tests.conftest import MAZE_FILE

from engine.batched_game import BatchedGame
from engine.constants import NUM_STATE_CHANNELS, ROWS, COLS, MAX_STEPS
from models.pacman_model import PacmanDQN
from utils.replay_buffer import ReplayBuffer


class TestGetState:
    """Verify get_state() produces correct observation tensors."""

    def test_state_shape(self):
        game = BatchedGame(n_envs=4, maze_file=MAZE_FILE)
        state = game.get_state()
        assert state.shape == (4, NUM_STATE_CHANNELS, ROWS, COLS)
        assert state.dtype == torch.float32

    def test_walls_channel_nonzero(self):
        game = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
        state = game.get_state()
        walls = state[0, 0]  # channel 0 = walls
        assert walls.sum() > 0, "Walls channel should have nonzero entries"

    def test_pacman_channel_single_cell(self):
        game = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
        state = game.get_state()
        pac = state[0, 3]  # channel 3 = pacman
        assert pac.sum() == 1.0, "Pacman should occupy exactly one cell"

    def test_pellets_match_maze(self):
        game = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
        state = game.get_state()
        pellets = state[0, 1]
        power = state[0, 2]
        total = pellets.sum() + power.sum()
        assert total == game.maze.total_pellets

    def test_state_batch_consistency(self):
        """All envs in a fresh batch should have identical states."""
        game = BatchedGame(n_envs=4, maze_file=MAZE_FILE)
        state = game.get_state()
        for i in range(1, 4):
            assert torch.equal(state[0], state[i]), f"Env 0 != Env {i}"


class TestGetActionMask:
    """Verify get_action_mask() returns sensible masks."""

    def test_mask_shape(self):
        game = BatchedGame(n_envs=4, maze_file=MAZE_FILE)
        mask = game.get_action_mask()
        assert mask.shape == (4, 4)
        assert mask.dtype == torch.bool

    def test_mask_has_valid_actions(self):
        game = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
        # Burn ready timer
        for _ in range(game.ready_timer[0].item()):
            game.step(torch.tensor([0]))
        mask = game.get_action_mask()
        assert mask[0].any(), "Must have at least one valid action"


class TestModel:
    """Verify the DQN model produces correct output shapes."""

    def test_forward_pass(self):
        model = PacmanDQN(in_channels=7, num_actions=4)
        x = torch.randn(8, 7, 31, 28)
        out = model(x)
        assert out.shape == (8, 4)

    def test_single_state(self):
        model = PacmanDQN(in_channels=7, num_actions=4)
        game = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
        state = game.get_state()
        with torch.no_grad():
            q = model(state)
        assert q.shape == (1, 4)


class TestReplayBuffer:
    """Verify replay buffer works with batched adds."""

    def test_single_add(self):
        buf = ReplayBuffer(100)
        state = np.random.randn(7, 31, 28).astype(np.float32)
        buf.add(state, 2, 0.5, state, False)
        assert len(buf) == 1

    def test_batch_add(self):
        buf = ReplayBuffer(100)
        states = np.random.randn(16, 7, 31, 28).astype(np.float32)
        actions = np.random.randint(0, 4, 16).astype(np.int64)
        rewards = np.random.randn(16).astype(np.float32)
        dones = np.zeros(16, dtype=np.float32)
        buf.add_batch(states, actions, rewards, states, dones)
        assert len(buf) == 16

    def test_batch_add_wraparound(self):
        buf = ReplayBuffer(10)
        states = np.random.randn(8, 7, 31, 28).astype(np.float32)
        actions = np.random.randint(0, 4, 8).astype(np.int64)
        rewards = np.zeros(8, dtype=np.float32)
        dones = np.zeros(8, dtype=np.float32)
        buf.add_batch(states, actions, rewards, states, dones)
        assert len(buf) == 8
        # Add 5 more to wrap around
        states2 = np.random.randn(5, 7, 31, 28).astype(np.float32)
        actions2 = np.random.randint(0, 4, 5).astype(np.int64)
        buf.add_batch(states2, actions2, rewards[:5], states2, dones[:5])
        assert len(buf) == 10  # capped at capacity

    def test_sample(self):
        buf = ReplayBuffer(100)
        for i in range(50):
            state = np.random.randn(7, 31, 28).astype(np.float32)
            buf.add(state, i % 4, 0.1, state, False)
        s, a, r, ns, d = buf.sample(16)
        assert s.shape == (16, 7, 31, 28)
        assert a.shape == (16,)


class TestAutoReset:
    """Verify environments can be selectively reset."""

    def test_partial_reset(self):
        game = BatchedGame(n_envs=4, maze_file=MAZE_FILE)
        # Step a few times
        for _ in range(20):
            game.step(torch.tensor([0, 1, 2, 3]))

        # Reset envs 1 and 3
        mask = torch.tensor([False, True, False, True])
        game.reset(mask)

        # Check reset envs have starting position
        sx, sy = game.maze.pacman_start
        assert game.pacman_pos[1].tolist() == [sx, sy]
        assert game.pacman_pos[3].tolist() == [sx, sy]
        assert game.score[1].item() == 0
        assert game.score[3].item() == 0

    def test_reset_preserves_non_masked(self):
        game = BatchedGame(n_envs=2, maze_file=MAZE_FILE)
        for _ in range(20):
            game.step(torch.tensor([2, 3]))

        score_before = game.score[0].item()
        mask = torch.tensor([False, True])
        game.reset(mask)

        # Env 0 should be unchanged
        assert game.score[0].item() == score_before


class TestTrainingSmoke:
    """End-to-end smoke test: a few training steps don't crash."""

    def test_training_loop_runs(self):
        """Run a minimal training loop for 10 steps."""
        game = BatchedGame(n_envs=2, maze_file=MAZE_FILE)
        model = PacmanDQN(in_channels=7, num_actions=4)
        buf = ReplayBuffer(1000)

        state = game.get_state()

        for step in range(50):
            # Random actions
            mask = game.get_action_mask()
            actions = torch.randint(0, 4, (2,))

            rewards, dones, _ = game.step(actions)
            next_state = game.get_state()
            pac_r = game.get_reward_pacman()

            # Store
            buf.add_batch(
                state.cpu().numpy(),
                actions.numpy().astype(np.int64),
                pac_r.cpu().numpy(),
                next_state.cpu().numpy(),
                dones.float().cpu().numpy(),
            )

            # Train if enough data
            if len(buf) >= 16:
                s, a, r, ns, d = buf.sample(16)
                s_t = torch.from_numpy(s)
                q = model(s_t)
                assert q.shape == (16, 4), "Model output shape wrong"

            # Auto-reset done envs
            if dones.any():
                game.reset(dones)
                next_state = game.get_state()

            state = next_state

    def test_reward_clipping(self):
        """Rewards from get_reward_pacman are clipped to [-1, 1]."""
        game = BatchedGame(n_envs=4, maze_file=MAZE_FILE)
        for _ in range(100):
            game.step(torch.randint(0, 4, (4,)))
            r = game.get_reward_pacman()
            assert r.min() >= -1.0
            assert r.max() <= 1.0
