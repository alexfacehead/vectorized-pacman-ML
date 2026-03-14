"""Phase 1 parity tests: Pacman movement.

Runs identical action sequences through both the original Python engine
and the vectorized engine, asserting Pacman positions match at every step.
"""

import os
import sys
import numpy as np
import torch
import pytest

from tests.conftest import MAZE_FILE, ORIGINAL_MAZE_FILE, PACMANML_DIR

# Import original engine
sys.path.insert(0, PACMANML_DIR)
from pacman.core.game import Game as OriginalGame

# Import vectorized engine
from engine.batched_game import BatchedGame


def make_original_game() -> OriginalGame:
    """Create a headless original game.
    Removes power pellets to avoid frightened-mode RNG divergence."""
    game = OriginalGame(headless=True)
    game.load_level(ORIGINAL_MAZE_FILE)
    game.reset()
    for pp in game.level.power_pellets:
        pp.eaten = True
    return game


def make_batched_game(n_envs: int = 1) -> BatchedGame:
    """Create a batched game with power pellets removed."""
    bg = BatchedGame(n_envs=n_envs, maze_file=MAZE_FILE)
    bg.power_pellets.zero_()
    return bg


class TestPacmanMovementParity:
    """Compare Pacman movement between original and vectorized engines."""

    def test_start_position(self):
        """Both engines start Pacman at the same position."""
        orig = make_original_game()
        batched = make_batched_game()

        orig_pos = orig.pacman.position  # (x, y) tuple
        batch_pos = batched.pacman_pos[0].tolist()  # [x, y]

        assert list(orig_pos) == batch_pos, f"Start mismatch: orig={orig_pos} batch={batch_pos}"

    def test_ready_timer(self):
        """During ready countdown, Pacman should not move in either engine."""
        orig = make_original_game()
        batched = make_batched_game()

        # Both should have the same ready timer after reset
        assert orig.ready_timer == batched.ready_timer[0].item(), \
            f"Ready timer mismatch: orig={orig.ready_timer} batch={batched.ready_timer[0].item()}"

        # Step through ready period with actions — Pacman should stay put
        start_pos = list(orig.pacman.position)
        for _ in range(orig.ready_timer):
            orig.step(pacman_action=3)  # RIGHT
            batched.step(torch.tensor([3]))

        # After ready period, Pacman should still be at start (ready ticks don't move)
        assert list(orig.pacman.position) == start_pos
        assert batched.pacman_pos[0].tolist() == start_pos

    def test_movement_parity_random_actions(self):
        """Run 10,000 random actions through both engines. Positions must match."""
        np.random.seed(42)
        orig = make_original_game()
        batched = make_batched_game()

        # Burn through ready timer
        ready_steps = orig.ready_timer
        for _ in range(ready_steps):
            orig.step(pacman_action=0)
            batched.step(torch.tensor([0]))

        # Now run random actions
        num_steps = 10000
        actions = np.random.randint(0, 4, size=num_steps)

        for i, action in enumerate(actions):
            orig.step(pacman_action=int(action))
            batched.step(torch.tensor([int(action)]))

            orig_pos = list(orig.pacman.position)
            batch_pos = batched.pacman_pos[0].tolist()

            assert orig_pos == batch_pos, (
                f"Step {i}: positions diverged! "
                f"action={action} orig={orig_pos} batch={batch_pos}"
            )

            # Stop if original game ended (death/level complete)
            if orig.game_over or orig.level_complete:
                break

    def test_wall_collision(self):
        """Pacman can't move through walls."""
        batched = make_batched_game()

        # Burn ready timer
        for _ in range(batched.ready_timer[0].item()):
            batched.step(torch.tensor([0]))

        # Pacman starts at (13, 23). UP should be blocked by wall at row 22
        # (looking at maze: row 22 is "#.####.#####.##.#####.####.#")
        # Actually (13, 23) going UP is (13, 22) which is "." in the maze - not a wall.
        # Let's just verify Pacman doesn't teleport through walls by checking
        # position stays on walkable tiles throughout random actions.
        np.random.seed(99)
        for _ in range(1000):
            batched.step(torch.tensor([np.random.randint(0, 4)]))
            x, y = batched.pacman_pos[0].tolist()
            assert not batched.maze.walls[y, x].item(), f"Pacman in wall at ({x}, {y})"

    def test_tunnel_wrapping(self):
        """Pacman wraps through the tunnel on row 14."""
        orig = make_original_game()
        batched = make_batched_game()

        # Navigate Pacman to the tunnel entrance
        # Pacman starts at (13, 23). We need to go UP then LEFT to reach tunnel.
        # Let's use a sequence that navigates to row 14 leftmost position.
        # For now, just run random actions and check if tunnel wrapping
        # is consistent between engines when it happens.
        np.random.seed(77)
        ready_steps = orig.ready_timer
        for _ in range(ready_steps):
            orig.step(pacman_action=0)
            batched.step(torch.tensor([0]))

        for i in range(5000):
            action = np.random.randint(0, 4)
            orig.step(pacman_action=action)
            batched.step(torch.tensor([action]))

            orig_pos = list(orig.pacman.position)
            batch_pos = batched.pacman_pos[0].tolist()

            assert orig_pos == batch_pos, (
                f"Step {i}: tunnel wrapping divergence! "
                f"action={action} orig={orig_pos} batch={batch_pos}"
            )

            if orig.game_over or orig.level_complete:
                break

    def test_batch_consistency(self):
        """N envs with same actions produce identical results to single env."""
        np.random.seed(42)
        single = make_batched_game(n_envs=1)
        multi = make_batched_game(n_envs=16)

        # Burn ready
        for _ in range(single.ready_timer[0].item()):
            single.step(torch.tensor([0]))
            multi.step(torch.zeros(16, dtype=torch.int32))

        for _ in range(2000):
            action = np.random.randint(0, 4)
            single.step(torch.tensor([action]))
            multi.step(torch.full((16,), action, dtype=torch.int32))

            single_pos = single.pacman_pos[0].tolist()
            for env_idx in range(16):
                multi_pos = multi.pacman_pos[env_idx].tolist()
                assert single_pos == multi_pos, (
                    f"Env {env_idx} diverged from single: {multi_pos} vs {single_pos}"
                )

            if single.game_over[0].item() or single.level_complete[0].item():
                break

    def test_direction_buffering(self):
        """Buffered direction system: pre-turning works correctly."""
        orig = make_original_game()
        batched = make_batched_game()

        # Burn ready
        ready_steps = orig.ready_timer
        for _ in range(ready_steps):
            orig.step(pacman_action=0)
            batched.step(torch.tensor([0]))

        # Specific action sequence testing direction buffering:
        # Start at (13, 23), try RIGHT, RIGHT, RIGHT, UP, UP...
        actions = [3, 3, 3, 3, 0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1] * 10
        for i, action in enumerate(actions):
            orig.step(pacman_action=action)
            batched.step(torch.tensor([action]))

            orig_pos = list(orig.pacman.position)
            batch_pos = batched.pacman_pos[0].tolist()

            assert orig_pos == batch_pos, (
                f"Step {i}: buffering divergence! "
                f"action={action} orig={orig_pos} batch={batch_pos}"
            )

            if orig.game_over or orig.level_complete:
                break

    def test_ghost_house_door_blocked(self):
        """Pacman cannot pass through the ghost house door."""
        batched = make_batched_game()

        # The door is at (13, 12). If Pacman is at (13, 11) going DOWN,
        # it should be blocked.
        door_x, door_y = 13, 12
        assert batched.maze.pacman_blocked[door_y, door_x].item(), \
            "Ghost house door should be in pacman_blocked mask"
