"""Phase 4 parity tests: Ghost movement, AI, house exit, collision detection.

Compares Blinky ghost behavior between original and vectorized engines.
All other ghosts remain frozen (exit_timer=999999) to isolate Blinky.

Note: Frightened mode uses random movement. The original uses Python's
random.choice while the batched uses torch.rand — different RNG sources.
Position parity tests stop if a power pellet is eaten to avoid this issue.
"""

import sys
import numpy as np
import torch
import pytest

from tests.conftest import MAZE_FILE, ORIGINAL_MAZE_FILE, PACMANML_DIR

sys.path.insert(0, PACMANML_DIR)
from pacman.core.game import Game as OriginalGame

from engine.batched_game import BatchedGame
from engine.constants import SCATTER, CHASE, FRIGHTENED, EATEN


def make_original_game(release_blinky: bool = False) -> OriginalGame:
    game = OriginalGame(headless=True)
    game.load_level(ORIGINAL_MAZE_FILE)
    game.reset()
    if release_blinky:
        game.ghosts[0].exit_timer = 0
    return game


def make_batched_game(release_blinky: bool = False, remove_power_pellets: bool = False) -> BatchedGame:
    bg = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
    if release_blinky:
        bg.ghost_exit_timer[0, 0] = 0
    if remove_power_pellets:
        bg.power_pellets.zero_()
    return bg


def burn_ready(orig, batched):
    for _ in range(orig.ready_timer):
        orig.step(pacman_action=0)
        batched.step(torch.tensor([0]))


def burn_ready_batched(batched):
    for _ in range(batched.ready_timer[0].item()):
        batched.step(torch.tensor([0]))


class TestGhostHouseExit:
    """Test ghost house exit mechanics."""

    def test_ghost_starts_in_house(self):
        batched = make_batched_game()
        assert batched.ghost_in_house.all()

    def test_ghost_frozen_with_large_timer(self):
        batched = make_batched_game()
        # Freeze Blinky too (override curriculum default)
        batched.ghost_exit_timer[0, 0] = 999999
        burn_ready_batched(batched)
        for _ in range(100):
            batched.step(torch.tensor([0]))
        assert batched.ghost_in_house.all()

    def test_blinky_exits_house(self):
        batched = make_batched_game(release_blinky=True)
        burn_ready_batched(batched)
        for _ in range(10):
            batched.step(torch.tensor([0]))
        assert not batched.ghost_in_house[0, 0].item()
        assert batched.ghost_in_house[0, 1:].all()

    def test_blinky_exit_parity(self):
        """Blinky exits at the same step in both engines."""
        orig = make_original_game(release_blinky=True)
        batched = make_batched_game(release_blinky=True)

        burn_ready(orig, batched)

        for i in range(20):
            orig.step(pacman_action=0)
            batched.step(torch.tensor([0]))

            orig_in = orig.ghosts[0].in_ghost_house
            batch_in = batched.ghost_in_house[0, 0].item()

            assert orig_in == batch_in, (
                f"Step {i}: house state diverged! orig={orig_in} batch={batch_in}"
            )

            if not orig_in:
                # Both exited — check position matches
                assert list(orig.ghosts[0].position) == batched.ghost_pos[0, 0].tolist(), \
                    f"Step {i}: exit position mismatch"
                break


class TestGhostMovementParity:
    """Compare Blinky movement (non-frightened only)."""

    def test_blinky_position_no_power_pellets(self):
        """Blinky position matches with power pellets removed (no frightened mode)."""
        np.random.seed(42)

        orig = make_original_game(release_blinky=True)
        # Remove power pellets from original too
        for pp in orig.level.power_pellets:
            pp.eaten = True

        batched = make_batched_game(release_blinky=True, remove_power_pellets=True)

        burn_ready(orig, batched)

        actions = np.random.randint(0, 4, size=2000)
        for i, action in enumerate(actions):
            orig.step(pacman_action=int(action))
            batched.step(torch.tensor([int(action)]))

            orig_b = orig.ghosts[0]
            batch_pos = batched.ghost_pos[0, 0].tolist()
            orig_pos = list(orig_b.position)

            if batch_pos != orig_pos:
                assert False, (
                    f"Step {i}: Blinky position diverged! "
                    f"batch={batch_pos} orig={orig_pos} "
                    f"orig_dir={orig_b.direction} batch_dir={batched.ghost_dir[0,0].tolist()} "
                    f"orig_state={orig_b.state} batch_state={batched.ghost_state[0,0].item()}"
                )

            if orig.game_over or orig.level_complete:
                break

    def test_blinky_state_parity(self):
        """Blinky state (scatter/chase) matches step-by-step (no power pellets)."""
        np.random.seed(42)
        orig = make_original_game(release_blinky=True)
        for pp in orig.level.power_pellets:
            pp.eaten = True
        batched = make_batched_game(release_blinky=True, remove_power_pellets=True)

        burn_ready(orig, batched)

        actions = np.random.randint(0, 4, size=500)
        for i, action in enumerate(actions):
            orig.step(pacman_action=int(action))
            batched.step(torch.tensor([int(action)]))

            orig_state = orig.ghosts[0].state
            batch_state = batched.ghost_state[0, 0].item()

            if orig_state != batch_state:
                assert False, (
                    f"Step {i}: Blinky state diverged! "
                    f"batch={batch_state} orig={orig_state}"
                )

            if orig.game_over or orig.level_complete:
                break


class TestCollisionDetection:
    """Test ghost-Pacman collision detection."""

    def test_game_over_parity_no_power(self):
        """Game over events match (no power pellets)."""
        np.random.seed(42)
        orig = make_original_game(release_blinky=True)
        for pp in orig.level.power_pellets:
            pp.eaten = True
        batched = make_batched_game(release_blinky=True, remove_power_pellets=True)

        burn_ready(orig, batched)

        actions = np.random.randint(0, 4, size=3000)
        for i, action in enumerate(actions):
            orig.step(pacman_action=int(action))
            batched.step(torch.tensor([int(action)]))

            orig_over = orig.game_over
            batch_over = batched.game_over[0].item()

            if orig_over != batch_over:
                assert False, (
                    f"Step {i}: game_over diverged! "
                    f"batch={batch_over} orig={orig_over} "
                    f"pac_pos_o={orig.pacman.position} pac_pos_b={batched.pacman_pos[0].tolist()} "
                    f"blinky_pos_o={orig.ghosts[0].position} blinky_pos_b={batched.ghost_pos[0,0].tolist()}"
                )

            if orig_over:
                break

    def test_score_parity_no_power(self):
        """Score matches with ghost active (no power pellets)."""
        np.random.seed(42)
        orig = make_original_game(release_blinky=True)
        for pp in orig.level.power_pellets:
            pp.eaten = True
        batched = make_batched_game(release_blinky=True, remove_power_pellets=True)

        burn_ready(orig, batched)

        actions = np.random.randint(0, 4, size=1000)
        for i, action in enumerate(actions):
            orig.step(pacman_action=int(action))
            batched.step(torch.tensor([int(action)]))

            if orig.score != batched.score[0].item():
                assert False, (
                    f"Step {i}: score diverged! "
                    f"batch={batched.score[0].item()} orig={orig.score}"
                )

            if orig.game_over or orig.level_complete:
                break


class TestModeTimer:
    """Test scatter/chase mode cycling."""

    def test_mode_parity(self):
        """Scatter/Chase mode matches step-by-step."""
        np.random.seed(42)
        orig = make_original_game(release_blinky=True)
        for pp in orig.level.power_pellets:
            pp.eaten = True
        batched = make_batched_game(release_blinky=True, remove_power_pellets=True)

        burn_ready(orig, batched)

        actions = np.random.randint(0, 4, size=500)
        for i, action in enumerate(actions):
            orig.step(pacman_action=int(action))
            batched.step(torch.tensor([int(action)]))

            if orig.current_mode != batched.current_mode[0].item():
                assert False, (
                    f"Step {i}: mode diverged! "
                    f"batch={batched.current_mode[0].item()} orig={orig.current_mode}"
                )

            if orig.game_over or orig.level_complete:
                break


class TestRewardWithGhost:
    """Test rewards with ghost active (no power pellets to avoid RNG divergence)."""

    def test_reward_parity_no_power(self):
        """Rewards match with ghost active (no power pellets)."""
        np.random.seed(42)
        orig = make_original_game(release_blinky=True)
        for pp in orig.level.power_pellets:
            pp.eaten = True
        batched = make_batched_game(release_blinky=True, remove_power_pellets=True)

        burn_ready(orig, batched)

        actions = np.random.randint(0, 4, size=500)
        for i, action in enumerate(actions):
            orig.step(pacman_action=int(action))
            batched.step(torch.tensor([int(action)]))

            diff = abs(orig._reward_pacman - batched.reward_pacman[0].item())

            if diff > 1e-4:
                assert False, (
                    f"Step {i}: reward diverged! "
                    f"orig={orig._reward_pacman:.6f} batch={batched.reward_pacman[0].item():.6f} "
                    f"diff={diff:.6f}"
                )

            if orig.game_over or orig.level_complete:
                break
