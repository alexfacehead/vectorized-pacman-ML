"""Phase 6 parity tests: All four ghosts with personality targeting.

Tests all ghosts active simultaneously with staggered exits.
Power pellets removed to avoid frightened RNG divergence.
"""

import sys
import numpy as np
import torch
import pytest

from tests.conftest import MAZE_FILE, ORIGINAL_MAZE_FILE, PACMANML_DIR

sys.path.insert(0, PACMANML_DIR)
from pacman.core.game import Game as OriginalGame

from engine.batched_game import BatchedGame
from engine.constants import SCATTER, CHASE


def make_games_all_ghosts(exit_timers=None):
    """Create both engines with all ghosts released.

    Args:
        exit_timers: list of 4 ints, or None for staggered (index * 300).
    """
    if exit_timers is None:
        exit_timers = [i * 300 for i in range(4)]

    orig = OriginalGame(headless=True)
    orig.load_level(ORIGINAL_MAZE_FILE)
    orig.reset()
    for i, timer in enumerate(exit_timers):
        orig.ghosts[i].exit_timer = timer

    # Remove power pellets from original
    for pp in orig.level.power_pellets:
        pp.eaten = True

    batched = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
    for i, timer in enumerate(exit_timers):
        batched.ghost_exit_timer[0, i] = timer
    batched.power_pellets.zero_()

    return orig, batched


def burn_ready(orig, batched):
    for _ in range(orig.ready_timer):
        orig.step(pacman_action=0)
        batched.step(torch.tensor([0]))


class TestAllGhostsPositionParity:
    """Compare all 4 ghost positions step-by-step."""

    def test_all_ghosts_position_parity_staggered(self):
        """All ghost positions match with staggered exit timers."""
        np.random.seed(42)
        orig, batched = make_games_all_ghosts()
        burn_ready(orig, batched)

        actions = np.random.randint(0, 4, size=2000)
        for i, action in enumerate(actions):
            orig.step(pacman_action=int(action))
            batched.step(torch.tensor([int(action)]))

            for gi in range(4):
                orig_pos = list(orig.ghosts[gi].position)
                batch_pos = batched.ghost_pos[0, gi].tolist()

                if orig_pos != batch_pos:
                    assert False, (
                        f"Step {i}: Ghost {gi} position diverged! "
                        f"orig={orig_pos} batch={batch_pos} "
                        f"orig_state={orig.ghosts[gi].state} "
                        f"batch_state={batched.ghost_state[0, gi].item()} "
                        f"orig_in_house={orig.ghosts[gi].in_ghost_house} "
                        f"batch_in_house={batched.ghost_in_house[0, gi].item()}"
                    )

            if orig.game_over or orig.level_complete:
                break

    def test_all_ghosts_position_parity_immediate(self):
        """All ghost positions match with all ghosts released immediately."""
        np.random.seed(99)
        orig, batched = make_games_all_ghosts(exit_timers=[0, 0, 0, 0])
        burn_ready(orig, batched)

        actions = np.random.randint(0, 4, size=1500)
        for i, action in enumerate(actions):
            orig.step(pacman_action=int(action))
            batched.step(torch.tensor([int(action)]))

            for gi in range(4):
                orig_pos = list(orig.ghosts[gi].position)
                batch_pos = batched.ghost_pos[0, gi].tolist()

                if orig_pos != batch_pos:
                    assert False, (
                        f"Step {i}: Ghost {gi} position diverged! "
                        f"orig={orig_pos} batch={batch_pos}"
                    )

            if orig.game_over or orig.level_complete:
                break


class TestAllGhostsStateParity:
    """Compare ghost states step-by-step."""

    def test_all_ghosts_state_parity(self):
        """Ghost states (scatter/chase) match for all 4 ghosts."""
        np.random.seed(42)
        orig, batched = make_games_all_ghosts()
        burn_ready(orig, batched)

        actions = np.random.randint(0, 4, size=1000)
        for i, action in enumerate(actions):
            orig.step(pacman_action=int(action))
            batched.step(torch.tensor([int(action)]))

            for gi in range(4):
                orig_state = orig.ghosts[gi].state
                batch_state = batched.ghost_state[0, gi].item()

                if orig_state != batch_state:
                    assert False, (
                        f"Step {i}: Ghost {gi} state diverged! "
                        f"orig={orig_state} batch={batch_state} "
                        f"orig_in_house={orig.ghosts[gi].in_ghost_house} "
                        f"batch_in_house={batched.ghost_in_house[0, gi].item()}"
                    )

            if orig.game_over or orig.level_complete:
                break


class TestStaggeredExit:
    """Test ghosts exit house in the correct staggered order."""

    def test_staggered_exit_order(self):
        """Ghosts exit house in index order with staggered timers."""
        np.random.seed(7)
        batched = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
        for i in range(4):
            batched.ghost_exit_timer[0, i] = i * 300
            batched.ghost_speed[0, i] = 3  # uniform speed for this test
        batched.power_pellets.zero_()

        # Burn ready
        for _ in range(batched.ready_timer[0].item()):
            batched.step(torch.tensor([0]))

        exit_order = []
        was_in_house = [True] * 4
        actions = np.random.randint(0, 4, size=1200)
        for step, action in enumerate(actions):
            batched.step(torch.tensor([int(action)]))
            for gi in range(4):
                if was_in_house[gi] and not batched.ghost_in_house[0, gi].item():
                    exit_order.append(gi)
                    was_in_house[gi] = False
            if batched.game_over[0].item():
                break

        # At minimum, exits we observed should be in order
        assert len(exit_order) >= 2, f"Expected at least 2 exits, got {exit_order}"
        assert exit_order == sorted(exit_order), \
            f"Exit order should be ascending, got {exit_order}"

    def test_staggered_exit_parity(self):
        """Ghost house exit timing matches original."""
        np.random.seed(42)
        orig, batched = make_games_all_ghosts()
        burn_ready(orig, batched)

        actions = np.random.randint(0, 4, size=1200)
        for i, action in enumerate(actions):
            orig.step(pacman_action=int(action))
            batched.step(torch.tensor([int(action)]))

            for gi in range(4):
                orig_in = orig.ghosts[gi].in_ghost_house
                batch_in = batched.ghost_in_house[0, gi].item()

                if orig_in != batch_in:
                    assert False, (
                        f"Step {i}: Ghost {gi} in_house diverged! "
                        f"orig={orig_in} batch={batch_in} "
                        f"orig_timer={orig.ghosts[gi].exit_timer} "
                        f"batch_timer={batched.ghost_exit_timer[0, gi].item()}"
                    )

            if orig.game_over or orig.level_complete:
                break


class TestAllGhostsCollision:
    """Test collision detection with all ghosts active."""

    def test_game_over_parity_all_ghosts(self):
        """Game over events match with all 4 ghosts active."""
        np.random.seed(42)
        orig, batched = make_games_all_ghosts()
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
                    f"orig={orig_over} batch={batch_over}"
                )

            if orig_over:
                break

    def test_score_parity_all_ghosts(self):
        """Score matches with all 4 ghosts active."""
        np.random.seed(42)
        orig, batched = make_games_all_ghosts()
        burn_ready(orig, batched)

        actions = np.random.randint(0, 4, size=2000)
        for i, action in enumerate(actions):
            orig.step(pacman_action=int(action))
            batched.step(torch.tensor([int(action)]))

            if orig.score != batched.score[0].item():
                assert False, (
                    f"Step {i}: score diverged! "
                    f"orig={orig.score} batch={batched.score[0].item()}"
                )

            if orig.game_over or orig.level_complete:
                break


class TestGhostTargeting:
    """Test individual ghost targeting logic."""

    def test_pinky_aims_ahead(self):
        """Pinky targets 4 tiles ahead of Pacman in chase mode."""
        from engine.ghosts import compute_chase_targets

        pacman_pos = torch.tensor([[14, 23]], dtype=torch.int32)  # arbitrary
        pacman_dir = torch.tensor([[1, 0]], dtype=torch.int32)    # moving RIGHT
        ghost_pos = torch.zeros(1, 4, 2, dtype=torch.int32)
        scatter = torch.zeros(4, 2, dtype=torch.int32)

        targets = compute_chase_targets(pacman_pos, pacman_dir, ghost_pos, scatter, torch.device('cpu'))
        # Pinky target = pacman_pos + dir * 4 = [14+4, 23+0] = [18, 23]
        assert targets[0, 1].tolist() == [18, 23]

    def test_blinky_chases_pacman(self):
        """Blinky targets Pacman's position directly."""
        from engine.ghosts import compute_chase_targets

        pacman_pos = torch.tensor([[10, 15]], dtype=torch.int32)
        pacman_dir = torch.tensor([[0, -1]], dtype=torch.int32)
        ghost_pos = torch.zeros(1, 4, 2, dtype=torch.int32)
        scatter = torch.zeros(4, 2, dtype=torch.int32)

        targets = compute_chase_targets(pacman_pos, pacman_dir, ghost_pos, scatter, torch.device('cpu'))
        assert targets[0, 0].tolist() == [10, 15]

    def test_clyde_switches_target(self):
        """Clyde targets Pacman when far, scatter target when close."""
        from engine.ghosts import compute_chase_targets

        pacman_pos = torch.tensor([[14, 23]], dtype=torch.int32)
        pacman_dir = torch.tensor([[1, 0]], dtype=torch.int32)
        scatter = torch.zeros(4, 2, dtype=torch.int32)
        scatter[3] = torch.tensor([0, 31])  # Clyde scatter corner

        # Clyde far from Pacman (dist > 8)
        ghost_pos = torch.zeros(1, 4, 2, dtype=torch.int32)
        ghost_pos[0, 3] = torch.tensor([1, 1])  # far away
        targets = compute_chase_targets(pacman_pos, pacman_dir, ghost_pos, scatter, torch.device('cpu'))
        assert targets[0, 3].tolist() == pacman_pos[0].tolist(), \
            "Clyde should target Pacman when far away"

        # Clyde close to Pacman (dist <= 8)
        ghost_pos[0, 3] = torch.tensor([14, 20])  # dist = |0| + |3| = 3 ≤ 8
        targets = compute_chase_targets(pacman_pos, pacman_dir, ghost_pos, scatter, torch.device('cpu'))
        assert targets[0, 3].tolist() == [0, 31], \
            "Clyde should scatter when close to Pacman"
