"""Phase 7 parity tests: Action masking."""

import sys
import numpy as np
import torch
import pytest

from tests.conftest import MAZE_FILE, ORIGINAL_MAZE_FILE, PACMANML_DIR

sys.path.insert(0, PACMANML_DIR)
from pacman.core.game import Game as OriginalGame

from engine.batched_game import BatchedGame


def make_games():
    orig = OriginalGame(headless=True)
    orig.load_level(ORIGINAL_MAZE_FILE)
    orig.reset()
    # Remove power pellets to avoid frightened-mode RNG divergence
    for pp in orig.level.power_pellets:
        pp.eaten = True

    batched = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
    batched.power_pellets.zero_()
    return orig, batched


def burn_ready(orig, batched):
    for _ in range(orig.ready_timer):
        orig.step(pacman_action=0)
        batched.step(torch.tensor([0]))


class TestActionMaskParity:
    """Compare action masks step-by-step."""

    def test_action_mask_at_start(self):
        """Action mask matches at starting position (before any movement)."""
        orig, batched = make_games()
        burn_ready(orig, batched)

        orig_mask = orig.get_action_mask()
        # Original applies no-reverse when no ghosts are out
        ghosts_out = any(not g.in_ghost_house for g in orig.ghosts)
        batch_mask = batched.get_action_mask(no_reverse=not ghosts_out)[0].numpy()

        assert np.array_equal(orig_mask, batch_mask), \
            f"Start mask mismatch: orig={orig_mask} batch={batch_mask}"

    def test_action_mask_parity_random_actions(self):
        """Action masks match step-by-step with random actions."""
        np.random.seed(42)
        orig, batched = make_games()
        burn_ready(orig, batched)

        actions = np.random.randint(0, 4, size=2000)
        for i, action in enumerate(actions):
            orig.step(pacman_action=int(action))
            batched.step(torch.tensor([int(action)]))

            orig_mask = orig.get_action_mask()
            ghosts_out = any(not g.in_ghost_house for g in orig.ghosts)
            batch_mask = batched.get_action_mask(no_reverse=not ghosts_out)[0].numpy()

            if not np.array_equal(orig_mask, batch_mask):
                assert False, (
                    f"Step {i}: mask diverged! "
                    f"orig={orig_mask} batch={batch_mask} "
                    f"pos_o={orig.pacman.position} pos_b={batched.pacman_pos[0].tolist()} "
                    f"dir_o={orig.pacman.direction} dir_b={batched.pacman_dir[0].tolist()}"
                )

            if orig.game_over or orig.level_complete:
                break

    def test_action_mask_always_has_valid(self):
        """At least one action is always valid."""
        np.random.seed(99)
        batched = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
        for _ in range(batched.ready_timer[0].item()):
            batched.step(torch.tensor([0]))

        actions = np.random.randint(0, 4, size=1000)
        for i, action in enumerate(actions):
            batched.step(torch.tensor([int(action)]))
            mask = batched.get_action_mask()[0]
            assert mask.any(), f"Step {i}: no valid actions! pos={batched.pacman_pos[0].tolist()}"
            if batched.game_over[0].item():
                break

    def test_action_mask_blocks_walls(self):
        """Actions leading into walls are masked as invalid."""
        batched = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
        for _ in range(batched.ready_timer[0].item()):
            batched.step(torch.tensor([0]))

        mask = batched.get_action_mask()[0]
        px, py = batched.pacman_pos[0].tolist()

        from engine.constants import ACTION_DIRS
        dirs = ACTION_DIRS.tolist()
        for a, (dx, dy) in enumerate(dirs):
            nx, ny = px + dx, py + dy
            if 0 <= nx < 28 and 0 <= ny < 31:
                blocked = batched.maze.pacman_blocked[ny, nx].item()
                if blocked:
                    assert not mask[a].item(), \
                        f"Action {a} leads to wall at ({nx},{ny}) but is marked valid"

    def test_reverse_direction_not_masked_without_no_reverse(self):
        """Reverse direction is NOT masked when no_reverse=False (default)."""
        np.random.seed(42)
        batched = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
        for _ in range(batched.ready_timer[0].item()):
            batched.step(torch.tensor([0]))

        # Step a few times to establish a direction
        for _ in range(5):
            batched.step(torch.tensor([2]))  # LEFT

        dx, dy = batched.pacman_dir[0].tolist()
        if dx == 0 and dy == 0:
            pytest.skip("No direction established")

        mask = batched.get_action_mask(no_reverse=False)[0]
        from engine.constants import ACTION_DIRS
        dirs = ACTION_DIRS.tolist()

        # Find the reverse action — it should be valid (not masked)
        rev_dx, rev_dy = -dx, -dy
        for a, (adx, ady) in enumerate(dirs):
            if adx == rev_dx and ady == rev_dy:
                # Check the tile is actually passable
                px, py = batched.pacman_pos[0].tolist()
                nx, ny = px + rev_dx, py + rev_dy
                if 0 <= nx < 28 and 0 <= ny < 31:
                    if not batched.maze.pacman_blocked[ny, nx].item():
                        assert mask[a].item(), \
                            f"Reverse action {a} should be valid (not masked)"
                break

    def test_no_reverse_masks_reverse(self):
        """With no_reverse=True, the reverse direction is masked."""
        np.random.seed(42)
        batched = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
        for _ in range(batched.ready_timer[0].item()):
            batched.step(torch.tensor([0]))

        # Move LEFT several steps to establish direction
        for _ in range(5):
            batched.step(torch.tensor([2]))  # LEFT

        dx, dy = batched.pacman_dir[0].tolist()
        if dx == 0 and dy == 0:
            pytest.skip("No direction established")

        # Without no_reverse: reverse should be valid
        mask_normal = batched.get_action_mask(no_reverse=False)[0]
        # With no_reverse: reverse should be masked
        mask_no_rev = batched.get_action_mask(no_reverse=True)[0]

        from engine.constants import ACTION_DIRS
        dirs = ACTION_DIRS.tolist()
        rev_dx, rev_dy = -dx, -dy
        for a, (adx, ady) in enumerate(dirs):
            if adx == rev_dx and ady == rev_dy:
                px, py = batched.pacman_pos[0].tolist()
                nx, ny = px + rev_dx, py + rev_dy
                if 0 <= nx < 28 and 0 <= ny < 31:
                    if not batched.maze.pacman_blocked[ny, nx].item():
                        assert mask_normal[a].item(), "Reverse should be valid without no_reverse"
                        assert not mask_no_rev[a].item(), "Reverse should be masked with no_reverse"
                break

    def test_no_reverse_always_has_valid(self):
        """With no_reverse=True, at least one action is always valid.

        If reverse is the only valid move, it should NOT be masked.
        """
        np.random.seed(99)
        batched = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
        for _ in range(batched.ready_timer[0].item()):
            batched.step(torch.tensor([0]))

        actions = np.random.randint(0, 4, size=1000)
        for i, action in enumerate(actions):
            batched.step(torch.tensor([int(action)]))
            mask = batched.get_action_mask(no_reverse=True)[0]
            assert mask.any(), f"Step {i}: no valid actions with no_reverse! pos={batched.pacman_pos[0].tolist()}"
            if batched.game_over[0].item():
                break

    def test_no_reverse_no_direction_allows_all(self):
        """When Pacman has no direction (0,0), no_reverse has no effect."""
        batched = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
        for _ in range(batched.ready_timer[0].item()):
            batched.step(torch.tensor([0]))

        # At start, direction is (0,0)
        assert batched.pacman_dir[0].tolist() == [0, 0]
        mask_normal = batched.get_action_mask(no_reverse=False)[0]
        mask_no_rev = batched.get_action_mask(no_reverse=True)[0]
        assert torch.equal(mask_normal, mask_no_rev), \
            "no_reverse should have no effect when direction is (0,0)"

    def test_batch_mask_consistency(self):
        """N=4 batch produces same masks as N=1 for same positions."""
        np.random.seed(42)
        single = BatchedGame(n_envs=1, maze_file=MAZE_FILE)
        multi = BatchedGame(n_envs=4, maze_file=MAZE_FILE)

        for _ in range(single.ready_timer[0].item()):
            single.step(torch.tensor([0]))
            multi.step(torch.tensor([0, 0, 0, 0]))

        actions = np.random.randint(0, 4, size=100)
        for i, action in enumerate(actions):
            single.step(torch.tensor([int(action)]))
            multi.step(torch.tensor([int(action)] * 4))

            s_mask = single.get_action_mask()[0]
            for env in range(4):
                m_mask = multi.get_action_mask()[env]
                assert torch.equal(s_mask, m_mask), \
                    f"Step {i}, env {env}: batch mask mismatch"

            if single.game_over[0].item():
                break
