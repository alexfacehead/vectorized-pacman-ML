"""Phase 3 parity tests: Visit heatmap, proximity reward, distance matrix.

Compares visit map, proximity rewards, and overall reward signals.
"""

import sys
import numpy as np
import torch
import pytest

from tests.conftest import MAZE_FILE, ORIGINAL_MAZE_FILE, PACMANML_DIR

sys.path.insert(0, PACMANML_DIR)
from pacman.core.game import Game as OriginalGame

from engine.batched_game import BatchedGame
from engine.distances import precompute_distances
from engine.maze import Maze


def make_original_game() -> OriginalGame:
    game = OriginalGame(headless=True)
    game.load_level(ORIGINAL_MAZE_FILE)
    game.reset()
    return game


def make_batched_game(n_envs: int = 1) -> BatchedGame:
    return BatchedGame(n_envs=n_envs, maze_file=MAZE_FILE)


def burn_ready(orig, batched):
    for _ in range(orig.ready_timer):
        orig.step(pacman_action=0)
        batched.step(torch.tensor([0]))


class TestDistanceMatrix:
    """Validate the precomputed distance matrix against BFS."""

    def test_distance_matrix_self_zero(self):
        """Distance from any tile to itself is 0."""
        maze = Maze(MAZE_FILE)
        dist = precompute_distances(maze)
        for i in range(maze.num_walkable):
            assert dist[i, i] == 0

    def test_distance_matrix_symmetric(self):
        """Distance matrix is symmetric."""
        maze = Maze(MAZE_FILE)
        dist = precompute_distances(maze)
        assert (dist == dist.T).all(), "Distance matrix is not symmetric"

    def test_distance_matrix_adjacent(self):
        """Adjacent walkable tiles have distance 1."""
        maze = Maze(MAZE_FILE)
        dist = precompute_distances(maze)

        # Check a known pair: (1, 1) and (2, 1) should both be walkable and adjacent
        # From maze: row 1 is "#............##............#"
        # (1,1) is '.', (2,1) is '.'
        idx_a = maze.tile_to_idx[1, 1].item()
        idx_b = maze.tile_to_idx[1, 2].item()
        assert idx_a >= 0 and idx_b >= 0
        assert dist[idx_a, idx_b] == 1

    def test_distance_matches_bfs(self):
        """Spot-check: precomputed distance matches BFS from original engine."""
        orig = make_original_game()
        maze = Maze(MAZE_FILE)
        dist_matrix = precompute_distances(maze)

        # Check distance from Pacman start to a few known pellet positions
        px, py = orig.pacman.position

        # BFS from original
        bfs_dist = orig._bfs_nearest_pellet()

        # Precomputed: find nearest pellet using distance matrix
        pac_idx = maze.tile_to_idx[py, px].item()
        assert pac_idx >= 0

        # Get all uneaten pellet positions
        min_dist = 9999
        for p in orig.level.pellets:
            if not p.eaten:
                x, y = p.position
                tidx = maze.tile_to_idx[y, x].item()
                if tidx >= 0:
                    d = dist_matrix[pac_idx, tidx].item()
                    min_dist = min(min_dist, d)
        for p in orig.level.power_pellets:
            if not p.eaten:
                x, y = p.position
                tidx = maze.tile_to_idx[y, x].item()
                if tidx >= 0:
                    d = dist_matrix[pac_idx, tidx].item()
                    min_dist = min(min_dist, d)

        assert bfs_dist == min_dist, f"BFS={bfs_dist} vs precomputed={min_dist}"

    def test_tunnel_wrapping_in_distances(self):
        """Tunnel tiles are connected in the distance matrix."""
        maze = Maze(MAZE_FILE)
        dist = precompute_distances(maze)

        ty = maze.tunnel_y
        # Left end (0, tunnel_y) and right end (27, tunnel_y) should be distance 1
        idx_left = maze.tile_to_idx[ty, 0].item()
        idx_right = maze.tile_to_idx[ty, 27].item()
        if idx_left >= 0 and idx_right >= 0:
            assert dist[idx_left, idx_right] == 1, \
                f"Tunnel endpoints should be adjacent, got dist={dist[idx_left, idx_right]}"


class TestVisitMapParity:
    """Compare visit heatmap behavior between engines."""

    def test_visit_map_decay(self):
        """Visit map decays by 0.85 each active step."""
        batched = make_batched_game()

        # Burn ready first
        for _ in range(batched.ready_timer[0].item()):
            batched.step(torch.tensor([0]))

        # Now set a visit value at a non-Pacman tile
        batched.visit_map[0, 1, 1] = 1.0

        # One active step should decay it
        batched.step(torch.tensor([0]))

        expected = 1.0 * 0.85
        actual = batched.visit_map[0, 1, 1].item()
        assert abs(actual - expected) < 1e-5, f"Visit decay: expected {expected}, got {actual}"

    def test_visit_map_stamp(self):
        """Pacman's position is stamped to 1.0 in visit map."""
        batched = make_batched_game()

        # Burn ready
        for _ in range(batched.ready_timer[0].item()):
            batched.step(torch.tensor([0]))

        # One active step stamps Pacman's position
        batched.step(torch.tensor([0]))

        px, py = batched.pacman_pos[0].tolist()
        assert batched.visit_map[0, py, px].item() == 1.0, "Pacman position should be stamped"

    def test_visit_map_parity(self):
        """Visit map matches original engine step-by-step."""
        np.random.seed(42)
        orig = make_original_game()
        batched = make_batched_game()

        # Remove power pellets to avoid frightened-mode RNG divergence
        for pp in orig.level.power_pellets:
            pp.eaten = True
        batched.power_pellets.zero_()

        burn_ready(orig, batched)

        actions = np.random.randint(0, 4, size=500)
        for i, action in enumerate(actions):
            orig.step(pacman_action=int(action))
            batched.step(torch.tensor([int(action)]))

            # Compare visit maps
            orig_visit = orig._visit_map
            batch_visit = batched.visit_map[0].numpy()

            if not np.allclose(orig_visit, batch_visit, atol=1e-5):
                max_diff = np.abs(orig_visit - batch_visit).max()
                diff_pos = np.unravel_index(np.argmax(np.abs(orig_visit - batch_visit)), orig_visit.shape)
                assert False, (
                    f"Step {i}: visit map diverged! "
                    f"max_diff={max_diff:.6f} at pos={diff_pos}"
                )

            if orig.game_over or orig.level_complete:
                break


class TestRewardParity:
    """Compare full reward signal between engines."""

    def test_reward_parity_with_visit_and_proximity(self):
        """Raw rewards match step-by-step (all components active)."""
        np.random.seed(42)
        orig = make_original_game()
        batched = make_batched_game()

        # Remove power pellets to avoid frightened-mode RNG divergence
        for pp in orig.level.power_pellets:
            pp.eaten = True
        batched.power_pellets.zero_()

        burn_ready(orig, batched)

        actions = np.random.randint(0, 4, size=3000)
        max_diff_seen = 0.0

        for i, action in enumerate(actions):
            orig.step(pacman_action=int(action))
            batched.step(torch.tensor([int(action)]))

            orig_reward = orig._reward_pacman
            batch_reward = batched.reward_pacman[0].item()

            diff = abs(orig_reward - batch_reward)
            max_diff_seen = max(max_diff_seen, diff)

            assert diff < 1e-4, (
                f"Step {i}: reward diverged! "
                f"orig={orig_reward:.6f} batch={batch_reward:.6f} diff={diff:.6f} "
                f"action={action} pos={batched.pacman_pos[0].tolist()}"
            )

            if orig.game_over or orig.level_complete:
                break

    def test_clipped_reward_matches(self):
        """Clipped rewards [-1, 1] match."""
        np.random.seed(42)
        orig = make_original_game()
        batched = make_batched_game()

        # Remove power pellets to avoid frightened-mode RNG divergence
        for pp in orig.level.power_pellets:
            pp.eaten = True
        batched.power_pellets.zero_()

        burn_ready(orig, batched)

        actions = np.random.randint(0, 4, size=1000)
        for i, action in enumerate(actions):
            orig.step(pacman_action=int(action))
            batched.step(torch.tensor([int(action)]))

            orig_clipped = orig.get_reward_pacman()
            batch_clipped = batched.get_reward_pacman()[0].item()

            diff = abs(orig_clipped - batch_clipped)
            assert diff < 1e-4, (
                f"Step {i}: clipped reward diverged! "
                f"orig={orig_clipped:.6f} batch={batch_clipped:.6f}"
            )

            if orig.game_over or orig.level_complete:
                break
