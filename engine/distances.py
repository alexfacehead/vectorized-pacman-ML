"""Precomputed all-pairs shortest path distances via Floyd-Warshall.

Computed once per maze. Replaces per-step BFS for nearest pellet distance.
"""

import torch

from .constants import COLS
from .maze import Maze


def precompute_distances(maze: Maze) -> torch.Tensor:
    """Compute all-pairs shortest path distances for walkable tiles.

    Uses Floyd-Warshall on the ~400 walkable tiles. One-time cost.

    Args:
        maze: Maze object with walls, tunnel_y, tile_to_idx.

    Returns:
        dist_matrix: (num_walkable, num_walkable) int16 tensor.
        Distance of 9999 means unreachable.
    """
    n = maze.num_walkable
    INF = 9999

    # Initialize distance matrix
    dist = torch.full((n, n), INF, dtype=torch.int16)
    for i in range(n):
        dist[i, i] = 0

    # Build adjacency: for each walkable tile, check 4 neighbors
    for i in range(n):
        x = maze.idx_to_tile[i, 0].item()
        y = maze.idx_to_tile[i, 1].item()

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy

            # Tunnel wrapping
            if ny == maze.tunnel_y:
                if nx < 0:
                    nx = COLS - 1
                elif nx >= COLS:
                    nx = 0

            # Check bounds
            if ny < 0 or ny >= maze.walls.shape[0]:
                continue
            if nx < 0 or nx >= COLS:
                continue

            # Check if walkable
            j = maze.tile_to_idx[ny, nx].item()
            if j >= 0:
                dist[i, j] = 1

    # Floyd-Warshall
    for k in range(n):
        # dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        via_k = dist[:, k:k+1] + dist[k:k+1, :]  # (n, n)
        dist = torch.minimum(dist, via_k.to(torch.int16))

    return dist
