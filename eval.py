"""Headless evaluation: run N games with a trained model and print stats.

Usage:
    python eval.py --model checkpoints/pacman_agent.pt --stage 5 --games 100
"""

import argparse
import os
import torch
import numpy as np

MAZE_FILE = os.path.join(os.path.dirname(__file__), "levels", "level_1.txt")


def evaluate(model_path: str, stage: int, n_games: int, max_steps: int = 3500, epsilon: float = 0.01):
    from engine.batched_game import BatchedGame
    from engine.constants import NUM_STATE_CHANNELS
    from training.train import BatchedPacmanAgent

    # Load model
    agent = BatchedPacmanAgent(in_channels=NUM_STATE_CHANNELS, device="cpu")
    agent.load(model_path)
    agent.epsilon = epsilon
    print(f"Loaded model: {model_path}")
    print(f"Stage: {stage} | Games: {n_games} | Max steps: {max_steps}")
    print("-" * 50)

    # Run games in batches
    batch_size = min(n_games, 128)
    total_wins = 0
    total_losses = 0
    total_scores = []
    total_pellets = []
    total_steps_list = []
    games_done = 0

    while games_done < n_games:
        this_batch = min(batch_size, n_games - games_done)
        game = BatchedGame(n_envs=this_batch, maze_file=MAZE_FILE)
        game.configure_stage(stage)

        steps = 0
        batch_done = torch.zeros(this_batch, dtype=torch.bool)
        hidden = agent.init_hidden(this_batch)

        while steps < max_steps and not batch_done.all():
            state = game.get_state()
            action_mask = game.get_action_mask(no_reverse=True)
            actions, hidden = agent.select_actions_batched(state, action_mask, hidden)
            rewards, dones, infos = game.step(actions)
            steps += 1

            # Track which games just finished
            newly_done = (game.game_over | game.level_complete) & ~batch_done
            batch_done = batch_done | game.game_over | game.level_complete

        # Collect stats
        wins = game.level_complete.sum().item()
        losses = (game.game_over & ~game.level_complete).sum().item()
        timeouts = this_batch - wins - losses
        total_wins += wins
        total_losses += losses

        for i in range(this_batch):
            total_scores.append(game.score[i].item())
            eaten = game.maze.total_pellets - (game.pellets[i].sum().item() + game.power_pellets[i].sum().item())
            total_pellets.append(int(eaten))

        games_done += this_batch
        batch_pct = 100 * wins / this_batch
        print(f"  Batch {games_done}/{n_games}: W {wins} / L {losses} / TO {timeouts} [{batch_pct:.0f}% win] | Avg score {np.mean(total_scores[-this_batch:]):.0f} | Avg pellets {np.mean(total_pellets[-this_batch:]):.0f}/299")

    # Final summary
    total = total_wins + total_losses + (n_games - total_wins - total_losses)
    win_pct = 100 * total_wins / n_games
    avg_score = np.mean(total_scores)
    avg_pellets = np.mean(total_pellets)

    print("=" * 50)
    print(f"  RESULTS ({n_games} games, stage {stage})")
    print(f"  Win rate   : {total_wins}/{n_games} ({win_pct:.1f}%)")
    print(f"  Avg score  : {avg_score:.0f}")
    print(f"  Avg pellets: {avg_pellets:.0f}/299")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Pac-Man DQN evaluation")
    parser.add_argument("--model", type=str, default="checkpoints/pacman_agent.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--stage", type=int, default=5, choices=[1, 2, 3, 4, 5, 6, 7],
                        help="Curriculum stage")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games to evaluate")
    parser.add_argument("--max-steps", type=int, default=3500,
                        help="Max steps per game")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Exploration epsilon (default: 0.01, use 0 for pure greedy)")
    args = parser.parse_args()
    evaluate(args.model, args.stage, args.games, args.max_steps, args.epsilon)
