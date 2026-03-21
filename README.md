# Vectorized Pac-Man DRQN

A Dueling Deep Recurrent Q-Network agent that learns to play Pac-Man through curriculum learning, trained on a custom vectorized engine that runs 128 games simultaneously on a single MacBook.

Inspired by [DeepMind's original DQN work](https://www.nature.com/articles/nature14236) on Atari games. Unlike DeepMind's approach (raw pixel input, reward clipping, no domain knowledge), this project uses structured semantic state channels and carefully shaped rewards — trading generality for sample efficiency and trainability on consumer hardware. The agent learns maze navigation, pellet collection, ghost evasion, and multi-ghost strategy entirely through reinforcement learning with no hardcoded pathfinding.

<p align="center">
  <img src="assets/pacman-dqn-demo.gif" alt="DRQN agent playing Pac-Man against 4 ghosts" width="400">
  <br>
  <em>DRQN agent vs. 4 ghosts (all at half speed) — 84.2% win rate evaluated over 3,072 games at greedy policy (epsilon = 0). Trained in ~243,000 games across a 6-stage curriculum on Apple Silicon.</em>
</p>

## Results

All eval win rates measured at epsilon = 0 (pure greedy policy) over 512+ games unless noted.

| Stage | Ghosts | Ghost Speed | Episodes | Eval Win Rate |
|-------|--------|-------------|----------|---------------|
| 1 | None | -- | 225 | 100% |
| 2 | Blinky | 50% | 150 | 90%+ |
| 5 | Blinky + Pinky + Inky | 50% | 428 | 83% |
| 6 | All 4 ghosts | 50% | 1,900 | **84.2%** (n=3072) |

**Total: ~1,900 episodes / ~243,000 games.** Each episode = 128 parallel games.

### Key Milestone: No-Reverse Mask Removed

The DRQN with LSTM is the first architecture in this project to succeed without the anti-oscillation reverse mask. Previous DQN architectures required hard-masking the reverse direction to prevent pathological oscillation. The LSTM's temporal memory handles this naturally -- the agent remembers "I was just going left" and commits to directions without explicit action-space constraints. Oscillation is near-zero under greedy play; minor stuttering occurs only in genuinely ambiguous multi-ghost situations.

### Previous Architecture (DQN) Comparison

The flat DQN (no LSTM) achieved ~70% against 1 ghost but collapsed to 40-50% against 3 ghosts. The DRQN surpassed this ceiling within 120 episodes of Stage 5 training, demonstrating that temporal reasoning is essential for multi-ghost evasion.

## Architecture

**Dueling DRQN (Double DQN + LSTM)** with episode-based sequence replay and a target network.

**Model:** 3-layer CNN (32 -> 64 -> 128 filters, stride-2 on layer 3) -> FC (24,960 -> 512) -> LSTM (512 hidden) -> Dueling heads (value stream + advantage stream). ~13M parameters.

```
Input (9, 31, 28) -> Conv 3x3 (32) -> Conv 3x3 (64) -> Conv 3x3 stride-2 (128)
  -> Flatten (24,960) -> FC (512) -> LSTM (512)
  -> Value:     FC 512->256->1
  -> Advantage: FC 512->256->4
  -> Q = V + (A - mean(A))
```

The LSTM supports two modes:
- **Single-step inference** (gameplay): processes one frame at a time, maintaining hidden state across steps
- **Sequence mode** (training): unrolls full sequences for BPTT, enabling temporal credit assignment

**State representation:** 9-channel tensor (9 x 31 x 28) with per-ghost channels:

| Channel | Contents |
|---------|----------|
| 0 | Walls |
| 1 | Pellets |
| 2 | Power pellets |
| 3 | Pac-Man position |
| 4 | Blinky (1.0 = dangerous, -1.0 = frightened, 0.0 = absent) |
| 5 | Pinky |
| 6 | Inky |
| 7 | Clyde |
| 8 | Visit heatmap (decaying) |

Per-ghost channels (vs. a single shared ghost channel) enable the LSTM to learn ghost-specific behaviors and track each ghost's trajectory independently. When transitioning between curriculum stages, trained ghost channels transfer cleanly while new channels converge from near-random initialization.

**Split CPU/GPU architecture:** The game engine runs on CPU using PyTorch tensors for vectorized integer operations (faster than GPU for small integer ops). Only the neural network forward/backward passes run on GPU (MPS on Apple Silicon, or CUDA).

## How It Works

### Vectorized Engine

Instead of training on one game at a time, the engine runs **128 games simultaneously** using batched tensor operations. Each game step produces 128 transitions, yielding ~2,800 env-steps/sec throughput.

### Prioritized Experience Replay (PER)

The replay buffer uses episode-level prioritized sampling based on max TD error per sequence. Episodes where the model's predictions were furthest off (i.e., the hardest games) are replayed more frequently. This replaced uniform sampling and was responsible for the jump from 78% to 84% eval win rate — the model was always capable, but uniform sampling wasted training cycles on easy games the agent had already mastered.

Key PER details:
- **Proportional sampling** with alpha = 0.6 (controls how much prioritization vs. uniform)
- **Importance-sampling weights** with beta annealing 0.4 → 1.0 (corrects for sampling bias)
- **Episode-level priorities** updated after each training step using max TD error across the sequence

### Sequence Replay Buffer

Unlike flat DQN replay buffers that store individual transitions, the DRQN uses an episode-based buffer. Complete episode trajectories are stored and fixed-length sequences (default 32 steps) are sampled for LSTM training via BPTT. This preserves temporal coherence that random single-transition sampling would destroy.

### Curriculum Learning

The agent progresses through stages of increasing difficulty, with each stage building on the previous checkpoint:

```bash
# Stage 1: Learn the maze (no ghosts)
python -m training.train --n-envs 128 --stage 1 --episodes 225 --allow-reverse

# Stage 2: One ghost (Blinky, half speed)
python -m training.train --n-envs 128 --stage 2 --episodes 200 --eps-start 0.5 --allow-reverse --resume

# Stage 5: Three ghosts (Blinky + Pinky + Inky, half speed)
python -m training.train --n-envs 128 --stage 5 --episodes 500 --eps-start 0.5 --allow-reverse --resume

# Stage 6: Four ghosts (all, half speed)
python -m training.train --n-envs 128 --stage 6 --episodes 800 --eps-start 0.25 --allow-reverse --resume
```

Each new stage resets epsilon to allow exploration of new ghost-avoidance strategies. The LSTM architecture means higher starting epsilon is tolerable -- the diverse experiences fill the replay buffer with novel multi-ghost configurations needed for learning, even though high epsilon degrades in-episode LSTM context.

**Stage transition insight:** When adding new ghosts, the corresponding input channels start with near-random weights since they were always zero during previous training. The agent initially ignores (or walks into) new ghosts, but converges quickly because the existing ghost-evasion features on trained channels transfer. Stage 6 (4th ghost) ramped from 9% to 65% in ~100 episodes.

### Ghost Proximity: BFS Distance

Ghost proximity penalties use precomputed Floyd-Warshall BFS distances rather than Manhattan distance. This prevents "phantom fear" where the agent avoids ghosts that are nearby in straight-line distance but unreachable behind walls. The BFS distance matrix is computed once from the maze layout and reused every step.

### Reward Structure

| Signal | Value | Notes |
|--------|-------|-------|
| Per-step penalty | -0.05 | Time pressure |
| Visit penalty | -(0.15 or 0.30) x heatmap | 0.30 no ghosts, 0.15 with ghosts |
| Direction change | -0.03 | When action differs from previous |
| Pellet eaten | +1.0 + 2.0 x progress | progress = pellets_eaten / total |
| Power pellet | +2.0 + 2.0 x progress | Same progress scaling |
| BFS proximity reward | +/-(0.1 to 0.3) x delta | Reward for moving toward nearest pellet |
| Ghost proximity penalty | -0.075 x (5 - dist) | BFS dist <= 4, non-house non-frightened ghosts |
| Ghost eaten | +0.5 x combo | Combo resets on power-up expiry |
| Death | -5.0 | 1 life = game over |
| Level complete | +5.0 + time_bonus | Time bonus up to +5.0 |

**Reward clipping was intentionally removed.** The original DQN paper clips rewards to [-1, 1], but this destroyed signal differentiation -- death (-5.0) clipped to -1.0 was barely worse than a revisit penalty. Removing clipping was critical for learning ghost avoidance.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.0001 (Adam) |
| Gamma | 0.99 |
| Epsilon decay | 0.99997 per env step |
| Epsilon floor | 0.05 |
| Replay buffer | 200,000 transitions (episode-based) |
| Batch size | 32 |
| Sequence length | 32 (LSTM unroll for BPTT) |
| Train every | 4 game steps |
| Target network update | Every 5,000 steps |
| Max steps/episode | 3,500 |
| Grad clip | max_norm 10.0 |

## Project Structure

```
vectorized-pacman-ML/
├── engine/              # Vectorized game engine (batched PyTorch tensors)
│   ├── batched_game.py  # Core: N-env game loop, stage config, rewards
│   ├── ghosts.py        # Ghost AI (classic Pac-Man algorithms, not ML)
│   ├── maze.py          # Maze parsing, wall logic, BFS distance matrix
│   ├── action_mask.py   # Action masking (reverse mask, optional)
│   ├── rewards.py       # Reward computation (BFS proximity, ghost penalty)
│   └── constants.py     # State channels, ghost configs
├── models/
│   └── pacman_model.py  # Dueling DRQN (CNN + LSTM + Dueling heads)
├── training/
│   └── train.py         # Training loop with CSV logging, checkpointing
├── utils/
│   └── replay_buffer.py # Episode-based sequence replay buffer
├── levels/
│   └── level_1.txt      # Maze layout
├── watch.py             # Watch trained agent play (uses PacmanML renderer)
├── eval.py              # Headless evaluation (configurable epsilon, game count)
└── checkpoints/         # Saved models (~233MB each) and training logs
    └── logs/            # Per-episode CSV logs
```

## Watching the Agent Play

```bash
# Watch the agent play against 4 ghosts
python watch.py --model checkpoints/pacman_agent.pt --stage 6

# Headless evaluation (2048 games, pure greedy)
python eval.py --model checkpoints/pacman_agent.pt --stage 6 --games 2048 --epsilon 0
```

Requires the [PacmanML](https://github.com/alexhugli/PacmanML) renderer installed alongside this project for `watch.py`.

## Research Journey

### Dead Ends

- **Reward clipping** (following Atari DQN paper): destroyed reward differentiation. Removing it was the single biggest early improvement.
- **Flat DQN for multi-ghost**: hit a hard ceiling at 40-50% against 3 ghosts. No amount of reward tuning or training volume could overcome the lack of temporal memory.
- **No-reverse mask with DQN**: required as a crutch to prevent pathological oscillation. Worked, but constrained the action space artificially.
- **Manhattan distance for ghost proximity**: caused phantom penalties through walls, teaching the agent to fear ghosts in adjacent corridors it couldn't reach. Replaced with BFS distance.
- **Sequence length 16 with 4 ghosts**: adequate for 1-3 ghosts, but 4 ghosts cover enough of the maze that longer planning horizons (seq_len 32) were needed to anticipate traps.
- **Sequence length 64**: diluted training signal with irrelevant ancient history, degrading eval from 78% to 70.6%. The sweet spot is 32.
- **Self-attention after LSTM**: catastrophic failure. LayerNorm reshapes feature distributions the dueling heads expect. Even a gated residual (gate=0.01) with LayerNorm removed only delayed collapse — attention gradually corrupted the policy over hundreds of episodes. Dead end without training from scratch.
- **Reward tuning (ghost proximity)**: reducing ghost proximity penalty from -0.075/radius 4 to -0.04/radius 3 showed no improvement. The original values were already well-calibrated.
- **Epsilon floor 0.02**: caused overfitting within ~75 episodes. The buffer became too homogeneous with the model replaying its own near-greedy trajectories. 0.05 is the safe floor.

### Breakthroughs

1. **Vectorized engine** (128 parallel games): provided the training volume needed to learn non-trivial strategies. Single-env training plateaued at ~38% on one ghost.
2. **DRQN with LSTM**: enabled temporal reasoning, eliminating the need for the reverse mask and breaking through the DQN's multi-ghost ceiling. The LSTM's hidden state provides implicit memory of ghost trajectories and movement history.
3. **Per-ghost state channels**: allowed the network to learn ghost-specific evasion strategies and enabled clean weight transfer during curriculum transitions.
4. **BFS ghost proximity**: accurate distance signal through maze topology, replacing misleading Manhattan distance.
5. **Curriculum learning with epsilon resets**: each stage inherits a strong prior from the previous stage, requiring only moderate exploration to adapt to new threats.
6. **Prioritized Experience Replay**: the single biggest post-architecture improvement. By replaying the hardest episodes more frequently, the model focused on its weaknesses instead of coasting on easy games. Responsible for a 6+ point eval improvement (78% → 84.2%).

### LSTM Phase Transition

A distinctive pattern emerged across all stages: the LSTM exhibits a "phase transition" where performance suddenly accelerates as epsilon drops. At high epsilon, random actions corrupt the LSTM's hidden state, preventing coherent temporal reasoning. As epsilon falls below ~0.15, the agent gets enough consecutive coherent actions for the hidden state to build meaningful context, creating a feedback loop: cleaner sequences -> better temporal reasoning -> better Q-values -> even better sequences.

Stage 5 (3 ghosts) progression: 0% (ep 376) -> 6% (ep 480) -> 27% (ep 525) -> 51% (ep 575) -> 83% eval (ep 800).

Stage 6 (4 ghosts) progression: 0-1% (ep 426) -> 6% (ep 480) -> 58% (ep 526) -> 78% (ep 1600, uniform replay) -> **84.2%** (ep 1900, prioritized replay).

## Future Work

- **Fractional ghost speeds**: replace integer speed timer with accumulator system to match original arcade ghost speeds (75-95% of Pac-Man's speed depending on level)
- **Multiple maze layouts**: forced generalization to prove the agent isn't memorizing a single maze
- **Adversarial ghost training**: self-play where a second DRQN controls the ghosts, bootstrapped from the trained Pac-Man agent

## Setup

```bash
# Clone
git clone https://github.com/alexhugli/vectorized-pacman-ML.git
cd vectorized-pacman-ML

# Install dependencies
pip install -r requirements.txt

# Train from scratch
python -m training.train --n-envs 128 --stage 1 --episodes 225 --allow-reverse

# Run tests
pytest tests/
```

Requires Python 3.10+, PyTorch 2.0+. Apple Silicon (MPS) or CUDA GPU recommended but CPU works.

## License

MIT
