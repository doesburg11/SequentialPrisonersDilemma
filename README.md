# Sequential Iterated Prisoner's Dilemma (RLlib)

## Overview

This project studies a sequential iterated Prisoner's Dilemma with two independent learning agents using RLlib 2.54.0. Policies are learned via self-play.

## Environment and MARL Setup

- Environment class: `envs/prisoners_dilemma_env.py`
- Agent IDs: `player_1`, `player_2`
- Action space: `0=cooperate`, `1=defect`
- Reward matrix:
  - `(C, C) -> (3, 3)`
  - `(C, D) -> (0, 5)`
  - `(D, C) -> (5, 0)`
  - `(D, D) -> (1, 1)`
- Turn order is sequential each round: Player 1 then Player 2
- Horizon modes: `fixed`, `random_revealed`, `random_continuation`
- Two independent RLlib policies are trained:
  - `policy_player_1` for `player_1`
  - `policy_player_2` for `player_2`

## Game Dynamics

<div align="center">
  <img src="assets/prisoners_dilemma_matrix.svg" alt="Prisoner's Dilemma payoff matrix" width="400" />
  <p><strong>Display 1: The reward after each round.</strong></p>
</div>

Each episode is a repeated game with one of three horizon regimes:

1. `fixed`: always run exactly `max_rounds`.
2. `random_revealed`: sample episode horizon in `[min_rounds, max_rounds]` and reveal it via observation progress/info.
3. `random_continuation`: after each round (after `min_rounds`), continue with probability `continuation_prob`; stop otherwise.

## Research Question and Hypotheses

This project is best framed as a finite-horizon RL question, not as a direct equilibrium solver.

- Research question:
  - In a fixed-horizon iterated Prisoner's Dilemma, do independently trained PPO agents converge to backward-induction-like defection, or to cooperative conventions?
- Hypothesis H1 (game-theoretic target):
  - If learning approximates subgame-perfect play, defection probability should be high from early rounds and remain high.
- Hypothesis H2 (RL/self-play behavior):
  - With function approximation and self-play dynamics, agents may sustain cooperation for many rounds and defect only near the end (or remain cooperative throughout).

PPO drawback (important):

- Independent PPO self-play is not an equilibrium-finding algorithm.
- In this setup, each agent optimizes against a moving opponent policy, but PPO does not directly solve the Nash fixed-point condition ("no unilateral profitable deviation").
- As opposed to equilibrium-focused methods (e.g., backward induction, CFR-style methods, or PSRO + best-response checks), PPO alone does not provide equilibrium guarantees.

Recommended reporting:

- Defection/cooperation rate by round index `t`
- Mean episode return
- Mean rounds (fixed at `max_rounds` by design)
- Multiple random seeds (to detect equilibrium-selection effects)

## Historical Background (Rapoport / Axelrod)

- Anatol Rapoport is closely associated with Tit-for-Tat in repeated Prisoner's Dilemma research, including work with Albert Chammah in the 1960s.
- In 1980 and 1981, political scientist Robert Axelrod ran computer tournaments for iterated Prisoner's Dilemma strategies.
- Rapoport submitted Tit-for-Tat, and it ranked first in both tournaments.
- Axelrod's 1984 book *The Evolution of Cooperation* made these results widely known and influential.

## Training and Evaluation (RLlib 2.54.0)

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train with two independent policies and evaluate:

```bash
python scripts/train_eval_rllib.py --train-iters 50 --eval-episodes 20
```

Evaluate only from a saved checkpoint:

```bash
python scripts/train_eval_rllib.py --from-checkpoint checkpoints/sequential_pd_ppo --eval-episodes 50
```

Useful options:

```bash
# Adjust episode length
python scripts/train_eval_rllib.py --max-rounds 100

# Random horizon (revealed at episode start): sample T in [10, 50]
python scripts/train_eval_rllib.py --horizon-mode random_revealed --min-rounds 10 --max-rounds 50

# Random continuation (unknown final round): continue each round with prob 0.95
python scripts/train_eval_rllib.py --horizon-mode random_continuation --min-rounds 1 --max-rounds 100 --continuation-prob 0.95
```

## Experiment: Fixed Horizon (50 Rounds)

Goal:

- Test whether the finite-horizon setup converges to all-defect behavior.

```bash
python scripts/train_eval_rllib.py \
  --train-iters 50 \
  --eval-episodes 20 \
  --horizon-mode fixed \
  --max-rounds 50
```

Observed eval summary:

- `mean_episode_reward`: `player_1=50.0`, `player_2=50.0`
- `cooperation_rate`: `player_1=0.0`, `player_2=0.0`
- `mean_rounds_per_episode`: `50.0`

Interpretation:

- This matches all-defect over 50 rounds: each round yields `(D,D) -> (1,1)`, totaling `50` per agent.
- This is the expected finite-horizon baseline in the standard window-less setup.
