# TitForTat

## Overview

This project demonstrates a Tit-for-Tat-inspired repeated Prisoner's Dilemma setup with two independent learning agents using RLlib 2.54.0.

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
- Episodes are fixed-horizon (`max_rounds`): no early stop on defection
- Two independent RLlib policies are trained:
  - `policy_player_1` for `player_1`
  - `policy_player_2` for `player_2`

## Game Dynamics

<div align="center">
  <img src="assets/prisoners_dilemma_matrix.svg" alt="Prisoner's Dilemma payoff matrix" width="400" />
  <p><strong>Display 1: The reward after each round.</strong></p>
</div>

Each episode is a repeated game with `max_rounds` rounds:

1. Player 1 acts.
2. Player 2 acts.
3. Round payoff is applied.
4. Next round starts, regardless of whether someone defected.

Optional: `reward_window` can be used so episode return only reflects the last `N` rounds (recency-weighted objective).

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

# Only count the last 10 round payoffs in cumulative return
python scripts/train_eval_rllib.py --reward-window 10
```
