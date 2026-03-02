"""Turn-based two-player Prisoner's Dilemma environment for RLlib."""

from __future__ import annotations

from collections import deque
from typing import Dict, Tuple

import numpy as np
from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

COOPERATE = 0
DEFECT = 1
ACTION_NAMES = {
    COOPERATE: "cooperate",
    DEFECT: "defect",
}
AGENT_IDS = ("player_1", "player_2")
ENV_NAME = "sequential_prisoners_dilemma"

PAYOFF_MATRIX: Dict[Tuple[int, int], Tuple[float, float]] = {
    (COOPERATE, COOPERATE): (3.0, 3.0),
    (COOPERATE, DEFECT): (0.0, 5.0),
    (DEFECT, COOPERATE): (5.0, 0.0),
    (DEFECT, DEFECT): (1.0, 1.0),
}


class SequentialPrisonersDilemmaEnv(MultiAgentEnv):
    """Two-agent turn-based repeated Prisoner's Dilemma.

    Rules:
    - Player 1 acts first, then Player 2 acts.
    - A payoff is assigned after both actions in the round are known.
    - Episode length is fixed: it always runs for `max_rounds`.
    - If `reward_window` is set (>0), the episode return tracks only the most
      recent N round payoffs via delta rewards.
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or {}

        self.max_rounds = int(config.get("max_rounds", 50))
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be > 0")
        reward_window = int(config.get("reward_window", 10))
        if reward_window <= 0:
            raise ValueError("reward_window must be > 0")
        self.reward_window = reward_window

        # Observation = [last_own_action, last_opponent_action, round_progress]
        # last actions use -1.0 before first complete round.
        shared_observation_space = Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        shared_action_space = Discrete(2)  # 0=cooperate, 1=defect

        # Preferred by RLlib's new API stack: per-agent space dicts.
        self.observation_spaces = {
            agent_id: shared_observation_space for agent_id in AGENT_IDS
        }
        self.action_spaces = {agent_id: shared_action_space for agent_id in AGENT_IDS}

        # Keep shared fields for compatibility with old stack code paths.
        self.observation_space = shared_observation_space
        self.action_space = shared_action_space

        self.possible_agents = list(AGENT_IDS)
        self.agents = list(AGENT_IDS)

        self._next_player = AGENT_IDS[0]
        self._pending_action_player_1 = None
        self._last_joint_actions = (-1, -1)
        self.rounds_completed = 0
        self._episode_done = False
        self._recent_round_rewards = {
            agent_id: deque(maxlen=self.reward_window) for agent_id in AGENT_IDS
        }

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self._next_player = AGENT_IDS[0]
        self._pending_action_player_1 = None
        self._last_joint_actions = (-1, -1)
        self.rounds_completed = 0
        self._episode_done = False
        for history in self._recent_round_rewards.values():
            history.clear()

        obs = {self._next_player: self._build_obs(self._next_player)}
        infos = {self._next_player: {"round": 1}}
        return obs, infos

    def step(self, action_dict):
        if self._episode_done:
            raise RuntimeError("step() called after episode is done; call reset().")
        if self._next_player not in action_dict:
            raise ValueError(f"Expected action for active agent {self._next_player}.")

        active_agent = self._next_player
        action = int(action_dict[active_agent])
        if action not in (COOPERATE, DEFECT):
            raise ValueError(f"Invalid action {action}; expected 0 (cooperate) or 1 (defect).")

        # Phase 1: Player 1 acts, then we hand the turn to Player 2.
        if active_agent == AGENT_IDS[0]:
            self._pending_action_player_1 = action
            self._next_player = AGENT_IDS[1]

            obs = {self._next_player: self._build_obs(self._next_player)}
            rewards = {AGENT_IDS[0]: 0.0, AGENT_IDS[1]: 0.0}
            terminateds = {"__all__": False}
            truncateds = {"__all__": False}
            infos = {self._next_player: {"round": self.rounds_completed + 1}}
            return obs, rewards, terminateds, truncateds, infos

        # Phase 2: Player 2 acts, round is complete, compute payoff.
        player_1_action = int(self._pending_action_player_1)
        player_2_action = action
        base_reward_player_1, base_reward_player_2 = PAYOFF_MATRIX[
            (player_1_action, player_2_action)
        ]
        reward_player_1 = self._windowed_reward(AGENT_IDS[0], base_reward_player_1)
        reward_player_2 = self._windowed_reward(AGENT_IDS[1], base_reward_player_2)

        self.rounds_completed += 1
        self._last_joint_actions = (player_1_action, player_2_action)
        self._pending_action_player_1 = None

        terminated_all = self.rounds_completed >= self.max_rounds
        truncated_all = False

        self._episode_done = terminated_all or truncated_all
        self._next_player = AGENT_IDS[0]

        if self._episode_done:
            # RLlib's new API stack expects final observations for ended agents,
            # especially for truncation value-bootstrapping.
            obs = {agent_id: self._build_obs(agent_id) for agent_id in AGENT_IDS}
            infos = {agent_id: {"round": self.rounds_completed} for agent_id in AGENT_IDS}
        else:
            obs = {self._next_player: self._build_obs(self._next_player)}
            infos = {self._next_player: {"round": self.rounds_completed + 1}}

        rewards = {
            AGENT_IDS[0]: reward_player_1,
            AGENT_IDS[1]: reward_player_2,
        }

        terminateds = {"__all__": terminated_all}
        truncateds = {"__all__": truncated_all}
        if self._episode_done:
            terminateds.update({AGENT_IDS[0]: terminated_all, AGENT_IDS[1]: terminated_all})
            truncateds.update({AGENT_IDS[0]: truncated_all, AGENT_IDS[1]: truncated_all})

        return obs, rewards, terminateds, truncateds, infos

    def _build_obs(self, agent_id: str):
        if agent_id == AGENT_IDS[0]:
            own_last, opp_last = self._last_joint_actions
        else:
            opp_last, own_last = self._last_joint_actions
        round_progress = float(self.rounds_completed) / float(self.max_rounds)
        return np.array([float(own_last), float(opp_last), round_progress], dtype=np.float32)

    def _windowed_reward(self, agent_id: str, round_reward: float) -> float:
        """Return a reward delta so episode return equals sum of last N payoffs."""
        history = self._recent_round_rewards[agent_id]
        dropped = history[0] if len(history) == self.reward_window else 0.0
        history.append(float(round_reward))
        return float(round_reward) - float(dropped)
