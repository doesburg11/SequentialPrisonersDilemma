#!/usr/bin/env python3
"""Train and evaluate two independent RLlib policies on a MARL PD environment."""

from __future__ import annotations

import argparse
import json
import random
import sys
from importlib import import_module
from pathlib import Path
from typing import Dict

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core import Columns
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_torch

# Resolve env imports for both:
# - `python scripts/train_eval_rllib.py` (script dir on sys.path)
# - `python -m scripts.train_eval_rllib` (project root on sys.path)
try:
    _env_module = import_module("envs.prisoners_dilemma_env")
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    _env_module = import_module("envs.prisoners_dilemma_env")

AGENT_IDS = _env_module.AGENT_IDS
COOPERATE = _env_module.COOPERATE
ENV_NAME = _env_module.ENV_NAME
SequentialPrisonersDilemmaEnv = _env_module.SequentialPrisonersDilemmaEnv

torch, _ = try_import_torch()


def env_creator(env_config):
    return SequentialPrisonersDilemmaEnv(env_config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    return f"policy_{agent_id}"


def build_algorithm(args) -> Algorithm:
    env_config = {
        "max_rounds": args.max_rounds,
        "min_rounds": args.min_rounds,
        "horizon_mode": args.horizon_mode,
        "continuation_prob": args.continuation_prob,
    }
    register_env(ENV_NAME, env_creator)

    policies = {f"policy_{agent_id}" for agent_id in AGENT_IDS}
    config = PPOConfig()
    if hasattr(config, "api_stack"):
        config = config.api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
    config = (
        config.environment(ENV_NAME, env_config=env_config)
        .framework(args.framework)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies),
        )
        .resources(num_gpus=args.num_gpus)
        .training(lr=args.lr)
    )
    if args.seed is not None and hasattr(config, "debugging"):
        config = config.debugging(seed=args.seed)

    # RLlib rollout worker APIs changed across versions; keep compatibility.
    if hasattr(config, "env_runners"):
        config = config.env_runners(num_env_runners=args.num_workers)
    elif hasattr(config, "rollouts"):
        config = config.rollouts(num_rollout_workers=args.num_workers)

    if args.train_batch_size is not None:
        config = config.training(train_batch_size=args.train_batch_size)

    if hasattr(config, "build_algo"):
        return config.build_algo()
    return config.build()


def extract_reward_mean(train_result: Dict) -> float:
    if "episode_reward_mean" in train_result:
        return float(train_result["episode_reward_mean"])
    env_runner_metrics = train_result.get("env_runners", {})
    if "episode_return_mean" in env_runner_metrics:
        return float(env_runner_metrics["episode_return_mean"])
    return float("nan")


def _to_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        return value.numpy()
    return np.asarray(value)


def _extract_first_action(action_batch) -> int:
    action_np = _to_numpy(action_batch)
    return int(np.asarray(action_np).reshape(-1)[0])


def _resolve_checkpoint_path(path: str) -> str:
    if "://" in path:
        return path
    return str(Path(path).expanduser().resolve())


def compute_eval_action(algo: Algorithm, policy_id: str, observation) -> int:
    module = algo.get_module(policy_id)
    if module is not None:
        obs_batch = np.expand_dims(np.asarray(observation, dtype=np.float32), axis=0)
        batch = {Columns.OBS: obs_batch}

        if getattr(module, "framework", None) == "torch" and torch is not None:
            batch[Columns.OBS] = torch.from_numpy(obs_batch)
            with torch.no_grad():
                module_out = module.forward_inference(batch)
        else:
            module_out = module.forward_inference(batch)

        if Columns.ACTIONS in module_out:
            return _extract_first_action(module_out[Columns.ACTIONS])
        if Columns.ACTIONS_FOR_ENV in module_out:
            return _extract_first_action(module_out[Columns.ACTIONS_FOR_ENV])
        if Columns.ACTION_DIST_INPUTS in module_out:
            action_dist_cls = module.get_inference_action_dist_cls()
            if action_dist_cls is None:
                raise ValueError(f"Inference distribution missing for module={policy_id!r}")
            action_dist = action_dist_cls.from_logits(module_out[Columns.ACTION_DIST_INPUTS])
            return _extract_first_action(action_dist.to_deterministic().sample())
        raise ValueError(
            f"Module output for {policy_id!r} missing action keys: "
            f"{list(module_out.keys())}"
        )

    # Backward-compat fallback if an old checkpoint restores policies only.
    policy = algo.get_policy(policy_id)
    if policy is None:
        raise ValueError(f"Policy not found for policy_id={policy_id!r}")

    actions, _state_out, _extra = policy.compute_actions([observation], explore=False)
    return int(actions[0])


def evaluate(algo: Algorithm, episodes: int, env_config: Dict) -> Dict:
    env = SequentialPrisonersDilemmaEnv(env_config)

    total_rewards = {agent: 0.0 for agent in AGENT_IDS}
    action_counts = {agent: 0 for agent in AGENT_IDS}
    cooperation_counts = {agent: 0 for agent in AGENT_IDS}
    rounds_per_episode = []

    for _ in range(episodes):
        obs, _infos = env.reset()
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        episode_rewards = {agent: 0.0 for agent in AGENT_IDS}

        while not (terminated["__all__"] or truncated["__all__"]):
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy_id = policy_mapping_fn(agent_id)
                action = compute_eval_action(algo, policy_id, agent_obs)
                actions[agent_id] = action
                action_counts[agent_id] += 1
                if action == COOPERATE:
                    cooperation_counts[agent_id] += 1

            obs, rewards, terminated, truncated, _infos = env.step(actions)
            for agent_id in AGENT_IDS:
                episode_rewards[agent_id] += float(rewards.get(agent_id, 0.0))

        rounds_per_episode.append(env.rounds_completed)
        for agent_id in AGENT_IDS:
            total_rewards[agent_id] += episode_rewards[agent_id]

    summary = {
        "episodes": episodes,
        "mean_episode_reward": {
            agent_id: total_rewards[agent_id] / max(episodes, 1) for agent_id in AGENT_IDS
        },
        "cooperation_rate": {
            agent_id: cooperation_counts[agent_id] / max(action_counts[agent_id], 1)
            for agent_id in AGENT_IDS
        },
        "mean_rounds_per_episode": sum(rounds_per_episode) / max(episodes, 1),
    }
    return summary


def checkpoint_to_path(checkpoint_obj) -> str:
    if isinstance(checkpoint_obj, str):
        return checkpoint_obj
    if hasattr(checkpoint_obj, "path"):
        return str(checkpoint_obj.path)
    if hasattr(checkpoint_obj, "checkpoint") and hasattr(checkpoint_obj.checkpoint, "path"):
        return str(checkpoint_obj.checkpoint.path)
    return str(checkpoint_obj)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-iters", type=int, default=50, help="PPO train iterations.")
    parser.add_argument(
        "--eval-episodes", type=int, default=20, help="Episodes for post-train evaluation."
    )
    parser.add_argument("--max-rounds", type=int, default=50, help="Max rounds per episode.")
    parser.add_argument(
        "--min-rounds",
        type=int,
        default=1,
        help="Minimum rounds for random horizon modes (<= max-rounds).",
    )
    parser.add_argument(
        "--horizon-mode",
        type=str,
        default="fixed",
        choices=["fixed", "random_revealed", "random_continuation"],
        help="How episode horizon is determined.",
    )
    parser.add_argument(
        "--continuation-prob",
        type=float,
        default=0.95,
        help=(
            "For random_continuation mode: probability to continue after each round "
            "(once min-rounds is reached)."
        ),
    )
    parser.add_argument("--framework", type=str, default="torch", choices=["torch", "tf2"])
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible training/evaluation runs.",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0, help="RLlib rollout workers.")
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/sequential_pd_ppo",
        help="Directory where checkpoints are written.",
    )
    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. If set, skip training and only evaluate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        if torch is not None and hasattr(torch, "manual_seed"):
            torch.manual_seed(args.seed)
    if args.min_rounds > args.max_rounds:
        raise ValueError("min-rounds must be <= max-rounds")
    env_config = {
        "max_rounds": args.max_rounds,
        "min_rounds": args.min_rounds,
        "horizon_mode": args.horizon_mode,
        "continuation_prob": args.continuation_prob,
    }

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    register_env(ENV_NAME, env_creator)

    if args.from_checkpoint:
        algo = Algorithm.from_checkpoint(_resolve_checkpoint_path(args.from_checkpoint))
    else:
        algo = build_algorithm(args)
        for i in range(1, args.train_iters + 1):
            result = algo.train()
            if i == 1 or i == args.train_iters or i % 10 == 0:
                reward_mean = extract_reward_mean(result)
                print(
                    f"[train] iter={i} reward_mean={reward_mean:.3f} "
                    f"timesteps_total={result.get('timesteps_total', 'n/a')}"
                )

        checkpoint_dir = _resolve_checkpoint_path(args.checkpoint_dir)
        checkpoint_path = checkpoint_to_path(algo.save(checkpoint_dir))
        print(f"[train] checkpoint saved to: {checkpoint_path}")

    if args.eval_episodes > 0:
        eval_summary = evaluate(algo, args.eval_episodes, env_config)
        print("[eval] summary:")
        print(json.dumps(eval_summary, indent=2))

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
