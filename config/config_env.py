"""Central runtime/environment settings for scripts/tune_eval_rllib.py.

Edit this file to control tune/eval behavior without CLI flags.
"""

config_env = {
    # Paths
    "ppo_config": "config/config_ppo.py",
    "checkpoint_dir": "checkpoints/sequential_pd_ppo",
    "from_checkpoint": None,
    "metrics_out": None,
    # Evaluation
    "eval_episodes": 20,
    # Environment horizon
    "n_sequential_games": 100,
    # Reproducibility
    "seed": None,
}

# Sweep-only settings for scripts/sweep_max_rounds_cooperation.py.
# These are intentionally outside `config_env` to keep tune/eval runtime
# parameters focused on training/evaluation and fixed-horizon env settings.
config_sweep_max_rounds = {
    "num_seeds": 10,
    "seed_start": 0,
    "ci_level": 0.95,
}
