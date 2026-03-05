"""Central PPO hyperparameters for scripts/tune_eval_rllib.py.

Edit this file to change PPO behavior without touching tuning code.
"""

config_ppo = {
    # Tune control
    "tune_iters": 100,
    # Core PPO learning
    "lr": 5e-4,
    "gamma": 0.99,
    "lambda_": 1.0,
    "train_batch_size_per_learner": 4000,
    "minibatch_size": 128,
    "num_epochs": 30,
    "entropy_coeff": 0.0,
    "vf_loss_coeff": 1.0,
    "clip_param": 0.3,
    # KL control
    "kl_coeff": 0.2,
    "kl_target": 0.01,
    # New API stack resource controls (single-node default)
    # Keep this schedulable on a typical workstation (1 GPU).
    "num_learners": 1,
    "num_gpus_per_learner": 1.0,
    "num_env_runners": 24,
    "num_envs_per_env_runner": 1,
    "num_cpus_per_env_runner": 1.0,
    "num_cpus_for_main_process": 1.0,
    "sample_timeout_s": 600.0,
    "rollout_fragment_length": "auto",
}
