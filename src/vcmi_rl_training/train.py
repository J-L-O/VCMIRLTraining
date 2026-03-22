"""
RL Training Loop for VCMI Battle AI

Uses the VCMIGym gymnasium environment with a transformer-based policy.
Supports vectorized (async) environments for batch rollouts.
Currently runs random rollouts as a starting point for PPO training.
"""

import argparse
import logging
import time

import gymnasium as gym
import numpy as np
import torch
from vcmigym import make_vcmi_env

from vcmi_rl_training.model import BattleTransformer

logger = logging.getLogger(__name__)


def obs_to_tensors(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Convert vectorized env observation dict to torch tensors.

    VectorEnv already returns arrays with shape (num_envs, ...), so no
    unsqueeze is needed — just convert to tensors.
    """
    return {
        k: torch.from_numpy(np.asarray(v)).float().to(device)
        if k != "attack_targets"
        else torch.from_numpy(np.asarray(v)).to(device)
        for k, v in obs.items()
    }


def run_rollouts(
    envs: gym.vector.VectorEnv,
    model: BattleTransformer,
    device: torch.device,
    num_steps: int,
) -> dict:
    """Run rollouts across all vectorized environments.

    Returns aggregate statistics over the completed episodes.
    """
    num_envs = envs.num_envs
    obs, info = envs.reset()
    total_steps = 0
    episode_rewards = np.zeros(num_envs)
    completed_episodes = []

    for step in range(num_steps):
        obs_t = obs_to_tensors(obs, device)

        with torch.no_grad():
            actions, log_probs, entropy, values = model.get_action_and_value(obs_t)

        actions_np = actions.cpu().numpy()  # (num_envs, 3)
        obs, rewards, terminated, truncated, infos = envs.step(actions_np)
        episode_rewards += rewards
        total_steps += num_envs

        # Check for completed episodes (auto-reset handled by VectorEnv)
        dones = terminated | truncated
        for i in range(num_envs):
            if dones[i]:
                completed_episodes.append(episode_rewards[i])
                episode_rewards[i] = 0.0

    return {
        "total_steps": total_steps,
        "completed_episodes": len(completed_episodes),
        "mean_reward": np.mean(completed_episodes) if completed_episodes else 0.0,
        "episode_rewards": completed_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="VCMI RL Training")
    parser.add_argument("--vcmi-client", required=True, help="Path to vcmiclient binary")
    parser.add_argument("--test-map", required=True, help="Map name relative to vcmi-cwd")
    parser.add_argument("--vcmi-cwd", default=None, help="Working directory for game process")
    parser.add_argument("--port", type=int, default=10000, help="Base port for game connections")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=512, help="Rollout steps per iteration")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--d-model", type=int, default=128, help="Transformer model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--device", default="auto", help="Device: cpu, cuda, or auto")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Suppress noisy per-pack deserialization logs from vcmigym
    logging.getLogger("vcmigym.vcmi_types").setLevel(logging.WARNING)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    model = BattleTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created: {n_params:,} parameters")

    envs = gym.vector.AsyncVectorEnv([
        lambda i=i: make_vcmi_env(
            i,
            vcmi_client=args.vcmi_client,
            test_map=args.test_map,
            port_base=args.port,
            vcmi_cwd=args.vcmi_cwd,
        )
        for i in range(args.num_envs)
    ])

    logger.info(
        f"Created {args.num_envs} async environments, "
        f"rollout={args.num_steps} steps/iteration"
    )

    try:
        for iteration in range(1, args.iterations + 1):
            t0 = time.time()
            stats = run_rollouts(envs, model, device, args.num_steps)
            elapsed = time.time() - t0
            sps = stats["total_steps"] / elapsed

            logger.info(
                f"Iter {iteration}/{args.iterations}: "
                f"episodes={stats['completed_episodes']}, "
                f"mean_reward={stats['mean_reward']:.1f}, "
                f"steps={stats['total_steps']}, "
                f"{sps:.0f} steps/s"
            )
    finally:
        envs.close()

    logger.info("Training complete")


if __name__ == "__main__":
    main()
