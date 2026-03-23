"""
PPO (Proximal Policy Optimization) implementation for VCMI Battle AI.

Follows the CleanRL-style single-file PPO with GAE, supporting
dict observation spaces and the BattleTransformer model.
"""

import numpy as np
import torch
import torch.nn as nn

from vcmigym import (
    MAX_STACKS,
    MAX_OBSTACLES,
    MAX_ATTACK_TARGETS,
    BATTLEFIELD_HEXES,
    STACK_FEATURES,
    OBSTACLE_FEATURES,
)


class RolloutBuffer:
    """Fixed-size buffer for storing PPO rollout data from vectorized envs."""

    def __init__(self, num_steps: int, num_envs: int, device: torch.device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.pos = 0

        # Observations (stored as numpy, converted to tensors on demand)
        self.obs_scalars = np.zeros((num_steps, num_envs, 19), dtype=np.float32)
        self.obs_stacks = np.zeros((num_steps, num_envs, MAX_STACKS, STACK_FEATURES), dtype=np.float32)
        self.obs_obstacles = np.zeros((num_steps, num_envs, MAX_OBSTACLES, OBSTACLE_FEATURES), dtype=np.float32)
        self.obs_reachable = np.zeros((num_steps, num_envs, BATTLEFIELD_HEXES), dtype=np.float32)
        self.obs_n_stacks = np.zeros((num_steps, num_envs, 1), dtype=np.int32)
        self.obs_attack_targets = np.zeros((num_steps, num_envs, MAX_ATTACK_TARGETS, 2), dtype=np.int32)

        # Actions and policy outputs
        self.actions = np.zeros((num_steps, num_envs, 3), dtype=np.int64)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)

        # Environment outputs
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)

        # Per-step side tracking (for self-play per-side GAE)
        self.sides = np.zeros((num_steps, num_envs), dtype=np.int32)

        # Computed after rollout
        self.advantages = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_envs), dtype=np.float32)

    def store(self, obs: dict, actions, log_probs, values, rewards, dones, sides):
        """Store one step of rollout data."""
        t = self.pos
        self.obs_scalars[t] = obs["scalars"]
        self.obs_stacks[t] = obs["stacks"]
        self.obs_obstacles[t] = obs["obstacles"]
        self.obs_reachable[t] = obs["reachable_hexes"]
        self.obs_n_stacks[t] = obs["n_stacks"]
        self.obs_attack_targets[t] = obs["attack_targets"]
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.sides[t] = sides
        self.pos += 1

    def compute_gae(self, last_value: np.ndarray, last_done: np.ndarray,
                    last_sides: np.ndarray,
                    gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute GAE advantages and discounted returns per side.

        In self-play, each side's steps form a separate episode for GAE
        purposes.  We only chain value bootstraps between consecutive
        steps of the *same* side, cutting whenever any done occurs in
        the intervening steps (battle ended during the other side's turn).
        """
        self.advantages[:] = 0.0
        N = self.num_steps

        for env in range(self.num_envs):
            for side in (0, 1):
                # Indices of steps where this side acted
                side_steps = np.where(self.sides[:N, env] == side)[0]
                if len(side_steps) == 0:
                    continue

                last_gae = 0.0
                for i in reversed(range(len(side_steps))):
                    t = side_steps[i]

                    if i == len(side_steps) - 1:
                        # Last same-side step in rollout — check for dones
                        # between here and rollout end
                        if self.dones[t:N, env].any():
                            next_non_terminal = 0.0
                            next_value = 0.0
                        elif last_sides[env] == side:
                            next_non_terminal = 1.0 - last_done[env]
                            next_value = last_value[env]
                        else:
                            # Last obs belongs to other side; can't bootstrap
                            next_non_terminal = 0.0
                            next_value = 0.0
                    else:
                        next_t = side_steps[i + 1]
                        # Check for any done between t (inclusive) and next_t
                        if self.dones[t:next_t, env].any():
                            next_non_terminal = 0.0
                            next_value = 0.0
                        else:
                            next_non_terminal = 1.0
                            next_value = self.values[next_t, env]

                    delta = (
                        self.rewards[t, env]
                        + gamma * next_value * next_non_terminal
                        - self.values[t, env]
                    )
                    last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
                    self.advantages[t, env] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """Yield mini-batch indices for PPO updates."""
        total = self.num_steps * self.num_envs
        indices = np.arange(total)
        np.random.shuffle(indices)
        for start in range(0, total, batch_size):
            yield indices[start:start + batch_size]

    def flatten_obs(self) -> dict[str, torch.Tensor]:
        """Flatten (steps, envs, ...) → (steps*envs, ...) as tensors."""
        S, E = self.num_steps, self.num_envs
        return {
            "scalars": torch.from_numpy(self.obs_scalars.reshape(S * E, 19)).float().to(self.device),
            "stacks": torch.from_numpy(self.obs_stacks.reshape(S * E, MAX_STACKS, STACK_FEATURES)).float().to(self.device),
            "obstacles": torch.from_numpy(self.obs_obstacles.reshape(S * E, MAX_OBSTACLES, OBSTACLE_FEATURES)).float().to(self.device),
            "reachable_hexes": torch.from_numpy(self.obs_reachable.reshape(S * E, BATTLEFIELD_HEXES)).float().to(self.device),
            "n_stacks": torch.from_numpy(self.obs_n_stacks.reshape(S * E, 1)).float().to(self.device),
            "attack_targets": torch.from_numpy(self.obs_attack_targets.reshape(S * E, MAX_ATTACK_TARGETS, 2)).to(self.device),
        }

    def flatten_actions(self) -> torch.Tensor:
        S, E = self.num_steps, self.num_envs
        return torch.from_numpy(self.actions.reshape(S * E, 3)).long().to(self.device)

    def flatten_log_probs(self) -> torch.Tensor:
        return torch.from_numpy(self.log_probs.flatten()).float().to(self.device)

    def flatten_advantages(self) -> torch.Tensor:
        return torch.from_numpy(self.advantages.flatten()).float().to(self.device)

    def flatten_returns(self) -> torch.Tensor:
        return torch.from_numpy(self.returns.flatten()).float().to(self.device)

    def reset(self):
        self.pos = 0


class PPOTrainer:
    """PPO trainer for the BattleTransformer model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        batch_size: int = 256,
        target_kl: float | None = None,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run PPO update on collected rollout data. Returns loss metrics."""
        all_obs = buffer.flatten_obs()
        all_actions = buffer.flatten_actions()
        all_old_log_probs = buffer.flatten_log_probs()
        all_advantages = buffer.flatten_advantages()
        all_returns = buffer.flatten_returns()

        # Normalize advantages
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        total_samples = buffer.num_steps * buffer.num_envs
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        n_updates = 0
        early_stop = False

        for epoch in range(self.update_epochs):
            if early_stop:
                break

            for batch_idx in buffer.get_batches(self.batch_size):
                batch_idx_t = torch.from_numpy(batch_idx).long()

                # Slice batch observations
                batch_obs = {k: v[batch_idx_t] for k, v in all_obs.items()}
                batch_actions = all_actions[batch_idx_t]
                batch_old_log_probs = all_old_log_probs[batch_idx_t]
                batch_advantages = all_advantages[batch_idx_t]
                batch_returns = all_returns[batch_idx_t]

                # Evaluate actions under current policy
                _, new_log_probs, entropy, new_values = self.model.get_action_and_value(
                    batch_obs, action=batch_actions
                )

                # Policy loss (clipped surrogate)
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = log_ratio.exp()
                pg_loss1 = -batch_advantages * ratio
                pg_loss2 = -batch_advantages * torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (MSE)
                v_loss = 0.5 * ((new_values - batch_returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss + self.value_coef * v_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()

                metrics["policy_loss"] += pg_loss.item()
                metrics["value_loss"] += v_loss.item()
                metrics["entropy"] += entropy_loss.item()
                metrics["total_loss"] += loss.item()
                metrics["approx_kl"] += approx_kl
                metrics["clip_fraction"] += clip_frac
                n_updates += 1

                # Early stopping on KL divergence
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    early_stop = True
                    break

        # Average metrics
        if n_updates > 0:
            for k in metrics:
                metrics[k] /= n_updates

        metrics["n_updates"] = n_updates
        return metrics
