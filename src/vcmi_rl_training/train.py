"""
RL Training Loop for VCMI Battle AI

PPO training with the BattleTransformer model and VCMIGym environments.
Supports vectorized (async) environments, Weights & Biases logging,
and a --playtest mode to play against the AI in a graphical game.
"""

import argparse
import logging
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from vcmigym import make_vcmi_env, MAX_STACKS, MAX_ATTACK_TARGETS, BATTLEFIELD_HEXES

from vcmi_rl_training.model import BattleTransformer
from vcmi_rl_training.ppo import RolloutBuffer, PPOTrainer

# Stack feature indices (must match env.py / model.py)
_S_ID = 0
_S_SIDE = 20
_S_ALIVE = 23
_S_CAN_SHOOT = 25
_S_WAITING = 29
_SC_ACTIVE_STACK_ID = 2
_SC_ACTIVE_SIDE = 3

logger = logging.getLogger(__name__)


def obs_to_tensors(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Convert vectorized env observation dict to torch tensors."""
    return {
        k: torch.from_numpy(np.asarray(v)).float().to(device)
        if k != "attack_targets"
        else torch.from_numpy(np.asarray(v)).to(device)
        for k, v in obs.items()
    }


def obs_to_tensors_single(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Convert single-env observation dict to batched torch tensors (batch dim=1)."""
    return {
        k: torch.from_numpy(np.asarray(v)).unsqueeze(0).float().to(device)
        if k != "attack_targets"
        else torch.from_numpy(np.asarray(v)).unsqueeze(0).to(device)
        for k, v in obs.items()
    }


def sample_random_valid_action(obs: dict[str, np.ndarray], env_idx: int) -> np.ndarray:
    """Sample a random valid action for one environment from its observation.

    Returns an action array [action_type, hex, target_stack_idx].
    Action types: 0=DEFEND, 1=WAIT, 2=WALK, 3=WALK_AND_ATTACK, 4=SHOOT.
    """
    scalars = obs["scalars"][env_idx]       # (19,)
    stacks = obs["stacks"][env_idx]         # (MAX_STACKS, 35)
    reachable = obs["reachable_hexes"][env_idx]  # (187,)
    attack_targets = obs["attack_targets"][env_idx]  # (MAX_ATTACK_TARGETS, 2)
    n_stacks = int(obs["n_stacks"][env_idx, 0])

    active_id = scalars[_SC_ACTIVE_STACK_ID]
    active_side = scalars[_SC_ACTIVE_SIDE]

    # Find active stack
    active_can_shoot = False
    active_waiting = False
    for i in range(n_stacks):
        if stacks[i, _S_ID] == active_id:
            active_can_shoot = bool(stacks[i, _S_CAN_SHOOT] > 0.5)
            active_waiting = bool(stacks[i, _S_WAITING] > 0.5)
            break

    reachable_hexes = np.where(reachable > 0.5)[0]

    # Collect valid attack target entries: (target_stack_idx, from_hex)
    valid_attacks = []
    for t in range(MAX_ATTACK_TARGETS):
        tid, from_hex = int(attack_targets[t, 0]), int(attack_targets[t, 1])
        if tid == 0 and from_hex == 0:
            break
        # Find stack index for this target unit id
        for si in range(n_stacks):
            if int(stacks[si, _S_ID]) == tid:
                valid_attacks.append((si, from_hex))
                break

    # Collect alive enemy stack indices for shooting
    enemy_indices = []
    if active_can_shoot:
        for i in range(n_stacks):
            if stacks[i, _S_ALIVE] > 0.5 and stacks[i, _S_SIDE] != active_side:
                enemy_indices.append(i)

    # Build list of valid (action_type, hex, target) tuples
    valid_actions = [(0, 0, 0)]  # DEFEND is always valid
    if not active_waiting:
        valid_actions.append((1, 0, 0))  # WAIT
    for h in reachable_hexes:
        valid_actions.append((2, int(h), 0))  # WALK
    for si, from_hex in valid_attacks:
        valid_actions.append((3, from_hex, si))  # WALK_AND_ATTACK
    for si in enemy_indices:
        valid_actions.append((4, 0, si))  # SHOOT

    choice = valid_actions[np.random.randint(len(valid_actions))]
    return np.array(choice, dtype=np.int64)


def _compute_and_dispatch(
    env_group: gym.vector.VectorEnv,
    obs_g: dict[str, np.ndarray],
    model: BattleTransformer,
    device: torch.device,
    epsilon: float,
    amp_dtype: torch.dtype | None = None,
) -> dict:
    """GPU inference + async env dispatch for one microbatch.

    Returns a dict with the pre-step data needed for buffer storage.
    """
    n_g = env_group.num_envs
    obs_t = obs_to_tensors(obs_g, device)

    action_mask = {
        "action_type": torch.from_numpy(obs_g["action_type_mask"]).to(device),
        "hex": torch.from_numpy(obs_g["hex_mask"]).to(device),
        "target": torch.from_numpy(obs_g["target_mask"]).to(device),
    }

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        actions, log_probs, _, values = model.get_action_and_value(obs_t, action_mask=action_mask)

    actions_np = actions.cpu().numpy()
    values_np = values.float().cpu().numpy()

    if epsilon > 0:
        explore_mask = np.random.random(n_g) < epsilon
        for i in range(n_g):
            if explore_mask[i]:
                actions_np[i] = sample_random_valid_action(obs_g, i)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            actions_t = torch.from_numpy(actions_np).long().to(device)
            _, log_probs, _, _ = model.get_action_and_value(obs_t, action=actions_t, action_mask=action_mask)

    log_probs_np = log_probs.float().cpu().numpy()
    sides = obs_g["scalars"][:, _SC_ACTIVE_SIDE].astype(np.int32)

    # Non-blocking: env subprocesses start stepping immediately
    env_group.step_async(actions_np)

    return {
        "obs": obs_g,
        "actions": actions_np,
        "log_probs": log_probs_np,
        "values": values_np,
        "sides": sides,
    }


def _store_group_step(buffer: RolloutBuffer, step: int, env_offset: int,
                      data: dict, rewards, dones_f32):
    """Write one microbatch's data into the buffer at [step, env_slice]."""
    n = len(rewards)
    sl = slice(env_offset, env_offset + n)
    buffer.obs_scalars[step, sl] = data["obs"]["scalars"]
    buffer.obs_stacks[step, sl] = data["obs"]["stacks"]
    buffer.obs_obstacles[step, sl] = data["obs"]["obstacles"]
    buffer.obs_reachable[step, sl] = data["obs"]["reachable_hexes"]
    buffer.obs_n_stacks[step, sl] = data["obs"]["n_stacks"]
    buffer.obs_attack_targets[step, sl] = data["obs"]["attack_targets"]
    buffer.mask_action_type[step, sl] = data["obs"]["action_type_mask"]
    buffer.mask_hex[step, sl] = data["obs"]["hex_mask"]
    buffer.mask_target[step, sl] = data["obs"]["target_mask"]
    buffer.actions[step, sl] = data["actions"]
    buffer.log_probs[step, sl] = data["log_probs"]
    buffer.values[step, sl] = data["values"]
    buffer.rewards[step, sl] = rewards
    buffer.dones[step, sl] = dones_f32
    buffer.sides[step, sl] = data["sides"]


def collect_rollout(
    env_groups: list[gym.vector.VectorEnv],
    model: BattleTransformer,
    buffer: RolloutBuffer,
    device: torch.device,
    obs_groups: list[dict[str, np.ndarray]],
    step_bar=None,
    epsilon: float = 0.0,
    amp_dtype: torch.dtype | None = None,
) -> tuple[dict, list[dict[str, np.ndarray]]]:
    """Collect one rollout with cross-step microbatch pipelining.

    Environments are split into *env_groups* (microbatches).  Each group
    advances through rollout steps independently: after collecting a
    group's result the GPU immediately computes and dispatches the next
    step for that group, even if other groups haven't finished their
    current step yet.  This overlaps GPU inference for one group with
    environment wall-clock time of the others.

    Returns (stats_dict, obs_groups) so the caller can carry obs across
    rollouts.
    """
    M = len(env_groups)
    num_envs = sum(g.num_envs for g in env_groups)
    S = buffer.num_steps

    episode_rewards = np.zeros(num_envs)
    episode_enemy_killed = np.zeros(num_envs)
    episode_own_lost = np.zeros(num_envs)
    completed_episodes = []
    completed_enemy_killed = []
    completed_own_lost = []
    total_steps = 0

    buffer.reset()

    # Precompute env-index offsets for each microbatch
    env_offsets = []
    offset = 0
    for g in range(M):
        env_offsets.append(offset)
        offset += env_groups[g].num_envs

    # Per-group pipeline state
    group_step = [0] * M          # next step index to fill for each group
    pending_data: list[dict | None] = [None] * M

    # Warm up: compute and dispatch step 0 for all groups.
    # Earlier groups start stepping while later ones are still computed.
    for g in range(M):
        pending_data[g] = _compute_and_dispatch(
            env_groups[g], obs_groups[g], model, device, epsilon,
            amp_dtype=amp_dtype,
        )

    # Progress tracking
    last_bar_step = 0

    # Pipeline: round-robin wait → store → compute next → dispatch
    while True:
        any_active = False
        for g in range(M):
            if group_step[g] >= S:
                continue
            any_active = True

            # Wait for this group's pending env step
            n_g = env_groups[g].num_envs
            next_obs, rewards, terminated, truncated, infos = (
                env_groups[g].step_wait()
            )
            dones = terminated | truncated
            dones_f32 = dones.astype(np.float32)

            # Store into buffer at this group's current step / env slice
            _store_group_step(
                buffer, group_step[g], env_offsets[g],
                pending_data[g], rewards, dones_f32,
            )

            # Update obs and advance step
            obs_groups[g] = next_obs
            group_step[g] += 1
            total_steps += n_g

            # Immediately compute & dispatch the *next* step for this
            # group so its envs start working while we process other
            # groups (or while the PPO update runs after the loop).
            if group_step[g] < S:
                pending_data[g] = _compute_and_dispatch(
                    env_groups[g], obs_groups[g], model, device, epsilon,
                )

            # ---- Episode stats for this group's envs ----
            sl = slice(env_offsets[g], env_offsets[g] + n_g)
            episode_rewards[sl] += rewards

            enemy_killed_arr = infos.get("enemy_killed_value", np.zeros(n_g))
            own_lost_arr = infos.get("own_lost_value", np.zeros(n_g))
            if isinstance(enemy_killed_arr, np.ndarray):
                episode_enemy_killed[sl] += enemy_killed_arr
            if isinstance(own_lost_arr, np.ndarray):
                episode_own_lost[sl] += own_lost_arr

            for i in range(n_g):
                if dones[i]:
                    gi = env_offsets[g] + i
                    completed_episodes.append(episode_rewards[gi])
                    completed_enemy_killed.append(episode_enemy_killed[gi])
                    completed_own_lost.append(episode_own_lost[gi])
                    episode_rewards[gi] = 0.0
                    episode_enemy_killed[gi] = 0.0
                    episode_own_lost[gi] = 0.0

            # Update progress bar when the slowest group advances
            if step_bar is not None:
                min_step = min(group_step)
                if min_step > last_bar_step:
                    step_bar.update(min_step - last_bar_step)
                    last_bar_step = min_step
                    step_bar.set_postfix(
                        ep=len(completed_episodes),
                    )

        if not any_active:
            break

    # Flush progress bar
    if step_bar is not None and last_bar_step < S:
        step_bar.update(S - last_bar_step)
        step_bar.set_postfix(
            ep=len(completed_episodes),
        )

    # Bootstrap value for GAE (per-side)
    last_obs = {
        k: np.concatenate([obs_groups[g][k] for g in range(M)])
        for k in obs_groups[0]
    }
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        obs_t = obs_to_tensors(last_obs, device)
        _, _, _, last_values = model.get_action_and_value(obs_t)
        last_values_np = last_values.float().cpu().numpy()

    last_all_dones = buffer.dones[S - 1]
    last_sides = last_obs["scalars"][:, _SC_ACTIVE_SIDE].astype(np.int32)
    buffer.compute_gae(last_values_np, last_all_dones, last_sides)

    stats = {
        "total_steps": total_steps,
        "completed_episodes": len(completed_episodes),
        "mean_reward": float(np.mean(completed_episodes)) if completed_episodes else 0.0,
        "min_reward": float(np.min(completed_episodes)) if completed_episodes else 0.0,
        "max_reward": float(np.max(completed_episodes)) if completed_episodes else 0.0,
        "mean_enemy_killed": float(np.mean(completed_enemy_killed)) if completed_enemy_killed else 0.0,
        "mean_own_lost": float(np.mean(completed_own_lost)) if completed_own_lost else 0.0,
    }
    return stats, obs_groups


def run_playtest(
    env: gym.Env,
    model: BattleTransformer,
    device: torch.device,
):
    """Run interactive playtest: human vs AI in a graphical game."""
    logger.info("Starting playtest — set up a game in the VCMI lobby.")
    logger.info("Configure one player as human and the other as AI.")

    obs, info = env.reset()
    logger.info("Battle started! AI is playing.")

    total_reward = 0.0
    steps = 0
    battles = 0

    try:
        while True:
            obs_t = obs_to_tensors_single(obs, device)
            action_mask = {
                "action_type": torch.from_numpy(obs["action_type_mask"]).unsqueeze(0).to(device),
                "hex": torch.from_numpy(obs["hex_mask"]).unsqueeze(0).to(device),
                "target": torch.from_numpy(obs["target_mask"]).unsqueeze(0).to(device),
            }

            with torch.no_grad():
                action, log_prob, entropy, value = model.get_action_and_value(obs_t, action_mask=action_mask)

            action_np = action[0].cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                battles += 1
                result = "WON" if info.get("win_bonus", 0) > 0 else "LOST" if info.get("win_bonus", 0) < 0 else "DRAW"
                logger.info(
                    f"Battle {battles} finished: {result} "
                    f"(reward={total_reward:.2f}, steps={steps})"
                )
                try:
                    obs, info = env.reset()
                    logger.info("Next battle started! AI is playing.")
                    total_reward = 0.0
                    steps = 0
                except RuntimeError:
                    logger.info("Game ended. Playtest complete.")
                    break
    except KeyboardInterrupt:
        logger.info("Playtest interrupted by user.")
    finally:
        env.close()

    logger.info(f"Playtest summary: {battles} battles played")


def main():
    parser = argparse.ArgumentParser(description="VCMI RL Training")

    # Environment
    parser.add_argument("--vcmi-client", required=True, help="Path to vcmiclient binary")
    parser.add_argument("--test-map", default=None, help="Map name relative to vcmi-cwd")
    parser.add_argument("--vcmi-cwd", default=None, help="Working directory for game process")
    parser.add_argument("--port", type=int, default=10000, help="Base port for game connections")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--microbatches", type=int, default=1,
                        help="Split envs into N microbatches to overlap GPU inference with env stepping")
    parser.add_argument("--max-steps", type=int, default=500, help="Truncate episodes after N steps (0 = no limit)")

    # Model
    parser.add_argument("--d-model", type=int, default=128, help="Transformer model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers")

    # PPO hyperparameters
    parser.add_argument("--num-steps", type=int, default=256, help="Rollout steps per iteration per env")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO update epochs per iteration")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size for PPO updates")
    parser.add_argument("--target-kl", type=float, default=None, help="Target KL for early stopping")
    parser.add_argument("--eps-start", type=float, default=0.1, help="Initial exploration rate for epsilon-greedy")
    parser.add_argument("--eps-end", type=float, default=0.0, help="Final exploration rate for epsilon-greedy")

    # Checkpointing and logging
    parser.add_argument("--checkpoint", default=None, help="Path to model-only checkpoint to load (weights only)")
    parser.add_argument("--resume", default=None, help="Path to full training checkpoint to resume from")
    parser.add_argument("--save-dir", default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save-interval", type=int, default=50, help="Save checkpoint every N iterations")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity name")
    parser.add_argument("--wandb-project", default="vcmi-rl", help="W&B project name")
    parser.add_argument("--wandb-name", default=None, help="W&B run name")

    # Modes
    parser.add_argument("--device", default="auto", help="Device: cpu, cuda, or auto")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--playtest", action="store_true",
                        help="Play against the AI in a graphical game")
    parser.add_argument("--progress", action="store_true",
                        help="Show tqdm progress bars during training")
    parser.add_argument("--fp8", action="store_true",
                        help="Enable FP8 mixed precision training (requires Ada/Hopper GPU)")
    parser.add_argument("--bf16", action="store_true",
                        help="Enable BF16 mixed precision training")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile to optimize the model")
    parser.add_argument("--sync-envs", action="store_true",
                        help="Use SyncVectorEnv instead of AsyncVectorEnv (for debugging)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("vcmigym.vcmi_types").setLevel(logging.WARNING)

    if not args.playtest and args.test_map is None:
        parser.error("--test-map is required for training mode")
    if args.fp8 and args.bf16:
        parser.error("--fp8 and --bf16 are mutually exclusive")
    if args.checkpoint and args.resume:
        parser.error("--checkpoint and --resume are mutually exclusive")

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

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        logger.info(f"Loaded model weights: {args.checkpoint}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        logger.info(f"Loaded model from resume checkpoint: {args.resume}")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created: {n_params:,} parameters")

    if args.fp8:
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        import torch.nn as nn

        def _fp8_module_filter(module: nn.Module, fqn: str) -> bool:
            """Only convert nn.Linear layers whose dimensions are both divisible by 16."""
            if not isinstance(module, nn.Linear):
                return False
            return module.in_features % 16 == 0 and module.out_features % 16 == 0

        config = Float8LinearConfig(pad_inner_dim=True)
        convert_to_float8_training(model, config=config, module_filter_fn=_fp8_module_filter)
        converted = sum(
            1 for _, m in model.named_modules()
            if type(m).__name__ == "Float8Linear"
        )
        total_linear = sum(
            1 for _, m in model.named_modules()
            if isinstance(m, (nn.Linear,)) or type(m).__name__ == "Float8Linear"
        )
        logger.info(f"FP8 training enabled: {converted}/{total_linear} linear layers converted")

    amp_dtype = None
    if args.bf16:
        amp_dtype = torch.bfloat16
        logger.info("BF16 mixed precision enabled")

    # Keep reference to unwrapped model for checkpoint save/load
    # (torch.compile adds _orig_mod. prefix to state dict keys)
    raw_model = model

    if args.compile:
        model = torch.compile(model)
        logger.info("Model compiled with torch.compile")

    # ---- Playtest mode ----
    if args.playtest:
        model.eval()
        env = make_vcmi_env(
            vcmi_client=args.vcmi_client,
            port_base=args.port,
            vcmi_cwd=args.vcmi_cwd,
            playtest=True,
        )
        run_playtest(env, model, device)
        return

    # ---- Training mode ----

    # Weights & Biases
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.wandb_name,
                config=vars(args),
            )
            wandb.watch(model, log="gradients", log_freq=100)
            logger.info(f"W&B run: {wandb_run.url}")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    num_mb = args.microbatches
    if args.num_envs % num_mb != 0:
        parser.error(f"--num-envs ({args.num_envs}) must be divisible by --microbatches ({num_mb})")
    envs_per_mb = args.num_envs // num_mb

    vec_env_cls = gym.vector.SyncVectorEnv if args.sync_envs else gym.vector.AsyncVectorEnv
    if args.sync_envs:
        logger.info("Using SyncVectorEnv (debug mode)")

    env_groups: list[gym.vector.VectorEnv] = []
    for mb in range(num_mb):
        start = mb * envs_per_mb
        group = vec_env_cls([
            lambda i=i: make_vcmi_env(
                i,
                vcmi_client=args.vcmi_client,
                test_map=args.test_map,
                port_base=args.port,
                vcmi_cwd=args.vcmi_cwd,
                max_steps=args.max_steps,
            )
            for i in range(start, start + envs_per_mb)
        ])
        env_groups.append(group)

    buffer = RolloutBuffer(args.num_steps, args.num_envs, device)
    trainer = PPOTrainer(
        model,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        target_kl=args.target_kl,
        amp_dtype=amp_dtype,
    )

    # Restore optimizer state and counters when resuming
    start_iteration = 1
    global_step = 0
    if args.resume:
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        start_iteration = ckpt["iteration"] + 1
        global_step = ckpt.get("global_step", 0)
        logger.info(f"Resumed training from iteration {ckpt['iteration']}, global_step={global_step}")
        del ckpt  # free memory

    logger.info(
        f"PPO training: {args.num_envs} envs × {args.num_steps} steps/iter "
        f"= {args.num_envs * args.num_steps} samples/iter, "
        f"iterations {start_iteration}..{args.iterations}"
        + (f", {num_mb} microbatches" if num_mb > 1 else "")
    )

    # Progress bars (optional)
    tqdm_cls = None
    if args.progress:
        try:
            from tqdm import tqdm
            tqdm_cls = tqdm
        except ImportError:
            logger.warning("tqdm not installed, disabling progress bars")

    iteration = start_iteration - 1  # track last completed iteration for final checkpoint
    obs_groups = [g.reset()[0] for g in env_groups]
    try:
        iter_range = range(start_iteration, args.iterations + 1)
        if tqdm_cls is not None:
            iter_bar = tqdm_cls(iter_range, desc="Training", unit="iter")
        else:
            iter_bar = iter_range

        for iteration in iter_bar:
            t0 = time.time()

            # Epsilon schedule: linear decay from eps_start to eps_end
            frac = (iteration - 1) / max(args.iterations - 1, 1)
            epsilon = args.eps_start + (args.eps_end - args.eps_start) * frac

            # Collect rollout
            model.eval()
            step_bar = None
            if tqdm_cls is not None:
                step_bar = tqdm_cls(
                    total=args.num_steps, desc="  Rollout", unit="step",
                    leave=False,
                )
            rollout_stats, obs_groups = collect_rollout(
                env_groups, model, buffer, device, obs_groups,
                step_bar=step_bar, epsilon=epsilon, amp_dtype=amp_dtype,
            )
            if step_bar is not None:
                step_bar.close()
            global_step += rollout_stats["total_steps"]

            # PPO update
            model.train()
            ppo_metrics = trainer.update(buffer)

            elapsed = time.time() - t0
            sps = rollout_stats["total_steps"] / elapsed

            # Update iteration progress bar
            if tqdm_cls is not None and hasattr(iter_bar, "set_postfix"):
                iter_bar.set_postfix(
                    reward=f"{rollout_stats['mean_reward']:.3f}",
                    eps=f"{epsilon:.2f}",
                    sps=f"{sps:.0f}",
                )

            # Log
            logger.info(
                f"Iter {iteration}/{args.iterations}: "
                f"episodes={rollout_stats['completed_episodes']}, "
                f"reward={rollout_stats['mean_reward']:.3f} "
                f"[{rollout_stats['min_reward']:.1f}, {rollout_stats['max_reward']:.1f}], "
                f"killed={rollout_stats['mean_enemy_killed']:.3f}, "
                f"lost={rollout_stats['mean_own_lost']:.3f}, "
                f"pg_loss={ppo_metrics['policy_loss']:.4f}, "
                f"v_loss={ppo_metrics['value_loss']:.4f}, "
                f"entropy={ppo_metrics['entropy']:.3f}, "
                f"kl={ppo_metrics['approx_kl']:.4f}, "
                f"eps={epsilon:.3f}, "
                f"{sps:.0f} steps/s"
            )

            if wandb_run is not None:
                import wandb
                wandb.log({
                    "rollout/episodes": rollout_stats["completed_episodes"],
                    "rollout/mean_reward": rollout_stats["mean_reward"],
                    "rollout/min_reward": rollout_stats["min_reward"],
                    "rollout/max_reward": rollout_stats["max_reward"],
                    "rollout/mean_enemy_killed": rollout_stats["mean_enemy_killed"],
                    "rollout/mean_own_lost": rollout_stats["mean_own_lost"],
                    "ppo/policy_loss": ppo_metrics["policy_loss"],
                    "ppo/value_loss": ppo_metrics["value_loss"],
                    "ppo/entropy": ppo_metrics["entropy"],
                    "ppo/approx_kl": ppo_metrics["approx_kl"],
                    "ppo/clip_fraction": ppo_metrics["clip_fraction"],
                    "ppo/total_loss": ppo_metrics["total_loss"],
                    "perf/sps": sps,
                    "perf/iteration_time": elapsed,
                    "exploration/epsilon": epsilon,
                    "global_step": global_step,
                }, step=iteration)

            # Save checkpoint
            if iteration % args.save_interval == 0:
                ckpt_path = save_dir / f"checkpoint_iter{iteration}.pt"
                torch.save({
                    "model": raw_model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "iteration": iteration,
                    "global_step": global_step,
                }, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    finally:
        for g in env_groups:
            g.close()
        # Save final checkpoint
        ckpt_path = save_dir / "checkpoint_latest.pt"
        torch.save({
            "model": raw_model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "iteration": iteration,
            "global_step": global_step,
        }, ckpt_path)
        logger.info(f"Saved final checkpoint: {ckpt_path}")

        if wandb_run is not None:
            import wandb
            wandb.finish()

    logger.info("Training complete")


if __name__ == "__main__":
    main()
