"""
Transformer-based battle AI model for VCMI.

Architecture:
    - 187 hex tokens (one per battlefield hex) + 2 hero tokens = 189 tokens
    - Each hex token encodes: unit features (if present), reachability, obstacle info
    - Each hero token encodes: side info, mana, spells cast, etc.
    - Global features (round, terrain, siege) are broadcast-added to all tokens
    - Transformer encoder processes all tokens with self-attention
    - Three output heads: action type, hex selection, target selection
    - Value head for actor-critic training
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vcmigym import (
    BATTLEFIELD_HEXES,
    MAX_STACKS,
    MAX_OBSTACLES,
    MAX_ATTACK_TARGETS,
    STACK_FEATURES,
    OBSTACLE_FEATURES,
    NUM_ACTION_TYPES,
)

# ---------------------------------------------------------------------------
# Observation feature layout (indices into the flat arrays from the env)
# ---------------------------------------------------------------------------

# Stack feature indices (in stacks array, dim=35 per stack)
_S = dict(
    ID=0, CREATURE_ID=1, COUNT=2, FIRST_HP=3, MAX_HP=4,
    TOTAL_HP=5, BASE_AMOUNT=6, KILLED=7,
    ATTACK=8, DEFENSE=9, RANGED_ATTACK=10, RANGED_DEFENSE=11,
    MIN_DMG=12, MAX_DMG=13, MIN_RANGED_DMG=14, MAX_RANGED_DMG=15,
    SPEED=16, INITIATIVE=17,
    POSITION=18, INITIAL_POS=19,
    SIDE=20, SLOT=21, OWNER=22,
    ALIVE=23, IS_SHOOTER=24, CAN_SHOOT=25,
    DOUBLE_WIDE=26, DEFENDING=27, MOVED=28, WAITING=29, CAN_MOVE=30,
    SHOTS_LEFT=31, SHOTS_TOTAL=32, RETALIATIONS_LEFT=33, LEVEL=34,
)

# Scalar observation indices (dim=19)
_SC = dict(
    BATTLE_ID=0, ROUND=1, ACTIVE_STACK_ID=2, ACTIVE_SIDE=3,
    TERRAIN=4, BATTLEFIELD=5, IS_SIEGE=6,
    ATK_COLOR=7, ATK_HAS_HERO=8, ATK_HERO_ID=9,
    ATK_SPELLS=10, ATK_MANA=11, ATK_ENCHANTER=12,
    DEF_COLOR=13, DEF_HAS_HERO=14, DEF_HERO_ID=15,
    DEF_SPELLS=16, DEF_MANA=17, DEF_ENCHANTER=18,
)

BATTLEFIELD_COLS = 17
BATTLEFIELD_ROWS = 11
NUM_CREATURE_TYPES = 256  # embedding table size (HoMM3 has ~200 creatures)

# Number of continuous features per hex (excluding creature embedding)
# has_unit, count_norm, hp_frac, atk, def, r_atk, r_def,
# min_dmg, max_dmg, min_r_dmg, max_r_dmg, speed, initiative,
# side, is_ally, alive, is_shooter, can_shoot, double_wide,
# defending, moved, waiting, can_move,
# shots_left_norm, retaliations_norm, level_norm, is_active,
# reachable, has_obstacle
HEX_CONTINUOUS_DIM = 29

# Number of continuous features per hero token
HERO_CONTINUOUS_DIM = 5   # has_hero, mana_norm, spells_norm, enchanter_norm, side

# Global features broadcast to all tokens
GLOBAL_DIM = 5            # round_norm, terrain, battlefield, is_siege, active_side


class BattleTransformer(nn.Module):
    """
    Transformer model for VCMI battle action prediction.

    Input: gymnasium observation dict
    Output: action_type logits, hex logits, target logits, state value
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        creature_embed_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        num_tokens = BATTLEFIELD_HEXES + 2  # 187 hexes + 2 heroes

        # --- Embeddings ---
        self.creature_embedding = nn.Embedding(
            NUM_CREATURE_TYPES, creature_embed_dim, padding_idx=0
        )
        self.hex_position_embedding = nn.Embedding(BATTLEFIELD_HEXES, d_model)
        # Token type: 0=hex, 1=attacker_hero, 2=defender_hero
        self.token_type_embedding = nn.Embedding(3, d_model)

        # --- Input projections ---
        hex_input_dim = HEX_CONTINUOUS_DIM + creature_embed_dim
        self.hex_proj = nn.Sequential(
            nn.Linear(hex_input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.hero_proj = nn.Sequential(
            nn.Linear(HERO_CONTINUOUS_DIM, d_model),
            nn.LayerNorm(d_model),
        )
        self.global_proj = nn.Linear(GLOBAL_DIM, d_model)

        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.post_norm = nn.LayerNorm(d_model)

        # --- Output heads ---
        # Action type: pool all tokens → predict action type
        self.action_type_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, NUM_ACTION_TYPES),
        )

        # Hex selection: per-hex score for destination
        self.hex_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Target selection: score per stack for attack/shoot target
        # Uses stack positions to index into hex token outputs
        self.target_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Value head: scalar state value
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Args:
            obs: dict with keys matching the gym observation space.
                 All tensors should have shape (batch, ...) and be on the same device.

        Returns:
            dict with:
                action_type_logits: (B, NUM_ACTION_TYPES)
                hex_logits: (B, BATTLEFIELD_HEXES)
                target_logits: (B, MAX_STACKS)
                value: (B,)
        """
        scalars = obs["scalars"]            # (B, 19)
        stacks = obs["stacks"]              # (B, MAX_STACKS, 35)
        obstacles = obs["obstacles"]        # (B, MAX_OBSTACLES, 7)
        reachable = obs["reachable_hexes"]  # (B, 187)
        n_stacks = obs["n_stacks"]          # (B, 1)

        B = scalars.shape[0]
        device = scalars.device

        # --- Build hex tokens ---
        hex_features = self._build_hex_features(
            scalars, stacks, obstacles, reachable, n_stacks, device
        )  # (B, 187, HEX_CONTINUOUS_DIM + creature_embed_dim)

        hex_tokens = self.hex_proj(hex_features)  # (B, 187, d_model)

        # Add positional embedding
        hex_pos_ids = torch.arange(BATTLEFIELD_HEXES, device=device)
        hex_tokens = hex_tokens + self.hex_position_embedding(hex_pos_ids)

        # Add token type embedding (type 0 = hex)
        hex_type = torch.zeros(1, dtype=torch.long, device=device)
        hex_tokens = hex_tokens + self.token_type_embedding(hex_type)

        # --- Build hero tokens ---
        hero_tokens = self._build_hero_tokens(scalars, device)  # (B, 2, d_model)

        # --- Concatenate all tokens ---
        tokens = torch.cat([hex_tokens, hero_tokens], dim=1)  # (B, 189, d_model)

        # --- Add global features ---
        global_feat = self._extract_global_features(scalars)  # (B, GLOBAL_DIM)
        global_bias = self.global_proj(global_feat).unsqueeze(1)  # (B, 1, d_model)
        tokens = tokens + global_bias

        # --- Run transformer ---
        tokens = self.transformer(tokens)  # (B, 189, d_model)
        tokens = self.post_norm(tokens)

        hex_out = tokens[:, :BATTLEFIELD_HEXES, :]  # (B, 187, d_model)
        hero_out = tokens[:, BATTLEFIELD_HEXES:, :]  # (B, 2, d_model)

        # --- Action type head (pool all tokens) ---
        pooled = tokens.mean(dim=1)  # (B, d_model)
        action_type_logits = self.action_type_head(pooled)  # (B, NUM_ACTION_TYPES)

        # --- Hex selection head ---
        hex_logits = self.hex_head(hex_out).squeeze(-1)  # (B, 187)

        # --- Target selection head ---
        # For each stack, look up its hex position and use that token's output
        target_logits = self._compute_target_logits(
            hex_out, stacks, n_stacks, device
        )  # (B, MAX_STACKS)

        # --- Value head ---
        value = self.value_head(pooled).squeeze(-1)  # (B,)

        return {
            "action_type_logits": action_type_logits,
            "hex_logits": hex_logits,
            "target_logits": target_logits,
            "value": value,
        }

    def _build_hex_features(
        self,
        scalars: Tensor,
        stacks: Tensor,
        obstacles: Tensor,
        reachable: Tensor,
        n_stacks: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Build per-hex feature vectors by scattering unit/obstacle data onto the grid."""
        B = scalars.shape[0]

        # Continuous features: (B, 187, HEX_CONTINUOUS_DIM)
        hex_cont = torch.zeros(B, BATTLEFIELD_HEXES, HEX_CONTINUOUS_DIM, device=device)
        # Creature IDs for embedding: (B, 187)
        hex_creature_ids = torch.zeros(B, BATTLEFIELD_HEXES, dtype=torch.long, device=device)

        active_stack_id = scalars[:, _SC["ACTIVE_STACK_ID"]]  # (B,)
        active_side = scalars[:, _SC["ACTIVE_SIDE"]]           # (B,)

        # Scatter stack features onto hex positions
        for b in range(B):
            ns = int(n_stacks[b, 0].item())
            for i in range(ns):
                s = stacks[b, i]
                pos = int(s[_S["POSITION"]].item())
                if pos < 0 or pos >= BATTLEFIELD_HEXES:
                    continue

                alive = s[_S["ALIVE"]]
                if alive < 0.5:
                    continue

                max_hp = s[_S["MAX_HP"]].clamp(min=1)
                is_active = (s[_S["ID"]] == active_stack_id[b]).float()
                is_ally = (s[_S["SIDE"]] == active_side[b]).float()

                creature_id = int(s[_S["CREATURE_ID"]].item())
                hex_creature_ids[b, pos] = min(creature_id, NUM_CREATURE_TYPES - 1)

                hex_cont[b, pos] = torch.tensor([
                    1.0,                                  # has_unit
                    s[_S["COUNT"]] / 1000.0,              # count normalized
                    s[_S["FIRST_HP"]] / max_hp,           # hp fraction
                    s[_S["ATTACK"]] / 100.0,
                    s[_S["DEFENSE"]] / 100.0,
                    s[_S["RANGED_ATTACK"]] / 100.0,
                    s[_S["RANGED_DEFENSE"]] / 100.0,
                    s[_S["MIN_DMG"]] / 100.0,
                    s[_S["MAX_DMG"]] / 100.0,
                    s[_S["MIN_RANGED_DMG"]] / 100.0,
                    s[_S["MAX_RANGED_DMG"]] / 100.0,
                    s[_S["SPEED"]] / 20.0,
                    s[_S["INITIATIVE"]] / 20.0,
                    s[_S["SIDE"]],                        # side (0 or 1)
                    is_ally,                              # is ally of active side
                    alive,
                    s[_S["IS_SHOOTER"]],
                    s[_S["CAN_SHOOT"]],
                    s[_S["DOUBLE_WIDE"]],
                    s[_S["DEFENDING"]],
                    s[_S["MOVED"]],
                    s[_S["WAITING"]],
                    s[_S["CAN_MOVE"]],
                    s[_S["SHOTS_LEFT"]] / 30.0,
                    s[_S["RETALIATIONS_LEFT"]] / 5.0,
                    s[_S["LEVEL"]] / 10.0,
                    is_active,                            # is this the active stack
                    0.0,                                  # reachable (filled below)
                    0.0,                                  # has_obstacle (filled below)
                ], device=device)

        # Fill reachable flags
        hex_cont[:, :, 27] = reachable

        # Scatter obstacle data
        for b in range(B):
            for i in range(MAX_OBSTACLES):
                o = obstacles[b, i]
                obs_id = o[0]
                if obs_id <= 0:
                    continue
                pos = int(o[2].item())  # obstacle position
                if 0 <= pos < BATTLEFIELD_HEXES:
                    hex_cont[b, pos, 28] = 1.0  # has_obstacle

        # Creature embedding
        creature_embed = self.creature_embedding(hex_creature_ids)  # (B, 187, embed_dim)

        return torch.cat([hex_cont, creature_embed], dim=-1)

    def _build_hero_tokens(self, scalars: Tensor, device: torch.device) -> Tensor:
        """Build 2 hero tokens from scalar observation."""
        B = scalars.shape[0]

        # Attacker hero
        atk = torch.stack([
            scalars[:, _SC["ATK_HAS_HERO"]],
            scalars[:, _SC["ATK_MANA"]] / 300.0,
            scalars[:, _SC["ATK_SPELLS"]] / 10.0,
            scalars[:, _SC["ATK_ENCHANTER"]] / 10.0,
            torch.zeros(B, device=device),          # side = 0 (attacker)
        ], dim=-1)  # (B, 5)

        # Defender hero
        defn = torch.stack([
            scalars[:, _SC["DEF_HAS_HERO"]],
            scalars[:, _SC["DEF_MANA"]] / 300.0,
            scalars[:, _SC["DEF_SPELLS"]] / 10.0,
            scalars[:, _SC["DEF_ENCHANTER"]] / 10.0,
            torch.ones(B, device=device),           # side = 1 (defender)
        ], dim=-1)  # (B, 5)

        atk_token = self.hero_proj(atk)  # (B, d_model)
        def_token = self.hero_proj(defn)  # (B, d_model)

        # Add token type embeddings (1=attacker_hero, 2=defender_hero)
        atk_type = torch.ones(1, dtype=torch.long, device=device)
        def_type = torch.full((1,), 2, dtype=torch.long, device=device)
        atk_token = atk_token + self.token_type_embedding(atk_type)
        def_token = def_token + self.token_type_embedding(def_type)

        return torch.stack([atk_token, def_token], dim=1)  # (B, 2, d_model)

    def _extract_global_features(self, scalars: Tensor) -> Tensor:
        """Extract global battle features from scalars."""
        return torch.stack([
            scalars[:, _SC["ROUND"]] / 50.0,
            scalars[:, _SC["TERRAIN"]] / 10.0,
            scalars[:, _SC["BATTLEFIELD"]] / 10.0,
            scalars[:, _SC["IS_SIEGE"]],
            scalars[:, _SC["ACTIVE_SIDE"]],
        ], dim=-1)

    def _compute_target_logits(
        self,
        hex_out: Tensor,
        stacks: Tensor,
        n_stacks: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Compute per-stack target logits using the hex output at each stack's position."""
        B = hex_out.shape[0]
        target_logits = torch.full((B, MAX_STACKS), -1e9, device=device)

        for b in range(B):
            ns = int(n_stacks[b, 0].item())
            for i in range(ns):
                pos = int(stacks[b, i, _S["POSITION"]].item())
                alive = stacks[b, i, _S["ALIVE"]].item()
                if pos < 0 or pos >= BATTLEFIELD_HEXES or alive < 0.5:
                    continue
                token_feat = hex_out[b, pos]  # (d_model,)
                target_logits[b, i] = self.target_head(token_feat).squeeze(-1)

        return target_logits

    def get_action_and_value(
        self,
        obs: dict[str, Tensor],
        action: Tensor | None = None,
        action_mask: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        For PPO-style training: sample or evaluate actions.

        Args:
            obs: observation dict
            action: if provided, evaluate this action; otherwise sample
            action_mask: optional dict with 'action_type' (B, 5), 'hex' (B, 187),
                        'target' (B, MAX_STACKS) boolean masks of valid actions

        Returns:
            action: (B, 3) tensor [action_type, hex, target]
            log_prob: (B,) log probability of the action
            entropy: (B,) entropy of the action distribution
            value: (B,) state value
        """
        out = self.forward(obs)

        at_logits = out["action_type_logits"]
        hex_logits = out["hex_logits"]
        tgt_logits = out["target_logits"]

        # Apply action masks
        if action_mask is not None:
            if "action_type" in action_mask:
                at_logits = at_logits.masked_fill(~action_mask["action_type"], -1e9)
            if "hex" in action_mask:
                hex_logits = hex_logits.masked_fill(~action_mask["hex"], -1e9)
            if "target" in action_mask:
                tgt_logits = tgt_logits.masked_fill(~action_mask["target"], -1e9)

        at_dist = torch.distributions.Categorical(logits=at_logits)
        hex_dist = torch.distributions.Categorical(logits=hex_logits)
        tgt_dist = torch.distributions.Categorical(logits=tgt_logits)

        if action is None:
            at = at_dist.sample()
            hx = hex_dist.sample()
            tg = tgt_dist.sample()
            action = torch.stack([at, hx, tg], dim=-1)
        else:
            at = action[:, 0]
            hx = action[:, 1]
            tg = action[:, 2]

        log_prob = at_dist.log_prob(at) + hex_dist.log_prob(hx) + tgt_dist.log_prob(tg)
        entropy = at_dist.entropy() + hex_dist.entropy() + tgt_dist.entropy()

        return action, log_prob, entropy, out["value"]
