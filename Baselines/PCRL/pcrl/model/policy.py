# pcrl/model/policy.py
from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
import torch
import torch.nn as nn

from gymnasium.spaces import Space
from gymnasium.spaces.dict import Dict as DictSpace
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule, TensorDict
from transformers import AutoModel  # encoder backbone

from pcrl.utils.warm_start import ActorCriticWarmStartMixin
from pcrl.algorithms.common.distributions import make_masked_proba_distribution


def labels_to_summary(input_batch, label_batch, tokenizer):
    summaries = []
    for input_ids, labels in zip(input_batch, label_batch):
        selected = [int(input_ids[i]) for i in range(len(input_ids)) if labels[i] == 1]
        summary = tokenizer.decode(selected, skip_special_tokens=True)
        summaries.append(summary)
    return summaries


class BatchTokenPolicy(BasePolicy, ActorCriticWarmStartMixin):
    """
    Token-level actor-critic policy over encoder hidden states.

    Key fixes:
      - Ensure input_ids / attention_mask are int64 (long) on the correct device.
      - Clip sequences to model's max_position_embeddings (keep right-most tokens).
      - Fit token-wise logits to the action space by pad/truncate before building the distribution.
    """
    def __init__(
        self,
        observation_space: DictSpace,
        action_space: Space,
        lr_schedule: Schedule,
        model_name: str,
        weight_decay: float = 1e-6,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs=None,
        state_dict: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(observation_space, action_space)
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        self._action_space = action_space
        self.action_dist = make_masked_proba_distribution(action_space)

        self._build_model_heads(model_name)
        self._setup_optimizer(optimizer_kwargs, weight_decay, optimizer_class)
        self.load_from_dict(state_dict)

    # ---------------- internal helpers ----------------
    def _prepare_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Ensure obs tensors are on self.device and have correct dtypes for HF:
          - input_ids: long
          - attention_mask: long
        Also clips to model max length (right side) if needed.
        """
        device = self.device
        out: Dict[str, torch.Tensor] = {}

        # input_ids
        x = obs["input_ids"]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        out["input_ids"] = x.to(device=device, dtype=torch.long)

        # attention_mask
        if "attention_mask" in obs and obs["attention_mask"] is not None:
            m = obs["attention_mask"]
            if not isinstance(m, torch.Tensor):
                m = torch.as_tensor(m)
            out["attention_mask"] = m.to(device=device, dtype=torch.long)
        else:
            # default: everything is real token
            out["attention_mask"] = torch.ones_like(out["input_ids"], device=device, dtype=torch.long)

        return self._clip_obs_to_model_max(out)

    def _clip_obs_to_model_max(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Ensure seq len <= model max (prevents embedding range crashes). Keeps the *last* tokens."""
        max_len = getattr(self._base_model.config, "max_position_embeddings", None)
        if max_len is None:
            return obs
        for k in ("input_ids", "attention_mask"):
            if k in obs and obs[k].size(-1) > max_len:
                obs[k] = obs[k][..., -max_len:]
        return obs

    def _last_token_hidden(self, last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Select the hidden state of the last *non-pad* token per sequence.
        last_hidden: (B, T, H)
        attn_mask:   (B, T)  with 1 for real tokens, 0 for pad (right-padded)
        """
        seq_len = attn_mask.sum(dim=-1)                # (B,)
        idx = (seq_len - 1).clamp(min=0).long()        # (B,)
        b_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        return last_hidden[b_idx, idx]                 # (B, H)

    def _fit_logits_to_action_space(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, T, 2) for per-token binary decision.
        Ensures T matches the expected per-token count derived from the action space.
        - If T > expected: keep last tokens.
        - If T < expected: right-pad with large negative logits.
        """
        total_expected = int(sum(self.action_dist.action_dims))  # e.g., 8192 for (4096 tokens * 2)
        if total_expected % 2 != 0:
            raise ValueError("Action space expects binary logits per token (even total).")
        t_expected = total_expected // 2

        B, T, C = logits.shape
        if T == t_expected:
            return logits
        if T > t_expected:
            return logits[:, -t_expected:, :]
        # pad
        pad_len = t_expected - T
        pad = logits.new_full((B, pad_len, C), fill_value=-1e9)
        return torch.cat([logits, pad], dim=1)

    # ---------------- build/opt ----------------
    def _build_model_heads(self, model_name: str):
        # Use an encoder backbone; for 4k context use a 4k-capable model (e.g., Longformer 4096)
        self._base_model = AutoModel.from_pretrained(model_name).to(self.device)

        hidden = self._base_model.config.hidden_size
        self._value_model = nn.Sequential(
            nn.Linear(hidden, 4096, bias=False),
            nn.GELU(),
            nn.Linear(4096, 1, bias=True),
        ).to(self.device)

        # token-wise binary logits (keep/remove) => shape (B, T, 2)
        self._policy_model = nn.Sequential(
            nn.Linear(hidden, 4096, bias=False),
            nn.GELU(),
            nn.Linear(4096, 2, bias=True),
        ).to(self.device)

    def _setup_optimizer(
        self,
        optimizer_kwargs: Dict[str, Any],
        weight_decay: float,
        optimizer_class: torch.optim.Optimizer,
    ):
        params = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'layer_norm.weight', 'layer_norm.bias']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        self.optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_kwargs)

    # ---------------- core API ----------------
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ):
        obs = self._prepare_obs(obs)

        with torch.no_grad():
            out = self._base_model(
                input_ids=obs['input_ids'],
                attention_mask=obs['attention_mask'],
                return_dict=True,
            )
        last_hidden = out.last_hidden_state  # (B, T, H)

        # Value from last non-pad token
        pooled = self._last_token_hidden(last_hidden, obs['attention_mask'])  # (B, H)
        values = self._value_model(pooled)  # (B, 1)

        # Policy logits per token (B, T, 2) -> fit to action space
        logits = self._policy_model(last_hidden)
        logits = self._fit_logits_to_action_space(logits)
        assert torch.all(torch.isfinite(logits)), "Non-finite logits in policy head"

        distribution = self.action_dist.proba_distribution(logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ):
        obs = self._prepare_obs(obs)

        with torch.no_grad():
            out = self._base_model(
                input_ids=obs['input_ids'],
                attention_mask=obs['attention_mask'],
                return_dict=True,
            )
        last_hidden = out.last_hidden_state

        pooled = self._last_token_hidden(last_hidden, obs['attention_mask'])
        values = self._value_model(pooled)

        logits = self._policy_model(last_hidden)
        logits = self._fit_logits_to_action_space(logits)
        assert torch.all(torch.isfinite(logits)), "Non-finite logits in policy head"

        distribution = self.action_dist.proba_distribution(logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        ent = distribution.entropy()
        return values, log_prob, ent

    def _predict(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        obs = self._prepare_obs(obs)

        with torch.no_grad():
            out = self._base_model(
                input_ids=obs['input_ids'],
                attention_mask=obs['attention_mask'],
                return_dict=True,
            )
            last_hidden = out.last_hidden_state
            logits = self._policy_model(last_hidden)
            logits = self._fit_logits_to_action_space(logits)
            assert torch.all(torch.isfinite(logits)), "Non-finite logits in policy head"

            distribution = self.action_dist.proba_distribution(logits)
            if action_masks is not None:
                distribution.apply_masking(action_masks)
            return distribution.get_actions(deterministic=deterministic)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.set_training_mode(False)
        obs, _ = self.obs_to_tensor(observation)
        with torch.no_grad():
            actions = self._predict(obs, deterministic=deterministic, action_masks=action_masks)
            actions = actions.cpu().numpy()
        return actions, state

    def predict_values(self, obs: TensorDict):
        obs = self._prepare_obs(obs)
        with torch.no_grad():
            out = self._base_model(
                input_ids=obs['input_ids'],
                attention_mask=obs['attention_mask'],
                return_dict=True,
            )
        last_hidden = out.last_hidden_state
        pooled = self._last_token_hidden(last_hidden, obs['attention_mask'])
        values = self._value_model(pooled)
        return values
