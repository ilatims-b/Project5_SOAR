# pcrl/rewards/longbench2_mc.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List, Tuple, Union
import numpy as np

LETTERS = ["A","B","C","D"]

def _flatten_to_text(x, tok=None) -> Union[str, List[str]]:
    """Coerce tuple/list/tensor/dict into a string (or list[str])."""
    # Already string(s)
    if isinstance(x, str):
        return x
    if isinstance(x, list) and x and isinstance(x[0], str):
        return " ".join(x)

    # Tuple/list of mixed things -> join their string forms
    if isinstance(x, (tuple, list)):
        parts = []
        for xx in x:
            parts.append(_flatten_to_text(xx, tok) if not isinstance(xx, (np.ndarray, torch.Tensor)) else _flatten_to_text(xx.tolist(), tok))
        # If any became list[str], flatten to one string
        flat = []
        for p in parts:
            if isinstance(p, list):
                flat.extend(p)
            else:
                flat.append(str(p))
        return " ".join(map(str, flat))

    # Dict with input_ids
    if isinstance(x, dict):
        if "input_ids" in x:
            ids = x["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.squeeze().tolist()
            if tok is not None and isinstance(ids, list) and (len(ids)==0 or isinstance(ids[0], (int, np.integer))):
                return tok.decode(ids, skip_special_tokens=True)
        # Fallback: try common fields
        for key in ("prompt", "compressed_prompt", "question"):
            if key in x:
                return _flatten_to_text(x[key], tok)
        return str(x)

    # Tensor/ndarray of token ids
    if isinstance(x, (np.ndarray, torch.Tensor)):
        arr = x
        if isinstance(arr, torch.Tensor):
            if arr.ndim == 0:
                return str(arr.item())
            arr = arr.cpu().numpy()
        if arr.ndim == 1 and tok is not None and arr.dtype.kind in "iu":
            return tok.decode(arr.tolist(), skip_special_tokens=True)
        return str(arr.tolist())

    return str(x)


def _pick_gold_letter(info_like: Any) -> str:
    """Extract gold answer letter if present; else empty string."""
    if isinstance(info_like, dict):
        for k in ("gold_letter","answer","gold","label","target"):
            v = info_like.get(k, None)
            if isinstance(v, str) and v.strip() in LETTERS:
                return v.strip()
        # Some datasets store 0..3 indices
        for k in ("label_idx","gold_idx"):
            v = info_like.get(k, None)
            if isinstance(v, (int, np.integer)) and 0 <= int(v) < len(LETTERS):
                return LETTERS[int(v)]
    # Nothing obvious
    return ""


def _compute_keep_ratio(gen_out: Dict, fixed_token_counts: Any) -> float:
    """
    Try to compute keep_ratio from what you showed:
      gen_out['compressed_token_counts'] -> list[int] per sample/segment
      fixed_token_counts -> sometimes dict of token id lists per section
    Fallback to 0.0 if we can't infer.
    """
    try:
        kept = 0
        if isinstance(gen_out, dict) and "compressed_token_counts" in gen_out:
            kept = int(np.sum(gen_out["compressed_token_counts"]))
        total = 0
        if isinstance(fixed_token_counts, dict):
            # Your print showed: {'instruction':[...ids...], 'input':[...], 'output':[...]}
            for v in fixed_token_counts.values():
                if isinstance(v, (list, tuple)):
                    total += len(v)
                elif isinstance(v, (np.ndarray, torch.Tensor)):
                    total += int(v.size if isinstance(v, np.ndarray) else v.numel())
        elif isinstance(fixed_token_counts, (list, tuple)):
            total = sum(int(x) for x in fixed_token_counts)
        if total > 0:
            return float(kept) / float(total)
    except Exception:
        pass
    return 0.0


class MCReward:
    def __init__(self, gen_model_name, device="cuda", penalty_lambda=0.5, max_new_tokens=2, verbose=False):
        print(f"Initializing MCReward with model {gen_model_name}")
        self.tok = AutoTokenizer.from_pretrained(gen_model_name, use_fast=True)
        self.lm  = AutoModelForCausalLM.from_pretrained(gen_model_name, torch_dtype="auto").to(device)
        self.device = device
        self.penalty_lambda = penalty_lambda
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose

    @torch.inference_mode()
    def __call__(self, infos, gen_output, fixed_token_counts):
        """
        Returns:
        rewards: float or np.ndarray (B,)
        extras:  {'comp': comp_values, 'sim': sim_values}
        """
        import numpy as np

        # --- 1) Build prompt text, gold letter, keep_ratio ---
        prompt_text = _flatten_to_text(
            infos.get("compressed_prompt", infos.get("prompt")) if isinstance(infos, dict) else infos,
            self.tok,
        )
        gold_letter = _pick_gold_letter(infos)
        keep_ratio = _compute_keep_ratio(gen_output, fixed_token_counts)

        # --- 2) Tokenize + generate (handles str or list[str]) ---
        enc = self.tok(prompt_text, return_tensors="pt", truncation=True, padding="longest").to(self.device)
        out = self.lm.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)  # temperature ignored when do_sample=False

        # --- 3) Decode prediction letters (batch-safe) ---
        B = out.shape[0]
        input_len = enc["input_ids"].shape[1]
        tails = out[:, input_len:]
        texts = [self.tok.decode(tails[i], skip_special_tokens=True) for i in range(B)]
        preds = [next((c for c in txt if c in LETTERS), "") for txt in texts]

        # --- 4) sim (correctness) and comp (length penalty) ---
        if gold_letter:
            sim = np.array([1.0 if p == gold_letter else 0.0 for p in preds], dtype=float)
        else:
            sim = np.zeros(B, dtype=float)

        # keep_ratio may be scalar or list; broadcast to B
        if isinstance(keep_ratio, (list, tuple, np.ndarray)):
            kr = np.asarray(keep_ratio, dtype=float).reshape(-1)
            if kr.size == 1:
                kr = np.repeat(kr, B)
            elif kr.size != B:
                kr = np.repeat(kr[0], B)  # fallback
        else:
            kr = np.full(B, float(keep_ratio), dtype=float)

        comp = self.penalty_lambda * kr                      # penalty component
        rewards = sim - comp                                 # final reward

        # --- 5) Return exactly TWO values (match trainer) ---
        if B == 1:
            return float(rewards[0]), {'comp': float(comp[0]), 'sim': float(sim[0])}
        else:
            return rewards, {'comp': comp.tolist(), 'sim': sim.tolist()}

