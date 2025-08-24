# -*- coding: utf-8 -*-
"""
LongBench-v2 MCQ: last-layer probs + LOO log-odds span attribution + inference output
with token-aware middle truncation and robust choice tokenization.

Tested with:
  - meta-llama/Llama-3.1-8B-Instruct
  - microsoft/Phi-3-mini-128k-instruct
"""

import re
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# 0) Model / tokenizer setup
# ----------------------------
MODEL_ID = "microsoft/Phi-3-mini-128k-instruct"  # or "microsoft/Phi-3-mini-128k-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IN_TOKENS = 4096  # input budget for scoring/generation (adjust to your memory)

print(f"Loading {MODEL_ID} on {DEVICE}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Ensure PAD is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id


# ----------------------------
# 1) Prompt building
# ----------------------------
def build_mcq_user_message(row: Dict) -> str:
    """Build a crisp MCQ user message."""
    choices = []
    for key in ["choice_A", "choice_B", "choice_C", "choice_D"]:
        if key in row and row[key] not in (None, ""):
            choices.append(key[-1] + ". " + str(row[key]))
    choices_text = "\n".join(choices) if choices else "A.\nB.\nC.\nD."

    ctx = row.get("context", "") or ""
    ques = row.get("question", "") or ""

    user_msg = (
        f"{ctx[:1000]}\n\n"
        "=== QUESTION ===\n"
        f"{ques}\n\n"
        "=== CHOICES ===\n"
        f"{choices_text}\n\n"
        "=== FORMAT ===\n"
        "Final answer: <A|B|C|D>"
    )
    return user_msg


def apply_chat_template_safe(messages: List[Dict[str, str]]) -> str:
    """
    Robust wrapper around tokenizer.apply_chat_template across model families.
    """
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        # Older sigs without add_generation_prompt
        return tokenizer.apply_chat_template(messages, tokenize=False)


def build_chat_prompt(row: Dict) -> str:
    msgs = [
        {"role": "system", "content": "You are a careful, concise MCQ solver. Respond with A, B, C, or D only."},
        {"role": "user", "content": build_mcq_user_message(row)},
    ]
    return apply_chat_template_safe(msgs)


# ----------------------------
# 2) Middle truncation
# ----------------------------
def middle_truncate_tokens(tokenizer, text: str, max_tokens: int, keep_head_ratio: float = 0.6) -> str:
    """
    Keep head and tail tokens, drop the middle if over budget.
    """
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text
    head = int(max_tokens * keep_head_ratio)
    tail = max_tokens - head
    kept = ids[:head] + ids[-tail:]
    return tokenizer.decode(kept, skip_special_tokens=True)


def safe_prompt(prompt: str, budget: int = MAX_IN_TOKENS - 16) -> str:
    """Reserve a little room for the final next token."""
    return middle_truncate_tokens(tokenizer, prompt, budget, keep_head_ratio=0.6)


def build_prompt_with_context(row: Dict, new_context: str) -> str:
    r = dict(row)
    r["context"] = new_context
    prompt = build_chat_prompt(r)
    return safe_prompt(prompt)


# ---------------------------------------
# 3) Choice tokenization + next-step probs/log-odds
# ---------------------------------------
def choice_token_ids(tokenizer, choices=("A", "B", "C", "D")) -> List[int]:
    """
    Robustly obtain token IDs for single-letter choices, trying both raw and leading-space tokens.
    """
    ids = []
    for c in choices:
        cand = []
        for s in (c, " " + c):
            enc = tokenizer(s, add_special_tokens=False).input_ids
            if len(enc) >= 1:
                cand.append(enc[-1])
        if not cand:
            # last resort
            cand.append(tokenizer.convert_tokens_to_ids(c))
        ids.append(cand[0])
    return ids


def first_step_probs_and_logodds(
    model,
    tokenizer,
    prompt: str,
    choices: Tuple[str, ...] = ("A", "B", "C", "D"),
    eps: float = 1e-8
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute next-token probabilities and log-odds for the choices (single forward pass).
      returns: (probs_dict, logodds_dict)
    """
    x = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**x).logits[:, -1, :]        # [1, V]
        probs  = torch.softmax(logits, dim=-1)[0]    # [V]

    cids = choice_token_ids(tokenizer, choices)
    p = torch.stack([probs[i] for i in cids])        # [K]
    p = torch.clamp(p, eps, 1 - eps)
    logodds = torch.log(p) - torch.log1p(-p)

    probs_dict   = {c: p_i.item() for c, p_i in zip(choices, p)}
    logodds_dict = {c: l.item()   for c, l   in zip(choices, logodds)}
    return probs_dict, logodds_dict


# ---------------------------------------
# 4) Greedy inference (to inspect model output)
# ---------------------------------------
def greedy_inference(model, tokenizer, prompt: str, max_new_tokens=16) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=False,
        )
    return tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------
# 5) LOO span attribution with log-odds Δᵢ
# ---------------------------------------
def split_into_spans(text: str, level: str = "sentence") -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    if level == "paragraph":
        return [s for s in re.split(r"\n\s*\n", text) if s.strip()]
    if level == "sentence":
        parts = re.split(r"(?<=[.?!])\s+", text)
        return [p for p in parts if p]
    m = re.match(r"chunk(\d+)", level)
    if m:
        n = int(m.group(1))
        toks = text.split()
        return [" ".join(toks[i:i + n]) for i in range(0, len(toks), n)]
    return [text]


def label_spans_by_loo_logodds(
    model,
    tokenizer,
    row: Dict,
    target: str = None,
    level: str = "sentence",
    choices: Tuple[str, ...] = ("A", "B", "C", "D"),
    eps: float = 1e-8,
) -> Dict:
    """
    Δ_i = logodds_baseline(target) - logodds_without_span_i(target)
    Positive Δ_i => span is HELPFUL (removing it hurts target odds).
    """
    context = row.get("context", "") or ""
    spans = split_into_spans(context, level=level)
    joined = " ".join(spans)

    base_prompt = build_prompt_with_context(row, joined)
    base_probs, base_logodds = first_step_probs_and_logodds(model, tokenizer, base_prompt, choices, eps)

    if target is None:
        target = max(base_probs, key=base_probs.get)

    results = []
    for i in range(len(spans)):
        ablated_context = " ".join(spans[:i] + spans[i + 1:])
        p_abl_prompt = build_prompt_with_context(row, ablated_context)
        _, abl_logodds = first_step_probs_and_logodds(model, tokenizer, p_abl_prompt, choices, eps)

        delta = base_logodds[target] - abl_logodds[target]  # matches the equation image
        label = "helpful" if delta > 1e-4 else ("harmful" if delta < -1e-4 else "redundant")

        results.append({
            "index": i,
            "span": spans[i][:160].replace("\n", " "),
            "logodds_base": base_logodds[target],
            "logodds_ablated": abl_logodds[target],
            "delta_logodds": delta,
            "label": label,
        })

    return {
        "target": target,
        "base_probs": base_probs,
        "base_logodds": base_logodds,
        "span_results": results,
    }


# ---------------------------------------
# 6) Demo
# ---------------------------------------
def pick_reasonable_row(ds) -> Dict:
    for r in ds:
        if isinstance(r.get("context", ""), str) and len(r["context"]) < 100000:
            return r
    return ds[0]


def normalize_gold(ans: str) -> str:
    if not ans:
        return ""
    s = ans.strip().upper()
    # Many LB-v2 MC tasks store 'A', but some store tokens like 'A.' or 'Answer: A'
    m = re.search(r"[ABCD]", s)
    return m.group(0) if m else s[:1]


def main():
    print("Loading LongBench-v2 (train split)...")
    ds = load_dataset("THUDM/LongBench-v2", split="train")

    row = pick_reasonable_row(ds)
    gold = normalize_gold(row.get("answer", ""))

    # Build and truncate the full chat prompt
    prompt_full = safe_prompt(build_chat_prompt(row))

    # 1) Last-layer probs (+ log-odds)
    probs_dict, logodds_dict = first_step_probs_and_logodds(model, tokenizer, prompt_full)
    print("\n== Next-token probs (A/B/C/D) ==")
    print({k: round(v, 6) for k, v in probs_dict.items()})
    print("Log-odds:", {k: round(v, 6) for k, v in logodds_dict.items()})
    if gold:
        print("Gold:", gold, "| Argmax:", max(probs_dict, key=probs_dict.get))

    # 2) Greedy inference output
    gen_out = greedy_inference(model, tokenizer, prompt_full, max_new_tokens=32)
    print("\n== Model greedy inference output ==")
    print(repr(gen_out))

    # 3) LOO sentence-level attribution (preview)
    print("\n== LOO sentence-level attribution (first 5 spans) ==")
    res = label_spans_by_loo_logodds(model, tokenizer, row, target=gold or None, level="sentence")
    for r in res["span_results"][:5]:
        print(
            f"[{r['index']:>3}] {r['label']:>9} Δlogodds({res['target']})={r['delta_logodds']:+.6f} "
            f"base={r['logodds_base']:.6f} -> ablated={r['logodds_ablated']:.6f}"
        )
        print("   Span preview:", r["span"], "...\n")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
