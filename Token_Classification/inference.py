# -*- coding: utf-8 -*-
"""
LongBench-v2 MCQ: last-layer probs + LOO span attribution + inference output
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
MODEL_ID = "microsoft/Phi-3-mini-128k-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
# 1) Data + prompt building
# ----------------------------
def build_mcq_user_message(row: Dict) -> str:
    """Your crisp MCQ prompt assembly for a single row."""
    choices = []
    for key in ["choice_A", "choice_B", "choice_C", "choice_D"]:
        if key in row and row[key] is not None and row[key] != "":
            choices.append(key[-1] + ". " + str(row[key]))
    choices_text = "\n".join(choices) if choices else "A.\nB.\nC.\nD."

    ctx = row.get("context", "")
    ques = row.get("question", "")

    user_msg = (
        f"{ctx[:100]}\n\n"
        "=== QUESTION ===\n"
        f"{ques}\n\n"
        "=== CHOICES ===\n"
        f"{choices_text}\n\n"
        "=== FORMAT ===\n"
        "Final answer: <A|B|C|D>"
    )
    return user_msg


def build_chat_prompt(row: Dict) -> str:
    """Wraps the message using the model's chat template."""
    msgs = [
        {"role": "system", "content": "You are a careful, concise MCQ solver. Respond with A, B, C, or D only."},
        {"role": "user", "content": build_mcq_user_message(row)},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ---------------------------------------
# 2) Last-layer probabilities utilities
# ---------------------------------------
def first_step_choice_probs(
    model, tokenizer, prompt: str, choices: Tuple[str, ...] = ("A", "B", "C", "D")
) -> Tuple[Dict[str, float], torch.Tensor, List[int]]:
    """
    Return P(next token = choices) using a single forward pass (no generate).
    - probs_dict: {choice: prob}
    - full_probs_vec: full softmax vector at next token (torch.Tensor [V])
    - choice_ids: tokenizer IDs corresponding to choices
    """
    x = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**x)  # logits: [B, T, V]
        print(out.logits.shape)
        logits = out.logits[:, -1, :]  # next-token logits
        print(logits.shape)
        probs = F.softmax(logits, dim=-1)  # [B, V]
        print(probs.shape)
    choice_ids = tokenizer.convert_tokens_to_ids(list(choices))
    # normalise the probs
    probs_dict = {c: probs[0, i].item() for c, i in zip(choices, choice_ids)}
    return probs_dict, probs[0], choice_ids


def greedy_inference(model, tokenizer, prompt: str, max_new_tokens=8) -> str:
    """Actually generate output (greedy)."""
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
    out = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return out


# ---------------------------------------
# 3) LOO span attribution (unchanged core)
# ---------------------------------------
def split_into_spans(text: str, level: str = "sentence") -> List[str]:
    text = text.strip()
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


def build_prompt_with_context(row: Dict, new_context: str) -> str:
    r = dict(row)
    r["context"] = new_context
    return build_chat_prompt(r)


# context[:x] + context[x+delta:]
# remove one word at a time
# calculate the delta
# three buckets: 10%, 80%, 10%
# token type - punctuations, stop words



def label_spans_by_loo(
    model,
    tokenizer,
    row: Dict,
    target: str = None,
    level: str = "sentence",
    choices: Tuple[str, ...] = ("A", "B", "C", "D"),
    eps: float = 1e-4,
) -> Dict:
    context = row.get("context", "")
    spans = split_into_spans(context, level=level)
    print(spans)
    joined = " ".join(spans)
    base_prompt = build_prompt_with_context(row, joined)

    base_probs_dict, _, _ = first_step_choice_probs(model, tokenizer, base_prompt, choices)

    if target is None:
        target = max(base_probs_dict, key=base_probs_dict.get)

    results = []
    for i in range(len(spans)):
        ablated_context = " ".join(spans[:i] + spans[i + 1:])
        p_abl = build_prompt_with_context(row, ablated_context)
        abl_probs_dict, _, _ = first_step_choice_probs(model, tokenizer, p_abl, choices)

        delta = (abl_probs_dict[target] - base_probs_dict[target]) / base_probs_dict[target]
        print("delta", delta)
        label = "helpful" if delta < -eps else "harmful" if delta > eps else "redundant"

        results.append({
            "index": i,
            "span": spans[i][:100].replace("\n", " "),  # preview
            "p_target_base": base_probs_dict[target],
            "p_target_ablated": abl_probs_dict[target],
            "delta_target": delta,
            "label": label,
        })

    return {"target": target, "base_probs": base_probs_dict, "span_results": results}


# ---------------------------------------
# 4) Demo
# ---------------------------------------
def pick_reasonable_row(ds) -> Dict:
    for r in ds:
        if isinstance(r.get("context", ""), str) and len(r["context"]) < 100000:
            return r
    return ds[0]


def main():
    print("Loading LongBench-v2 (train split)...")
    ds = load_dataset("THUDM/LongBench-v2", split="train")

    row = pick_reasonable_row(ds)
    gold = (row.get("answer") or "").strip().upper()
    prompt_full = build_chat_prompt(row)

    # 1) Last-layer probs
    probs_dict, _, _ = first_step_choice_probs(model, tokenizer, prompt_full)
    print("\n== Last-layer next-token probs (A/B/C/D) ==")
    print({k: round(v, 6) for k, v in probs_dict.items()})
    if gold:
        print("Gold:", gold, "| Argmax:", max(probs_dict, key=probs_dict.get))

    # 2) Greedy inference output
    gen_out = greedy_inference(model, tokenizer, prompt_full, max_new_tokens=100)
    print("\n== Model greedy inference output ==")
    print(repr(gen_out))

    # 3) LOO sentence attribution (preview)
    print("\n== LOO sentence-level attribution (first 5 spans) ==")
    res = label_spans_by_loo(model, tokenizer, row, target=gold or None, level="sentence")
    for r in res["span_results"][:5]:
        print(
            f"[{r['index']:>3}] {r['label']:>9} ΔP({res['target']})={r['delta_target']:+.6f} "
            f"base={r['p_target_base']:.6f} -> ablated={r['p_target_ablated']:.6f}"
        )
        print("   Span preview:", r["span"], "...\n")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
