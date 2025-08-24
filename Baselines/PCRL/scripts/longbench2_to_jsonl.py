# scripts/longbench2_to_jsonl.py
import argparse, json, os
from datasets import load_dataset

TEMPLATE = (
    "You are a careful reasoner.\n"
    "Question: {question}\n"
    "Options:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
    "Read the context and answer with a single letter (A, B, C, or D).\n"
    "Context:\n{context}\n\nAnswer:"
)

LETTER2IDX = {"A":0,"B":1,"C":2,"D":3}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_name", default="THUDM/LongBench-v2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_path", default="data/longbench2/train.jsonl")
    ap.add_argument("--max_examples", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    ds = load_dataset(args.hf_name, split=args.split)
    n = len(ds) if args.max_examples <= 0 else min(args.max_examples, len(ds))

    with open(args.out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            ex = ds[i]
            prompt = TEMPLATE.format(
                question=ex["question"],
                A=ex["choice_A"], B=ex["choice_B"], C=ex["choice_C"], D=ex["choice_D"],
                context=ex["context"]
            )
            gold_letter = ex["answer"].strip()
            rec = {
                "id": ex["_id"],
                "domain": ex.get("domain",""),
                "sub_domain": ex.get("sub_domain",""),
                "difficulty": ex.get("difficulty",""),
                "length": ex.get("length",""),
                "prompt": prompt,
                "gold": gold_letter,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
