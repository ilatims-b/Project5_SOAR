# pcrl/datasets/longbench2.py
import json
from torch.utils.data import Dataset

class LongBench2Jsonl(Dataset):
    def __init__(self, path):
        self.path = path
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            i = 0
            for line in f:
                # if i > 10:
                #     break
                self.rows.append(json.loads(line))
                i += 1
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "id": r["id"],
            "prompt": r["prompt"],    # long text the policy will compress
            "gold": r["gold"],        # "A"/"B"/"C"/"D"
            "meta": {
                "domain": r.get("domain",""),
                "sub_domain": r.get("sub_domain",""),
                "difficulty": r.get("difficulty",""),
                "length": r.get("length","")
            }
        }
