"""Mean next-token loss of a causal language model on the first --docs chunks of --ctx tokens of
the WikiText-103 training text.

python loss.py --model Qwen/Qwen2.5-1.5B --docs 2000 --ctx 1024 [--batch 16]
"""

import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROWS = 2000


def chunks(tok, docs, ctx):
    rows = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"]
    ids = []
    for i in range(0, len(rows), ROWS):
        ids += tok("".join(rows[i : i + ROWS]), add_special_tokens=False)["input_ids"]
        if len(ids) >= docs * ctx:
            break
    return torch.tensor(ids[: docs * ctx]).view(docs, ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--docs", type=int, required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}", flush=True)
    x = chunks(AutoTokenizer.from_pretrained(a.model), a.docs, a.ctx)
    m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
    total, t, said = 0.0, time.time(), 0
    with torch.no_grad():
        for i in range(0, a.docs, a.batch):
            b = x[i : i + a.batch].to(dev)
            total += m(input_ids=b, labels=b).loss.item() * len(b)
            done = i + len(b)
            if done * 10 // a.docs > said:
                said = done * 10 // a.docs
                print(f"{done}/{a.docs} docs  {time.time() - t:.0f} s", flush=True)
    print(f"tok_s {a.docs * a.ctx / (time.time() - t):.0f}")
    print(f"loss {total / a.docs:.4f}")


if __name__ == "__main__":
    main()
