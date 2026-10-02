"""Does an option's answer follow its text or its slot? Plus current headroom on the
option-block budget, measured rather than assumed."""
import json, glob, torch
import torch.nn.functional as Fn
from rwkv_system1.cache_layout import checkpoint, load_head
from rwkv_system1.pipeline.config import Config
from rwkv_system1.backbone import DecisionBackbone, PromptTemplate
from rwkv_system1.features import extract
from rwkv_system1.train import head_logits
from rwkv_system1 import datasets
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

cfg = Config()
bb = "rwkv7-g1d-0.1b"
tok = TRIE_TOKENIZER("weights/rwkv_vocab_v20230424.txt")
t = PromptTemplate()

print("=== option-block length under our 96-token cap (from the cache sidecars) ===")
for p in sorted(glob.glob("cache/*_test_*.json")):
    o = json.load(open(p, encoding="utf-8"))
    key = p.split("0.1b_")[1].split("_test")[0]
    block = t.suffix(o["qtype_name"], "", o["options"])
    print(f"  {key:<18} k={len(o['options']):>3}  block (instructions left out) = "
          f"{len(tok.encode(block)):>4} tokens")

N = 64
key = "ag_news"
backbone = DecisionBackbone(checkpoint(bb),
                            max_suffix_tokens=cfg.max_suffix_tokens)
ex = datasets.load_ag_news(n_train=0, n_dev=0, n_test=N)["test"][:N]
q = ex[0].question
k = len(q.options)
head, temps = load_head(bb, key, cfg)
T = float(list(temps.values())[0]) if temps else 1.0


def score(order):
    opts = [q.options[i] for i in order]
    qs = [(e.question.type.label, e.question.instructions, opts) for e in ex]
    fs = extract(backbone, [e.state for e in ex], qs, [e.answer for e in ex], batch_size=8)
    return head_logits(head, fs).argmax(-1)


ident = list(range(k))
rot = [(i + 2) % k for i in range(k)]
rev = list(reversed(range(k)))
ai, ar, av = score(ident), score(rot), score(rev)
gold = torch.tensor([e.answer for e in ex])


def pct(x):
    return f"{100.0 * float(x):5.1f}%"


print(f"\n=== text vs slot, n={N} rows, same trained head ===")
print(f"  rotate by 2 : answer follows the TEXT {pct((torch.tensor(rot)[ar] == ai).float().mean())}"
      f"   follows the SLOT {pct((ar == ai).float().mean())}")
print(f"  reversed    : answer follows the TEXT {pct((torch.tensor(rev)[av] == ai).float().mean())}"
      f"   follows the SLOT {pct((av == ai).float().mean())}")
print(f"  accuracy on these rows: original {pct((ai == gold).float().mean())}"
      f"   rotated {pct((torch.tensor(rot)[ar] == gold).float().mean())}"
      f"   reversed {pct((torch.tensor(rev)[av] == gold).float().mean())}")
print(f"  most-picked slot, original order: "
      f"{[int((ai == j).sum()) for j in range(k)]}")
