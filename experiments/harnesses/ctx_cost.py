"""What does a longer prompt actually cost on this implementation, and do wide option sets
encode now?  Three things, measured rather than asserted:

1. cache names must not have moved -- the suffix cap is no longer a truncation, so if the
   fingerprint were recomputed differently every existing split would silently orphan;
2. the cost curve in tokens, and in batch size, because rwkv7.py runs the recurrence as a
   Python loop over time steps -- which decides what "just support 8192" buys us;
3. a 20-option MASSIVE block and the 77 banking77 label names through encode_query, which is
   the thing that raised before.
"""
import glob, gzip, json, time

import torch

from rwkv_system1.cache_layout import cache_name, checkpoint, feature_fingerprint
from rwkv_system1.pipeline.config import Config
from rwkv_system1.backbone import DecisionBackbone

cfg = Config()
print("1) cache-name stability")
fp = feature_fingerprint(cfg)
on_disk = {p.rsplit("_", 1)[-1].split(".")[0] for p in glob.glob("cache/*.pt")}
print(f"   fingerprint now {fp}   fingerprints already on disk {sorted(on_disk)}"
      f"   -> unchanged: {fp in on_disk}")

print("\n2) cost curve (0.1b, CPU, batch 1 unless noted)")
bb = DecisionBackbone(checkpoint("rwkv7-g1d-0.1b"))
toks = torch.randint(0, 65000, (32, 8192))


def run(b, t, reps=1):
    x = toks[:b, :t].clone()
    t0 = time.perf_counter()
    for _ in range(reps):
        bb.model.forward(x, return_hidden=False)
    return (time.perf_counter() - t0) / reps


for t in (192, 512, 1024, 2048, 4096, 8192):
    dt = run(1, t)
    print(f"   b=1  t={t:>5}  {dt:6.2f} s   {1000 * dt / t:5.2f} ms/token"
          f"   -> {dt:5.2f} s for one example of this length")
for b in (1, 4, 16, 32):
    dt = run(b, 512)
    print(f"   t=512 b={b:>3}  {dt:6.2f} s total   {1000 * dt / b / 512:5.2f} ms per"
          f" example-token   ({dt / b:.2f} s/example)")

print("\n3) wide option sets")
b77 = json.load(open("D:/tmp/b77_infos.json", encoding="utf-8"))["default"]["features"]["label"][
    "names"
]
massive = sorted({json.loads(l)["label"] for l in gzip.open("D:/tmp/massive_test_en.json.gz", "rt",
                                                           encoding="utf-8")})[:20]
for name, opts in (("MASSIVE intent (20 sampled)", massive), ("banking77 (77 labels)", b77),
                   ("ag_news (4, the current worst case)", ["World", "Sports", "Business", "Sci/Tech"])):
    ids, markers = bb.encode_query("choice", "Which category does this belong to?", opts)
    print(f"   {name:<36} block {ids.shape[1]:>4} tokens   markers {len(markers)} of {len(opts)}"
          f"   encodes: {len(markers) == len(opts)}")
print(f"   headroom: ctx {bb.ctx_tokens} tokens for state + option block together")
