"""Prove the hand-written RWKV-7 matches the official `rwkv` package on CPU.

The two implementations are structurally different on purpose: ours returns hidden
states and lets us clone the recurrent state, the official one returns only logits.
If their logits agree to float noise on the same weights and the same tokens, then
our recurrence, the v_first carry, the token shift and the group norm are all right.
"""

import gc
import os
import sys
from pathlib import Path

os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"
os.environ["RWKV_JIT_ON"] = "0"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "weights" / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
VOCAB = ROOT / "weights" / "rwkv_vocab_v20230424.txt"

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "\u5ba2\u6237\u6295\u8bc9\u8bf4\u4ed6\u7684\u8ba2\u5355\u88ab\u91cd\u590d\u6263\u6b3e\u4e86\uff0c\u5e0c\u671b\u9000\u6b3e\u3002",
]


def official_logits(prompts: list[str]) -> list[torch.Tensor]:
    from rwkv.model import RWKV_x070
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

    tok = TRIE_TOKENIZER(str(VOCAB))
    model = RWKV_x070(str(CKPT.with_suffix("")), "cpu fp32")
    out = []
    for text in prompts:
        ids = tok.encode(text)
        logits, _ = model.forward(ids, None, full_output=True)
        out.append(logits.float().clone())
    del model
    gc.collect()
    return out


def main() -> int:
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

    from rwkv_system1.rwkv7 import RWKV7

    print("=== running official implementation ===", flush=True)
    ref = official_logits(PROMPTS)

    print("\n=== running rwkv_system1 implementation ===", flush=True)
    tok = TRIE_TOKENIZER(str(VOCAB))
    ours = RWKV7.from_file(CKPT)

    failures = 0
    for text, ref_logits in zip(PROMPTS, ref):
        ids = torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)
        hidden, logits, _ = ours.forward(ids)
        assert hidden.shape == (1, ids.shape[1], ours.n_embd), hidden.shape

        diff = (logits[0].float() - ref_logits).abs()
        rel = diff.max().item() / max(ref_logits.abs().max().item(), 1e-9)
        same_argmax = bool((logits[0].argmax(-1) == ref_logits.argmax(-1)).all())

        print(f"\nprompt: {text!r}")
        print(f"  tokens        : {ids.shape[1]}")
        print(f"  max |delta|   : {diff.max().item():.6f}")
        print(f"  max relative  : {rel:.3e}")
        print(f"  argmax match  : {same_argmax}")
        print(f"  last-token top5: {[tok.decode([int(i)]) for i in logits[0, -1].topk(5).indices]}")
        ok = rel < 1e-5 and same_argmax
        failures += 0 if ok else 1
        print(f"  -> {'PASS' if ok else 'FAIL'}")

    print()
    if failures:
        print(f"FAILED: {failures}/{len(PROMPTS)} prompts diverged")
        return 1
    print(f"OK: all {len(PROMPTS)} prompts match the official implementation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
