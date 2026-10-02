"""One model, two containers: ``model.pth`` and ``model.safetensors``.

Why a shared module instead of two ad-hoc dumps: publishing both formats means publishing the same
weights twice, and the failure mode of two independent writers is that they stop agreeing -- a tensor
skipped in one, a key spelled differently in the other, a dtype decided differently at save time.
So the mapping between "base section + N head sections" and "a flat key/value store" lives here
once, and the build compares the two files it produced tensor by tensor before it will ship them.

The stored tensors are **raw**: exactly what the upstream checkpoint holds (all 402 of them are
bfloat16).  Widening to fp32 and fusing ``ln0`` into the embedding happens in
``rwkv7.normalise_native`` at load time, on both paths, so a container choice can never change the
arithmetic -- which is the whole reason a second format is allowed to exist here.

Keys, because safetensors has no nesting:

    base.<upstream key>              e.g.  base.emb.weight
    heads/<head id>::<param>         e.g.  heads/rwkv7-g1d-0.1b_boolq_..._h1234::proj.weight

``::`` separates head id from parameter name because both contain dots and underscores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch

BASE = "base."
HEAD = "heads/"
SEP = "::"
FORMATS = ("pth", "safetensors")


def flat_items(base: dict[str, torch.Tensor], heads: dict[str, dict[str, torch.Tensor]]
               ) -> Iterator[tuple[str, torch.Tensor]]:
    for k, v in base.items():
        yield BASE + k, v
    for head_id, state in heads.items():
        for k, v in state.items():
            yield f"{HEAD}{head_id}{SEP}{k}", v


def unflat(flat: dict[str, torch.Tensor]) -> tuple[dict, dict[str, dict]]:
    """Inverse of :func:`flat_items`, with the container-agnostic checks in one place."""
    base: dict[str, torch.Tensor] = {}
    heads: dict[str, dict[str, torch.Tensor]] = {}
    stray = []
    for k, v in flat.items():
        if k.startswith(BASE):
            base[k[len(BASE):]] = v
        elif k.startswith(HEAD) and SEP in k:
            head_id, _, param = k[len(HEAD):].partition(SEP)
            heads.setdefault(head_id, {})[param] = v
        else:
            stray.append(k)
    if stray:
        raise ValueError(f"unrecognised keys (neither {BASE!r} nor {HEAD!r}+{SEP!r}): "
                         f"{stray[:5]}{'…' if len(stray) > 5 else ''}")
    if not base:
        raise ValueError("no base section")
    return base, heads


def save(path: str | Path, base: dict[str, torch.Tensor],
         heads: dict[str, dict[str, torch.Tensor]]) -> dict:
    """Write one container and return its identity (name, size, sha256).

    ``contiguous()`` is a no-op on tensors that already are, and safetensors refuses views that
    share storage -- so the call is here for the day a loader hands us one, not for today.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.contiguous() for k, v in flat_items(base, heads)}
    if p.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(tensors, str(p))
    elif p.suffix in (".pth", ".pt"):
        torch.save({"base": base, "heads": heads}, p)
    else:
        raise ValueError(f"{p.name}: unknown container, expected .pth or .safetensors")
    return identity(p)


def load(path: str | Path) -> tuple[dict, dict[str, dict]]:
    """Read either container back into ``(base, heads)``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        return unflat(load_file(str(p)))
    raw = torch.load(str(p), map_location="cpu", mmap=True, weights_only=True)
    if "base" in raw and "heads" in raw:      # the .pth keeps the nested shape
        return raw["base"], raw["heads"]
    return unflat(raw)                        # …and can be read as a flat map too


def identity(path: str | Path) -> dict:
    """Name, size and full sha256 -- not a prefix, so a mismatch message can show both in full."""
    import hashlib

    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return {"name": p.name, "size": p.stat().st_size, "sha256": h.hexdigest()}


def diff(a: tuple[dict, dict], b: tuple[dict, dict]) -> list[str]:
    """Per-tensor disagreements between two containers, as messages.  Empty means identical.

    Compares shape and dtype before values and reports the *first* disagreements rather than a
    boolean, because "the two files differ" without a name is not actionable at 3 a.m.  Values are
    compared widened to fp32: the stored tensors are bf16, and `torch.equal` on two bf16 tensors
    that are bit-equal is the same answer -- widening is only so a shape/dtype mistake cannot hide
    behind a dtype mismatch on one side.
    """
    problems: list[str] = []
    (ba, ha), (bb, hb) = a, b
    if set(ba) != set(bb):
        problems.append(f"base keys differ: {sorted(set(ba) ^ set(bb))[:5]}")
    if set(ha) != set(hb):
        problems.append(f"head ids differ: {sorted(set(ha) ^ set(hb))[:5]}")
    for k in sorted(set(ba) & set(bb)):
        x, y = ba[k], bb[k]
        if tuple(x.shape) != tuple(y.shape) or x.dtype != y.dtype:
            problems.append(f"base.{k}: {tuple(x.shape)}/{x.dtype} vs {tuple(y.shape)}/{y.dtype}")
        elif not torch.equal(x.to(torch.float32), y.to(torch.float32)):
            problems.append(f"base.{k}: same shape, different values")
    for head_id in sorted(set(ha) & set(hb)):
        sa, sb = ha[head_id], hb[head_id]
        if set(sa) != set(sb):
            problems.append(f"{head_id}: params differ {sorted(set(sa) ^ set(sb))[:4]}")
            continue
        for k in sorted(sa):
            if not torch.equal(sa[k].to(torch.float32), sb[k].to(torch.float32)):
                problems.append(f"{head_id}.{k}: different values")
    return problems
