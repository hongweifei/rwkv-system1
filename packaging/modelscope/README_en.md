# RWKV-7 decision heads (a System 1 classification / scoring model)

[中文](README.md) · English

One line: **give it a state (a ticket, a dialogue, an alert) and it picks among a fixed set of
options, returning a calibrated probability distribution.** It does not generate text.

The backbone is the frozen
[RWKV7-G1 "GooseOne" 0.1B](https://www.modelscope.cn/models/RWKV/rwkv7-g1) (Apache-2.0), and **its
weights are inside this package** — nothing else to download. That does not make them ours: the 402
base tensors were trained by the RWKV project (attribution in [NOTICE.md](NOTICE.md)); the new work
here is the decision heads on top of them.

The weights ship in **two containers, `model.safetensors` and `model.pth`, holding the same
tensors**: the build reloads each one, requires 402/402 base tensors identical to reading the upstream
file and a forward pass with exactly zero drift, then compares the two files against each other and
refuses to emit a package where any tensor differs. `config.json` records a sha256 per file and a
digest per head, checked before a single tensor is used. Nothing about the arithmetic changes with the
container: the stored tensors are raw (all bfloat16), and the widening plus `ln0` fusion happen in one
shared `normalise_native` at load time. safetensors is the default; `DecisionModel(".", fmt="pth")`
reads the other. <!-- generated:counts -->(filled by the build script)<!-- end:generated -->

## Getting started

```bash
pip install -r requirements.txt
python examples/selfcheck.py     # must end with: OK, this model reproduces the numbers on its card.
```

```python
from modeling.service import DecisionModel
model = DecisionModel(".")                        # default: model.safetensors
# fmt="pth" for the other container; or a repo id: DecisionModel("陆白tuber/rwkv7-system1")
model.score("laya_needs_human", '{"thread": [...]}')
```

`confidence` is Laya's `1 − H(p)/log k` — how decided the model is, not the top probability: 0 for a
uniform distribution, 1 for certainty.

## Results

<!-- generated:eval -->
(filled by the build script -- accuracy and ECE are at T=1, on each dataset's own test split)
<!-- end:generated -->

## What it is bad at, stated plainly

- **XNLI (8 languages) is at chance.** Three-way, baseline 33.3%; it scores 34.0–42.0%. Eight times
  more data and a wider head both leave it flat, so this is the frozen backbone losing the
  information, not under-training.
- **`boolq` is below "always answer yes"** (59.17% vs a 61.7% majority class) and needs T=2.0 to stop
  over-claiming.
- **It does not read the rubric.** Re-ordering a trained head's option list drops `ag_news` accuracy
  from 87.5% to 21.9%, so treat it as fixed-label classification, not as a criteria reader.
- **The ten MASSIVE rows share one head.** Those table rows are one model read on ten test splits,
  not ten models.

## Method, briefly

Two-stage training (cross-entropy warm-up, then RLCD on a strictly proper score), then one
temperature per `(question type, option count)` bucket fitted on the dev split under a one-standard-
error guard, so a rescaling is published only where the dev set can resolve it. Dev always comes from
each dataset's own validation split; the test splits are never used to tune anything. Headline
numbers are reported at T=1.

## Provenance

<!-- generated:seeds -->
(filled by the build script -- which seeds are in this package, and what the ensemble measured)
<!-- end:generated -->

Before writing any file, the build script refuses to package unless, for every task: the head
reproduces the published accuracy/ECE/NLL **digit for digit** from cached features; the gather rule
in the cache record matches the one the head will be served; and one real example run through the
full live path picks the same answer as the cached path. `examples/selfcheck.py` replays that third
check anywhere (measured worst deviation 1e-4 on probabilities, tolerance 0.05).

Code and heads: Apache-2.0, see [NOTICE.md](NOTICE.md). Base weights and tokenizer table belong to
the RWKV project and are validated by name and digest rather than redistributed.
