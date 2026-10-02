# rwkv-system1

[中文](README.md) · English

A frozen RWKV-7 backbone with a per-option scoring head: **give it a piece of state (a ticket, a dialogue, an alert) and it picks one of a fixed set of options, returning a calibrated probability distribution**. It does not generate text.

How to use the model itself and where it fails: the [model card, `packaging/modelscope/README_en.md`](packaging/modelscope/README_en.md) (that card is what the ModelScope page shows). **This file is about the repository**: how the code is laid out, how one experiment runs, why the artifacts stay out of git, and what has to pass before a release.

## What it is

| | |
|---|---|
| Backbone | `rwkv7-g1d-0.1b` (12 layers, d_model 768, vocab 65536), **frozen throughout**; one forward pass per example to gather features |
| Head | per-option scoring MLP, width 256, no set attention, no slot embeddings, ~260k parameters |
| Recipe | CE warm-up 60 epochs → RLCD 30 epochs → temperature fitting; lr 3e-3, batch 64, AdamW wd 1e-2 |
| Ensemble | 3 seeds averaged **in softmax-probability space**, never in logit space |
| Shared head | the ten MASSIVE languages share one head (`joint_massive`) |
| Hardware | CPU only (PyTorch, no CUDA dependency) |

## Where it stands (34 tasks, each on its dataset's own official test split)

| | |
|---|---|
| Suite mean accuracy | **52.04%** |
| Best few | `prompt_injections` 97.4% · `ag_news` 91.6% · `laya_category` 81.0% · `banking77` 76.8% |
| MASSIVE mean over 10 languages | **53.3%** (the task picks one of 60 intents; uniform guessing ≈ 1.7%) |
| Essentially no signal | XNLI across 8 languages, 34.0–39.7% (three-way baseline 33.3%) |
| Below "always answer the majority class" | `boolq` 59.83% vs 61.67% |

Read from `experiments/results/eval_shipped_ens3.json`, the shipped table. Per-task accuracy / ECE / NLL / fitted temperature live in the card's generated table, which the build script computes from the artifacts rather than carrying by hand.

## Getting the repository running

```bash
git clone <this repository> rwkv-system1 && cd rwkv-system1
uv sync                                          # Python 3.14+, torch 2.14 CPU

uv run python tests/invariants.py                # 12 checks, seconds: no network, no checkpoint
```

The suite is green in a **fresh clone**: two of the checks compute a feature identity, and that identity includes a digest of the base checkpoint -- those two stand up their own placeholder files, so the checks cannot silently turn into fake passes for want of a few hundred MB of weights (see `stand_in_weights` in `tests/invariants.py`).

Training or evaluation additionally needs the backbone:

```bash
uv run modelscope download --model RWKV/rwkv7-g1 \
  --include "rwkv7-g1d-0.1b-20260129-ctx8192.pth" --local-dir weights
```

(that is the exact command `cache_layout.checkpoint()` prints when the file is missing).

## One full pass

```bash
python -m rwkv_system1.pipeline data     --tasks banking77   # what the rendered prompt looks like
python -m rwkv_system1.pipeline extract --tasks banking77   # features -> cache/, ~0.68 s/example
python -m rwkv_system1.pipeline train   --tasks banking77 --ensemble-seeds 3
python -m rwkv_system1.pipeline eval    --tasks banking77 --ensemble-seeds 3
python -m rwkv_system1.pipeline predict --tasks boolq --n 3  # live path: tokenise -> prefill -> options -> markers -> head
python -m rwkv_system1.pipeline ablate  --tasks banking77     # one forward pass, several heads
```

A head takes about 45 s; a full extraction runs from minutes to hours per task. Flags such as `--feature-rule {last,pooled,two_layer}`, `--n-train/-dev/-test` and `--targets {gold,crossfit,teacher,hard}` **change the artifact's name**, they do not merely change behaviour.

## Layout

| Directory | Contents |
|---|---|
| `rwkv_system1/` | the package. `backbone.py` templates and prefill; `features.py` marker → vector and the gather rules; `cache_layout.py` **names only**; `train.py` / `rlcd.py` / `decision.py`; `rwkv7.py` a hand-written RWKV-7 forward pass (`scripts/verify_rwkv7.py` checks it against the official `rwkv` package on CPU); `datasets/` one loader per dataset; `pipeline/` the CLI and its five commands |
| `experiments/` | `harnesses/` one file per measurement, `chains/` shell that strings them together and **prints `*_RC=` per step**, `results/` the evidence kept in the repo (JSON tables plus TXT summaries) |
| `packaging/modelscope/` | the hand-authored half of the release: the model card (zh/en), `NOTICE.md`, `modeling/service.py` (the thin service users import after download), `examples/selfcheck.py` |
| `scripts/` | `build_modelscope_pkg.py` (assembly + gates), `upload_modelscope.py`, `smoke_train.py` (exercise the head without extraction), `verify_rwkv7.py`, `prune_caches.py` |
| `tests/invariants.py` | 12 invariants, one per real incident |

## What is not in git, and why

One rule: **anything reproducible from repo contents plus one command stays out of history.**

| Not tracked | How to get it back | Price |
|---|---|---|
| `weights/` | the `modelscope download` above | bandwidth |
| `cache/` | `pipeline extract` | ~0.68 s/example -- the expensive one |
| `runs/` | `pipeline train` | ~45 s per head |
| `dist/` | `python scripts/build_modelscope_pkg.py` | tens of minutes, gates included |
| `.venv/`, `refs/`, `.env` | `uv sync`, a package index, your own keys | — |

The authored files under `packaging/modelscope/` **are** tracked; `dist/` is only their assembled copy. The published package (`model.pth` + `model.safetensors`) **does bundle the backbone's 402 tensors** and redistributes them under Apache-2.0; the **repository carries no backbone weights**. Do not conflate the two.

## The naming rule (where this repo has repeatedly shot itself)

A cache filename encodes every input that shapes its vectors: the template, token budgets, the gather rule, digests of the vocabulary table and of the checkpoint, `data_version`, and the three split sizes. A head name adds the recipe knobs (width, layers, slots, option permutation, targets, temperature fitting, seed...). Therefore:

- change any one of them and the name changes, so the old artifact **expires** instead of being silently reused;
- **a name must describe the artifact, never the request** -- one name once said `n11000` for a file holding 10,930 rows;
- the identity stored inside an artifact and the hash in its name must come from the same function, always (the shared fix behind six bugs of this class).

One detail that bites: the hash in a cache **name** excludes `data_version` (the version is appended separately as `_d2`), the sidecar record **includes** it, and a head's `feature_inputs` hashes the record -- or, for a shared head, the whole group's records. So deciding "is this artifact still reachable" requires **both bases to agree**; using one alone labelled 393 of 407 healthy files dead. That happened.

## Gates before a release

`python scripts/build_modelscope_pkg.py` does three things per task and **refuses to build** if any fails:

1. re-run that task's official test with the heads inside the package and the cached features, requiring accuracy / ECE / NLL to match the card **digit for digit** ("close" is not the criterion);
2. check the `marker_rule` recorded in the cache against what the registry resolves -- `last` and `pooled` differ by about 1e2 in vector magnitude, so a wrong pairing computes the wrong thing silently;
3. take one real example through the full live path and require the same answer as the cached path.

Plus one between containers: `model.pth` and `model.safetensors` must hold the same tensors, compared tensor by tensor at build time.

Run `python examples/selfcheck.py` **inside the built package** (`cd dist/modelscope`) and it replays step 3 on any machine, reporting the probability drift. `scripts/upload_modelscope.py` **refuses to touch the network** unless that self-check passes in the built directory; it reads the token from the environment or `.env` (ignored, and never from argv); after pushing it reads the remote file list back and fails if anything local is missing -- a partial upload must not be able to look like a success.

## Licence and provenance

- This repository's code and decision heads: Apache-2.0. Attribution and the third-party source list: [`packaging/modelscope/NOTICE.md`](packaging/modelscope/NOTICE.md). There is **no LICENSE file in the tree yet**; add the Apache-2.0 text and let it govern both the code and the cards' claim.
- Backbone weights and vocabulary table come from the **RWKV project** (`RWKV/rwkv7-g1`, Apache-2.0): the repository does not carry them, the published package redistributes them with attribution.
- Evaluation data comes from public datasets (Laya typed-decisions, MASSIVE, XNLI, Banking77, go_emotions, amazon_reviews_multi, ag_news, sst5, boolq, ...), each under its own licence; cite the dataset when quoting a number derived from it.
