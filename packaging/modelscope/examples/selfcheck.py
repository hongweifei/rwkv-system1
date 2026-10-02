"""Does this installed model reproduce the numbers on its own card?

Run from inside the downloaded folder:

    python examples/selfcheck.py            # or: python examples/selfcheck.py /path/to/the/folder

For every task it re-runs the whole live path -- tokenise, one prefill, the option block, the marker
gather, the head, the fitted temperature -- on the state recorded in `config.json`, and compares the
distribution against the one the training pipeline measured from cached features at build time.

Two things are required and they mean different things:

* **same answer** -- the label with the most mass, or for a `score` question the same expectation to
  within half a printed step.  If this fails, a head is reading a layout it was not trained on.
  The one that matters: the ten MASSIVE tasks gather the mean of each option's token span, and every
  other head reads the terminator itself; those vectors differ by ~1e2 in magnitude.  A build that got
  the rule wrong is caught here, not on a benchmark you have to trust.
* **probabilities within 0.05** -- live features are not bit-identical to the cache because the
  backbone ran at a different batch shape, so float reduction order changes.  Measured at build time
  as max|delta| = 6.2e-2 on the vectors (norm ~22) and 0 on every reported metric of a 292-row split,
  so this bound is headroom, not a claim.

It also checks the container: `model.pth` must hash to the sha256 `config.json` records, and every
head section must hash to the digest recorded for it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    # Not cosmetic: on a GBK console a single un-encodable character kills the run
    # with UnicodeEncodeError partway through the report.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

PROB_TOLERANCE = 0.05
#: A `score` question's "answer" is the distribution's expectation printed to four decimals, so a
#: 1e-5 probability wobble flips its last digit.  Comparing that as a string fails on a model that
#: reproduces perfectly -- measured on `laya_urgency`: max|dp| 0.0001, "2.3493" vs "2.3494".
SCORE_TOLERANCE = 0.005


def answers_agree(qtype: str, got: str, want: str) -> bool:
    if qtype == "score":
        try:
            return abs(float(got) - float(want)) <= SCORE_TOLERANCE
        except ValueError:
            return False
    return got == want


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else ROOT
    config_path = root / "config.json"
    if not config_path.exists():
        print(f"FAIL  no {config_path}: is {root} a model folder?")
        return 1
    manifest = json.loads(config_path.read_text(encoding="utf-8"))
    for field in ("weight_file", "seeds", "heads", "tasks", "base_name"):
        if field not in manifest:
            print(f"FAIL  {config_path} has no {field!r}: this folder was built by an older "
                  f"build_modelscope_pkg.py, rebuild it")
            return 1
    containers = manifest.get("containers") or {}
    if not containers:
        print(f"FAIL  {config_path} describes the unpacked layout; this check ships the "
              f"bundled model")
        return 1
    parts = " + ".join(f"{i['name']} ({i['size'] / 1e6:.0f} MB)" for i in containers.values())
    print(f"{manifest['package']}: {len(manifest['tasks'])} tasks, "
          f"{manifest['heads']} heads x {len(manifest['seeds'])} seeds, in {parts}, "
          f"base {manifest['base_name']}\n")

    from modeling.service import DecisionModel

    try:
        model = DecisionModel(root)
    except Exception as exc:
        print(f"FAIL  the model refused to load: {type(exc).__name__}: {exc}")
        return 1
    assert model.ensemble_seeds == len(manifest["seeds"]), \
        f"loaded {model.ensemble_seeds} of {len(manifest['seeds'])} shipped seeds"

    # The other container, if the package ships one.  Both files hold the same model because the
    # build compared them tensor by tensor, so this checks that claim where it matters: on the
    # answers.  Features are computed once and fed to both, so the comparison isolates the
    # container -- a second live pass would only re-measure the backbone.
    others = {k: v for k, v in manifest.get("containers", {}).items() if k != model.fmt}
    alt = None
    if others:
        try:
            alt = DecisionModel(root, fmt=next(iter(others)))
        except Exception as exc:
            print(f"FAIL  the second container ({next(iter(others))}) refused to load: "
                  f"{type(exc).__name__}: {exc}")
            return 1

    bad, worst, worst_alt = [], 0.0, 0.0
    for key in model.tasks():
        spec = model.spec(key)
        want = spec["selfcheck"]
        feats = model.features_for(want["state"], [key])[key]
        got = model.answer(key, want["state"])
        delta = max(abs(a - b) for a, b in zip(got["probs"], want["probs"]))
        worst = max(worst, delta)
        ok = answers_agree(spec["qtype"], got["answer"], want["answer"]) and delta <= PROB_TOLERANCE
        note = ""
        if alt is not None:
            from modeling.decision import QTYPES, DecisionHead

            qtype = torch.tensor([QTYPES[spec["qtype"]]])
            t = model.temperature_for(key)
            # Scored inside the block: a lambda defined under no_grad but *called* outside it runs
            # with autograd on, which is correct but prints a UserWarning about requires_grad in a
            # script whose whole job is to look clean when the model is fine.
            with torch.no_grad():
                def served(m):
                    return torch.stack([DecisionHead.probabilities(
                        h(feats.unsqueeze(0), qtype), t) for h in m.heads(key)]).mean(0)[0]

                pa, pb = served(model), served(alt)
            cross = float((pa - pb).abs().max())
            worst_alt = max(worst_alt, cross)
            note = f"  {model.fmt}-vs-{alt.fmt} {cross:.1e}"
            if cross > 0.0:
                ok = False
                bad.append((key, "containers disagree", "", round(cross, 6)))
        if not ok:
            bad.append((key, got["answer"], want["answer"], round(delta, 4)))
        print(f"  {'ok  ' if ok else 'FAIL'}  {key:<20} {spec['gather_rule']:<7} "
              f"answer {str(got['answer']):<14} conf {got['confidence']:.3f}  "
              f"max|dp| {delta:.4f}{note}")
    print(f"\nworst probability deviation over {len(model.tasks())} tasks: {worst:.4f} "
          f"(tolerance {PROB_TOLERANCE})")
    if alt is not None:
        print(f"worst {model.fmt} vs {alt.fmt} deviation: {worst_alt:.1e} "
              f"({'the two containers are the same model' if worst_alt == 0.0 else 'MISMATCH'})")
    if bad:
        print(f"{len(bad)} task(s) disagree with the build-time record:")
        for row in bad:
            print("   ", row)
        return 1
    print("OK: this model reproduces the numbers on its card.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
