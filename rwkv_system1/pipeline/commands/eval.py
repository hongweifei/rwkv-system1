"""Score the trained heads and print the reliability curve."""

from __future__ import annotations

import json
import time

import torch

from ...cache_layout import (RUNS, group_members, head_temperature, load_features, load_head,
                             load_heads)
from ...rlcd import reliability_curve
from ...train import evaluate, head_probabilities
from ..config import Config


def cmd_eval(cfg: Config, args) -> None:
    n_seeds = getattr(args, "ensemble_seeds", 1)
    header = (
        f"{'task':<17}{'k':>3}{'acc':>8}{'Bayes':>8}{'Gold':>7}{'ECE':>8}{'NLL':>8}{'ECEnt':>8}"
        f"{'postL1':>8}{'postKL':>8}{'ECEfit':>8}{'NLLfit':>8}{'Tfit':>6}{'ms':>8}"
    )
    if n_seeds > 1:
        print(f"ENSEMBLE: every row below averages the softmax of {n_seeds} heads "
              f"(seeds 0..{n_seeds - 1}), one recipe differing only in seed.  The T=1 columns are "
              f"what was measured; the fitted columns apply seed 0's temperature to every member, "
              f"which is a choice and has not been scored.")
    print(header)
    summary = {}
    for key in args.task_list:
        if n_seeds > 1:
            heads, temps = load_heads(args.backbone, key, cfg, n_seeds)
        else:
            one, temps = load_head(args.backbone, key, cfg)
            heads = [one]
        fs = load_features(args.backbone, key, "test", cfg)
        # Ten of these rows can come out of one head, and a table that does not say so reads as
        # ten independent models.
        members = group_members(key)
        if members:
            print(f"  [{key}] head shared with {len(members) - 1} other task(s): "
                  f"{'+'.join(m.split('_')[-1] for m in members)}; this row is that one head "
                  f"read on this task's test split")
        temp = head_temperature(temps, fs.qtype, fs.n_options)
        t1 = evaluate(heads, fs, temperature=1.0)
        fit = evaluate(heads, fs, temperature=temp)
        if fs.true_probs is not None:
            # The ceiling is the analytic posterior's own argmax scored against the labels, NOT
            # `mean(max posterior)` -- that is only the ceiling when the label is a *draw* from the
            # posterior, and for the typed-decisions corpora the label is its *mode*.  Measured
            # 2026-09-24: laya_urgency's mean top posterior is 56.62% while the posterior's argmax
            # reproduces the recorded label 98.00% of the time, so quoting the former as "best
            # achievable" understated the headroom by 41 points.
            t1["bayes_accuracy"] = float((fs.true_probs.argmax(-1) == fs.labels).float().mean())
            t1["gold_top_prob"] = float(fs.true_probs.max(-1).values.mean())
        else:
            t1["bayes_accuracy"] = float("nan")
            t1["gold_top_prob"] = float("nan")
        qtype = torch.full((len(fs),), fs.qtype, dtype=torch.long)
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(5):
                heads[0](fs.features, qtype)
            ms = (time.perf_counter() - t0) / 5 * 1000
        # `head_ms` stays one member, so the column is comparable with every table shipped before
        # the ensemble existed; the served model's cost is that times the member count.
        t1["head_ms"] = ms
        t1["ensemble_members"] = len(heads)
        t1["served_ms"] = ms * len(heads)
        t1["ece_at_fitted_t"] = fit["ece"]
        t1["nll_at_fitted_t"] = fit["nll"]
        t1["fitted_temperature"] = temp
        summary[key] = t1
        print(
            f"{key:<17}{fs.n_options:>3}{t1['accuracy']:>8.4f}{t1['bayes_accuracy']:>8.4f}"
            f"{t1['gold_top_prob']:>7.4f}"
            f"{t1['ece']:>8.4f}{t1['nll']:>8.4f}{t1['ece_entropy']:>8.4f}"
            f"{t1.get('posterior_l1', float('nan')):>8.4f}"
            f"{t1.get('posterior_kl', float('nan')):>8.4f}"
            f"{fit['ece']:>8.4f}{fit['nll']:>8.4f}{temp:>6.3f}{ms:>8.2f}"
        )
        # The curve is the served distribution, so it is scored off the same probabilities the
        # accuracy column came from rather than off a second unbatched call.
        probs = head_probabilities(heads, fs, 1.0)
        print(f"  reliability ({key}, at T=1):")
        for conf, acc, n in reliability_curve(probs, fs.labels, n_bins=8):
            print(f"    conf {conf:.3f}  acc {acc:.3f}  n={n:>4}  {'#' * int(round(acc * 40))}")
    print(
        "\n  Headline columns are at T=1, i.e. no temperature fitted."
        "\n  ECEfit/NLLfit/Tfit = the same head after the dev-fitted rescaling. Three-seed"
        "\n    replicates say that rescaling is earned only where dev is large enough to"
        "\n    resolve it (helps on ag_news' 500-row dev, hurts on the 75- and 300-row ones)."
        "\n  Bayes = the analytic posterior's own argmax scored against the recorded labels: the"
        "\n    accuracy of the predictor this pipeline already calls `gold`. It saturates near 100%"
        "\n    whenever the label is the posterior's *mode* rather than a draw from it (laya_urgency"
        "\n    98.00%, go_emotions 100.00%), because then reproducing the posterior reproduces the"
        "\n    label exactly -- so it measures how much the labels owe to the annotators'"
        "\n    distribution, not how much a head can extract from the text."
        "\n  Gold = the mean top posterior probability, i.e. the annotators' own disagreement --"
        "\n    56.62% on laya_urgency where Bayes is 98.00%. It is NOT a bound on the head: it is"
        "\n    the accuracy of the posterior's argmax only when the label is a draw from the"
        "\n    posterior, so a head can and does score above it. The interpretable gap is"
        "\n    postL1/postKL, how far the head's distribution is from the posterior."
        "\n  postL1/KL = distance to the analytic posterior (0 is perfect)."
        "\n  ms = head forward for the whole test split, backbone excluded."
        "\n  " + (f"This table averages {n_seeds} seeds per task (seeds 0..{n_seeds - 1}): the members differ"
        f"\n    only in seed, and their softmaxes are averaged, not their logits. Measured effect:"
        f"\n    +0.80pt accuracy and -0.027 nats over the suite, 34/34 tasks on NLL, but the binned-ECE"
        f"\n    gain does not appear in entropy-confidence ECE, so the honest claim is the likelihood."
        f"\n    The ensemble has no fitted temperature of its own -- Tfit is seed 0's, applied to every"
        f"\n    member before averaging."
        if n_seeds > 1 else
        "\n  One seed only (cfg.seed, which now also fixes the head's weight init, not just the"
        "\n    batch order). Three-seed replicates of this exact per-task arm on the fixed"
        "\n    prompt give sd 0.6-1.9pt on accuracy and 0.6-2.1pt on top-label ECE, so a"
        "\n    single-run difference smaller than that is not evidence of anything. Averaging those"
        "\n    three is `--ensemble-seeds 3`, and `head_ms` above is one member.")
        +
        "\n  The head carries no option-slot embedding unless head_layers > 0 needs one; with it"
        "\n    on, re-ordering the option list cost 51pt of accuracy on this same test split."
    )

    (RUNS / f"{args.backbone}_eval.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    # What the table *is* has to travel with it. A single-seed table and a three-seed one have the
    # same shape and the same file name, and reading the wrong one as the other is a silent
    # misattribution of the model, not of the number.
    (RUNS / f"{args.backbone}_eval.meta.json").write_text(
        json.dumps({"ensemble_seeds": n_seeds, "seeds": list(range(n_seeds)) if n_seeds > 1
                    else [cfg.seed], "backbone": args.backbone,
                    "tasks": len(summary)}, indent=1),
        encoding="utf-8",
    )
