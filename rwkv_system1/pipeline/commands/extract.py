"""Run the frozen backbone once per split and cache the marker vectors."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ...backbone import DecisionBackbone
from ...features import SHIPPED_RULE, cache_mismatch, extract
from ...cache_layout import (CACHE, RUNS, cache_name, checkpoint, data_version, feature_inputs,
                             resolve_rule)
from ...examples import build_examples, queries_for
from ..config import Config


def cmd_extract(cfg: Config, args) -> None:
    ckpt = checkpoint(args.backbone)
    print(f"loading backbone {ckpt.name}")
    backbone = DecisionBackbone(ckpt, ctx_tokens=cfg.ctx_tokens)
    print(
        f"  {backbone.model.n_layer} layers, d_model {backbone.model.n_embd}, "
        f"terminator token id {backbone.terminator_id}"
    )

    data = build_examples(cfg, args.task_list)
    CACHE.mkdir(exist_ok=True)
    for key, splits in data.items():
        # Per task, not once for the run: the gather rule is pinned per task, so both the identity
        # record and the rule the extractor is handed have to be resolved after the key is known.
        rule = resolve_rule(cfg.feature_rule, key)
        inputs = {**feature_inputs(cfg, args.backbone, data_version(key), key)}
        if cfg.feature_rule is None and rule != SHIPPED_RULE:
            print(f"  {key}: gather rule {rule!r} from the task's pin")
        for split_name, examples in splits.items():
            dest = CACHE / cache_name(args.backbone, key, split_name, cfg)
            queries = queries_for(examples)
            if Path(str(dest) + ".pt").exists() and not args.force:
                why = cache_mismatch(dest, inputs, queries, [e.state for e in examples],
                                     [e.answer for e in examples])
                if why:
                    print(f"  {dest.name}: cache no longer matches this code ({why}), "
                          f"re-extracting")
                else:
                    print(f"  {dest.name}: cached, skipping")
                    continue
            print(f"  {dest.name}: {len(examples)} examples ...", flush=True)
            fs = extract(
                backbone,
                [e.state for e in examples],
                queries,
                [e.answer for e in examples],
                [e.true_probs for e in examples],
                batch_size=cfg.batch_extract,
                inputs=inputs,
                rule=rule,
            )
            fs.save(dest)
            print(f"    -> {tuple(fs.features.shape)}")

    RUNS.mkdir(exist_ok=True)
    # The only writer of this file: it records the pipeline's recipe for later commands to
    # resume.  `train` deliberately does not rewrite it -- see the note at the end of
    # `commands/train.py` for what that cost.
    #
    # `feature_rule` is left out.  It is decided per task (`RealTask.feature_rule`) or on the
    # command line, so one value here cannot express it -- and a stale one *did* silently override
    # every pin, producing an all-`last` "shipped" table while the pins were in place.  Which rule
    # a cache was built under is recorded where it belongs: in the cache's own identity and name.
    recipe = asdict(cfg)
    recipe.pop("feature_rule", None)
    (RUNS / f"{args.backbone}_config.json").write_text(
        json.dumps(recipe, indent=1), encoding="utf-8"
    )
