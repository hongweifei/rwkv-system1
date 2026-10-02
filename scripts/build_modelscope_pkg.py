"""Assemble the upload-ready ModelScope package.  Run: python scripts/build_modelscope_pkg.py [--out dist/modelscope] [--limit N]

Why a build script and not a folder someone copies into:

the package has to carry, for every head, the **exact prompt text that head was trained against** --
the option strings and the instruction line -- and that text lives in the dataset loaders, not in
`runs/`.  A hand-assembled directory is how a card ends up advertising a label list the weights never
saw, which is the same defect `features.cache_mismatch` was written to catch on the training side.  So
every field here is read off the artifacts, and the two things that cannot be read off them (the
instruction line, the option labels) come from the loaders in a **subprocess per task**, because a
process that loads several real corpora at once has taken this machine down before.

It also verifies what it writes, per task, before shipping it:

* the head file it copies is the one `run_name` resolves for that task -- so a package cannot ship a
  head the pipeline would not load;
* the gather rule recorded for the task is the one its cache sidecar was built under;
* scoring the task's own test features through the *package's* construction path reproduces the
  shipped table's accuracy/ECE/NLL exactly, and scoring one row through the *live* feature path
  reproduces the same argmax (live features differ from cached ones by ~6e-2 on vectors of norm ~22,
  which is batch-shape float order, not a different model -- measured 2026-10-02 on 292 rows of
  `massive_zh`, all four metrics identical).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = "陆白tuber/rwkv7-system1"
sys.path.insert(0, str(ROOT))

SRC = ROOT / "packaging" / "modelscope"
#: The runtime the published model needs, in the order it reads.  Copied verbatim rather than
#: imported from the installed package: a published model must not depend on this repository.
RUNTIME = ("rwkv7.py", "decision.py", "backbone.py", "features.py", "containers.py",
           "service.py")
#: What a buyer of the package needs to see per task, off the shipped table.
EVAL_FIELDS = ("accuracy", "ece", "nll", "ece_entropy", "fitted_temperature")


def child(task: str) -> dict:
    """One corpus, one process: the first test example's prompt text and state."""
    code = f"""
import contextlib, io, json, sys
sys.path.insert(0, {str(ROOT)!r})
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    from rwkv_system1.examples import build_examples
    from rwkv_system1.pipeline.config import Config
    rows = build_examples(Config(), [{task!r}])[{task!r}]
    e = rows["test"][0]
out = {{"state": e.state, "instructions": e.question.instructions,
        "options": e.question.options, "labels": e.question.answer_labels,
        "qtype": e.question.type.label, "name": e.question.name}}
# Explicit UTF-8 bytes: this box's console codepage is GBK and a state text carrying CJK or a
# typographic dash dies in `sys.stdout.write` with a UnicodeEncodeError long after the work is done.
sys.__stdout__.buffer.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         encoding="utf-8")
    if res.returncode != 0:
        raise SystemExit(f"{task}: prompt-text subprocess failed\n{res.stderr[-2000:]}")
    return json.loads(res.stdout)


def render_eval(specs: dict, language: str) -> str:
    """The results table, built from the artifacts the package actually contains.

    Not hand-written because a hand-written card drifts: the figures a reader trusts and the heads
    they download are then two different claims, and nothing compares them.
    """
    import math

    zh = language == "zh"
    head = ("| 任务 | 选项 | 准确率 | 瞎猜 | 高出 | ECE | NLL | 均匀 NLL | Tfit | 测试行数 | 说明 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|") if zh else (
            "| task | k | acc | chance | lift | ECE | NLL | uniform NLL | Tfit | n_test | note |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for key in sorted(specs, key=lambda k: -(specs[k]["eval"]["accuracy"] -
                                              1 / len(specs[k]["options"]))):
        s = specs[key]
        e = s["eval"]
        k = len(s["options"])
        share = len(s["shared_with"])
        note = []
        if share > 1:
            note.append(f"{share} 个任务共用一个头" if zh else f"1 head for {share} tasks")
        if s["gather_rule"] != "last":
            note.append(f"取特征规则 {s['gather_rule']}" if zh else f"gathers '{s['gather_rule']}'")
        rows.append((e["accuracy"] - 1 / k,
                     f"| `{key}` | {k} | {e['accuracy'] * 100:.2f}% | {100 / k:.1f}% "
                     f"| {(e['accuracy'] - 1 / k) * 100:+.1f} pt | {e['ece']:.4f} "
                     f"| {e['nll']:.3f} | {math.log(k):.3f} | {e['fitted_temperature']:.2f} "
                     f"| {s['n_test']} | {'; '.join(note) or '—'} |"))
    weak = sorted(k for k, s in specs.items() if s["eval"]["nll"] > math.log(len(s["options"])))
    listed = "、".join(f"`{k}`" for k in weak) if zh else ", ".join(f"`{k}`" for k in weak)
    closer = ("——在这些任务上它输出的概率没有信息量。" if zh
              else " -- on these tasks the reported probabilities carry no information and "
                   "should not be used.")
    title = (f"**NLL 比「每个选项都给 1/k」还高的任务（{len(weak)} 个）：** " if zh
             else f"**Tasks whose NLL is worse than predicting uniformly ({len(weak)}): **")
    return (head + "\n" + "\n".join(r for _, r in sorted(rows, reverse=True))
            + "\n\n" + title + listed + closer)


def inject(text: str, marker: str, body: str, name: str) -> str:
    start, end = f"<!-- generated:{marker} -->", "<!-- end:generated -->"
    if start not in text or end not in text:
        print(f"  note: {name} has no generated:{marker} block; shipping its authored text as-is")
        return text
    pre, rest = text.split(start, 1)
    _, post = rest.split(end, 1)
    return pre + start + "\n" + body + "\n" + end + post



def seeds_note(seeds: list[int], lang: str, paired: dict | None = None) -> str:
    """What the release actually contains on the seed axis, and what that buys *here*.

    Generated because the sentence is a claim about the package: a hand-written "seed 0 only" above a
    directory holding three heads per task is the card/weights disagreement this builder exists to
    prevent.

    The deltas come from the two tables this pipeline just produced, not from the harness numbers.
    They are different questions: ``seed_ensemble.py`` measured the average against **the mean of
    three single-head metric sets**, while a package replacing a published single-seed release is
    measured against **seed 0's one head**. Quoting the first under the second baseline overstates
    the win, and NLL's "34/34" belongs to the first comparison only.
    """
    n = len(seeds)
    if n == 1:
        return ("只发布 **seed 0** 一份权重。多种子概率平均的实测见 N>1 的构建（本包未包含）。"
                if lang == "zh" else
                "Ships **seed 0** only. The multi-seed average is measured in the N>1 build; it is "
                "not part of this release.")
    if paired is None:
        return (f"发布 **seed {seeds[0]}..{seeds[-1]}**，推理时平均 {n} 个概率分布（不是平均 logits）。"
                if lang == "zh" else
                f"Ships **seeds {seeds[0]}..{seeds[-1]}**; inference averages the {n} probability "
                f"distributions, never the logits.")
    p = paired
    zh = lang == "zh"
    body = (f"发布 **seed {seeds[0]}..{seeds[-1]}**，共 {n} 份权重；推理时**平均 {n} 个概率分布"
            f"（不是平均 logits）**。相对本包替换掉的那一份（seed 0 单头），在 {p['n']} 个任务上实测："
            f"准确率 **{p['accuracy']:+.2f} pt**（{p['acc_wins']}/{p['n']} 个变好，符号检验 "
            f"p={p['acc_p']:.3f}）、ECE {p['ece']:+.2f} pt（p={p['ece_p']:.3f}）、"
            f"NLL {p['nll']:+.3f} nats（{p['nll_wins']}/{p['n']}，p={p['nll_p']:.3f}）。"
            f"**要说清楚的边界**：熵归一化置信度的 ECE 变化 {p['ece_entropy']:+.2f} pt 没被分辨出来"
            f"（p={p['ee_p']:.2f}），所以能声称的是似然变好，不是校准变好；而 harness 里那个 "
            f"\"NLL 34/34 全赢\" 是相对*三个单头指标的平均*量的，不是相对这一份 seed 0，两个基线"
            f"别混着用。" if zh else
            f"Ships **seeds {seeds[0]}..{seeds[-1]}** ({n} weight files); inference averages the "
            f"{n} probability distributions, never the logits. Against the single seed-0 head this "
            f"release replaces, over {p['n']} tasks: accuracy **{p['accuracy']:+.2f} pt** "
            f"({p['acc_wins']}/{p['n']}, sign p={p['acc_p']:.3f}), ECE {p['ece']:+.2f} pt "
            f"(p={p['ece_p']:.3f}), NLL {p['nll']:+.3f} nats ({p['nll_wins']}/{p['n']}, "
            f"p={p['nll_p']:.3f}). **The boundary**: entropy-confidence ECE moved "
            f"{p['ece_entropy']:+.2f} pt and is not resolved (p={p['ee_p']:.2f}), so the claim is "
            f"likelihood, not calibration -- and the \"NLL better on 34/34\" figure belongs to a "
            f"different baseline (the mean of three single-head metric sets, not this one head).")
    if p.get("losers"):
        body += ("\n\n变差的任务：" + "、".join(f"`{k}` {v:+.2f}pt" for k, v in p["losers"])
                 if zh else "\n\nTasks that moved down: "
                 + ", ".join(f"`{k}` {v:+.2f}pt" for k, v in p["losers"]))
    return body

def paired_stats(single_name: str, ens_name: str) -> dict | None:
    """Paired deltas between the single-head table and the ensemble table, computed here.

    Returns None when the comparison table is not on disk, in which case the card falls back to
    stating only what the package contains -- a missing number is honest, a borrowed one is not.
    """
    import math
    import statistics as st
    from pathlib import Path

    def sign_p(w, n):
        k = min(w, n - w)
        return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)

    sp, ep = ROOT / "experiments" / "results" / single_name, ROOT / "experiments" / "results" / ens_name
    if not sp.exists():
        return None
    one = json.loads(sp.read_text(encoding="utf-8"))
    ens = json.loads(ep.read_text(encoding="utf-8"))
    keys = [k for k in ens if k in one]
    out = {"n": len(keys)}
    for metric, better, scale, key in (("accuracy", 1, 100, "accuracy"), ("ece", -1, 100, "ece"),
                                       ("nll", -1, 1, "nll"), ("ece_entropy", -1, 100, "ece_entropy")):
        d = [(ens[k][metric] - one[k][metric]) * scale for k in keys]
        wins = sum(x * better > 0 for x in d)
        out[metric] = st.mean(d)
        out["acc_wins" if metric == "accuracy" else f"{metric}_wins"] = wins
        out["ee_p" if metric == "ece_entropy" else f"{metric}_p"] = sign_p(wins, len(d))
        if metric == "ece":
            out["ece_p"] = sign_p(wins, len(d))
        if metric == "accuracy":
            out["acc_p"] = sign_p(wins, len(d))
            out["losers"] = sorted(((k, x) for k, x in zip(keys, d) if x < 0), key=lambda r: r[1])[:8]
    return out


def head_digest(state: dict) -> str:
    """The same six lines ``modeling/service.py::head_tensor_identity`` runs, copied not imported.

    It cannot be shared: the published folder must stand alone, and a relative import back into this
    repository would be exactly the dependency a buyer should not have.  The two copies are checked
    against each other by ``examples/selfcheck.py``, which loads every head through the service and
    refuses the file if the hashes disagree -- so drift fails loudly rather than silently.
    """
    import hashlib

    h = hashlib.sha1()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist" / "modelscope"))
    ap.add_argument("--limit", type=int, default=0,
                    help="package only the first N tasks (smoke build; 0 = all)")
    ap.add_argument("--seeds", type=int, default=3,
                    help="how many seed replicas to ship and average (1 = the single head)")
    ap.add_argument("--bundle", default="model",
                    help="weight file stem, written once per --format; pass '' to keep the "
                         "unpacked heads/*.pt layout instead")
    ap.add_argument("--formats", nargs="+", default=["pth", "safetensors"],
                    choices=["pth", "safetensors"],
                    help="containers to publish; with both, the build proves they hold the same "
                         "model tensor by tensor before shipping")
    ap.add_argument("--table", default=None,
                    help="results table to publish; default eval_shipped_ens<N>.json when N>1")
    args = ap.parse_args(argv)

    from dataclasses import replace as dc_replace

    from rwkv_system1.cache_layout import (BACKBONES, BACKBONE_REPO, CACHE, DEFAULT_BACKBONE,
                                          RUNS, cache_name, head_inputs, load_features, resolve_rule,
                                          run_name)
    from rwkv_system1.pipeline.config import Config
    from rwkv_system1.task_registry import REAL_TASKS
    import rwkv_system1.cache_layout as cl

    cfg = Config()
    seeds = list(range(args.seeds))
    default_table = ("eval_shipped_ens3.json" if args.seeds > 1 else "eval_shipped.json")
    table_name = args.table or default_table
    table_path = ROOT / "experiments" / "results" / table_name
    if not table_path.exists():
        raise SystemExit(f"{table_path} does not exist; run the training and eval it before "
                         f"packaging {args.seeds} seeds (or pass --table)")
    table = json.loads(table_path.read_text(encoding="utf-8"))
    print(f"publishing table {table_name}, {len(seeds)} seed(s) {seeds}")
    keys = list(REAL_TASKS)[: args.limit or None]

    # Head per task, resolved the way the pipeline resolves it -- and a task whose head is shared is
    # recorded as shared, because ten rows out of one head is a fact a card has to state.
    heads: dict[str, dict] = {}
    specs: dict[str, dict] = {}
    missing = []
    for key in keys:
        fs = load_features(DEFAULT_BACKBONE, key, "test", cfg)
        names = [run_name(DEFAULT_BACKBONE, key, dc_replace(cfg, seed=s), fs) for s in seeds]
        absent = [f"{s}:{n}" for s, n in zip(seeds, names) if not (RUNS / f"{n}.pt").exists()]
        if absent:
            missing.append(f"{key} (no head for {', '.join(absent)})")
            continue
        metas = [json.loads((RUNS / f"{n}.json").read_text(encoding="utf-8")) for n in names]
        hi = metas[0]["head_inputs"]
        # Every member must be the same recipe except for `seed`, or the "ensemble" is averaging
        # different models and the card's single recipe table would describe none of them.
        recipe = {k: v for k, v in hi.items() if k != "seed"}
        off = [s for s, m in zip(seeds[1:], metas[1:])
               if {k: v for k, v in m["head_inputs"].items() if k != "seed"} != recipe]
        if off:
            raise SystemExit(f"{key}: seeds {off} carry a different recipe than seed {seeds[0]}. "
                             f"Averaging those is not an ensemble of one model, so the card would "
                             f"describe a recipe none of the members has.")
        side = json.loads((CACHE / (cache_name(DEFAULT_BACKBONE, key, "test", cfg) + ".json"))
                          .read_text(encoding="utf-8"))
        rule = resolve_rule(cfg.feature_rule, key)
        # The rule recorded must be the rule the *vectors* were built with, or the package would
        # gather them a different way from the one the weights were fitted on.
        recorded = side["inputs"].get("marker_rule")
        if recorded != cl.GATHER_RULES[rule]:
            raise SystemExit(f"{key}: cache sidecar says marker_rule={recorded}, the registry "
                             f"resolves to {rule!r}. Do not package this disagreement.")
        for n, meta in zip(names, metas):
            if n not in heads:
                heads[n] = {"file": RUNS / f"{n}.pt", "meta": meta}
        text = child(key)
        if text["options"] != side["options"]:
            raise SystemExit(f"{key}: the loader's option list is not the one in the cache sidecar; "
                             f"the cache is stale or the loader moved")
        if text["qtype"] != side["qtype_name"]:
            raise SystemExit(f"{key}: question type {side['qtype_name']!r} in the cache, "
                             f"{text['qtype']!r} from the loader")
        group = sorted(k for k in REAL_TASKS if cl.group_members(k) and key in cl.group_members(k))
        specs[key] = {
            "name": text["name"], "head_files": [f"heads/{n}.pt" for n in names],
            "head_keys": names,
            "head_digests": {n: head_digest(torch.load(str(RUNS / f"{n}.pt"), map_location="cpu",
                                                       weights_only=True)) for n in names},
            "seeds": seeds, "shared_with": group, "gather_rule": rule,
            "qtype": text["qtype"], "instructions": text["instructions"],
            "options": text["options"], "labels": text["labels"],
            "in_dim": hi["in_dim"], "d_model": hi["d_model"], "n_layers": hi["n_layers"],
            "dropout": hi["dropout"], "slot_embeddings": hi["slot_embeddings"],
            "n_test": int(len(fs)),
            "temperatures": metas[0]["history"]["temperatures"],
            "temperature_source": f"seed {seeds[0]}'s fitted table, applied to every member",
            "targets": hi["targets"],
            "feature_fingerprint": hi["feature_inputs"],
            "eval": {m: table[key][m] for m in EVAL_FIELDS if m in table[key]},
            "selfcheck": {"state": text["state"], "probs": None, "answer": None},
        }

    if missing:
        print(f"  skipping {len(missing)} task(s) with no trained head: {missing}")

    # Score every task's own test split through the package's construction path and require the
    # shipped numbers back.  Done before any file is written, so a package that cannot reproduce the
    # card is never produced.
    from rwkv_system1.decision import (QTYPES, DecisionHead, DecisionType, Question, temp_bucket)
    from rwkv_system1.train import evaluate
    from rwkv_system1.features import markers_for_state
    from rwkv_system1.backbone import DecisionBackbone

    print("\nverifying before writing:")
    backbone = None
    for key, s in specs.items():
        fs = load_features(DEFAULT_BACKBONE, key, "test", cfg)
        names = [run_name(DEFAULT_BACKBONE, key, dc_replace(cfg, seed=sd), fs) for sd in seeds]
        members = []
        for n in names:
            member = DecisionHead(in_dim=s["in_dim"], d_model=s["d_model"],
                                  n_layers=s["n_layers"], dropout=s["dropout"],
                                  slot_embeddings=s["slot_embeddings"]).eval()
            member.load_state_dict(torch.load(RUNS / f"{n}.pt", map_location="cpu",
                                              weights_only=True))
            members.append(member)
        if [f"heads/{n}.pt" for n in names] != s["head_files"]:
            raise SystemExit(f"{key}: the head files in the manifest are not the ones the pipeline "
                             f"resolves for seeds {seeds}")

        bucket = temp_bucket(QTYPES[s["qtype"]], len(s["options"]))
        t = float(s["temperatures"].get(bucket, 1.0))
        # Two comparisons, because they catch different mistakes.  The shipped table's headline
        # columns are T=1, so checking those at T=1 is what proves the right vectors went into the
        # right heads; checking the *fitted* columns against this bucket proves the bucket lookup the
        # package does is the same one the pipeline did -- a head that silently ran at T=1 would pass
        # the first test and fail the second.
        #
        # `evaluate` is handed the list, so this measures the *average* distribution: the number on
        # the card is the number the ensemble produces, not one member's.
        head_t1 = evaluate(members, fs, temperature=1.0)
        head_fit = evaluate(members, fs, temperature=t)
        want = s["eval"]
        bad = [m for m in ("accuracy", "ece", "nll") if abs(head_t1[m] - want[m]) > 1e-9]
        if abs(head_fit["ece"] - table[key]["ece_at_fitted_t"]) > 1e-9:
            bad.append("ece_at_fitted_t")
        if abs(head_fit["nll"] - table[key]["nll_at_fitted_t"]) > 1e-9:
            bad.append("nll_at_fitted_t")
        if abs(t - table[key]["fitted_temperature"]) > 1e-9:
            bad.append("fitted_temperature")
        if bad:
            raise SystemExit(f"{key}: the heads {s['head_files']} do not reproduce the published "
                             f"table on {bad}: T=1 acc {head_t1['accuracy']:.4f} vs "
                             f"{want['accuracy']:.4f}, fitted T {t} vs "
                             f"{table[key]['fitted_temperature']}")
        # One row through the live path, to bind the prompt text that ships to the weights it is
        # paired with.  Row 0 of the cache is row 0 of the corpus, so an order change is a failure
        # here rather than a silent comparison of two different examples.
        if fs.states[0] != s["selfcheck"]["state"]:
            raise SystemExit(f"{key}: the corpus's first test state is not the one recorded in the "
                             f"cache (order moved between the loader and the extraction)")
        if backbone is None:
            backbone = DecisionBackbone(cl.checkpoint(DEFAULT_BACKBONE), ctx_tokens=cfg.ctx_tokens)
        qtype_t = torch.tensor([QTYPES[s["qtype"]]])
        feats = markers_for_state(backbone, s["selfcheck"]["state"],
                                  [(s["qtype"], s["instructions"], s["options"])],
                                  rule=s["gather_rule"])[0]
        with torch.no_grad():
            def ensemble(feat):
                return torch.stack([DecisionHead.probabilities(m(feat.unsqueeze(0), qtype_t), t)
                                    for m in members]).mean(0)[0]

            live_probs = ensemble(feats)
            cached_probs = ensemble(fs.features[0])
        dl1 = float((live_probs - cached_probs).abs().max())
        if int(live_probs.argmax()) != int(cached_probs.argmax()):
            raise SystemExit(f"{key}: live features pick a different option than the cached ones")
        question = Question(name=s["name"], instructions=s["instructions"],
                            type=DecisionType[s["qtype"].upper()], options=list(s["options"]),
                            labels=list(s["labels"]) if s.get("labels") else None)
        label, conf = DecisionHead.point_answer(question, cached_probs)
        s["selfcheck"]["probs"] = [round(float(p), 6) for p in cached_probs]
        s["selfcheck"]["answer"] = str(label)
        s["selfcheck"]["confidence"] = round(conf, 4)
        s["selfcheck"]["live_delta"] = round(dl1, 5)
        s["selfcheck"]["bucket"] = bucket
        del members
        print(f"  {key:<20} T=1 {head_t1['accuracy']:.4f}/{head_t1['ece']:.4f}/"
              f"{head_t1['nll']:.4f} == shipped, fit T={t:.2f}   "
              f"live-vs-cached max|dp| {dl1:.2e}")

    weight_identity: dict[str, dict] = {}
    out = Path(args.out)
    (out / "modeling").mkdir(parents=True, exist_ok=True)
    (out / "modeling" / "heads").mkdir(exist_ok=True)
    # A package, not a pile of modules: `from modeling.service import DecisionModel` needs this, and
    # `service.py` uses relative imports so the published code cannot quietly reach back into the
    # repository it was built in.
    (out / "modeling" / "__init__.py").write_text(
        '"""The published decision model. See service.py."""\n', encoding="utf-8")
    for f in RUNTIME:
        shutil.copy2(SRC / "modeling" / f if f == "service.py"
                     else ROOT / "rwkv_system1" / f, out / "modeling" / f)
    if args.bundle:
        # Both containers, one model.  The base section is stored *as the upstream file has it* --
        # 402 raw bfloat16 tensors, before `normalise_native` widens them and fuses ln0 -- so the
        # arithmetic is decided once, at load, by code both files share.  Writing two files without
        # comparing them is how two published formats drift into two different models.
        from rwkv_system1 import containers as C
        from rwkv_system1.rwkv7 import RWKV7

        base_path = ROOT / "weights" / BACKBONES[DEFAULT_BACKBONE]
        base = torch.load(str(base_path), map_location="cpu", mmap=True, weights_only=True)
        head_states = {n: torch.load(str(r["file"]), map_location="cpu", weights_only=True)
                       for n, r in heads.items()}
        loose = RWKV7.from_file(base_path)
        probe = torch.randint(1, 5000, (1, 8))
        reference, _, _ = loose.forward(probe, return_hidden=True)
        written: dict[str, dict] = {}
        for fmt in args.formats:
            target = out / f"{args.bundle}{'.pth' if fmt == 'pth' else '.safetensors'}"
            written[fmt] = C.save(target, base, head_states)
            got_base, got_heads = C.load(target)
            problems = C.diff((base, head_states), (got_base, got_heads))
            if problems:
                raise SystemExit(f"{target.name} does not round-trip: {problems[:4]}")
            packed = RWKV7.from_bundle(dict(got_base), target.name)
            drift = float((packed.forward(probe, return_hidden=True)[0] - reference).abs().max())
            if drift != 0.0 or set(packed.z) != set(loose.z):
                raise SystemExit(f"{target.name}: its base forward drifts by {drift:.3e} from the "
                                 f"file the heads were trained against -- do not ship it")
            del got_base, got_heads, packed
            print(f"  {fmt:12} {target.name}: {written[fmt]['size'] / 1e6:7.1f} MB, "
                  f"{len(base)} base tensors + {len(head_states)} heads, round trip exact, "
                  f"forward drift {drift:.1e}")
        # The two files must be identical models, tensor by tensor -- not just each faithful to the
        # source, which the loop above already proved.
        a, b = C.load(out / f"{args.bundle}.pth"), C.load(out / f"{args.bundle}.safetensors")
        across = C.diff(a, b)
        if across:
            raise SystemExit(f"the two containers disagree: {across[:4]}")
        print(f"  containers agree: {len(a[0])} base + {sum(len(v) for v in a[1].values())} head "
              f"tensors identical across .pth and .safetensors")
        weight_identity = written
        del base, head_states, loose, a, b
        heads_dir = out / "modeling" / "heads"
        if heads_dir.exists():
            shutil.rmtree(heads_dir)
    else:
        for name, rec in heads.items():
            shutil.copy2(rec["file"], out / "modeling" / "heads" / f"{name}.pt")
            (out / "modeling" / "heads" / f"{name}.json").write_text(
                json.dumps(rec["meta"], indent=1), encoding="utf-8")
    vocab = ROOT / "weights" / "rwkv_vocab_v20230424.txt"
    shutil.copy2(vocab, out / "modeling" / vocab.name)
    for f in ("examples", "NOTICE.md", "requirements.txt"):
        src = SRC / f
        if src.is_dir():
            shutil.copytree(src, out / f, dirs_exist_ok=True)
        elif src.exists():
            shutil.copy2(src, out / f)
        else:
            print(f"  note: {f} is not authored yet")
    for f, lang in (("README.md", "zh"), ("README_en.md", "en")):
        src = SRC / f
        if not src.exists():
            print(f"  note: {f} is not authored yet")
            continue
        paired = paired_stats("eval_shipped_seed0.json", table_name) if args.seeds > 1 else None
        if args.seeds > 1 and paired is None:
            print("  note: no eval_shipped_seed0.json to pair against; the card will state the "
                  "seed count only, not a delta")
        text = src.read_text(encoding="utf-8")
        text = inject(text, "eval", render_eval(specs, lang), f)
        n_heads = len({tuple(s["head_files"]) for s in specs.values()})
        counts = (f"**{n_heads} 个头 × {len(seeds)} 个种子 = {len(heads)} 份权重**，服务 {len(specs)} 个任务" if lang == "zh" else
                  f"**{n_heads} heads × {len(seeds)} seeds = {len(heads)} weight files** serving {len(specs)} tasks")
        text = inject(text, "counts", counts, f)
        text = inject(text, "seeds", seeds_note(seeds, lang, paired), f)
        (out / f).write_text(text, encoding="utf-8")

    standing = json.loads((ROOT / "runs" / f"{DEFAULT_BACKBONE}_config.json").read_text(
        encoding="utf-8"))
    manifest = {
        "package": "rwkv7-decision-heads",
        "repo": DEFAULT_REPO_ID,
        "base_repo": BACKBONE_REPO,
        "base": {"checkpoint": cl._file_identity(ROOT / "weights" / BACKBONES[DEFAULT_BACKBONE]),
                 "vocab": cl._file_identity(vocab), "ctx_tokens": cfg.ctx_tokens},
        "vocab_file": vocab.name,
        "recipe": {k: standing[k] for k in
                   ("warmup_epochs", "rlcd_epochs", "lr", "batch_size", "targets") if k in standing},
        "seeds": seeds, "results_table": table_name, "heads": len(heads),
        "weight_file": (f"{args.bundle}.pth" if args.bundle else None),
        "containers": weight_identity,
        "base_name": BACKBONES[DEFAULT_BACKBONE],
        "tasks": specs,
        "built_from_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
    }
    (out / "config.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"\nwrote {out}: {len(specs)} tasks over {len(heads)} head(s)"
          + (", in " + " + ".join(f"{i['name']} ({i['size'] / 1e6:.0f} MB)"
                                  for i in weight_identity.values())
             if weight_identity else " as loose heads/*.pt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
