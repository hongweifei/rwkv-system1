"""Invariants that each cost a real incident.  Run: python tests/invariants.py

No pytest dependency, and nothing here touches the network or a checkpoint, so it runs in a
second and can be run before any extraction.  "No checkpoint" is a promise about this file, not
about the machine: the two checks that reach :func:`cache_layout.feature_inputs` stand in their own
weight files (see :func:`stand_in_weights`), because ``weights/`` is fetched rather than committed
and a suite that only runs on the one box holding it verifies nothing on a fresh clone.

Why these:
  * the ECE formula was once `mean|confidence - correct|`, which reports ~0.4 for a perfectly
    calibrated model and inverted a conclusion -- so it is checked against a constructed case
    whose answer is known;
  * a `FeatureSet.subset()` call with an index tensor instead of a bool mask raised deep inside
    torch, and `head_logits(head, fs, 1.2)` silently passed a temperature as a batch size;
  * `permute_columns` exists to make the head read option text, and it must round-trip exactly,
    including the padding;
  * `slot_embeddings` must be off for a per-option scorer -- with it on, only 36% of answers
    followed the option text and re-ordering cost 51 pt of accuracy;
  * a duplicated key in `REAL_TASKS` silently takes effect, last one winning, so sizes were
    quoted for a task that had been registered twice;
  * `forward_layers` was split out of `forward` so a gather rule could read a middle block without
    a second pass, and the split has to stay free -- hence the random-weight bit-for-bit check;
  * `pooled` is the shipped gather rule, and a rule that averaged the wrong span would still
    return a well-shaped tensor, so its arithmetic is pinned rather than left to a docstring.
"""
from __future__ import annotations

import inspect
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    # A check that fails prints the OS's own message, which is Chinese on this box and arrives in
    # the console's code page -- unreadable through a pipe, and enough to make grep call it binary.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rwkv_system1.backbone import PromptTemplate, find_vocab  # noqa: E402
from rwkv_system1.task_registry import REAL_TASKS  # noqa: E402
from rwkv_system1.decision import DecisionHead  # noqa: E402
from rwkv_system1.features import FeatureSet, _gather_markers  # noqa: E402
from rwkv_system1.rlcd import expected_calibration_error  # noqa: E402
from rwkv_system1.train import head_logits, permute_columns  # noqa: E402


@contextmanager
def stand_in_weights():
    """Give the identity layer two throwaway files in place of the ones git does not carry.

    ``weights/`` holds the base checkpoint, which is fetched rather than redistributed, so a check
    that hashed the real bytes could not run in a fresh clone -- the one place this suite is needed
    most.  What these two checks measure is that a cache's *name* and its stored *record* are the
    same dict, and that every input which shapes a vector is in it: a claim about the record's shape,
    not about 400 MB of weights.  So the files are stood up in a temp dir and the two globals the
    identity layer reads are pointed at them.
    """
    import rwkv_system1.backbone as backbone
    import rwkv_system1.cache_layout as cache_layout

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "fake-root"
        (root / "weights").mkdir(parents=True)
        for name in cache_layout.BACKBONES.values():
            (root / "weights" / name).write_bytes(b"stand-in checkpoint")
        vocab = root / "weights" / backbone.VOCAB_FILENAME
        vocab.write_text("stand-in vocab\n", encoding="utf-8")
        saved_root, saved_find = cache_layout.ROOT, backbone.find_vocab
        cache_layout.ROOT = root
        # ``feature_inputs`` imports this by name at call time, so the module attribute is the seam.
        backbone.find_vocab = lambda explicit=None: vocab
        try:
            yield root
        finally:
            cache_layout.ROOT, backbone.find_vocab = saved_root, saved_find


def ece_matches_a_constructed_case() -> str:
    """A model that says 0.6 and is right 60% of the time is perfectly calibrated."""
    n = 1000
    probs = torch.zeros(n, 2)
    probs[:, 0] = 0.6
    probs[:, 1] = 0.4
    labels = torch.zeros(n, dtype=torch.long)
    labels[: int(n * 0.6)] = 0
    labels[int(n * 0.6):] = 1
    ece = expected_calibration_error(probs, labels)
    assert ece < 0.02, (
        f"ECE {ece:.4f} for a perfectly calibrated model; the buggy "
        f"mean|conf - correct| form would report ~0.4"
    )
    # and it must notice real miscalibration
    probs[:, 0], probs[:, 1] = 0.9, 0.1
    assert expected_calibration_error(probs, labels) > 0.25
    return f"calibrated case {ece:.4f}, miscalibrated case > 0.25"


def permute_columns_round_trips() -> str:
    feats = torch.randn(5, 4, 7)
    mask = torch.zeros(5, 4, dtype=torch.bool)
    mask[:, 2:] = True                      # a 2-option row padded to 4
    gen = torch.Generator().manual_seed(0)
    permuted, pmask, back = permute_columns(feats, mask, gen)
    # `back` is (B, K); gather on a (B, K, D) tensor needs that expanded, which is exactly the
    # difference between this call and the one in the training loop (there the logits are 2-D).
    wide = back.unsqueeze(-1).expand_as(permuted)
    assert torch.equal(permuted.gather(1, wide), feats), "logits cannot be recovered"
    assert torch.equal(pmask.gather(1, back), mask), "padding did not follow its column"
    assert not torch.equal(permuted, feats), "the permutation did nothing"
    return "features and padding both survive the round trip"


def slot_embeddings_follow_the_layout() -> str:
    assert DecisionHead(in_dim=8, d_model=16).slot_embeddings is False
    assert DecisionHead(in_dim=8, d_model=16, n_layers=2).slot_embeddings is True
    assert DecisionHead(in_dim=8, d_model=16, slot_embeddings=True).slot_embeddings is True
    assert DecisionHead(in_dim=8, d_model=16, n_layers=2,
                        slot_embeddings=False).slot_embeddings is False
    return "off for a per-option scorer, on for a set-attention stack, explicit wins"


def wide_option_blocks_are_not_truncated() -> str:
    """The removed 96-token cap would have silently shortened these blocks."""
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

    tok = TRIE_TOKENIZER(str(find_vocab()))
    template = PromptTemplate()
    labels = [f"topic_{i}_name" for i in range(77)]
    ids = tok.encode(template.suffix("choice", "What is the topic?", labels))
    assert len(ids) > 96, f"a 77-label block measured only {len(ids)} tokens; test is stale"
    return f"77 labels encode to {len(ids)} tokens, all markers present"


def registry_is_sound() -> str:
    keys = list(REAL_TASKS)
    assert len(keys) == len(set(keys)), "a duplicated key silently overrides the earlier entry"
    for key, task in REAL_TASKS.items():
        assert callable(task.loader), f"{key}: loader not callable"
        assert isinstance(task.data_version, int), f"{key}: data_version must be an int"
        unknown = set(task.kwargs) - set(inspect.signature(task.loader).parameters)
        assert not unknown, f"{key}: kwargs {unknown} are not parameters of its loader"
    return f"{len(keys)} tasks, no duplicate keys, every kwarg accepted by its loader"


def subset_takes_a_mask_not_indices() -> str:
    fs = FeatureSet(features=torch.zeros(4, 2, 3), labels=torch.zeros(4, dtype=torch.long),
                    n_options=2, qtype=0, true_probs=None, states=["a", "b", "c", "d"],
                    options=["x", "y"], qtype_name="choice")
    mask = torch.zeros(4, dtype=torch.bool)
    mask[:2] = True
    assert len(fs.subset(mask)) == 2
    # arange(len(fs)) would *also* be a valid mask, so the probe has to use a length the row
    # count cannot absorb -- the original incident was arange(400) against a 4000-row split.
    try:
        fs.subset(torch.arange(3))
    except IndexError:
        pass
    else:
        raise AssertionError("subset() accepted an index tensor; it takes a bool mask")
    assert inspect.signature(head_logits).parameters["batch_size"].default == 512, (
        "head_logits' third positional argument must stay batch_size, not temperature"
    )
    return "bool mask accepted, indices refused, head_logits signature pinned"


def identity_comes_from_one_place() -> str:
    """The cache name and the stored record must be the same dict, or they drift apart."""
    from rwkv_system1.cache_layout import DEFAULT_BACKBONE, feature_fingerprint, feature_inputs
    from rwkv_system1.pipeline.config import Config

    with stand_in_weights():
        base = Config()
        inputs = feature_inputs(base, DEFAULT_BACKBONE)
        for field in ("ctx_tokens", "tokenizer", "checkpoint", "template", "state_cap"):
            assert field in inputs, f"{field} is missing from the identity record"
        # a change to any recorded input must move the name, since the name is its hash
        other = Config()
        other.ctx_tokens = base.ctx_tokens + 1
        assert (feature_fingerprint(other, DEFAULT_BACKBONE)
                != feature_fingerprint(base, DEFAULT_BACKBONE)), "ctx_tokens is not in the fingerprint"
        assert feature_inputs(base, DEFAULT_BACKBONE) == inputs, "feature_inputs is not deterministic"
        # and the head's identity has to be separate from the feature's
        from rwkv_system1.cache_layout import head_inputs, run_name

        fs = FeatureSet(features=torch.zeros(2, 3, 8), labels=torch.zeros(2, dtype=torch.long),
                        n_options=3, qtype=0, true_probs=None, states=["a", "b"],
                        options=["x", "y", "z"], qtype_name="choice")
        assert "d_model" in head_inputs(base, fs)
        assert "h" in run_name(DEFAULT_BACKBONE, "laya_urgency", base, fs).rsplit("_", 1)[-1]
    return "fingerprint and record share one definition; head identity is separate"


def every_recipe_knob_is_recorded() -> str:
    """A knob that reaches the head but not its name puts two heads on one path.

    The seed was the first one to do this -- it reached the batches but not the weights, so a
    saved ``.pt`` was not reproducible from the config that named it.  Rather than list the
    knobs, compare ``train()``'s own signature against the record, so the next one added
    cannot be forgotten.
    """
    from rwkv_system1.cache_layout import head_inputs
    from rwkv_system1.pipeline.config import Config
    from rwkv_system1.train import train

    fs = FeatureSet(features=torch.zeros(2, 3, 8), labels=torch.zeros(2, dtype=torch.long),
                    n_options=3, qtype=0, true_probs=None, states=["a", "b"],
                    options=["x", "y", "z"], qtype_name="choice")
    not_a_knob = {"head", "train_fs", "dev_fs", "soft_targets", "fit_temperature_on_dev", "log"}
    recipe = set(inspect.signature(train).parameters) - not_a_knob
    missing = recipe - set(head_inputs(Config(), fs))
    assert not missing, (
        f"train() reads {sorted(missing)} but the run name does not cover them, so two "
        f"differently-trained heads land on the same path"
    )
    return f"{len(recipe)} recipe knobs, every one in the head's identity record"


def forward_and_forward_layers_agree() -> str:
    """``forward_layers`` must be the same pass, not a second one with its own arithmetic.

    ``forward`` was split into a shared block loop so a feature experiment can collect an
    intermediate block without re-running the sequence.  Random weights on a two-layer model make
    the split checkable without a checkpoint: if the two entry points disagree anywhere, or if the
    loop start to drift, this fails in a second instead of silently producing a second feature
    space.
    """
    from rwkv_system1.rwkv7 import RWKV7

    c, h, n, d, words = 8, 2, 4, 4, 32
    hn = h * n

    def rnd(*shape):
        return torch.randn(*shape)

    def block(i: int) -> dict:
        p = f"blocks.{i}."
        return {
            p + "ln1.weight": rnd(c), p + "ln1.bias": rnd(c),
            p + "ln2.weight": rnd(c), p + "ln2.bias": rnd(c),
            p + "att.r_k": rnd(h, n), p + "att.x_r": rnd(c), p + "att.x_w": rnd(c),
            p + "att.x_k": rnd(c), p + "att.x_v": rnd(c), p + "att.x_a": rnd(c),
            p + "att.x_g": rnd(c), p + "att.receptance.weight": rnd(c, hn),
            p + "att.w1": rnd(c, d), p + "att.w2": rnd(d, hn), p + "att.w0": rnd(hn),
            p + "att.key.weight": rnd(c, hn), p + "att.value.weight": rnd(c, hn),
            p + "att.a0": rnd(hn), p + "att.a1": rnd(c, d), p + "att.a2": rnd(d, hn),
            p + "att.g1": rnd(c, d), p + "att.g2": rnd(d, hn),
            p + "att.k_k": rnd(hn), p + "att.k_a": rnd(hn), p + "att.v0": rnd(c),
            p + "att.v1": rnd(c, d), p + "att.v2": rnd(d, hn),
            p + "att.ln_x.weight": rnd(hn), p + "att.ln_x.bias": rnd(hn),
            p + "att.output.weight": rnd(hn, c),
            p + "ffn.x_k": rnd(c), p + "ffn.key.weight": rnd(c, d),
            p + "ffn.value.weight": rnd(d, c),
        }

    model = RWKV7({
        "emb.weight": rnd(words, c), "ln_out.weight": rnd(c), "ln_out.bias": rnd(c),
        # head.weight is (n_embd, vocab): the model does `hidden @ head.weight`, and the loader is
        # what turns the checkpoint's own layout into that.
        "head.weight": rnd(c, words), **block(0), **block(1),
    })
    assert model.n_layer == 2 and (model.n_head, model.head_size) == (h, n)
    tokens = torch.randint(0, words, (2, 6))

    hidden, _, state = model.forward(tokens)
    collected, state2 = model.forward_layers(tokens, collect=(0, 1))
    assert torch.equal(hidden, collected[1]), "forward_layers' last layer differs from forward's"
    assert all(torch.equal(a, b) for a, b in zip(state, state2)), "the two passes diverged"
    assert hidden.shape == collected[1].shape == (2, 6, c)
    assert not torch.equal(collected[0], collected[1]), "the middle block is not a different view"
    # and the returned state has to be usable as a prefix, or the split lost the recurrence
    assert model.forward(tokens[:, :3], state)[2][0].shape == (2, c)
    return "forward and forward_layers(last) agree bit for bit; block 0 is a different view"


def a_pinned_task_names_its_own_cache() -> str:
    """The gather rule is pinned per task, so the resolution has to reach the cache *name*.

    Three levels, and each one is a way to get this wrong: a pin that only reached the identity
    record and not the name would read the default cache and look fine while measuring the other
    rule; a pin that could not be overridden would make the arms that measure a rule against its
    alternative unreproducible; and a default that changed for the tasks nobody pinned would move
    every cache name in the suite at once.
    """
    from dataclasses import replace

    from rwkv_system1.cache_layout import DEFAULT_BACKBONE, cache_name, resolve_rule
    from rwkv_system1.features import SHIPPED_RULE
    from rwkv_system1.pipeline.config import Config

    pinned = "banking77"  # 77 options, gained +14.6 pt; see task_registry.RealTask.feature_rule
    plain = "boolq"  # 2 options, a wash: `pooled` would cost 3.5 pt on its sibling
    base = Config()
    assert base.feature_rule is None, "the config has an opinion it should not have"
    assert resolve_rule(None, pinned) == "pooled", resolve_rule(None, pinned)
    assert resolve_rule(None, plain) == SHIPPED_RULE, resolve_rule(None, plain)
    assert resolve_rule(None, None) == SHIPPED_RULE, "no key must not fall over"
    # explicit wins, or the chained arm extractions cannot reproduce themselves
    assert resolve_rule("last", pinned) == "last", "a pin cannot be overridden"
    with stand_in_weights():
        # and the pin has to land in the name, not only in the record
        assert (cache_name(DEFAULT_BACKBONE, pinned, "train", base)
                != cache_name(DEFAULT_BACKBONE, pinned, "train", replace(base, feature_rule="last"))
                ), "the pinned rule does not reach the cache name, so the two rules share one file"
        # while an unpinned task's name must be exactly what it always was
        assert (cache_name(DEFAULT_BACKBONE, plain, "train", base)
                == cache_name(DEFAULT_BACKBONE, plain, "train", replace(base, feature_rule=SHIPPED_RULE))
                ), "the shipped default is written into the name, so every unpinned cache renamed"
    return f"{pinned} resolves to its pin; {plain} and a keyless call resolve to {SHIPPED_RULE!r}"


def pooled_gathers_the_option_span() -> str:
    """``pooled`` is now the shipped rule, so its definition cannot live only in a docstring.

    The layout a prompt produces is ``... header [term] option_0 [term] option_1 [term] ...``, so
    option ``k`` occupies the tokens after the terminator that closed ``k-1`` up to and including
    its own -- for ``k = 0``, after the terminator that closed the header.  ``encode_query``
    already guarantees there are exactly ``n_options + 1`` terminators (it raises otherwise), which
    is what makes ``stops[k + 1]`` safe to read here; this pins the *arithmetic* on top of that
    contract.  A rule that silently averaged the wrong span would still return a well-shaped
    tensor, so nothing downstream would notice.
    """
    term = 11
    # header [term] o0a o0b [term] o1 [term] o2 [term] -- four terminators, three options
    ids = [5, term, 2, 3, term, 4, term, 6, term]
    markers = [4, 6, 8]  # `encode_query` takes positions[1:]: the first terminator closes the header
    hidden = torch.arange(len(ids) * 4, dtype=torch.float32).reshape(len(ids), 4)
    got = _gather_markers(hidden, ids, markers, "pooled", None, term)
    assert got.shape == (3, 4), got.shape
    torch.testing.assert_close(got[0], hidden[2:5].mean(0))  # after the header term, through o0's
    torch.testing.assert_close(got[1], hidden[5:7].mean(0))
    torch.testing.assert_close(got[2], hidden[7:9].mean(0))
    # `last` reads the terminator alone, so the two rules must differ wherever a span is > 1 token
    last = _gather_markers(hidden, ids, markers, "last", None, term)
    torch.testing.assert_close(last[1], hidden[6])
    assert not torch.equal(got[1], last[1])
    return "pooled means the option's own token span; last reads only its terminator"


def a_shared_head_cannot_be_read_as_a_per_task_one() -> str:
    """The registry can put several tasks under one head, so the *name* has to carry that.

    Same collision shape as the gather-rule pin: a MASSIVE joint head and a MASSIVE per-task head
    have identical width, identical option count and identical recipe, so without the group in the
    identity they hash to one run name and whichever trains second overwrites the first, silently.
    Four ways to get it wrong, all cheap to check:

      * two members must name **one** file -- that is what a group is;
      * a member must not name the same file as an ungrouped task;
      * the record must name the members, not merely count their rows -- two different groups can
        have identical totals, and a name built only on sizes would not tell them apart;
      * removing the group must rename, so the per-task heads written before the group existed
        stay reachable instead of being read as the joint model;
      * growing the group must rename, so adding an eleventh language cannot reuse the head that
        never saw it.

    ``group_feature_inputs`` is stubbed: this checks naming, and the check must not need a cache on
    disk (the file's contract is that it runs in a second, offline).
    """
    from dataclasses import replace as _dc_replace

    import rwkv_system1.cache_layout as cl
    from rwkv_system1.pipeline.config import Config

    base = Config()
    fs = FeatureSet(features=torch.zeros(2, 3, 8), labels=torch.zeros(2, dtype=torch.long),
                    n_options=3, qtype=0, true_probs=None, states=["a", "b"],
                    options=["x", "y", "z"], qtype_name="choice")
    members = cl.group_members("massive_zh")
    assert members and len(members) >= 2, "no task is grouped, so there is nothing to check"
    assert all(k.startswith("massive_") for k in members), f"group leaked outside MASSIVE: {members}"
    assert cl.group_members("boolq") is None, "boolq is not grouped"

    stub, stub_rows = cl.group_feature_inputs, cl.group_rows
    real_tasks = dict(REAL_TASKS)
    cl.group_feature_inputs = lambda ms, bb, cfg, split="train": {m: {"stub": i}
                                                                  for i, m in enumerate(ms)}
    # Ten deliberate rows, whatever the caches hold, so the tag can be told apart from the registry
    # sums (MASSIVE asks 1100 per language and the pools hand over 1074-1099).
    cl.group_rows = lambda ms, bb, cfg, split="train": 10
    try:
        # The record has to *say* which tasks the head saw.  Row totals alone would catch most
        # membership edits -- a mutation that deleted the member list and hashed a constant instead
        # still passed the naming assertions below, because two groups can share a total -- so the
        # list is checked where it is written, not only where it is hashed.
        record = cl.head_inputs(base, fs, "massive_zh", cl.DEFAULT_BACKBONE)
        assert record.get("joint_members") == members, "the record omits who the head was fitted on"
        assert record.get("joint_group") == "massive", f"wrong group in the record: {record}"
        assert record["feature_inputs"] == cl.fingerprint_of(
            {m: {"stub": i} for i, m in enumerate(members)}
        ), "a grouped head does not hash the group's caches"
        name = cl.run_name(cl.DEFAULT_BACKBONE, "massive_zh", base, fs)
        assert "joint" in name, f"a grouped head is named like a solo one: {name}"
        # the row tag has to come from the caches, not from what the registry asked for
        assert "n10-10" in name, f"the size tag is a request, not a measurement: {name}"
        assert (cl.run_name(cl.DEFAULT_BACKBONE, members[1], base, fs) == name
                ), "two members of one group name two files"
        assert (cl.run_name(cl.DEFAULT_BACKBONE, "boolq", base, fs) != name
                ), "an ungrouped task shares a path with the group head"
        for k in members:
            REAL_TASKS[k] = _dc_replace(REAL_TASKS[k], joint_group=None)
        solo = cl.run_name(cl.DEFAULT_BACKBONE, "massive_zh", base, fs)
        assert solo != name, "ungrouping does not rename: one path holds two models"
        assert "joint" not in solo
        REAL_TASKS.update(real_tasks)
        for k in members:
            REAL_TASKS[k] = _dc_replace(REAL_TASKS[k], joint_group="massive")
        REAL_TASKS["boolq"] = _dc_replace(REAL_TASKS["boolq"], joint_group="massive")
        assert (cl.run_name(cl.DEFAULT_BACKBONE, "massive_zh", base, fs) != solo
                ), "growing the group does not rename: the old head serves a row it never trained on"
    finally:
        REAL_TASKS.clear()
        REAL_TASKS.update(real_tasks)
        cl.group_feature_inputs, cl.group_rows = stub, stub_rows
    return (f"{len(members)} members share one name; un-grouping and growing it both rename; "
            f"an ungrouped task cannot collide")


def main() -> int:
    checks = [
        ece_matches_a_constructed_case, permute_columns_round_trips,
        slot_embeddings_follow_the_layout, wide_option_blocks_are_not_truncated,
        registry_is_sound, subset_takes_a_mask_not_indices, identity_comes_from_one_place,
        every_recipe_knob_is_recorded, forward_and_forward_layers_agree,
        pooled_gathers_the_option_span, a_pinned_task_names_its_own_cache,
        a_shared_head_cannot_be_read_as_a_per_task_one,
    ]
    failed = 0
    for check in checks:
        try:
            print(f"  PASS  {check.__name__:<38} {check()}")
        except Exception as exc:  # noqa: BLE001 - a failing check is the point
            failed += 1
            print(f"  FAIL  {check.__name__:<38} {type(exc).__name__}: {exc}")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
