"""Print what a task contains: sizes, option counts, a rendered prompt."""

from __future__ import annotations

import torch

from ...examples import build_examples
from ..config import Config


def cmd_data(cfg: Config, args) -> None:
    data = build_examples(cfg, args.task_list)
    print(
        f"{'task':<17}{'qtype':<9}{'k':>3}{'split':<7}{'n':>6}"
        f"{'Bayes acc':>11}{'ambiguous':>11}"
    )
    for key, splits in data.items():
        for split_name, examples in splits.items():
            if examples[0].true_probs is not None:
                probs = torch.tensor([e.true_probs for e in examples])
                top = probs.max(dim=-1).values
                stats = f"{float(top.mean()):>11.4f}{float((top < 0.5).float().mean()):>11.3f}"
            else:
                # Real corpora ship hard labels only, so there is no posterior to report.
                stats = f"{'n/a':>11}{'n/a':>11}"
            print(
                f"{key:<17}{examples[0].question.type.label:<9}"
                f"{len(examples[0].question.options):>3}{split_name:<7}{len(examples):>6}{stats}"
            )
    print("\nrendered prompts:")
    e = next(iter(data.values()))["train"][0]
    q = e.question
    print(f"  state   : {e.state[:150]}")
    print(f"  question: {q.type.label} question: {q.instructions}")
    for o in q.options[:6]:
        print(f"            {o}\\n")
    print(f"  answer  : {q.answer_labels[e.answer]}")
    n_amb = sum(
        1
        for k, v in data.items()
        for ex in v["train"]
        if ex.true_probs is not None and max(ex.true_probs) < 0.9
    )
    n_tot = sum(len(v["train"]) for v in data.values())
    print(
        f"\n{n_amb}/{n_tot} training decisions carry a posterior below 0.9 -- those are the"
        f" ones that punish over-confidence. Corpora with hard labels cannot be scored this way."
    )
