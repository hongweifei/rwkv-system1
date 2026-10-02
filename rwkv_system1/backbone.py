"""Frozen RWKV-7 backbone wrapper: tokenisation, prompt layout, marker extraction.

Laya's layout is ``[CLS] <qtype> question: ... [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]``
and it works because ModernBERT is bidirectional: an option's ``[MASK]`` slot can attend
forward to the state that follows it.

RWKV is causal, so the layout has to be reordered -- everything an option must see has to
come *before* it, and the scoring slot has to come after the option text:

    <state> \\n  |  <qtype> question: <instructions> \\n  <opt0> \\n  <opt1> \\n ...

Everything before the ``|`` is the prefix.  Two consequences fall out, both good:

* The prefix depends only on the state, so **one prefill serves every question asked about
  that state**.  Laya gets cross-question reuse from a shared forward pass; RWKV gets it from
  reusing a fixed-size recurrent state, which is cheaper and is the reason this backbone was
  worth choosing.
* Markers are just the newline that terminates each option, so no vocabulary surgery is
  needed to invent a ``[MASK]`` token.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .rwkv7 import RWKV7

VOCAB_FILENAME = "rwkv_vocab_v20230424.txt"


def find_vocab(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    candidates = [
        # The table is model-independent (v20230424 serves every RWKV-7 checkpoint, and it is
        # byte-identical to the one bundled in the `rwkv` package apart from line endings), so
        # it is vendored beside the weights rather than inside one model's directory.
        Path(__file__).resolve().parent.parent / "weights" / VOCAB_FILENAME,
    ]
    try:
        import rwkv

        candidates.append(Path(rwkv.__file__).resolve().parent / VOCAB_FILENAME)
    except ImportError:
        pass
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"{VOCAB_FILENAME} not found; looked in {[str(c) for c in candidates]}"
    )


TERMINATOR = "\n"


@dataclass(frozen=True)
class PromptTemplate:
    state_suffix: str = TERMINATOR
    question_header: str = " question: "
    terminator: str = TERMINATOR

    def prefix(self, state: str) -> str:
        return f"{state}{self.state_suffix}"

    def suffix(self, qtype_name: str, instructions: str, options: list[str]) -> str:
        head = f"{qtype_name}{self.question_header}{instructions}{self.terminator}"
        return head + "".join(f"{o}{self.terminator}" for o in options)


class DecisionBackbone:
    def __init__(
        self,
        checkpoint: str | Path,
        vocab: str | Path | None = None,
        template: PromptTemplate | None = None,
        ctx_tokens: int = 8192,
    ):
        self._bind(RWKV7.from_file(checkpoint), vocab, template, ctx_tokens)

    @classmethod
    def from_model(
        cls,
        model: RWKV7,
        vocab: str | Path | None = None,
        template: PromptTemplate | None = None,
        ctx_tokens: int = 8192,
    ) -> "DecisionBackbone":
        """Wrap an RWKV7 that is already in memory -- a single-file release hands over a bundle.

        Shares `_bind` with `__init__` on purpose: the terminator must be one token for marker
        positions to be unambiguous, and that guarantee should not exist in one constructor only.
        """
        self = cls.__new__(cls)
        self._bind(model, vocab, template, ctx_tokens)
        return self

    def _bind(self, model: RWKV7, vocab: str | Path | None,
              template: PromptTemplate | None, ctx_tokens: int) -> None:
        from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

        self.model = model
        self.tokenizer = TRIE_TOKENIZER(str(find_vocab(vocab)))
        self.template = template or PromptTemplate()
        #: The one length bound that remains.  RWKV-7 has no positional-encoding ceiling, so this
        #: is the context the checkpoint was trained at, not an architectural limit, and neither
        #: half of the prompt is truncated any more -- see ``check_room``.  What stops us going
        #: longer is compute: ``rwkv7.RWKV7`` runs the recurrence as a Python loop over time
        #: steps, measured at 45 s for one 8192-token pass.
        self.ctx_tokens = ctx_tokens
        self.feature_dim = self.model.n_embd
        self._nl = self.tokenizer.encode(self.template.terminator)
        if len(self._nl) != 1:
            raise RuntimeError(
                f"terminator {self.template.terminator!r} is not a single token "
                f"({self._nl}); marker positions would be ambiguous"
            )
        self.terminator_id = self._nl[0]

    def encode(self, text: str, limit: int | None = None) -> torch.Tensor:
        ids = self.tokenizer.encode(text)
        if limit is not None:
            ids = ids[:limit]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def encode_state(self, state: str) -> torch.Tensor:
        """The state prefix, whole.

        This used to be cut at 192 tokens -- a budget inherited from the synthetic corpora, whose
        states are ~190 tokens by construction.  On real text it silently dropped the tail of
        68-78% of the typed-decisions rows (median state 270 tokens, max 524), which are the rows
        the published comparisons rest on, and a silent cut is indistinguishable from a model
        that ignored the text.  The cap is gone; ``check_room`` is the only bound left.
        """
        return self.encode(self.template.prefix(state))

    def check_room(self, state_tokens: int, block_tokens: int) -> None:
        """Refuse a prompt that does not fit the context the checkpoint was trained at.

        Called where *both* halves are known -- the state is tokenised in ``extract`` and the
        block in ``encode_query``, and neither knows the other.  Raising beats truncating: a
        truncated prompt trains and scores exactly like a full one.
        """
        total = state_tokens + block_tokens
        if total > self.ctx_tokens:
            raise ValueError(
                f"state {state_tokens} + options {block_tokens} = {total} tokens, over the"
                f" {self.ctx_tokens}-token context this checkpoint was trained at"
            )

    def encode_query(
        self, qtype_name: str, instructions: str, options: list[str]
    ) -> tuple[torch.Tensor, list[int]]:
        """Tokenise the question+options block and locate the marker of each option.

        Markers are the terminators that follow an option, found by scanning for the
        terminator token rather than by re-tokenising each option in isolation: the
        tokenizer is greedy longest-match, so ``a + b`` does not generally tokenise to
        ``tok(a) + tok(b)``.
        """
        text = self.template.suffix(qtype_name, instructions, options)
        ids = self.tokenizer.encode(text)
        if len(ids) > self.ctx_tokens:
            raise ValueError(
                f"the {len(options)}-option block alone needs {len(ids)} tokens, more than the"
                f" {self.ctx_tokens}-token context this checkpoint was trained at"
            )
        positions = [i for i, t in enumerate(ids) if t == self.terminator_id]
        markers = positions[1:]  # the first terminator closes the question header
        if len(markers) != len(options):
            raise ValueError(
                f"expected {len(options)} option markers, found {len(markers)}; "
                f"an option probably contains {self.template.terminator!r}"
            )
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0), markers
