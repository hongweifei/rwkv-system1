"""Pure-PyTorch RWKV-7 ("Goose") inference.

Transcribed from the official implementation shipped in the ``rwkv`` PyPI package
(``rwkv/model.py``, the ``RWKV_CUDA_ON != '1'`` branches of ``RWKV_x070_TMix_seq``
and ``RWKV_x070_CMix_seq``).  We re-implement it rather than import it so that we
can run the official native ``.pth`` weights on CPU without TorchScript, without the
CUDA extension, and without the Triton-dependent ``fla`` package -- and so that the
recurrent state can be cloned and reused across calls (the option-scoring trick).

The recurrent state of a layer ``i`` is three tensors:

* ``att_x_prev``  ``(C,)``      last post-ln1 activation, for the token shift
* ``att_kv``      ``(H, N, N)`` the WKV-7 memory, the actual "context summary"
* ``ffn_x_prev``  ``(C,)``      last post-ln2 activation, for the FFN token shift

Because ``att_kv`` is fixed-size, a forward pass over the context can be done once
and every candidate option scored against the *same* state.  This is the property a
Transformer does not have.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

LN_EPS = 64e-5
DECAY_SCALE = 0.606531  # exp(-0.5)

_TRANSPOSED_SUFFIXES = (
    "key.weight",
    "value.weight",
    "receptance.weight",
    "output.weight",
    "head.weight",
)


def normalise_native(raw: dict[str, torch.Tensor], name: str,
                     dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Turn a BlinkDL checkpoint dict into the dict this forward pass reads.

    Split out of :func:`load_native_checkpoint` because a single-file release hands us the base
    tensors inside a bundle rather than as their own file.  The two paths must be the *same*
    arithmetic: tensors are widened to ``dtype`` *before* ln0 is fused into the embedding, and doing
    that fusion in bf16 costs roughly a bf16 ulp per embedding entry, which the 24 recurrent layers
    then amplify into visible logit drift.
    """
    # A DeepEmbed checkpoint (the `rwkv7a-*` files) trains an extra token-conditioned term into
    # every FFN and needs a different forward. All the plain-x070 keys are still present, so
    # without this guard we would load it, never read those tensors, and quietly produce
    # embeddings-minus-a-term instead of raising.
    deep = sorted(k for k in raw if "s_emb" in k)
    if deep:
        raise ValueError(
            f"{name} is a DeepEmbed checkpoint ({len(deep)} `s_emb` tensors, "
            f"e.g. {deep[0]}); this loader implements plain x070 only -- use a rwkv7-* file"
        )

    z: dict[str, torch.Tensor] = {}
    for k, tensor in list(raw.items()):
        if k.endswith(_TRANSPOSED_SUFFIXES):
            tensor = tensor.t().contiguous()
        tensor = tensor.squeeze()
        z[k] = tensor.to(dtype) if tensor.is_floating_point() else tensor
        del raw[k]

    # Layer 0 has no previous value to mix in, so its v_lora slots are dead weight.
    for i in range(3):
        z[f"blocks.0.att.v{i}"] = torch.empty(0, dtype=dtype)

    # ln0 is fused into the embedding lookup by the official loader.
    z["emb.weight"] = F.layer_norm(
        z["emb.weight"],
        (z["emb.weight"].shape[1],),
        weight=z["blocks.0.ln0.weight"],
        bias=z["blocks.0.ln0.bias"],
    )
    return z


def load_native_checkpoint(
    path: str | Path, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """Load a BlinkDL-format ``.pth`` and normalise it the way the official loader does."""
    raw = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
    return normalise_native(raw, Path(path).name, dtype)


class RWKV7:
    """Frozen RWKV-7 backbone. CPU, fp32, no autograd through the recurrence."""

    def __init__(self, weights: dict[str, torch.Tensor], dtype: torch.dtype = torch.float32):
        self.z = {k: v.to(dtype) if v.is_floating_point() else v for k, v in weights.items()}
        self.n_head, self.head_size = self.z["blocks.0.att.r_k"].shape
        self.vocab_size, self.n_embd = self.z["emb.weight"].shape
        self.n_layer = 1 + max(
            int(k.split(".")[1]) for k in self.z if k.startswith("blocks.")
        )
        self.dtype = dtype

    @classmethod
    def from_file(cls, path: str | Path, dtype: torch.dtype = torch.float32) -> "RWKV7":
        return cls(load_native_checkpoint(path, dtype), dtype)

    @classmethod
    def from_bundle(cls, base: dict[str, torch.Tensor], name: str = "<bundle base>",
                   dtype: torch.dtype = torch.float32) -> "RWKV7":
        """The same construction from tensors already in hand -- a single-file release's base section.

        Deliberately shares :func:`normalise_native` with ``from_file`` instead of re-deriving it:
        two loaders for one format is how the two paths drift, and the drift would show up as a
        difference in the answers, not as an error.
        """
        return cls(normalise_native(base, name, dtype), dtype)

    def zero_state(self, batch: int = 1) -> list[torch.Tensor]:
        h, n, c = self.n_head, self.head_size, self.n_embd
        state: list[torch.Tensor] = []
        for _ in range(self.n_layer):
            state.append(torch.zeros(batch, c, dtype=self.dtype))
            state.append(torch.zeros(batch, h, n, n, dtype=torch.float32))
            state.append(torch.zeros(batch, c, dtype=self.dtype))
        return state

    @staticmethod
    def clone_state(state: list[torch.Tensor]) -> list[torch.Tensor]:
        return [t.clone() for t in state]

    def _time_mix(
        self,
        layer: int,
        x: torch.Tensor,
        x_prev: torch.Tensor,
        kv: torch.Tensor,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.z
        p = f"blocks.{layer}.att."
        b, t, c = x.shape
        h, n = self.n_head, self.head_size

        delta = torch.cat((x_prev.unsqueeze(1), x[:, :-1]), dim=1) - x
        xr = x + delta * z[p + "x_r"]
        xw = x + delta * z[p + "x_w"]
        xk = x + delta * z[p + "x_k"]
        xv = x + delta * z[p + "x_v"]
        xa = x + delta * z[p + "x_a"]
        xg = x + delta * z[p + "x_g"]

        r = xr @ z[p + "receptance.weight"]
        w = torch.tanh(xw @ z[p + "w1"]) @ z[p + "w2"]
        k = xk @ z[p + "key.weight"]
        v = xv @ z[p + "value.weight"]
        a = torch.sigmoid(z[p + "a0"] + (xa @ z[p + "a1"]) @ z[p + "a2"])
        g = torch.sigmoid(xg @ z[p + "g1"]) @ z[p + "g2"]

        kk = F.normalize((k * z[p + "k_k"]).view(b, t, h, n), dim=-1, p=2.0).view(b, t, h * n)
        k = k * (1 + (a - 1) * z[p + "k_a"])
        if layer == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(z[p + "v0"] + (xv @ z[p + "v1"]) @ z[p + "v2"])

        # decay is strictly inside (0, 1); the log-space form the CUDA kernel takes
        # is exp()'d here because we run the recurrence directly.  w0 is the decay
        # LoRA's output bias, not an optional extra.
        w = torch.exp(-DECAY_SCALE * torch.sigmoid(z[p + "w0"] + w).float())

        out = torch.empty(b, t, h * n, dtype=self.dtype)
        r_h = r.view(b, t, h, n)
        w_h = w.view(b, t, h, n)
        k_h = k.view(b, t, h, n)
        v_h = v.view(b, t, h, n)
        kk_h = kk.view(b, t, h, n)
        a_h = a.view(b, t, h, n)

        for s in range(t):
            rs, ws = r_h[:, s], w_h[:, s]
            ks, vs = k_h[:, s], v_h[:, s]
            kks, as_ = kk_h[:, s], a_h[:, s]

            # The official CPU path materialises the O(N^3) products
            #   state @ ((-kk) (x) (kk*a))   and   v (x) k
            # every step.  Both are rank-1, so
            #   S @ ((-kk) (x) (kk*a)) == -(S @ kk) (x) (kk*a)
            # which removes the matmul and keeps the recurrence O(N^2).  Same value,
            # different rounding order.
            u = (kv * kks.unsqueeze(-2)).sum(-1)
            kv = (
                kv * ws.unsqueeze(-2)
                - u.unsqueeze(-1) * (kks * as_).unsqueeze(-2)
                + vs.unsqueeze(-1) * ks.unsqueeze(-2)
            )
            out[:, s] = (kv.to(self.dtype) * rs.unsqueeze(-2)).sum(-1).view(b, h * n)

        o = F.group_norm(
            out.view(b * t, h * n),
            num_groups=h,
            weight=z[p + "ln_x.weight"],
            bias=z[p + "ln_x.bias"],
            eps=LN_EPS,
        ).view(b, t, h * n)
        o = o + (
            (r.view(b, t, h, n) * k.view(b, t, h, n) * z[p + "r_k"]).sum(dim=-1, keepdim=True)
            * v.view(b, t, h, n)
        ).view(b, t, h * n)
        return (o * g) @ z[p + "output.weight"], x[:, -1], kv, v_first

    def _channel_mix(self, layer: int, x: torch.Tensor, x_prev: torch.Tensor):
        p = f"blocks.{layer}.ffn."
        delta = torch.cat((x_prev.unsqueeze(1), x[:, :-1]), dim=1) - x
        k = x + delta * self.z[p + "x_k"]
        k = F.relu(k @ self.z[p + "key.weight"]) ** 2
        return k @ self.z[p + "value.weight"], x[:, -1]

    @torch.inference_mode()
    def forward(
        self,
        tokens: torch.Tensor,
        state: list[torch.Tensor] | None = None,
        return_hidden: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Run one sequence.

        ``tokens`` is ``(B, T)``.  Returns ``(hidden, logits, state)`` where ``hidden``
        is the post-``ln_out`` activation ``(B, T, C)`` -- the feature space the
        decision head consumes.
        """
        x, new_state, _ = self._run(tokens, state, collect=frozenset())
        hidden = self._ln_out(x)
        logits = hidden @ self.z["head.weight"]
        return hidden, logits, new_state

    @torch.inference_mode()
    def forward_layers(
        self,
        tokens: torch.Tensor,
        state: list[torch.Tensor] | None = None,
        collect: tuple[int, ...] = (),
    ) -> tuple[dict[int, torch.Tensor], list[torch.Tensor]]:
        """The post-``ln_out`` activation after each block in ``collect``, plus the new state.

        One pass, not one per layer: this exists so a feature experiment can ask for an
        intermediate block alongside the last one without paying for a second forward over the
        sequence.  ``self.n_layer - 1`` is the layer :meth:`forward` returns.  Every collected
        tensor is normalised with ``ln_out``, so a concatenation of two of them lives in one
        space rather than mixing a raw residual stream with a normalised one.
        """
        x, new_state, collected = self._run(tokens, state, collect=frozenset(collect))
        return collected, new_state

    def _ln_out(self, x: torch.Tensor) -> torch.Tensor:
        # ln0 is already fused into the embedding, so ln_out is the only final norm.
        return F.layer_norm(
            x, (self.n_embd,), weight=self.z["ln_out.weight"], bias=self.z["ln_out.bias"]
        )

    def _run(
        self,
        tokens: torch.Tensor,
        state: list[torch.Tensor] | None,
        collect: frozenset[int],
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[int, torch.Tensor]]:
        """The block loop.  ``forward`` and ``forward_layers`` differ only in what they keep."""
        if state is None:
            state = self.zero_state(tokens.shape[0])
        x = self.z["emb.weight"][tokens]
        v_first = None
        new_state: list[torch.Tensor] = []
        collected: dict[int, torch.Tensor] = {}

        for i in range(self.n_layer):
            p = f"blocks.{i}."
            ax, kv, fx = state[3 * i], state[3 * i + 1], state[3 * i + 2]
            xx = F.layer_norm(
                x, (self.n_embd,), weight=self.z[p + "ln1.weight"], bias=self.z[p + "ln1.bias"]
            )
            xx, ax, kv, v_first = self._time_mix(i, xx, ax, kv, v_first)
            x = x + xx

            xx = F.layer_norm(
                x, (self.n_embd,), weight=self.z[p + "ln2.weight"], bias=self.z[p + "ln2.bias"]
            )
            xx, fx = self._channel_mix(i, xx, fx)
            x = x + xx

            new_state += [ax, kv, fx]
            if i in collect:
                collected[i] = self._ln_out(x)

        return x, new_state, collected
