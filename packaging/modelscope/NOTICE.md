Attribution and licensing
========================

This package contains two clearly separate things, and they are not the same author's work.

1. What is published here (this repository's own work)

   - the decision heads inside `model.pth` -- one section per trained head, fitted on public
     benchmark data.
   - `modeling/service.py`  -- the read path that ties a head to its prompt and temperature bucket.
   - `modeling/rwkv7.py`, `modeling/decision.py`, `modeling/backbone.py`, `modeling/features.py`
     -- the inference code, copied verbatim from the training repository at build time so the
     published model does not depend on that repository being installed.
   - `modeling/rwkv_vocab_v20230424.txt` -- the RWKV tokenizer table, vendored because it is 1 MB,
     model-independent, and required for any inference.

   These are offered under the Apache License, Version 2.0.

       Copyright 2026 the author of this package

       Licensed under the Apache License, Version 2.0 (the "License"); you may not use these
       files except in compliance with the License. You may obtain a copy of the License at

           http://www.apache.org/licenses/LICENSE-2.0

       Unless required by applicable law or agreed to in writing, software distributed under the
       License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
       either express or implied. See the License for the specific language governing permissions
       and limitations under the License.

   The licence above is what *this package declares*. Change it before publishing if you want a
   different one; nothing in the code depends on which.

2. The base model inside `model.pth`, which is someone else's work

   - The `base` section is a verbatim copy of `rwkv7-g1d-0.1b-20260129-ctx8192.pth`, the base model
     by the RWKV project (`RWKV/rwkv7-g1` on ModelScope, `license: apache-2.0` in its model card,
     382 MB). It is redistributed here under that licence, with attribution, because the heads only
     mean anything against it -- but it was **not trained by the author of this package**, and no
     number on the card is credit for it. The upstream file's name, size and head/tail digest are
     recorded in `config.json` beside the section, so the copy can be checked against the original.
   - `model.pth` carries a sha256 in `config.json`, verified on load before a single tensor is read,
     so a truncated or substituted download fails with a message instead of quietly changing the
     answers.
   - The training/evaluation corpora: Laya typed-decisions, MASSIVE, XNLI, Banking77, go_emotions,
     amazon_reviews_multi, ag_news, sst5, boolq, prompt-injections. None of their data ships here;
     only the heads that were fitted on subsets of them. Each keeps its own licence, and any
     comparison to published figures should cite the original dataset papers, not this package.

3. Reference implementations consulted

   - Laya's typed-decisions work (ModernBERT backbone + set-attention decision head), whose two-stage
     CE/RLCD recipe, `1 - H(p)/log k` confidence and temperature fitting this design follows.
   - The official RWKV-7 implementation (`RWKV/rwkv` PyPI package, used here only for its tokenizer).
