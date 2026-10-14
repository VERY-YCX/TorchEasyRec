# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OneRank structured tokenization (paper 2.1 / 2.2).

Every candidate is expanded into a group of ``K + 1`` tokens -- the
candidate token followed by the task tokens::

    X   = [ prefix (contextual + UIH, length H) | G_1 | G_2 | ... | G_N ]
    G_i = [ e^C_i, t_1, t_2, ..., t_K ]                        |G_i| = K + 1

The task tokens ``t_1..t_K`` are ``K`` learned vectors shared by every
candidate and every request; what makes ``t_k`` candidate-specific is the
mask, not the content.

The mask gives ``t_k`` the visible set ``[0, H) u {e^C_i} u {t_k}`` --
three column intervals, because the mutually invisible ``t_1..t_{k-1}``
sit in between.  HSTU's arbitrary-mask path compiles with
``HSTU_ARBITRARY_NFUNC = 5``, exposing exactly ``(NFUNC + 1) // 2 = 3``
intervals per query row (see :func:`build_onerank_func_tensor`), so the
paper layout encodes directly with no candidate replicas.

The cost is ``N * (K + 1)`` tokens per request, so ``N`` must stay in
the few-candidates-per-request regime a rerank stage works in (a few to
a few dozen); it does not scale to full-corpus candidate sets.

The **group mixer** (``group_mixer`` on ``family_tokens``; see
:class:`OneRankGroupMixer`) moves the in-group half of that attention out
of the STU stack: a RankMixer block -- parameter-free token-mixing
transpose plus per-slot FFNs with slot-isolated parameters -- is spliced
in right after tokenization, and the group rows' func-tensor columns
shrink to prefix + self.  The tokenization, the ``W_f`` projections, the
readout and the prefix's causal attention are unchanged, so the arm
isolates exactly one variable against ``family_tokens``: the in-group
interaction mechanism (attention vs. transpose + per-slot FFN).
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.profiler import record_function

from tzrec.modules.gr.hstu_transducer import HSTUTransducer
from tzrec.modules.gr.stu import STU, STULayer
from tzrec.modules.utils import BaseModule
from tzrec.ops.hstu_attention_utils import (
    _FAM_MASKS,
    FAM_MASK_GROUP_NULL,
    LAYOUT_BLOCK,
    LAYOUT_PAPER,
    LAYOUTS,
    STUTruncationPlan,
    build_onerank_func_tensor,
)
from tzrec.ops.jagged_tensors import concat_2D_jagged, split_2D_jagged
from tzrec.utils.fx_util import fx_int_item

torch.fx.wrap(fx_int_item)


def _fx_check_max_seq_len(
    max_prefix_len: int, max_new_targets: int, max_seq_len: int
) -> None:
    """Raise when a request's inflated sequence exceeds ``max_seq_len``.

    An fx-wrapped leaf: under ``torch.fx`` symbolic tracing the bounds are
    proxies whose comparison cannot be evaluated as a Python bool, so the
    check must be recorded as a graph node and run for real at execution
    time (``fx_int_item`` returns a proxy while tracing, an ``int``
    otherwise).
    """
    if max_prefix_len + max_new_targets > max_seq_len:
        raise ValueError(
            f"a request's inflated sequence reaches "
            f"{max_prefix_len + max_new_targets} tokens "
            f"({max_prefix_len} history + {max_new_targets} expanded "
            f"candidate tokens), more than max_seq_len ({max_seq_len}); "
            f"every jagged kernel was autotuned for at most max_seq_len "
            f"tokens. Raise max_seq_len or filter the input."
        )


torch.fx.wrap(_fx_check_max_seq_len)


class _PerSlotFFN(torch.nn.Module):
    """Two-layer MLP with parameters private to one slot (eq. 6-7)."""

    def __init__(self, dim: int, hidden_multiplier: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden_multiplier * dim)
        self.fc2 = torch.nn.Linear(hidden_multiplier * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class OneRankGroupMixer(BaseModule):
    """RankMixer block over a candidate's token group (``group_mixer``).

    Replaces the in-group attention of the ``family_tokens`` block layout
    with RankMixer's two operators (Fan et al., RankMixer):

    * **Multi-head token mixing** (eq. 3-5): each token is cut into
      ``H = T`` heads (``T`` = group size); head ``h`` of the mixed
      output is the concatenation of head ``h`` of every input token --
      i.e. the (T, D) token matrix, viewed as (T, H, D/H), is transposed
      to (H, T, D/H) and re-flattened to (T, D).  Parameter-free, and
      ``H = T`` is what keeps the output shaped like the residual.  In
      attention terms it is a rank-free all-to-all mixing: no QK inner
      products between heterogeneous feature spaces (the paper's
      diagnosis of why attention underperforms here).
    * **Per-slot FFN** (eq. 6-7): every slot gets its own two-layer MLP
      (``fc1 -> Gelu -> fc2``) with slot-isolated parameters -- the
      ablation axis the paper found largest (its Table 2).

    Both sub-blocks are residual + LayerNorm (eq. 5).  ``D`` need not be
    divisible by ``T`` (the production probe is D=512, T=12): the stream
    is zero-padded to ``D_pad = ceil(D / T) * T`` for the whole block and
    sliced back to ``D`` at the exit -- the only deviation from the
    paper, whose feature tokens are built with a divisible ``D`` by
    construction.

    Capacity note: under the group_mixer func tensor no row reads a
    group column other than its own and ``_compose_output`` keeps only
    the task-slot rows, so the *family*-slot FFNs receive no gradient
    (they stay at init; 5 of 12 slots, ~42% of the parameter budget
    above in the production probe).  The family embeddings still train
    -- token mixing reads every input token -- and the task-slot FFNs,
    the shared norms and the family projections are all live.

    Input / output: the flat expanded group segment
    ``(total_targets * T, D)`` -- exactly what :class:`OneRankTokenizer`
    produces and what ``concat_2D_jagged`` consumes, so the mixer splices
    in between the two with a single ``view``.

    Args:
        group_size (int): tokens per candidate group ``T`` (``K + F``).
        embedding_dim (int): STU embedding dim ``D``.
        ffn_multiplier (int): hidden multiplier ``k`` of the per-slot FFN.
        num_layers (int): stacked RankMixer blocks.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        group_size: int,
        embedding_dim: int,
        ffn_multiplier: int = 1,
        num_layers: int = 1,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        if group_size <= 0:
            raise ValueError(f"group_size must be positive; got {group_size}")
        if embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive; got {embedding_dim}"
            )
        if ffn_multiplier <= 0:
            raise ValueError(
                f"ffn_multiplier must be positive; got {ffn_multiplier}"
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive; got {num_layers}")
        self._group_size: int = group_size
        self._embedding_dim: int = embedding_dim
        # ceil division without importing math.ceil twice; 512 with T=12
        # -> 516.
        self._padded_dim: int = -(-embedding_dim // group_size) * group_size
        self._layers = torch.nn.ModuleList(
            torch.nn.ModuleList(
                _PerSlotFFN(self._padded_dim, ffn_multiplier)
                for _ in range(group_size)
            )
            for _ in range(num_layers)
        )
        self._norms = torch.nn.ModuleList(
            torch.nn.ModuleList(
                torch.nn.LayerNorm(self._padded_dim) for _ in range(2)
            )
            for _ in range(num_layers)
        )

    @property
    def group_size(self) -> int:
        """Tokens per candidate group ``T``."""
        return self._group_size

    @property
    def padded_dim(self) -> int:
        """The stream width inside the block: ``ceil(D/T)*T``."""
        return self._padded_dim

    def forward(self, group_embeddings: torch.Tensor) -> torch.Tensor:
        """Mix each candidate's ``[t_1..t_K, x^1..x^F]`` block in place.

        Args:
            group_embeddings (torch.Tensor): the expanded group segment
                ``(total_targets * T, D)``; candidate ``i`` owns rows
                ``[i*T, (i+1)*T)``.

        Returns:
            torch.Tensor: same shape; every token carries the whole
            group's information (token mixing) passed through its own
            FFN.
        """
        T = self._group_size
        D = self._embedding_dim
        D_pad = self._padded_dim
        pad = D_pad - D  # static: decided at construction, fx-safe
        x = group_embeddings.view(-1, T, D)
        if pad:
            x = F.pad(x, (0, pad))
        for ffns, norms in zip(self._layers, self._norms):
            # --- Token mixing (eq. 3-5): (N, T, D_pad) viewed as
            # (N, token, head, D_pad/T) -> heads permuted onto the token
            # axis -> (N, T, D_pad).  With H = T this is the transpose of
            # the token matrix and carries no parameters.
            heads = x.view(-1, T, T, D_pad // T)
            mixed = heads.permute(0, 2, 1, 3).reshape(-1, T, D_pad)
            x = norms[0](x + mixed)
            # --- Per-slot FFN (eq. 6-7): slot-isolated parameters.
            x = norms[1](
                x
                + torch.stack(
                    [ffns[t](x[:, t, :]) for t in range(T)], dim=1
                )
            )
        if pad:
            x = x[..., :D]
        return x.reshape(-1, D)


class OneRankTokenizer(BaseModule):
    """Expands each candidate into its token group.

    Two layouts, selected by whether family info is given:

    - paper (default): ``[e^C_i, t_1..t_K]`` (``K + 1`` tokens);
      ``forward`` receives the preprocessor's projected candidate
      embeddings.
    - block (``family_dims`` given): ``[t_1..t_K, x^1..x^F]`` (``K + F``
      tokens); ``forward`` receives the *raw* candidate features,
      because family boundaries live in the raw feature layout.  Each
      family token ``x^f = W_f(gather_f(raw))`` is projected to the STU
      embedding dim by a per-family ``Linear`` shared across tasks -- the
      task axis is carried by the task tokens' attention over the family
      tokens, not by the projections.

    Args:
        num_tasks (int): number of task tokens ``K``.
        embedding_dim (int): STU embedding dimension.
        is_inference (bool): whether to run in inference mode.
        family_dims (Optional[List[int]]): dimension of each family's
            slice of the raw candidate features; ``None`` selects the
            paper layout.
        family_feature_ids (Optional[List[int]]): family id of each
            *dimension* of the raw candidate features (the model layer
            expands the proto's per-feature ids); required iff
            ``family_dims`` is given.
        family_names (Optional[List[str]]): human-readable family names,
            kept only for reading the learned projections off a trained
            checkpoint.
    """

    def __init__(
        self,
        num_tasks: int,
        embedding_dim: int,
        is_inference: bool = False,
        family_dims: Optional[List[int]] = None,
        family_feature_ids: Optional[List[int]] = None,
        family_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(is_inference=is_inference)
        if num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive; got {num_tasks}")
        self._num_tasks: int = num_tasks
        self._embedding_dim: int = embedding_dim
        self._task_tokens: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty((num_tasks, embedding_dim)).normal_(
                0.0, 1.0 / math.sqrt(embedding_dim)
            )
        )
        if (family_dims is None) != (family_feature_ids is None):
            raise ValueError(
                "family_dims and family_feature_ids must be given together "
                f"(got family_dims={'set' if family_dims else 'None'}, "
                f"family_feature_ids={'set' if family_feature_ids else 'None'})"
            )
        self._family_names: Optional[List[str]] = None
        self._family_dims: Optional[List[int]] = None
        if family_dims is not None:
            fam_dims = [int(d) for d in family_dims]
            fam_ids = [int(i) for i in family_feature_ids]
            n_fam = len(fam_dims)
            if n_fam <= 0:
                raise ValueError("family_dims must not be empty")
            if any(d <= 0 for d in fam_dims):
                raise ValueError(
                    f"every family must own at least one dimension; got "
                    f"family_dims={fam_dims}"
                )
            if len(fam_ids) != sum(fam_dims):
                raise ValueError(
                    f"family_feature_ids has {len(fam_ids)} entries but "
                    f"family_dims sum to {sum(fam_dims)}: one id per raw "
                    f"candidate dimension is required"
                )
            if min(fam_ids, default=0) < 0 or max(fam_ids, default=-1) >= n_fam:
                raise ValueError(
                    f"family ids must lie in [0, {n_fam}); got ids spanning "
                    f"[{min(fam_ids)}, {max(fam_ids)}]"
                )
            # Per-family column indices of the raw features: family f
            # owns a possibly discontiguous set of columns (FI families
            # interleave in embedding order), so each family gets an index
            # buffer and forward gathers with one index_select per family
            # -- single-op, fx-friendly.  No cached index-tensor list: the
            # buffer references would go stale across .to(device)/DMP, so
            # forward looks the buffers up live (see _forward_block).
            indices = [[] for _ in range(n_fam)]
            for j, fid in enumerate(fam_ids):
                indices[fid].append(j)
            for f, idx in enumerate(indices):
                self.register_buffer(
                    f"_family_idx_{f}", torch.as_tensor(idx, dtype=torch.long)
                )
            self._family_dims = fam_dims
            self._family_names = list(family_names or [
                f"family_{i}" for i in range(n_fam)
            ])
            if len(self._family_names) != n_fam:
                raise ValueError(
                    f"family_names has {len(self._family_names)} entries "
                    f"but family_dims imply {n_fam} families"
                )
            self._family_projs = torch.nn.ModuleList(
                torch.nn.Linear(d, embedding_dim) for d in fam_dims
            )

    @property
    def num_tasks(self) -> int:
        """Number of task tokens ``K``."""
        return self._num_tasks

    @property
    def layout(self) -> str:
        """Token layout: :data:`LAYOUT_PAPER` or :data:`LAYOUT_BLOCK`."""
        return LAYOUT_BLOCK if self._family_dims is not None else LAYOUT_PAPER

    @property
    def num_families(self) -> Optional[int]:
        """Number of family tokens ``F``; ``None`` in the paper layout."""
        return None if self._family_dims is None else len(self._family_dims)

    @property
    def family_names(self) -> Optional[List[str]]:
        """Family names, for reading learned projections; ``None`` if off."""
        return self._family_names

    @property
    def group_size(self) -> int:
        """Tokens per candidate group: ``K + 1`` paper, ``K + F`` block."""
        if self._family_dims is not None:
            return self._num_tasks + len(self._family_dims)
        return self._num_tasks + 1

    def forward(self, target_embeddings: torch.Tensor) -> torch.Tensor:
        """Expand the candidate segment of a jagged sequence.

        Args:
            target_embeddings (torch.Tensor): candidate embeddings
                ``(total_targets, D)``, jagged and grouped by request.
                ``D`` is the preprocessor output dim in the paper layout,
                the *raw* candidate feature dim (e.g. 736) in the block
                layout.

        Returns:
            torch.Tensor: ``(total_targets * group_size, D_out)``, each
                candidate replaced in place by its token group --
                ``[e^C_i, t_1, ..., t_K]`` paper, or
                ``[t_1, ..., t_K, x^1, ..., x^F]`` block (``D_out`` is
                the STU embedding dim in both).
        """
        if self._family_dims is not None:
            return self._forward_block(target_embeddings)
        return self._forward_paper(target_embeddings)

    def _forward_paper(self, target_embeddings: torch.Tensor) -> torch.Tensor:
        dim = target_embeddings.size(-1)
        tokens = self._task_tokens.to(target_embeddings.dtype)
        tokens = tokens.unsqueeze(0).expand(target_embeddings.size(0), -1, -1)
        # (total_targets, K + 1, D) -> flatten preserves the
        # candidate-then-task-token order the mask assumes.
        return torch.cat([target_embeddings.unsqueeze(1), tokens], dim=1).reshape(
            -1, dim
        )

    def _forward_block(self, candidate_raw: torch.Tensor) -> torch.Tensor:
        """``[t_1..t_K, x^1..x^F]`` per candidate, block layout."""
        # family tokens: gather each family's (discontiguous) columns and
        # project them to the STU dim.  Buffer lookups are live
        # (self._buffers[...]) so .to(device)/DMP cannot leave a stale
        # index tensor behind.
        fam_tokens = []
        for f in range(len(self._family_dims)):
            seg = torch.index_select(
                candidate_raw, 1, self._buffers[f"_family_idx_{f}"]
            )
            fam_tokens.append(self._family_projs[f](seg))
        # (total_targets, F, D)
        fam_tokens = torch.stack(fam_tokens, dim=1)
        # task queries, broadcast to every candidate; the family tokens
        # follow, so slot k in [0, K) is t_k and the readout keeps
        # ``_compose_output``'s task-slot convention.
        tokens = self._task_tokens.to(candidate_raw.dtype)
        tokens = tokens.unsqueeze(0).expand(candidate_raw.size(0), -1, -1)
        dim = self._embedding_dim
        return torch.cat([tokens, fam_tokens], dim=1).reshape(-1, dim)


class OneRankSTULayer(STULayer):
    """``STULayer`` masked with the OneRank group layout instead of SLA.

    Args:
        group_size (int): tokens per candidate group, ``num_tasks + 1``
            in the paper layout, ``num_tasks + num_families`` in the
            block layout.
        layout (str): :data:`LAYOUT_PAPER` (default) or
            :data:`LAYOUT_BLOCK` (the ``family_tokens`` probe).
        group_mask (str): block layout only; the in-group visibility
            strategy, one of the :data:`FAM_MASK_GROUP_NULL` /
            :data:`FAM_MASK_TASK_ISO` constants (see
            :func:`build_onerank_func_tensor`).
        num_tasks (Optional[int]): number of task tokens ``K``; required
            in the block layout (the ``FAM_MASK_TASK_ISO`` mask splits
            task rows from family rows at slot ``K``), ignored otherwise.
        group_attn (bool): whether group rows attend their own group
            inside attention.  ``False`` (the ``group_mixer`` probe)
            restricts them to prefix + self and requires the block
            layout; see :class:`OneRankGroupMixer`.
        **kwargs: forwarded verbatim to :class:`STULayer`.
    """

    def __init__(
        self,
        group_size: int,
        layout: str = LAYOUT_PAPER,
        group_mask: str = FAM_MASK_GROUP_NULL,
        num_tasks: Optional[int] = None,
        group_attn: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}; got {layout!r}")
        if group_size < 2:
            raise ValueError(
                f"group_size must be at least 2 (candidate token + one task "
                f"token, i.e. num_tasks + 1); got {group_size}"
            )
        if layout == LAYOUT_BLOCK:
            # build_onerank_func_tensor re-validates on every call; checked
            # here too so a bad config fails at construction, not at the
            # first forward.
            if group_mask not in _FAM_MASKS:
                raise ValueError(
                    f"group_mask must be one of {_FAM_MASKS}; got {group_mask!r}"
                )
            if num_tasks is None or not (1 <= num_tasks < group_size):
                raise ValueError(
                    f"the block layout requires 1 <= num_tasks < group_size "
                    f"(K task tokens + F >= 1 family tokens); got "
                    f"num_tasks={num_tasks}, group_size={group_size}"
                )
        if not group_attn and layout != LAYOUT_BLOCK:
            raise ValueError(
                f"group_attn=False (the group_mixer probe) requires the "
                f"block layout (family_tokens); got layout={layout!r}"
            )
        self._group_size: int = group_size
        self._layout: str = layout
        self._group_mask: str = group_mask
        self._group_num_tasks: Optional[int] = num_tasks
        self._group_attn: bool = group_attn
        if self._sla_k1 > 0 or self._sla_k2 > 0:
            raise ValueError(
                f"OneRank builds its own NFUNC mask and cannot also apply SLA; "
                f"keep stu.sla_k1 / stu.sla_k2 at 0, got sla_k1={self._sla_k1}, "
                f"sla_k2={self._sla_k2}."
            )
        if self._contextual_seq_len != 0:
            raise ValueError(
                f"OneRank requires stu.contextual_seq_len == 0, got "
                f"{self._contextual_seq_len}. The NFUNC mask encodes "
                f"(HSTU_ARBITRARY_NFUNC + 1) // 2 column intervals per query "
                f"row and the group layout already spends all of them on "
                f"`prefix + own group`; a bidirectional contextual block "
                f"would need another. With 0 the contextual tokens are "
                f"ordinary causal prefix; note this differs from the "
                f"DlrmHSTU baseline, whose unset (-1) sentinel resolves to "
                f"the number of contextual features and gives contextual "
                f"tokens a bidirectional block."
            )
        if self._max_attn_len > 0:
            raise ValueError(
                f"OneRank does not support stu.max_attn_len (got "
                f"{self._max_attn_len}); a local window would have to be "
                f"folded into the func tensor's intervals, which the group "
                f"layout already uses up."
            )
        if not self._causal:
            raise ValueError("OneRank requires stu.causal = true.")

    @property
    def uses_arbitrary_mask(self) -> bool:
        """OneRank always drives attention through its own func tensor."""
        return True

    @property
    def uses_noncausal_func_mask(self) -> bool:
        """Whether this layer's func tensor reads future columns.

        The paper layout stays causal: every interval ends at or before
        the query row (``{e^C_i}`` sits before ``t_k``).  Both block
        masks look ahead -- ``FAM_MASK_GROUP_NULL`` reads the whole group,
        ``FAM_MASK_TASK_ISO`` reads the trailing family block -- so the
        CUTLASS attention window must be lifted from ``(-1, 0)`` to
        ``(-1, -1)`` or the kernel would silently clip the family
        columns out of every group row.

        ``group_attn=False`` (the group_mixer probe) instead shrinks
        every group row's visible set to prefix + self -- every visible
        column is again at or left of the query row, so the causal
        window is a superset and the lift would be a no-op.
        """
        if not self._group_attn:
            return False
        return self._layout == LAYOUT_BLOCK

    @property
    def attn_func_static_sig(self) -> str:
        """OneRank NFUNC cache key.

        Prefixed so it can never collide with ``STULayer``'s SLA
        signature in a mixed stack.  ``layout``, ``group_mask`` and
        ``group_attn`` must be part of the key: the func tensor is
        cached across layers keyed on this string, and two OneRank
        layers with different token layouts, in-group masks or
        in-group attention modes must not share one tensor.
        """
        return (
            f"onerank:{self._layout}:{self._group_size}:"
            f"{self._num_heads}:{self._group_mask}:"
            f"{'attn' if self._group_attn else 'mixer'}"
        )

    def _build_attn_func(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Build the OneRank group mask.

        ``num_targets`` is used unconditionally: unlike SLA, the group
        layout is meaningless without the prefix boundary, so
        ``target_aware`` cannot switch it off.
        """
        return build_onerank_func_tensor(
            nheads=self._num_heads,
            seq_offsets=x_offsets,
            total_q=x.size(0),
            num_targets=num_targets,
            group_size=self._group_size,
            layout=self._layout,
            group_mask=self._group_mask,
            num_tasks=self._group_num_tasks,
            group_attn=self._group_attn,
        )

    def truncate_input(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: Optional[torch.Tensor],
        *,
        truncate_tail_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, STUTruncationPlan]:
        """Unsupported: truncation would desynchronize the group layout."""
        raise NotImplementedError(
            "OneRank does not support mid-stack attention truncation: the "
            "func tensor is cached across layers keyed on a static signature, "
            "and truncation changes total_q and the prefix boundary. Leave "
            "attn_truncation_split_layer / attn_truncation_tail_len at 0."
        )

    def cached_forward(
        self,
        delta_x: torch.Tensor,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Unsupported: the incremental path cannot carry a func tensor."""
        raise NotImplementedError(
            "OneRank has no incremental (KV-cache) serving path: "
            "delta_hstu_mha takes no attn_func and downgrades Kernel.CUTLASS "
            "to Kernel.TRITON, so the task-token mask would be silently "
            "dropped. Serve with the full forward pass."
        )


class OneRankHSTUTransducer(HSTUTransducer):
    """``HSTUTransducer`` that tokenizes candidates into OneRank groups.

    The expansion is spliced in between the positional encoder and the STU
    stack by overriding :meth:`_preprocess`: the base implementation runs
    the input preprocessor plus positional encoding, and only then are the
    candidates replaced by their groups.  Task tokens therefore carry no
    positional / time encoding (they are pure learned queries) while the
    candidate token keeps the encoding of the candidate it heads.  In the
    block layout there is no candidate token: the family tokens are
    projected from the *raw* candidate features inside the tokenizer (see
    :class:`OneRankTokenizer`), so no member of the group carries a
    positional encoding.

    Output is ``(total_targets, K, D)`` -- the per-candidate per-task
    representation ``r^i_k`` of the paper -- instead of the baseline's
    ``(total_targets, D)``.

    Args:
        num_task_tokens (int): number of task tokens ``K``; must equal the
            number of task towers.
        max_seq_len (int): upper bound on the inflated sequence length
            per request (history tokens + expanded candidate groups);
            requests that exceed it fail with a clear error instead of
            silently overflowing the static max sequence length the
            jagged kernels were autotuned for.
        group_mask (str): block layout only; the in-group visibility
            strategy forwarded to every :class:`OneRankSTULayer` (see
            :func:`build_onerank_func_tensor`).
        family_dims (Optional[List[int]]): dimension of each family's
            slice of the raw candidate features; given -> the block
            layout (``family_tokens``, Layer 2 probe), ``None`` -> the
            paper layout.  Must sum to ``target_embedding_dim``.
        family_feature_ids (Optional[List[int]]): family id of each
            *dimension* of the raw candidate features (the model layer
            expands the proto's per-feature ids); required iff
            ``family_dims`` is given.
        family_names (Optional[List[str]]): human-readable family names,
            kept for reading the learned projections off a checkpoint.
        group_mixer (Optional[Tuple[int, int]]): ``(ffn_multiplier,
            num_layers)`` of the RankMixer in-group mixer (see
            :class:`OneRankGroupMixer`); given -> in-group mixing moves out
            of attention into the mixer block (group rows read prefix +
            self only) and the block layout is required.  ``None`` ->
            in-group interaction stays inside attention (the
            ``family_tokens`` behavior).
        **kwargs: forwarded verbatim to :class:`HSTUTransducer`.
    """

    def __init__(
        self,
        num_task_tokens: int,
        max_seq_len: int,
        group_mask: str = FAM_MASK_GROUP_NULL,
        family_dims: Optional[List[int]] = None,
        family_feature_ids: Optional[List[int]] = None,
        family_names: Optional[List[str]] = None,
        group_mixer: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> None:
        if num_task_tokens <= 0:
            raise ValueError(f"num_task_tokens must be positive; got {num_task_tokens}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive; got {max_seq_len}")
        if (family_dims is None) != (family_feature_ids is None):
            raise ValueError(
                "family_dims and family_feature_ids must be given together"
            )
        if family_dims is not None:
            if group_mask not in _FAM_MASKS:
                raise ValueError(
                    f"group_mask must be one of {_FAM_MASKS}; got {group_mask!r}"
                )
            raw_dim = int(kwargs.get("target_embedding_dim", 0))
            if raw_dim <= 0 or sum(family_dims) != raw_dim:
                raise ValueError(
                    f"family_dims must sum to the raw candidate feature "
                    f"dim (target_embedding_dim): sum(family_dims)="
                    f"{sum(family_dims)}, target_embedding_dim={raw_dim}"
                )
        if group_mixer is not None:
            if family_dims is None:
                raise ValueError(
                    "group_mixer requires family_tokens (the block layout): "
                    "it replaces the in-group attention of the "
                    "family-token groups"
                )
            if (
                len(group_mixer) != 2
                or group_mixer[0] <= 0
                or group_mixer[1] <= 0
            ):
                raise ValueError(
                    f"group_mixer must be (ffn_multiplier, num_layers), "
                    f"both positive ints; got {group_mixer!r}"
                )
        self._max_seq_len: int = max_seq_len
        stu = dict(kwargs.pop("stu"))
        # Routed through the stu dict because `_build_stu_layer` runs inside
        # `super().__init__()`, before subclass attributes exist.
        if family_dims is not None:
            stu["group_size"] = num_task_tokens + len(family_dims)
            stu["layout"] = LAYOUT_BLOCK
            stu["group_mask"] = group_mask
            stu["num_tasks"] = num_task_tokens
        else:
            stu["group_size"] = num_task_tokens + 1
            stu["layout"] = LAYOUT_PAPER
        # group_mixer (RankMixer probe): in-group mixing leaves attention,
        # so the group rows' visible set shrinks to prefix + self (see
        # build_onerank_func_tensor).
        stu["group_attn"] = group_mixer is None
        # The paper layout spends all three func-tensor column intervals
        # on `prefix + own group`, so contextual tokens must be ordinary
        # causal prefix. Resolve the `-1` sentinel ("inherit from the
        # preprocessor", which the base class would resolve to the number
        # of contextual features, >= 1) to 0 here, so a `dlrm_hstu` config
        # block can be reused verbatim; an explicit non-zero value is left
        # for the `OneRankSTULayer` check to reject with a clear message.
        if stu.get("contextual_seq_len", -1) < 0:
            stu["contextual_seq_len"] = 0
        # Rejected at construction rather than at the first training step:
        # a reused `dlrm_hstu` block may carry truncation tuning, and the
        # stack would otherwise only surface `OneRankSTULayer`'s
        # NotImplementedError after torchrun, sharding and the data
        # pipeline are fully up.
        if (
            kwargs.get("attn_truncation_split_layer", 0) != 0
            or kwargs.get("attn_truncation_tail_len", 0) != 0
        ):
            raise ValueError(
                "OneRank does not support mid-stack attention truncation: "
                "the func tensor is cached across layers keyed on a static "
                "signature, and truncation changes total_q and the prefix "
                "boundary. Leave attn_truncation_split_layer / "
                "attn_truncation_tail_len at 0."
            )
        super().__init__(stu=stu, **kwargs)
        if self._return_full_embeddings:
            raise ValueError(
                "OneRank does not return full sequence embeddings; the "
                "expanded sequence is an internal layout."
            )
        # Interleaving reorders the candidate segment before this module
        # sees it, which would break the fixed group stride. Checked via
        # the public construction-time accessor: `interleave_targets()` is
        # train/eval dependent and would read False at construction.
        if self._input_preprocessor.has_interleaving():
            raise ValueError(
                "OneRank requires an input preprocessor without target "
                "interleaving (use `contextual_preprocessor`, not "
                "`contextual_interleave_preprocessor`)."
            )
        self._tokenizer: OneRankTokenizer = OneRankTokenizer(
            num_tasks=num_task_tokens,
            embedding_dim=stu["embedding_dim"],
            is_inference=self._is_inference,
            family_dims=family_dims,
            family_feature_ids=family_feature_ids,
            family_names=family_names,
        )
        self._group_mixer: Optional[OneRankGroupMixer] = (
            None
            if group_mixer is None
            else OneRankGroupMixer(
                group_size=stu["group_size"],
                embedding_dim=stu["embedding_dim"],
                ffn_multiplier=group_mixer[0],
                num_layers=group_mixer[1],
                is_inference=self._is_inference,
            )
        )

    def _build_stu_layer(self, stu: Dict[str, Any]) -> STU:
        return OneRankSTULayer(**stu)

    def _preprocess(
        self, grouped_features: Dict[str, torch.Tensor]
    ) -> Tuple[
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            max_seq_len,
            total_uih_len,
            total_targets,
            seq_lengths,
            seq_offsets,
            seq_timestamps,
            seq_embeddings,
            num_targets,
        ) = super()._preprocess(grouped_features)

        group_size = self._tokenizer.group_size
        with record_function("onerank_tokenizer"):
            prefix_lengths = seq_lengths - num_targets
            prefix_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                prefix_lengths
            )
            target_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(num_targets)
            prefix_embeddings, target_embeddings = split_2D_jagged(
                values=seq_embeddings,
                max_seq_len=max_seq_len,
                total_len_left=total_uih_len,
                total_len_right=total_targets,
                offsets_left=prefix_offsets,
                offsets_right=target_offsets,
                kernel=self.kernel(),
            )
            prefix_timestamps, target_timestamps = split_2D_jagged(
                values=seq_timestamps.unsqueeze(-1),
                max_seq_len=max_seq_len,
                total_len_left=total_uih_len,
                total_len_right=total_targets,
                offsets_left=prefix_offsets,
                offsets_right=target_offsets,
                kernel=self.kernel(),
            )

            if self._tokenizer.layout == LAYOUT_BLOCK:
                # Family boundaries live in the *raw* candidate feature
                # layout (the FI families interleave in embedding order),
                # so the block tokenizer consumes the pre-preprocessor
                # candidate features from grouped_features and the
                # preprocessor's projected target segment above is
                # dropped: its parameters keep serving the UIH prefix.
                # Same jagged candidate order as target_embeddings (the
                # input preprocessor reads this very key), so the
                # timestamps split above stay aligned.
                tokens = self._tokenizer(
                    grouped_features["candidate.sequence"]
                )
                if self._group_mixer is not None:
                    # The group_mixer probe replaces the in-group
                    # attention: mix each [t_1..t_K, x^1..x^F] block
                    # here (parameter-free transpose + per-slot FFN), so
                    # the STU layers only have to route group rows to
                    # the prefix (their func-tensor second interval is
                    # the self column).  The input is exactly the flat
                    # (total_targets * group_size, D) segment the
                    # tokenizer just produced.
                    tokens = self._group_mixer(tokens)
                # The .to() anchors the dtype to the embedding pipeline:
                # under AMP the dense candidate features arrive in bf16
                # (so the projected family tokens are bf16 too) while the
                # UIH prefix stays fp32, and concat_2D_jagged requires
                # both sides to match (triton: "Mismatched type for
                # in_ptrs"). The paper path needs no cast -- its input IS
                # target_embeddings. No-op outside AMP.
                expanded_embeddings = tokens.to(target_embeddings.dtype)
            else:
                expanded_embeddings = self._tokenizer(target_embeddings)
            # Every token of a group inherits the candidate's timestamp, so
            # the output postprocessor's time features are unchanged.
            expanded_timestamps = target_timestamps.repeat_interleave(group_size, dim=0)
            new_num_targets = num_targets * group_size
            new_target_offsets = target_offsets * group_size

            # Two D->H syncs for the padding bounds of the jagged concat.
            # `_preprocess` is already sync-bound (the input preprocessor
            # does four), and passing a loose bound would pad the dense
            # intermediate to the full expanded length for every sample.
            max_prefix_len = fx_int_item(prefix_lengths.max())
            max_new_targets = fx_int_item(new_num_targets.max())
            _fx_check_max_seq_len(max_prefix_len, max_new_targets, self._max_seq_len)

            new_seq_embeddings = concat_2D_jagged(
                values_left=prefix_embeddings,
                values_right=expanded_embeddings,
                max_len_left=max_prefix_len,
                max_len_right=max_new_targets,
                offsets_left=prefix_offsets,
                offsets_right=new_target_offsets,
                kernel=self.kernel(),
            )
            new_seq_timestamps = concat_2D_jagged(
                values_left=prefix_timestamps,
                values_right=expanded_timestamps,
                max_len_left=max_prefix_len,
                max_len_right=max_new_targets,
                offsets_left=prefix_offsets,
                offsets_right=new_target_offsets,
                kernel=self.kernel(),
            ).squeeze(-1)
            new_seq_lengths = prefix_lengths + new_num_targets
            new_seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                new_seq_lengths
            )

        return (
            max_prefix_len + max_new_targets,
            total_uih_len,
            total_targets * group_size,
            new_seq_lengths,
            new_seq_offsets,
            new_seq_timestamps,
            new_seq_embeddings,
            new_num_targets,
        )

    def _compose_output(
        self,
        encoded_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        max_seq_len: int,
        total_uih_len: int,
        total_targets: int,
        num_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Split off the expanded candidate segment and keep the task slots."""
        group_embeddings, full = super()._compose_output(
            encoded_embeddings=encoded_embeddings,
            seq_timestamps=seq_timestamps,
            seq_lengths=seq_lengths,
            seq_offsets=seq_offsets,
            max_seq_len=max_seq_len,
            total_uih_len=total_uih_len,
            total_targets=total_targets,
            num_targets=num_targets,
        )
        # (total_targets * group_size, D) -> (total_targets, K, D): which
        # slots hold the task tokens' outputs r^i_k depends on the layout.
        dim = group_embeddings.size(-1)
        if self._tokenizer.layout == LAYOUT_BLOCK:
            # Block: the first K slots of each [t_1..t_K, x^1..x^F] group;
            # the family-token outputs are the trunk's internal
            # cross-feature carriers, not read out.
            return (
                group_embeddings.view(-1, self._tokenizer.group_size, dim)[
                    :, : self._tokenizer.num_tasks, :
                ],
                full,
            )
        # Paper: slots 1..K of each [e^C_i, t_1..t_K] group.
        return (
            group_embeddings.view(-1, self._tokenizer.group_size, dim)[:, 1:, :],
            full,
        )
