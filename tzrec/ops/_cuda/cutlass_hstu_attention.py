# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch

from tzrec.ops.hstu_attention_utils import HSTU_ARBITRARY_NFUNC
from tzrec.utils.logging_util import logger

# Eager hstu import: registers fbgemm::hstu_varlen_fwd_{80,90} schema
# (needed by aoti_load_package at predict time) and the register_fake
# metas (needed by torch.export's FX trace). Optional on CPU images.
try:
    import hstu.hstu_ops_gpu  # noqa: F401
    from hstu import hstu_attn_varlen_func
except Exception as e:
    # Not just ImportError: hstu/library.py skips loading its .so when
    # torch.cuda.is_available() is false, and hstu_ops_gpu.py then registers
    # fakes for ops that were never registered, raising RuntimeError.
    logger.debug("fbgemm_gpu_hstu not available: %s", e)
    hstu_attn_varlen_func = None  # type: ignore[assignment]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


@torch.fx.wrap
def cutlass_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    attn_func: Optional[torch.Tensor] = None,
    scaling_seqlen: int = -1,
    func_noncausal: bool = False,
) -> torch.Tensor:
    """CUTLASS-based HSTU multi-head attention.

    Supports two mask modes:

    - Standard fixed-mask path (``attn_func=None``): causal/local/
      context/target masking driven by ``causal``, ``max_attn_len``,
      ``contextual_seq_len`` and ``num_targets``.
    - Arbitrary-mask NFUNC path (``attn_func`` provided): the kernel
      interprets ``attn_func`` as a jagged
      ``(nheads, HSTU_ARBITRARY_NFUNC, total_q)`` int32 tensor encoding
      ``(NFUNC + 1) // 2`` disjoint column-intervals per query row. The
      caller is responsible for constructing it (see
      ``build_sla_func_tensor`` for the SLA case). In this mode the
      wrapper pins the kernel window to ``(-1, 0)`` (causal) so
      ``causal`` and ``max_attn_len`` are redundant and must be left at
      their defaults (``causal=True, max_attn_len=0``). Pass
      ``func_noncausal=True`` to lift the window to ``(-1, -1)`` so the
      func tensor may also encode future columns (bidirectional masks,
      e.g. OneRank's block layout); the kernel then honors the full
      interval union instead of clipping it to ``col <= row``. The
      non-causal window requires ``num_targets=None`` and
      ``contextual_seq_len=0`` -- the kernel rejects context/target
      masking combined with a non-causal window, and an NFUNC caller
      encodes those regions in the func tensor itself anyway.

    The CUTLASS kernel uses int32 cu_seqlens internally, so the cumulative
    sum ``seq_offsets[-1]`` (total token count in the batch) must fit in
    int32 (< 2**31 ≈ 2.1B tokens).

    Args:
        max_seq_len: maximum sequence length in the batch.
        alpha: scaling factor for attention scores.
        q: query tensor of shape (total, nheads, attn_dim).
        k: key tensor of shape (total, nheads, attn_dim).
        v: value tensor of shape (total, nheads, hidden_dim).
        seq_offsets: cumulative sequence offsets of shape (batch_size + 1,).
        causal: whether to apply causal masking (fixed-mask path only).
        num_targets: number of target tokens per batch element.
        max_attn_len: maximum attention window length (fixed-mask path;
            0 means unlimited).
        contextual_seq_len: number of contextual tokens per sequence.
        attn_func: pre-built arbitrary-mask func tensor of shape
            ``(nheads, HSTU_ARBITRARY_NFUNC, total_q)``, int32. When
            provided, selects the NFUNC mask path; ``causal`` and
            ``max_attn_len`` must be at defaults.
        scaling_seqlen: divisor used to scale the attention output inside
            the kernel. ``-1`` (default) falls back to ``max_seq_len`` so
            the behavior matches the legacy code path.
        func_noncausal: NFUNC path only -- lift the kernel window from
            causal ``(-1, 0)`` to fully open ``(-1, -1)`` so the func
            tensor may attend to columns past the query row. Requires
            ``num_targets=None`` and ``contextual_seq_len=0``.

    Returns:
        output tensor of shape (total, nheads, hidden_dim).
    """
    if hstu_attn_varlen_func is None:
        raise RuntimeError(
            "fbgemm_gpu_hstu wheel is not installed; cannot run CUTLASS "
            "HSTU attention. Install via -f https://tzrec.oss-accelerate."
            "aliyuncs.com/third_party/hstu/${DEVICE}/repo.html (cu126/cu129/cu130)."
        )
    if q.shape[2] != v.shape[2]:
        raise ValueError(
            f"CUTLASS hstu_attn requires attention_dim == hidden_dim, "
            f"got q.shape[2]={q.shape[2]} != v.shape[2]={v.shape[2]}"
        )
    if q.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"CUTLASS hstu_attn supports fp16 and bf16, got {q.dtype}. "
            f"Set train_config.mixed_precision to 'BF16' or 'FP16'."
        )
    if contextual_seq_len < 0:
        raise ValueError(
            f"contextual_seq_len must be non-negative; got {contextual_seq_len}"
        )
    if attn_func is not None:
        if not causal:
            raise ValueError(
                "attn_func requires causal=True; the NFUNC mask path "
                "forces window_size_right=0 so causal=False has no effect."
            )
        if max_attn_len > 0:
            raise ValueError(
                f"attn_func is mutually exclusive with max_attn_len "
                f"(got max_attn_len={max_attn_len}); any local window "
                "must be encoded by the func tensor itself."
            )
        torch._assert(
            attn_func.dtype == torch.int32,
            "attn_func must be int32",
        )
        torch._assert(attn_func.dim() == 3, "attn_func must be 3-D")
        torch._assert(
            attn_func.shape[0] == q.shape[1]
            and attn_func.shape[1] == HSTU_ARBITRARY_NFUNC
            and attn_func.shape[2] == q.shape[0],
            f"attn_func must have shape (nheads, {HSTU_ARBITRARY_NFUNC}, total_q); "
            "it must match the HSTU_ARBITRARY_NFUNC the installed "
            "fbgemm_gpu_hstu wheel was built with (see its fn<N> version tag)",
        )
        torch._assert(
            attn_func.device == q.device,
            "attn_func must be on the same device as q",
        )
        attn_func = attn_func.contiguous()

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cu_seqlens = seq_offsets.to(torch.int32)

    num_targets_int32: Optional[torch.Tensor] = None
    if num_targets is not None:
        num_targets_int32 = num_targets.to(torch.int32)

    num_contexts_tensor: Optional[torch.Tensor] = None
    if contextual_seq_len > 0:
        batch_size = seq_offsets.size(0) - 1
        num_contexts_tensor = torch.full(
            (batch_size,),
            contextual_seq_len,
            dtype=torch.int32,
            device=q.device,
        )

    if attn_func is not None:
        if func_noncausal:
            if num_targets is not None or contextual_seq_len > 0:
                raise ValueError(
                    "func_noncausal requires num_targets=None and "
                    "contextual_seq_len=0; the CUTLASS kernel rejects "
                    "context/target masking combined with a non-causal "
                    "window (encode those regions in the func tensor "
                    "instead)."
                )
            window_size_left, window_size_right = -1, -1
        else:
            window_size_left, window_size_right = -1, 0
    elif causal:
        if max_attn_len > 0:
            window_size_left, window_size_right = max_attn_len, 0
        else:
            window_size_left, window_size_right = -1, 0
    else:
        window_size_left, window_size_right = -1, -1

    if scaling_seqlen == -1:
        scaling_seqlen = max_seq_len

    out = hstu_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=max_seq_len,
        max_seqlen_k=max_seq_len,
        scaling_seqlen=scaling_seqlen,
        num_contexts=num_contexts_tensor,
        num_targets=num_targets_int32,
        target_group_size=1,
        window_size=(window_size_left, window_size_right),
        alpha=alpha,
        rab=None,
        has_drab=False,
        func=attn_func,
    )
    # ABI-patched fn5 kernel wheel: the output TensorImpl carries a stale
    # numel_ cache (the T5-built kernel writes the cache at a field offset
    # that maps onto different storage under torch 2.11), so downstream
    # view()/flatten()/numel() see a garbage count (2**48 | true_numel)
    # while sizes/strides read fine. Rebuilding the view metadata via
    # as_strided is zero-copy and autograd-transparent (verified: fwd loss
    # and all input grads bit-identical vs the clone path).
    if out.dim() == 3:
        out = torch.as_strided(out, out.shape, out.stride(), out.storage_offset())
    return out


# ---- ABI-patched fn5 kernel wheel: stale-numel metadata guard ----------------
# The T5-built kernel writes some TensorImpl field at an offset that maps
# onto 2.11's numel_ for the tensors it *constructs* (fwd out, bwd
# dq/dk/dv/dRab), so numel() reads 2**48 | true_numel while sizes/strides
# stay correct; any downstream view()/flatten()/reshape() then dies with
# "shape is invalid for input of size 281474976710656...".  The fwd output
# is purified at the return above; the backward outputs are produced inside
# HstuAttnVarlenFunc.backward, so patch that staticmethod once at import
# time (module load precedes any graph construction) to rebuild the view
# metadata of every tensor gradient it returns.  as_strided is zero-copy
# and autograd-transparent (verified: fwd loss and all input grads
# bit-identical vs the clone path).
def _purify_meta(t: "torch.Tensor") -> "torch.Tensor":
    if isinstance(t, torch.Tensor) and t.dim() >= 1:
        return torch.as_strided(t, t.shape, t.stride(), t.storage_offset())
    return t


try:
    from hstu.cuda_hstu_attention import HstuAttnVarlenFunc as _HstuFn

    if not getattr(_HstuFn, "_tzrec_numel_purified", False):
        # plain function either way: bare `def backward` or @staticmethod
        # both surface as a function on class attribute access.
        _orig_backward = getattr(
            _HstuFn.backward, "__func__", None
        ) or _HstuFn.backward

        def _purified_backward(ctx, dout, *args):  # noqa: ANN001, ANN202
            grads = _orig_backward(ctx, dout, *args)
            if isinstance(grads, tuple):
                return tuple(_purify_meta(g) for g in grads)
            return _purify_meta(grads)

        _HstuFn.backward = staticmethod(_purified_backward)
        _HstuFn._tzrec_numel_purified = True
except Exception as e:  # pragma: no cover - CPU images
    logger.debug("HstuAttnVarlenFunc backward purify not applied: %s", e)
