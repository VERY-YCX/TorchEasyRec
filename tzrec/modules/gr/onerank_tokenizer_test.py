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

"""Unit tests for ``tzrec.modules.gr.onerank_tokenizer``.

The core assertion is :meth:`OneRankFuncTensorTest.test_matches_reference_mask`:
the NFUNC=5 func tensor, once decoded to a dense mask, must equal the mask
written out directly from the paper's group layout.  Everything downstream --
task-token isolation, cross-candidate isolation, the causal prefix -- is a
property of that mask, so it is the one place the whole tokenization scheme
can be checked for correctness without a GPU.
"""

import unittest
from typing import Dict, List, Optional, Tuple

import torch
from parameterized import parameterized
from torch import nn

from tzrec.modules.gr.onerank_tokenizer import (
    OneRankHSTUTransducer,
    OneRankSTULayer,
    OneRankTokenizer,
)
from tzrec.ops import Kernel
from tzrec.ops._pytorch.pt_hstu_attention import _decode_attn_func_to_mask
from tzrec.ops.hstu_attention_utils import (
    FAM_MASK_GROUP_NULL,
    FAM_MASK_TASK_ISO,
    LAYOUT_BLOCK,
    build_onerank_func_tensor,
)
from tzrec.utils.test_util import (
    TestGraphType,
    create_test_module,
    mark_ci_scope,
)

_GRAPH_TYPES = [TestGraphType.NORMAL, TestGraphType.FX_TRACE]

# (name, group_size = K + 1, [(prefix_len, num_candidates), ...]) per batch
# sample.
_LAYOUT_CASES = [
    ("single_task", 2, [(4, 1)]),
    ("no_prefix", 3, [(0, 2)]),
    ("seven_tasks", 8, [(5, 3)]),
    ("no_candidates", 4, [(6, 0)]),
    ("mixed_batch", 4, [(3, 2), (7, 1), (1, 4)]),
    ("ragged_with_empty", 3, [(0, 0), (5, 2), (2, 1)]),
]


def _xprod(cases: List[Tuple]) -> List[Tuple]:
    return [
        (f"{case[0]}_{gt.name.lower()}", gt, *case[1:])
        for gt in _GRAPH_TYPES
        for case in cases
    ]


class _BuildOneRankFuncTensorWrapper(nn.Module):
    def __init__(self, group_size: int, nheads: int = 1) -> None:
        super().__init__()
        self._group_size = group_size
        self._nheads = nheads

    def forward(
        self,
        x: torch.Tensor,
        seq_offsets: torch.Tensor,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        return build_onerank_func_tensor(
            nheads=self._nheads,
            seq_offsets=seq_offsets,
            total_q=x.size(0),
            num_targets=num_targets,
            group_size=self._group_size,
        )


def _reference_mask(
    layout: List[Tuple[int, int]], group_size: int, padded_len: int
) -> torch.Tensor:
    """Write out the paper's group layout as a dense mask, row by row.

    ``X = [prefix (H) | G_1 | ... | G_N]`` with
    ``G_i = [e^C_i, t_1, ..., t_K]``:

    * prefix rows are plain causal;
    * the candidate token sees the whole prefix plus itself;
    * a task token sees the whole prefix plus the candidate and itself.

    Returns:
        torch.Tensor: ``(B, padded_len, padded_len)`` bool.
    """
    mask = torch.zeros(len(layout), padded_len, padded_len, dtype=torch.bool)
    for b, (prefix_len, num_candidates) in enumerate(layout):
        for q in range(prefix_len):
            mask[b, q, : q + 1] = True
        for i in range(num_candidates):
            base = prefix_len + i * group_size
            # slot 0: the candidate token e^C_i.
            mask[b, base, :prefix_len] = True
            mask[b, base, base] = True
            # slots 1..K: task tokens, paired with the candidate at slot 0.
            for k in range(1, group_size):
                q = base + k
                mask[b, q, :prefix_len] = True
                mask[b, q, base] = True
                mask[b, q, q] = True
    return mask


def _build_func_tensor(
    layout: List[Tuple[int, int]],
    group_size: int,
    nheads: int = 1,
    graph_type: TestGraphType = TestGraphType.NORMAL,
    num_targets_override: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Run ``build_onerank_func_tensor`` on an expanded jagged batch."""
    seq_lengths = [p + n * group_size for p, n in layout]
    num_targets = (
        num_targets_override
        if num_targets_override is not None
        else [n * group_size for _, n in layout]
    )
    seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
        torch.tensor(seq_lengths, dtype=torch.int32)
    )
    # Dummy x so that total_q = x.size(0) is a Proxy under fx-trace, which
    # is how STULayer.forward calls it.
    x = torch.zeros(int(sum(seq_lengths)), 1)
    module = create_test_module(
        _BuildOneRankFuncTensorWrapper(group_size=group_size, nheads=nheads),
        graph_type,
    )
    func = module(x, seq_offsets, torch.tensor(num_targets, dtype=torch.int32))
    return func, seq_offsets, max(seq_lengths + [1])


@mark_ci_scope("h20", "gpu")
class OneRankFuncTensorTest(unittest.TestCase):
    """Verify the NFUNC=5 encoding of the OneRank group layout."""

    @parameterized.expand(_xprod(_LAYOUT_CASES))
    def test_matches_reference_mask(
        self,
        _name: str,
        graph_type: TestGraphType,
        group_size: int,
        layout: List[Tuple[int, int]],
    ) -> None:
        """Decoded func tensor == mask built directly from the layout.

        ``_decode_attn_func_to_mask`` is the PyTorch kernel's own reader of
        the encoding, so comparing against it checks exactly what attention
        will see -- not our re-reading of the convention.
        """
        func, seq_offsets, padded_len = _build_func_tensor(
            layout, group_size, graph_type=graph_type
        )
        self.assertEqual(tuple(func.shape), (1, 5, int(seq_offsets[-1])))
        got = _decode_attn_func_to_mask(func, seq_offsets, padded_len)
        want = _reference_mask(layout, group_size, padded_len)
        torch.testing.assert_close(got[:, 0], want)

    @parameterized.expand([(gt.name.lower(), gt) for gt in _GRAPH_TYPES])
    def test_all_heads_share_one_row(
        self, _name: str, graph_type: TestGraphType
    ) -> None:
        """The mask is head-independent, so it is broadcast, not copied."""
        func, _, _ = _build_func_tensor(
            [(4, 2)], group_size=6, nheads=4, graph_type=graph_type
        )
        self.assertEqual(func.size(0), 4)
        for head in range(1, 4):
            torch.testing.assert_close(func[head], func[0])

    def test_task_tokens_within_a_group_are_isolated(self) -> None:
        """``t_k`` never sees ``t_j`` for ``j != k``, nor another candidate.

        This is the property the three-interval encoding exists to buy: a
        task token must not read another task's state, or the cross-task
        head in the tower would be modelling an attention path that
        already leaked.
        """
        prefix_len, num_candidates, group_size = 3, 3, 5
        func, seq_offsets, padded_len = _build_func_tensor(
            [(prefix_len, num_candidates)], group_size
        )
        mask = _decode_attn_func_to_mask(func, seq_offsets, padded_len)[0, 0]

        for i in range(num_candidates):
            base = prefix_len + i * group_size
            for k in range(1, group_size):
                q = base + k  # a task token
                visible = mask[q].nonzero().flatten().tolist()
                self.assertEqual(visible, list(range(prefix_len)) + [base, q])

        # Cross-group: no row of group 0 sees any column of group 1.
        g0 = slice(prefix_len, prefix_len + group_size)
        g1 = slice(prefix_len + group_size, prefix_len + 2 * group_size)
        self.assertFalse(mask[g0, g1].any())
        self.assertFalse(mask[g1, g0].any())
        # And no prefix row sees any group column.
        self.assertFalse(mask[:prefix_len, prefix_len:].any())

    def test_prefix_is_strictly_causal(self) -> None:
        """Prefix rows keep the baseline's causal mask, contextual included.

        ``OneRankSTULayer`` forces ``contextual_seq_len = 0``, so there is
        no bidirectional block to carve out here.
        """
        func, seq_offsets, padded_len = _build_func_tensor([(6, 1)], group_size=2)
        mask = _decode_attn_func_to_mask(func, seq_offsets, padded_len)[0, 0]
        for q in range(6):
            self.assertEqual(mask[q].nonzero().flatten().tolist(), list(range(q + 1)))

    def test_num_targets_over_length_is_clamped(self) -> None:
        """A prefix boundary can never go negative.

        Silent-NaN guard: a negative boundary would make ``col_max0``
        negative, every interval empty, and the attention denominator zero.
        """
        func, seq_offsets, padded_len = _build_func_tensor(
            [(2, 1)], group_size=4, num_targets_override=[12]
        )
        self.assertGreaterEqual(int(func.min()), 0)
        mask = _decode_attn_func_to_mask(func, seq_offsets, padded_len)[0, 0]
        # Boundary clamped to 0 -> every row is a group row, so each row
        # still attends to at least itself.
        self.assertTrue(mask.any(dim=-1).all())

    def test_group_size_below_two_raises(self) -> None:
        """``group_size`` is ``num_tasks + 1``.

        Below 2 there is no room for even the candidate token plus one
        task token.
        """
        seq_offsets = torch.tensor([0, 4], dtype=torch.int32)
        for bad in (0, 1, -2):
            with self.assertRaisesRegex(ValueError, "group_size"):
                build_onerank_func_tensor(
                    nheads=1,
                    seq_offsets=seq_offsets,
                    total_q=4,
                    num_targets=torch.tensor([2], dtype=torch.int32),
                    group_size=bad,
                )


class OneRankTokenizerTest(unittest.TestCase):
    """Verify the ``[e^C_i, t_1, ..., t_K]`` group expansion."""

    def test_expansion_layout(self) -> None:
        """Slot 0 is the candidate, slot ``k`` is task token ``t_k``.

        The func tensor's slot rule is hard-coded, so a layout that put
        the task tokens first would mask correctly and mean the wrong
        thing.
        """
        num_tasks, dim, total_targets = 3, 8, 4
        tokenizer = OneRankTokenizer(num_tasks=num_tasks, embedding_dim=dim)
        candidates = torch.randn(total_targets, dim)

        out = tokenizer(candidates)
        self.assertEqual(tuple(out.shape), (total_targets * (num_tasks + 1), dim))
        self.assertEqual(tokenizer.group_size, num_tasks + 1)
        for i in range(total_targets):
            base = i * (num_tasks + 1)
            torch.testing.assert_close(out[base], candidates[i])
            for k in range(num_tasks):
                torch.testing.assert_close(out[base + 1 + k], tokenizer._task_tokens[k])

    def test_gradients_follow_the_single_candidate_token(self) -> None:
        """The candidate row appears once; task tokens repeat per group.

        The candidate gradient must be exactly its single row's weight,
        and each task token's gradient must sum over all candidate groups
        -- the shared parameters see every group, not just one.
        """
        num_tasks, dim = 4, 5
        tokenizer = OneRankTokenizer(num_tasks=num_tasks, embedding_dim=dim)
        candidates = torch.randn(2, dim, requires_grad=True)

        out = tokenizer(candidates)
        # Weight each expanded row distinctly so a mis-aggregated gradient
        # cannot cancel.
        weights = torch.arange(1, out.size(0) + 1, dtype=out.dtype).unsqueeze(-1)
        (out * weights).sum().backward()

        group = num_tasks + 1
        # Candidate i lives at row i * group; its gradient is that row's
        # weight, broadcast over dim.
        want_candidates = torch.stack(
            [weights[i * group, 0].expand(dim) for i in range(2)]
        )
        torch.testing.assert_close(candidates.grad, want_candidates)
        # Task token k lives at slot k + 1 of every group; its gradient
        # sums those rows' weights over both candidates.
        want_tokens = torch.zeros_like(tokenizer._task_tokens)
        for i in range(2):
            for k in range(num_tasks):
                want_tokens[k] += weights[i * group + 1 + k, 0]
        torch.testing.assert_close(tokenizer._task_tokens.grad, want_tokens)

    def test_task_tokens_are_shared_across_candidates(self) -> None:
        """``K`` parameters total, not ``K`` per candidate."""
        tokenizer = OneRankTokenizer(num_tasks=7, embedding_dim=16)
        self.assertEqual(
            [(n, tuple(p.shape)) for n, p in tokenizer.named_parameters()],
            [("_task_tokens", (7, 16))],
        )

    def test_non_positive_num_tasks_raises(self) -> None:
        """There is no group layout without at least one task token."""
        for bad in (0, -1):
            with self.assertRaisesRegex(ValueError, "num_tasks must be positive"):
                OneRankTokenizer(num_tasks=bad, embedding_dim=8)


class OneRankSTULayerTest(unittest.TestCase):
    """Verify the construction-time contract of the OneRank STU layer."""

    _STU_KWARGS = {
        "embedding_dim": 16,
        "num_heads": 2,
        "hidden_dim": 8,
        "attention_dim": 8,
    }

    def _build(self, **overrides: object) -> OneRankSTULayer:
        kwargs: Dict[str, object] = dict(self._STU_KWARGS)
        kwargs.update(overrides)
        return OneRankSTULayer(group_size=kwargs.pop("group_size", 4), **kwargs)

    def test_always_uses_its_own_func_tensor(self) -> None:
        """``uses_arbitrary_mask`` gates the NFUNC path in ``STUStack``."""
        layer = self._build()
        self.assertTrue(layer.uses_arbitrary_mask)

    def test_cache_signature_cannot_collide_with_sla(self) -> None:
        """The cross-layer func-tensor cache is keyed on this string.

        A collision with plain SLA in a mixed stack would silently reuse
        the wrong mask.
        """
        layer = self._build(group_size=8, num_heads=4)
        self.assertEqual(
            layer.attn_func_static_sig,
            "onerank:LAYOUT_PAPER:8:4:FAMILY_MASK_GROUP_NULL:attn",
        )
        self.assertNotEqual(
            layer.attn_func_static_sig,
            self._build(group_size=8, num_heads=2).attn_func_static_sig,
        )

    @parameterized.expand(
        [
            ("sla_k1", {"sla_k1": 2}, "sla_k1"),
            ("sla_k2", {"sla_k2": 2}, "sla_k2"),
            ("contextual", {"contextual_seq_len": 4}, "contextual_seq_len"),
            ("max_attn_len", {"max_attn_len": 8}, "max_attn_len"),
            ("non_causal", {"causal": False}, "causal"),
            ("small_group", {"group_size": 1}, "group_size"),
        ]
    )
    def test_unsupported_options_are_rejected(
        self, _name: str, overrides: dict, message: str
    ) -> None:
        """Most of these would need a fourth column interval in NFUNC=5.

        The paper layout spends all three intervals on ``prefix + own
        group``, so SLA / a contextual block / a local window cannot also
        apply; a group below two tokens is not a group at all.  Rejecting
        at construction time rather than producing a mask that quietly
        drops one of the requirements.
        """
        with self.assertRaisesRegex(ValueError, message):
            self._build(**overrides)

    def test_truncation_and_kv_cache_are_unsupported(self) -> None:
        """Both paths would drop the mask; fail loudly instead."""
        layer = self._build()
        x = torch.zeros(6, 16)
        offsets = torch.tensor([0, 6], dtype=torch.int32)
        num_targets = torch.tensor([4], dtype=torch.int32)
        with self.assertRaises(NotImplementedError):
            layer.truncate_input(
                x, offsets, max_seq_len=6, num_targets=num_targets, truncate_tail_len=2
            )
        with self.assertRaises(NotImplementedError):
            layer.cached_forward(x, num_targets)


@mark_ci_scope("h20", "gpu")
class OneRankComposeOutputTest(unittest.TestCase):
    """The transducer output must be the *task* slots of the groups.

    ``_compose_output`` slices ``view(-1, K + 1, D)[:, 1:, :]`` off the
    expanded sequence.  A wiring slip taking the candidate slot
    ``[:, 0, :]`` still yields a correctly shaped output that differs
    across K (every candidate token sits at a different position under a
    different mask), so shape/inequality assertions cannot catch it.
    Here every row of the (stubbed) STU output is stamped with its own
    position in the expanded sequence, and the returned tensor must
    carry exactly the task-slot stamps.
    """

    def test_returns_the_task_slot_rows(self) -> None:
        K, D = 3, 4
        group_size = K + 1
        prefix = [1, 1]
        candidates = [2, 4]
        num_targets = [c * group_size for c in candidates]
        lengths = [p + n for p, n in zip(prefix, num_targets)]
        total_q = sum(lengths)

        # ``object.__new__`` skips the full pipeline construction; only
        # the attributes the base ``_compose_output`` reads are stubbed,
        # and plain ``object.__setattr__`` bypasses nn.Module's
        # (uninitialized) ``__setattr__`` bookkeeping.
        transducer = object.__new__(OneRankHSTUTransducer)
        object.__setattr__(
            transducer, "_tokenizer", OneRankTokenizer(num_tasks=K, embedding_dim=D)
        )
        object.__setattr__(transducer, "_return_full_embeddings", False)
        object.__setattr__(transducer, "_is_inference", False)
        object.__setattr__(transducer, "_kernel", Kernel.PYTORCH)

        class _StubPreprocessor:
            def interleave_targets(self) -> bool:
                return False

        class _StubPostprocessor:
            def __call__(self, seq_embeddings, seq_timestamps):
                return seq_embeddings

        object.__setattr__(transducer, "_input_preprocessor", _StubPreprocessor())
        object.__setattr__(transducer, "_output_postprocessor", _StubPostprocessor())

        # Stamp every expanded row with its own position.
        encoded = (
            torch.arange(total_q, dtype=torch.float32)
            .unsqueeze(-1)
            .expand(-1, D)
            .contiguous()
        )
        timestamps = torch.arange(total_q, dtype=torch.float32)

        out, full = transducer._compose_output(
            encoded_embeddings=encoded,
            seq_timestamps=timestamps,
            seq_lengths=torch.tensor(lengths),
            seq_offsets=torch.tensor([0, lengths[0], total_q]),
            max_seq_len=max(lengths),
            total_uih_len=sum(prefix),
            total_targets=sum(num_targets),
            num_targets=torch.tensor(num_targets),
        )

        # Expected stamp of the task-token row of candidate c, task k,
        # request b: request start + prefix + c * (K + 1) + (k + 1).
        expected = []
        start = 0
        for b in range(2):
            for c in range(candidates[b]):
                expected.append(
                    [
                        float(start + prefix[b] + c * group_size + k + 1)
                        for k in range(K)
                    ]
                )
            start += lengths[b]
        self.assertTrue(
            torch.equal(out[:, :, 0], torch.tensor(expected)),
            msg=f"task-slot stamps wrong; got\n{out[:, :, 0]}",
        )
        self.assertEqual(out.shape, (sum(candidates), K, D))
        self.assertIsNone(full)


# (name, K task tokens, F family tokens, [(prefix_len, num_candidates), ...])
# per batch sample; group_size = K + F.
_BLOCK_LAYOUT_CASES = [
    ("single_task", 1, 1, [(3, 2)]),
    ("three_two", 3, 2, [(4, 2)]),
    ("seven_five", 7, 5, [(5, 3)]),
    ("no_prefix", 2, 2, [(0, 2)]),
    ("mixed_batch", 3, 1, [(3, 2), (7, 1), (1, 4)]),
]


def _reference_block_mask(
    layout: List[Tuple[int, int]],
    group_size: int,
    num_tasks: int,
    padded_len: int,
    group_mask: str = FAM_MASK_GROUP_NULL,
) -> torch.Tensor:
    """Write out the block (``family_tokens``) layout as a dense mask.

    ``X = [prefix (H) | G_1 | ... | G_N]`` with
    ``G_i = [t_1, ..., t_K, x^1, ..., x^F]``:

    * prefix rows are plain causal;
    * ``FAM_MASK_GROUP_NULL``: every group row -- task or family --
      sees the whole prefix plus its whole group, both directions;
    * ``FAM_MASK_TASK_ISO``: a task row sees the whole prefix, itself
      and the family block; a family row sees the whole prefix and the
      family block, never any task token.
    """
    mask = torch.zeros(len(layout), padded_len, padded_len, dtype=torch.bool)
    for b, (prefix_len, num_candidates) in enumerate(layout):
        for q in range(prefix_len):
            mask[b, q, : q + 1] = True
        for i in range(num_candidates):
            base = prefix_len + i * group_size
            fam_lo, fam_hi = base + num_tasks, base + group_size
            for slot in range(group_size):
                q = base + slot
                mask[b, q, :prefix_len] = True
                if group_mask == FAM_MASK_GROUP_NULL:
                    mask[b, q, base : base + group_size] = True
                elif slot < num_tasks:  # a task row
                    mask[b, q, q] = True
                    mask[b, q, fam_lo:fam_hi] = True
                else:  # a family row
                    mask[b, q, fam_lo:fam_hi] = True
    return mask


class _BuildOneRankBlockFuncTensorWrapper(nn.Module):
    def __init__(
        self,
        group_size: int,
        num_tasks: int,
        nheads: int = 1,
        group_mask: str = FAM_MASK_GROUP_NULL,
    ) -> None:
        super().__init__()
        self._group_size = group_size
        self._num_tasks = num_tasks
        self._nheads = nheads
        self._group_mask = group_mask

    def forward(
        self,
        x: torch.Tensor,
        seq_offsets: torch.Tensor,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        return build_onerank_func_tensor(
            nheads=self._nheads,
            seq_offsets=seq_offsets,
            total_q=x.size(0),
            num_targets=num_targets,
            group_size=self._group_size,
            layout=LAYOUT_BLOCK,
            group_mask=self._group_mask,
            num_tasks=self._num_tasks,
        )


def _build_block_func_tensor(
    layout: List[Tuple[int, int]],
    group_size: int,
    num_tasks: int,
    group_mask: str = FAM_MASK_GROUP_NULL,
    nheads: int = 1,
    graph_type: TestGraphType = TestGraphType.NORMAL,
    num_targets_override: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Run ``build_onerank_func_tensor`` on an expanded block-layout batch."""
    seq_lengths = [p + n * group_size for p, n in layout]
    num_targets = (
        num_targets_override
        if num_targets_override is not None
        else [n * group_size for _, n in layout]
    )
    seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
        torch.tensor(seq_lengths, dtype=torch.int32)
    )
    x = torch.zeros(int(sum(seq_lengths)), 1)
    module = create_test_module(
        _BuildOneRankBlockFuncTensorWrapper(
            group_size=group_size,
            num_tasks=num_tasks,
            nheads=nheads,
            group_mask=group_mask,
        ),
        graph_type,
    )
    func = module(x, seq_offsets, torch.tensor(num_targets, dtype=torch.int32))
    return func, seq_offsets, max(seq_lengths + [1])


@mark_ci_scope("h20", "gpu")
class OneRankBlockFuncTensorTest(unittest.TestCase):
    """Verify the NFUNC=5 encoding of the block (``family_tokens``) layout."""

    @parameterized.expand(
        _xprod([(c[0], c[1] + c[2], c[1], c[3]) for c in _BLOCK_LAYOUT_CASES])
    )
    def test_group_null_matches_reference_mask(
        self,
        _name: str,
        graph_type: TestGraphType,
        group_size: int,
        num_tasks: int,
        layout: List[Tuple[int, int]],
    ) -> None:
        """Decoded GROUP_NULL tensor == mask built directly from the layout."""
        func, seq_offsets, padded_len = _build_block_func_tensor(
            layout, group_size, num_tasks,
            group_mask=FAM_MASK_GROUP_NULL, graph_type=graph_type,
        )
        self.assertEqual(tuple(func.shape), (1, 5, int(seq_offsets[-1])))
        got = _decode_attn_func_to_mask(func, seq_offsets, padded_len)
        want = _reference_block_mask(
            layout, group_size, num_tasks, padded_len, FAM_MASK_GROUP_NULL
        )
        torch.testing.assert_close(got[:, 0], want)

    @parameterized.expand(
        _xprod([(c[0], c[1] + c[2], c[1], c[3]) for c in _BLOCK_LAYOUT_CASES])
    )
    def test_task_iso_matches_reference_mask(
        self,
        _name: str,
        graph_type: TestGraphType,
        group_size: int,
        num_tasks: int,
        layout: List[Tuple[int, int]],
    ) -> None:
        """Decoded TASK_ISO tensor == mask built directly from the layout."""
        func, seq_offsets, padded_len = _build_block_func_tensor(
            layout, group_size, num_tasks,
            group_mask=FAM_MASK_TASK_ISO, graph_type=graph_type,
        )
        got = _decode_attn_func_to_mask(func, seq_offsets, padded_len)
        want = _reference_block_mask(
            layout, group_size, num_tasks, padded_len, FAM_MASK_TASK_ISO
        )
        torch.testing.assert_close(got[:, 0], want)

    def test_task_iso_task_rows_are_isolated(self) -> None:
        """Under TASK_ISO, ``t_k`` sees prefix + itself + family block only.

        Never ``t_j`` for ``j != k`` -- the unbundled twin of the paper
        layout's task isolation, now with the family block shared instead
        of the candidate token.
        """
        prefix_len, num_candidates, K, F = 3, 3, 2, 2
        group_size = K + F
        func, seq_offsets, padded_len = _build_block_func_tensor(
            [(prefix_len, num_candidates)], group_size, K,
            group_mask=FAM_MASK_TASK_ISO,
        )
        mask = _decode_attn_func_to_mask(func, seq_offsets, padded_len)[0, 0]
        for i in range(num_candidates):
            base = prefix_len + i * group_size
            fam = list(range(base + K, base + group_size))
            for k in range(K):
                q = base + k
                visible = mask[q].nonzero().flatten().tolist()
                self.assertEqual(
                    visible, list(range(prefix_len)) + [q] + fam
                )

    def test_task_iso_family_rows_never_see_task_tokens(self) -> None:
        """A family summary is a pure function of the shared features."""
        prefix_len, num_candidates, K, F = 4, 2, 3, 2
        group_size = K + F
        func, seq_offsets, padded_len = _build_block_func_tensor(
            [(prefix_len, num_candidates)], group_size, K,
            group_mask=FAM_MASK_TASK_ISO,
        )
        mask = _decode_attn_func_to_mask(func, seq_offsets, padded_len)[0, 0]
        for i in range(num_candidates):
            base = prefix_len + i * group_size
            task_cols = set(range(base, base + K))
            for slot in range(K, group_size):
                q = base + slot
                visible = set(mask[q].nonzero().flatten().tolist())
                self.assertEqual(
                    visible,
                    set(range(prefix_len))
                    | set(range(base + K, base + group_size)),
                )
                self.assertFalse(visible & task_cols)

    def test_group_null_every_row_reads_whole_group(self) -> None:
        """GROUP_NULL: the whole group is bidirectional, family included."""
        prefix_len, K, F = 5, 3, 2
        group_size = K + F
        func, seq_offsets, padded_len = _build_block_func_tensor(
            [(prefix_len, 2)], group_size, K, group_mask=FAM_MASK_GROUP_NULL
        )
        mask = _decode_attn_func_to_mask(func, seq_offsets, padded_len)[0, 0]
        for q in range(prefix_len, prefix_len + group_size):
            self.assertEqual(
                mask[q].nonzero().flatten().tolist(),
                list(range(prefix_len))
                + list(range(prefix_len, prefix_len + group_size)),
            )

    def test_block_layout_rejects_bad_group_mask(self) -> None:
        """Unknown mask names fail loudly, not by falling into a branch."""
        seq_offsets = torch.tensor([0, 5], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "group_mask"):
            build_onerank_func_tensor(
                nheads=1,
                seq_offsets=seq_offsets,
                total_q=5,
                num_targets=torch.tensor([5], dtype=torch.int32),
                group_size=5,
                layout=LAYOUT_BLOCK,
                group_mask="FAMILY_MASK_NOPE",
                num_tasks=3,
            )

    def test_block_layout_requires_num_tasks(self) -> None:
        """Without ``K`` the TASK_ISO row split is undefined."""
        seq_offsets = torch.tensor([0, 5], dtype=torch.int32)
        for bad in (None, 0, 5):
            with self.assertRaisesRegex(ValueError, "num_tasks"):
                build_onerank_func_tensor(
                    nheads=1,
                    seq_offsets=seq_offsets,
                    total_q=5,
                    num_targets=torch.tensor([5], dtype=torch.int32),
                    group_size=5,
                    layout=LAYOUT_BLOCK,
                    group_mask=FAM_MASK_TASK_ISO,
                    num_tasks=bad,
                )


class OneRankBlockTokenizerTest(unittest.TestCase):
    """Verify the ``[t_1..t_K, x^1..x^F]`` block expansion."""

    _FAMILY_DIMS = [3, 2, 4]
    _FAMILY_IDS = [0, 0, 0, 1, 1, 2, 2, 2, 2]

    def _tokenizer(self) -> OneRankTokenizer:
        return OneRankTokenizer(
            num_tasks=3,
            embedding_dim=8,
            family_dims=self._FAMILY_DIMS,
            family_feature_ids=self._FAMILY_IDS,
        )

    def test_layout_and_group_size(self) -> None:
        """Block layout: K + F tokens, F projections, F index buffers."""
        tokenizer = self._tokenizer()
        self.assertEqual(tokenizer.layout, LAYOUT_BLOCK)
        self.assertEqual(tokenizer.group_size, 3 + 3)
        self.assertEqual(tokenizer.num_families, 3)
        self.assertEqual(tokenizer.family_names, ["family_0", "family_1", "family_2"])

    def test_expansion_matches_projection_definition(self) -> None:
        """Slot ``K + f`` of the group is exactly ``W_f(gather_f(raw))``."""
        torch.manual_seed(0)
        tokenizer = self._tokenizer()
        raw = torch.randn(2, sum(self._FAMILY_DIMS))
        out = tokenizer(raw)
        K = tokenizer.num_tasks
        for f, d in enumerate(self._FAMILY_DIMS):
            idx = torch.as_tensor(
                [j for j, fid in enumerate(self._FAMILY_IDS) if fid == f],
                dtype=torch.long,
            )
            want = tokenizer._family_projs[f](raw[:, idx])
            for i in range(2):
                torch.testing.assert_close(out[i * tokenizer.group_size + K + f], want[i])
        # Task slots are the shared learned queries.
        for i in range(2):
            for k in range(K):
                torch.testing.assert_close(
                    out[i * tokenizer.group_size + k], tokenizer._task_tokens[k]
                )

    def test_gradients_flow_through_every_raw_column(self) -> None:
        """Every raw column feeds exactly one family projection.

        A family id typo that orphaned a column would leave its gradient
        at zero.
        """
        torch.manual_seed(0)
        tokenizer = self._tokenizer()
        raw = torch.randn(2, sum(self._FAMILY_DIMS), requires_grad=True)
        out = tokenizer(raw)
        (out * torch.randn_like(out)).sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertFalse((raw.grad == 0).any(dim=0).any())

    @parameterized.expand(
        [
            ("id_count_mismatch", {"family_dims": [3, 2], "family_feature_ids": [0, 0, 1]}),
            ("empty_family", {"family_dims": [3, 0, 2], "family_feature_ids": [0, 0, 0, 2, 1, 1]}),
            ("id_out_of_range", {"family_dims": [3, 2], "family_feature_ids": [0, 0, 0, 5, 1]}),
        ]
    )
    def test_bad_family_config_raises(
        self, _name: str, kwargs: Dict[str, object]
    ) -> None:
        with self.assertRaises(ValueError):
            OneRankTokenizer(num_tasks=2, embedding_dim=8, **kwargs)


class OneRankBlockComposeOutputTest(unittest.TestCase):
    """The block-layout transducer output is the first ``K`` slots."""

    def test_block_layout_returns_the_first_k_slots(self) -> None:
        K, F, D = 3, 2, 4
        group_size = K + F
        prefix = [1, 1]
        candidates = [2, 4]
        num_targets = [c * group_size for c in candidates]
        lengths = [p + n for p, n in zip(prefix, num_targets)]
        total_q = sum(lengths)

        transducer = object.__new__(OneRankHSTUTransducer)
        object.__setattr__(
            transducer,
            "_tokenizer",
            OneRankTokenizer(
                num_tasks=K,
                embedding_dim=D,
                family_dims=[2, 2],
                family_feature_ids=[0, 0, 1, 1],
            ),
        )
        object.__setattr__(transducer, "_return_full_embeddings", False)
        object.__setattr__(transducer, "_is_inference", False)
        object.__setattr__(transducer, "_kernel", Kernel.PYTORCH)

        class _StubPreprocessor:
            def interleave_targets(self) -> bool:
                return False

        class _StubPostprocessor:
            def __call__(self, seq_embeddings, seq_timestamps):
                return seq_embeddings

        object.__setattr__(transducer, "_input_preprocessor", _StubPreprocessor())
        object.__setattr__(transducer, "_output_postprocessor", _StubPostprocessor())

        encoded = (
            torch.arange(total_q, dtype=torch.float32)
            .unsqueeze(-1)
            .expand(-1, D)
            .contiguous()
        )
        timestamps = torch.arange(total_q, dtype=torch.float32)

        out, full = transducer._compose_output(
            encoded_embeddings=encoded,
            seq_timestamps=timestamps,
            seq_lengths=torch.tensor(lengths),
            seq_offsets=torch.tensor([0, lengths[0], total_q]),
            max_seq_len=max(lengths),
            total_uih_len=sum(prefix),
            total_targets=sum(num_targets),
            num_targets=torch.tensor(num_targets),
        )

        # Expected stamp of the task-token row of candidate c, task k,
        # request b: request start + prefix + c * (K + F) + k -- the first
        # K slots of each block, not slots 1..K.
        expected = []
        start = 0
        for b in range(2):
            for c in range(candidates[b]):
                expected.append(
                    [
                        float(start + prefix[b] + c * group_size + k)
                        for k in range(K)
                    ]
                )
            start += lengths[b]
        self.assertTrue(
            torch.equal(out[:, :, 0], torch.tensor(expected)),
            msg=f"task-slot stamps wrong; got\n{out[:, :, 0]}",
        )
        self.assertEqual(out.shape, (sum(candidates), K, D))
        self.assertIsNone(full)


class OneRankBlockSTULayerTest(unittest.TestCase):
    """Block-layout ``OneRankSTULayer`` construction contracts."""

    _STU_KWARGS = {
        "embedding_dim": 16,
        "num_heads": 2,
        "hidden_dim": 8,
        "attention_dim": 8,
    }

    def _build(
        self,
        group_size: int = 12,
        num_tasks: int = 7,
        group_mask: str = FAM_MASK_TASK_ISO,
        **overrides: object,
    ) -> OneRankSTULayer:
        kwargs: Dict[str, object] = dict(self._STU_KWARGS)
        kwargs.update(overrides)
        return OneRankSTULayer(
            group_size=group_size,
            layout=LAYOUT_BLOCK,
            group_mask=group_mask,
            num_tasks=num_tasks,
            **kwargs,
        )

    def test_block_mask_is_noncausal(self) -> None:
        """Both block masks read future columns; the paper layout does not."""
        self.assertTrue(self._build().uses_noncausal_func_mask)
        paper_kwargs: Dict[str, object] = dict(self._STU_KWARGS)
        paper = OneRankSTULayer(group_size=8, **paper_kwargs)
        self.assertFalse(paper.uses_noncausal_func_mask)

    def test_cache_signature_separates_layouts_and_masks(self) -> None:
        """Different layouts / group masks must not share a func tensor."""
        block_null = self._build(group_mask=FAM_MASK_GROUP_NULL)
        block_iso = self._build(group_mask=FAM_MASK_TASK_ISO)
        self.assertNotEqual(
            block_null.attn_func_static_sig, block_iso.attn_func_static_sig
        )
        paper_kwargs: Dict[str, object] = dict(self._STU_KWARGS)
        paper = OneRankSTULayer(group_size=12, **paper_kwargs)
        self.assertNotEqual(
            block_iso.attn_func_static_sig, paper.attn_func_static_sig
        )

    def test_bad_block_options_are_rejected(self) -> None:
        """Unknown mask / missing num_tasks fail at construction."""
        with self.assertRaisesRegex(ValueError, "group_mask"):
            self._build(group_mask="FAMILY_MASK_NOPE")
        with self.assertRaisesRegex(ValueError, "num_tasks"):
            self._build(num_tasks=None)


if __name__ == "__main__":
    unittest.main()
