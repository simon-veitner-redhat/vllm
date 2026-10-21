# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request attention blocks (`vllm.attention_blocks`): validation, the
scheduler's no-split rule and the per-block prefix-cache keys."""

from types import SimpleNamespace

import pytest
import torch

from vllm.attention_blocks import (
    AttentionBlocksError,
    block_aligned_chunk_end,
    check_attention_blocks,
    ends_inside_block,
    get_attention_blocks,
    parse_attention_blocks,
)
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from .utils import create_scheduler

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


@pytest.fixture(autouse=True)
def _init_none_hash():
    init_none_hash(sha256)


def make_request(
    request_id: str,
    prompt: list[int],
    blocks: list[list[int]] | None,
    max_tokens: int = 1,
) -> Request:
    params = SamplingParams(
        max_tokens=max_tokens,
        extra_args=None if blocks is None else {"attention_blocks": blocks},
    )
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


# -- Validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        [],
        [[0]],
        [[0, 1, 2]],
        [[-1, 3]],
        [[5, 3]],
        [[0, 9], [9, 12]],
        [[10, 12], [0, 4]],
        [["0", "9"]],
        [[True, 9]],
        "0-9",
    ],
)
def test_parse_rejects(value):
    with pytest.raises(AttentionBlocksError):
        parse_attention_blocks(value)


def test_parse_and_check():
    blocks = parse_attention_blocks([[0, 98], (99, 156)])
    assert blocks == ((0, 98), (99, 156))
    assert get_attention_blocks(SamplingParams()) is None
    assert get_attention_blocks(
        SamplingParams(extra_args={"attention_blocks": [[0, 3]]})
    ) == ((0, 3),)
    check_attention_blocks(blocks, prompt_len=157, max_num_batched_tokens=157)
    with pytest.raises(AttentionBlocksError, match="past the prompt"):
        check_attention_blocks(blocks, prompt_len=156, max_num_batched_tokens=8192)
    with pytest.raises(AttentionBlocksError, match="one scheduler step"):
        check_attention_blocks(blocks, prompt_len=157, max_num_batched_tokens=156)


def test_ends_inside_block():
    blocks = ((0, 99), (100, 149))
    assert [e for e in range(151) if not ends_inside_block(e, blocks)] == [0, 100, 150]


def _input_processor(request_attention_blocks: bool):
    return SimpleNamespace(
        model_config=SimpleNamespace(request_attention_blocks=request_attention_blocks),
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_scheduled_tokens=None, max_num_batched_tokens=256
            )
        ),
    )


def test_input_processor_validates_blocks():
    validate = InputProcessor._validate_attention_blocks
    params = SamplingParams(extra_args={"attention_blocks": ((0, 98), [99, 156])})
    validate(_input_processor(True), params, 157, "token")
    assert params.extra_args == {"attention_blocks": [[0, 98], [99, 156]]}

    plain = SamplingParams()
    validate(_input_processor(False), plain, 157, "token")

    for processor, prompt_len, input_type, match in [
        (_input_processor(False), 157, "token", "block_bidirectional"),
        (_input_processor(True), 157, "multimodal", "text-only"),
        (_input_processor(True), 119, "token", "past the prompt"),
        (_input_processor(True), 300, "token", "one scheduler step"),
    ]:
        params = SamplingParams(extra_args={"attention_blocks": [[0, 98], [99, 119]]})
        with pytest.raises(VLLMValidationError, match=match):
            validate(processor, params, prompt_len, input_type)


# -- Where a step may end ---------------------------------------------------

PREFIX_SUFFIX = ((0, 99), (100, 149))


@pytest.mark.parametrize(
    "start,budget_end,expected",
    [
        (0, 150, 150),  # the whole prompt fits
        (0, 120, 100),  # stop at the end of the prefix block
        (0, 80, 0),  # cannot finish the prefix block: wait
        (50, 120, 100),  # a cache hit inside the prefix still finishes it
        (100, 149, 100),  # cannot finish the suffix block: wait
        (149, 150, 150),  # the last token of a full-prompt hit
    ],
)
def test_chunk_end_without_mamba_alignment(start, budget_end, expected):
    end = block_aligned_chunk_end(PREFIX_SUFFIX, start, budget_end, prefill_end=150)
    assert end == expected


@pytest.mark.parametrize(
    "blocks,start,budget_end,align_end,expected",
    [
        # The split's stop at the last Mamba block boundary (128) lies in the
        # suffix block; the step runs to the prompt's end instead.
        (PREFIX_SUFFIX, 0, 150, 128, 150),
        # Stopping at 100 would leave the next step unable to finish the
        # suffix block without crossing 128 from inside a Mamba block.
        (PREFIX_SUFFIX, 0, 120, 64, 0),
        # ... unless the suffix block ends at the Mamba block boundary.
        (((0, 99), (100, 127)), 0, 110, 64, 100),
        (((0, 99), (100, 127)), 100, 128, 128, 128),
        # A partial hit inside a Mamba block may not cross its end ...
        (((0, 99), (100, 199)), 96, 200, 128, 96),
        # ... unless the split allows it (an in-chunk checkpoint).
        (((0, 99), (100, 199)), 96, 200, 200, 200),
        # A split end that is allowed is kept.
        (((0, 63), (64, 199)), 0, 200, 64, 64),
    ],
)
def test_chunk_end_with_mamba_alignment(blocks, start, budget_end, align_end, expected):
    end = block_aligned_chunk_end(
        blocks,
        start,
        budget_end,
        prefill_end=blocks[-1][1] + 1,
        align_end=align_end,
        mamba_block_size=64,
    )
    assert end == expected


# -- Prefix-cache keys -----------------------------------------------------


def make_manager() -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=8192,
        enable_caching=True,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
    )


def compute(manager: KVCacheManager, request: Request) -> int:
    """Look the request up, then compute and cache its whole prompt. Returns
    the cache hit."""
    computed, num_computed, _ = manager.get_computed_blocks(request)
    manager.allocate_slots(
        request,
        request.num_tokens - num_computed,
        num_computed,
        computed,
    )
    request.num_computed_tokens = request.num_tokens
    return num_computed


PREFIX = list(range(1000, 1040))  # 40 tokens: blocks 0 and 1, part of block 2


def two_blocks(prompt: list[int], prefix_len: int = len(PREFIX)) -> list[list[int]]:
    return [[0, prefix_len - 1], [prefix_len, len(prompt) - 1]]


def test_cache_hit_inside_the_prefix_block():
    manager = make_manager()
    q0 = PREFIX + [1] * 30
    assert compute(manager, make_request("q0", q0, two_blocks(q0))) == 0
    # Another question on the same state reads the prefix block's full hash
    # blocks; block 2 also holds suffix tokens, which differ.
    q1 = PREFIX + [2] * 30
    assert compute(manager, make_request("q1", q1, two_blocks(q1))) == 32


def test_warm_up_of_the_prefix_block_is_reused():
    manager = make_manager()
    assert compute(manager, make_request("warm", PREFIX, [[0, 39]])) == 0
    q = PREFIX + [3] * 30
    assert compute(manager, make_request("q", q, two_blocks(q))) == 32


def test_questions_sharing_suffix_tokens_do_not_share_cache():
    """The suffix block's KV depends on its later tokens, so two questions
    whose text starts alike must not share the blocks of that text."""
    prefix = PREFIX[:32]
    q0 = prefix + [7] * 20 + [8] * 10
    q1 = prefix + [7] * 20 + [9] * 10

    causal = make_manager()
    compute(causal, make_request("q0", q0, None))
    assert compute(causal, make_request("q1", q1, None)) == 48

    blocks = make_manager()
    compute(blocks, make_request("q0", q0, two_blocks(q0, 32)))
    assert compute(blocks, make_request("q1", q1, two_blocks(q1, 32))) == 32


def test_blocks_are_part_of_the_cache_key():
    manager = make_manager()
    q = PREFIX + [4] * 30
    compute(manager, make_request("blocks", q, two_blocks(q)))
    # The same tokens, causal or cut elsewhere, are different KV.
    assert compute(manager, make_request("causal", q, None)) == 0
    assert compute(manager, make_request("cut", q, two_blocks(q, 30))) == 0


# -- Scheduler ---------------------------------------------------------------


def test_scheduler_waits_instead_of_splitting_a_block():
    def second_request_tokens(blocks) -> int | None:
        scheduler = create_scheduler(max_num_batched_tokens=100, block_size=BLOCK_SIZE)
        first = make_request("first", list(range(50)), None)
        q = PREFIX + [5] * 30
        second = make_request("second", q, two_blocks(q) if blocks else None)
        scheduler.add_request(first)
        scheduler.add_request(second)
        output = scheduler.schedule()
        assert output.num_scheduled_tokens["first"] == 50
        return output.num_scheduled_tokens.get("second")

    # Causally the 50 tokens left are a chunk; with blocks that chunk would
    # end inside the suffix block, so the step stops at the prefix block.
    assert second_request_tokens(blocks=False) == 50
    assert second_request_tokens(blocks=True) == 40

    scheduler = create_scheduler(max_num_batched_tokens=100, block_size=BLOCK_SIZE)
    q = PREFIX + [5] * 30
    scheduler.add_request(make_request("alone", q, two_blocks(q)))
    assert scheduler.schedule().num_scheduled_tokens["alone"] == 70


def test_scheduler_ends_steps_on_block_boundaries():
    scheduler = create_scheduler(max_num_batched_tokens=60, block_size=BLOCK_SIZE)
    q = PREFIX + [6] * 20  # blocks [0, 39] and [40, 59]
    scheduler.add_request(make_request("filler", list(range(30)), None))
    scheduler.add_request(make_request("q", q, two_blocks(q)))
    output = scheduler.schedule()
    # 30 tokens are left: not enough for the prefix block.
    assert "q" not in output.num_scheduled_tokens


def make_hybrid_align_scheduler(max_num_batched_tokens: int = 8192) -> Scheduler:
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=8192,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        model_config=ModelConfig(
            model="facebook/opt-125m",
            trust_remote_code=True,
            dtype="float16",
            seed=42,
            skip_tokenizer_init=True,
        ),
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
    )
    vllm_config.cache_config.num_gpu_blocks = 100
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["fa"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=BLOCK_SIZE,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        log_stats=True,
    )


@pytest.mark.parametrize("blocks,expected", [(False, 64), (True, 70)])
def test_mamba_align_stop_inside_a_block_is_skipped(blocks, expected):
    """Align mode stops at the last Mamba block boundary (64) to cache its
    state; that boundary lies in the suffix block, so a request with blocks
    computes its whole prompt in one step instead."""
    scheduler = make_hybrid_align_scheduler()
    q = PREFIX + [5] * 30
    scheduler.add_request(make_request("q", q, two_blocks(q) if blocks else None))
    assert scheduler.schedule().num_scheduled_tokens["q"] == expected


def test_unresumable_partial_hit_is_dropped():
    """A hit inside a Mamba block that cannot reach the next boundary without
    ending inside an attention block is recomputed from scratch."""
    request = make_request("q", list(range(200)), [[0, 99], [100, 199]])
    split_end = {}
    stub = SimpleNamespace(
        need_mamba_block_aligned_split=True,
        cache_config=SimpleNamespace(block_size=64),
        kv_cache_manager=SimpleNamespace(empty_kv_cache_blocks="empty"),
        _mamba_block_aligned_split=lambda req, n, local, ext: split_end[local] - local,
    )
    stub._attention_block_split = lambda *args: Scheduler._attention_block_split(
        stub, *args
    )
    fit = Scheduler._fit_attention_block_hit

    # The split stops the partial hit at 96 at its Mamba block's end (128),
    # inside the suffix block.
    split_end[96] = 128
    assert fit(stub, request, "hit", 96, 0) == ("empty", 0)
    # With an in-chunk checkpoint the split runs to the end, and the hit stays.
    split_end[96] = 200
    assert fit(stub, request, "hit", 96, 0) == ("hit", 96)
    # A hit on a Mamba block boundary can always finish the prompt.
    split_end[64] = 128
    assert fit(stub, request, "hit", 64, 0) == ("hit", 64)


# -- Guards ------------------------------------------------------------------


def _config_stub(**changes):
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="m", request_attention_blocks=True),
        kv_transfer_config=None,
        speculative_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
    )
    for key, value in changes.items():
        setattr(config, key, value)
    return config


def test_config_refuses_paths_that_drop_blocks():
    verify = VllmConfig._verify_attention_blocks
    verify(_config_stub())
    verify(_config_stub(kv_transfer_config=object(), model_config=None))
    with pytest.raises(ValueError, match="KV transfer connectors"):
        verify(_config_stub(kv_transfer_config=object()))
    with pytest.raises(ValueError, match="speculative decoding"):
        verify(_config_stub(speculative_config=object()))


def test_v2_model_state_passes_blocks_by_batch_row():
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    state = SimpleNamespace(attention_blocks={"b": ((0, 9), (10, 19))})
    ranges = DefaultModelState.attention_block_ranges(state, ["a", "b", "c"])
    assert ranges == {1: [(0, 9), (10, 19)]}
    state.attention_blocks = {}
    assert DefaultModelState.attention_block_ranges(state, ["a"]) is None
