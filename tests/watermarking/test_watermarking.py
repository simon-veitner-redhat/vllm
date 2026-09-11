# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import SamplingParams
from vllm.config.watermarking import WatermarkConfig
from vllm.platforms import current_platform
from vllm.v1.watermarking import create_watermarker
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.watermarking.watermarker import Watermarker, WatermarkSample
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.watermark import (
    _watermark_prep_cpu,
    repeated_context_mask,
    watermark_prep,
)


class StubWatermarker(Watermarker):
    context_width = 1

    def _sample_watermarked(self, logits, contexts):
        return WatermarkSample(torch.tensor([7, 7]), logits + 10)


@pytest.mark.parametrize("algorithm", ["gumbel"])
def test_watermarker_contract(algorithm: str):
    watermarker = create_watermarker(
        WatermarkConfig(algorithm=algorithm, key=42, context_width=4)
    )
    logits = torch.zeros(2, 128)
    contexts = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    random_sample = lambda sample_logits: sample_logits.argmax(dim=-1)

    first = watermarker.sample(logits, contexts, random_sample)
    second = watermarker.sample(logits, contexts, random_sample)

    assert first.token_ids.shape == (2,)
    assert first.logits.shape == logits.shape
    assert torch.equal(first.token_ids, second.token_ids)


@pytest.mark.parametrize(
    "config_overrides",
    [
        {"deduplicate_contexts": "none"},
        {"deduplicate_contexts_max_history": 255},
    ],
)
def test_gumbel_config_warns_when_context_deduplication_is_weak(
    monkeypatch, config_overrides
):
    messages: list[str] = []
    monkeypatch.setattr(
        "vllm.config.watermarking.logger.warning_once",
        lambda message, *, scope: messages.append(message),
    )

    WatermarkConfig(key=42)
    WatermarkConfig(key=42, deduplicate_contexts_max_history=256)
    WatermarkConfig(key=42, deduplicate_contexts_max_history=None)
    WatermarkConfig(key=42, **config_overrides)

    assert messages == [
        (
            "Single-key Gumbel-max watermarking with context deduplication disabled "
            "or limited to fewer than 256 positions may increase the frequency of "
            "degenerate generations, including repetition loops. Use "
            "deduplicate_contexts='single_turn' or 'all' with "
            "deduplicate_contexts_max_history at least 256 or null to mitigate this."
        )
    ]


def test_context_deduplication_history_can_be_unbounded():
    config = WatermarkConfig(key=42, deduplicate_contexts_max_history=None)

    assert config.deduplicate_contexts_max_history is None


def test_large_context_width_warns_but_is_allowed():
    config = WatermarkConfig(key=42, context_width=17)

    with pytest.warns(UserWarning, match="reduce robustness to edits"):
        watermarker = create_watermarker(config)

    assert watermarker.context_width == 17


def test_sampling_params_can_disable_watermarking():
    assert SamplingParams().watermarking
    assert not SamplingParams.from_optional(watermarking=False).watermarking


def test_gpu_sampler_warns_when_watermarking_is_enabled_for_greedy(monkeypatch):
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarking = SimpleNamespace(np=np.ones(1, dtype=bool))
    messages: list[str] = []
    monkeypatch.setattr(Sampler, "add_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "vllm.v1.watermarking.gpu_sampler.logger.warning_once", messages.append
    )

    sampler.add_request(0, 1, SamplingParams(temperature=0))
    sampler.add_request(0, 1, SamplingParams(temperature=1))
    sampler.add_request(0, 1, SamplingParams(temperature=0, watermarking=False))

    assert messages == [
        (
            "Watermarking is enabled, but greedy decoding (temperature=0) cannot be "
            "watermarked. This request will use ordinary greedy sampling."
        )
    ]


def test_gpu_sampler_respects_mixed_request_watermarking(monkeypatch):
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = StubWatermarker()
    sampler.deduplicate_contexts = "single_turn"
    sampler.watermarking = SimpleNamespace(
        np=np.array([True, False]), gpu=torch.tensor([True, False])
    )
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.ones(2), gpu=torch.ones(2)),
        seeds=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int64)),
    )
    sampler.use_fp64_gumbel = False
    sampler._prepare_watermark_inputs = lambda expanded_idx_mapping: (
        torch.zeros(2, 1, dtype=torch.int64),
        torch.tensor([False, True]),
    )
    monkeypatch.setattr(
        "vllm.v1.watermarking.gpu_sampler.gumbel_sample",
        lambda *args, **kwargs: torch.tensor([3, 4]),
    )
    logits = torch.zeros(2, 8)

    sampled, output_logits = sampler._sample_random(
        logits,
        torch.tensor([0, 1]),
        np.array([0, 1]),
        torch.zeros(2, dtype=torch.int64),
        None,
        None,
        False,
    )

    assert torch.equal(sampled, torch.tensor([7, 4]))
    assert torch.equal(output_logits[0], torch.full((8,), 10.0))
    assert torch.equal(output_logits[1], logits[1])


def test_gpu_sampler_skips_watermarking_for_repeated_contexts(monkeypatch):
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = StubWatermarker()
    sampler.deduplicate_contexts = "single_turn"
    sampler.watermarking = SimpleNamespace(
        np=np.array([True, True]), gpu=torch.tensor([True, True])
    )
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.ones(2), gpu=torch.ones(2)),
        seeds=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int64)),
    )
    sampler.use_fp64_gumbel = False
    sampler._prepare_watermark_inputs = lambda expanded_idx_mapping: (
        torch.zeros(2, 1, dtype=torch.int64),
        torch.tensor([True, False]),
    )
    monkeypatch.setattr(
        "vllm.v1.watermarking.gpu_sampler.gumbel_sample",
        lambda *args, **kwargs: torch.tensor([3, 4]),
    )
    logits = torch.zeros(2, 8)

    sampled, output_logits = sampler._sample_random(
        logits,
        torch.tensor([0, 1]),
        np.array([0, 1]),
        torch.zeros(2, dtype=torch.int64),
        None,
        None,
        False,
    )

    assert torch.equal(sampled, torch.tensor([3, 7]))
    assert torch.equal(output_logits[0], logits[0])
    assert torch.equal(output_logits[1], torch.full((8,), 10.0))


def test_gpu_sampler_can_disable_context_deduplication(monkeypatch):
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = StubWatermarker()
    sampler.deduplicate_contexts = "none"
    sampler.watermarking = SimpleNamespace(
        np=np.array([True, True]), gpu=torch.tensor([True, True])
    )
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.ones(2), gpu=torch.ones(2)),
        seeds=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int64)),
    )
    sampler.use_fp64_gumbel = False
    sampler.deduplicate_contexts_max_history = 8192
    sampler.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.tensor([[1, 2], [3, 4]])),
        prompt_len=SimpleNamespace(gpu=torch.tensor([0, 0])),
        total_len=SimpleNamespace(gpu=torch.tensor([2, 2])),
    )
    scans: list[bool] = []
    monkeypatch.setattr(
        "vllm.v1.watermarking.gpu_sampler.watermark_prep",
        lambda *args, scan, **kwargs: (
            scans.append(scan),
            (torch.zeros(2, 1, dtype=torch.int64), torch.tensor([False, False])),
        )[1],
    )
    monkeypatch.setattr(
        "vllm.v1.watermarking.gpu_sampler.gumbel_sample",
        lambda *args, **kwargs: torch.tensor([3, 4]),
    )
    logits = torch.zeros(2, 8)

    sampled, output_logits = sampler._sample_random(
        logits,
        torch.tensor([0, 1]),
        np.array([0, 1]),
        torch.zeros(2, dtype=torch.int64),
        None,
        None,
        False,
    )

    assert scans == [False]
    assert torch.equal(sampled, torch.tensor([7, 7]))
    assert torch.equal(output_logits, torch.full((2, 8), 10.0))


def test_repeated_context_mask_ignores_prompt_tokens():
    all_token_ids = torch.tensor(
        [
            [8, 9, 1, 2, 1, 2],
            [3, 4, 1, 2, 3, 4],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    req_indices = torch.tensor([0, 1, -1])
    prompt_lens = torch.tensor([2, 2, 0])
    total_lens = torch.tensor([6, 6, 0])
    contexts = torch.tensor([[1, 2], [3, 4], [-1, -1]])

    repeated = repeated_context_mask(
        all_token_ids,
        req_indices,
        prompt_lens,
        total_lens,
        contexts,
    )

    assert torch.equal(repeated, torch.tensor([True, False, False]))


def test_repeated_context_mask_can_include_prompt_tokens():
    all_token_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4]])
    req_indices = torch.tensor([0])
    prompt_lens = torch.tensor([4])
    total_lens = torch.tensor([10])
    contexts = torch.tensor([[1, 2, 3, 4]])

    single_turn = repeated_context_mask(
        all_token_ids, req_indices, prompt_lens, total_lens, contexts
    )
    all_history = repeated_context_mask(
        all_token_ids,
        req_indices,
        prompt_lens,
        total_lens,
        contexts,
        include_prompt=True,
    )

    assert not single_turn.item()
    assert all_history.item()


def test_gpu_sampler_all_history_uses_prompt_context():
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = SimpleNamespace(context_width=4)
    sampler.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.tensor([[10, 11, 12, 13, 1, 2]])),
        prompt_len=SimpleNamespace(gpu=torch.tensor([4])),
        total_len=SimpleNamespace(gpu=torch.tensor([6])),
    )
    request_indices = torch.tensor([0])

    sampler.deduplicate_contexts = "single_turn"
    single_turn = sampler._get_contexts(request_indices)
    sampler.deduplicate_contexts = "all"
    all_history = sampler._get_contexts(request_indices)

    assert torch.equal(single_turn, torch.tensor([[-1, -1, 1, 2]]))
    assert torch.equal(all_history, torch.tensor([[12, 13, 1, 2]]))


def test_repeated_context_mask_respects_max_history():
    all_token_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4]])
    req_indices = torch.tensor([0])
    prompt_lens = torch.tensor([0])
    total_lens = torch.tensor([10])
    contexts = torch.tensor([[1, 2, 3, 4]])

    full_history = repeated_context_mask(
        all_token_ids, req_indices, prompt_lens, total_lens, contexts
    )
    last_six = repeated_context_mask(
        all_token_ids, req_indices, prompt_lens, total_lens, contexts, max_history=6
    )
    last_five = repeated_context_mask(
        all_token_ids, req_indices, prompt_lens, total_lens, contexts, max_history=5
    )

    assert full_history.item()
    assert last_six.item()
    assert not last_five.item()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("max_history", [5, 6])
@pytest.mark.parametrize("include_prompt", [False, True])
def test_repeated_context_mask_max_history_accelerator_parity(
    max_history: int, include_prompt: bool
):
    inputs = (
        torch.tensor([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4]]),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([10]),
        torch.tensor([[1, 2, 3, 4]]),
    )

    expected = repeated_context_mask(
        *inputs, max_history=max_history, include_prompt=include_prompt
    )
    actual = repeated_context_mask(
        *(value.cuda() for value in inputs),
        max_history=max_history,
        include_prompt=include_prompt,
    ).cpu()

    assert torch.equal(actual, expected)


def _unaligned_max_history_inputs() -> tuple[torch.Tensor, ...]:
    """One row whose only repeated context ends at position 680.

    680 is not a multiple of the kernel's 512-position block, so the scan window
    selected by ``max_history`` starts in the middle of a block.
    """
    history = list(range(1_000, 2_200))
    history[676:680] = [1, 2, 3, 4]
    return (
        torch.tensor([history], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([len(history)], dtype=torch.int32),
        torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
    )


def _watermark_prep_fixture(seed: int, scope: str) -> dict:
    """A random batch with padding, opted-out, greedy and short-history rows."""
    generator = torch.Generator().manual_seed(seed)
    num_reqs = 6
    context_width = 4
    max_len = 40
    all_token_ids = torch.randint(
        0, 12, (num_reqs, max_len), dtype=torch.int32, generator=generator
    )
    prompt_lens = torch.tensor([8, 8, 8, 8, 8, 0], dtype=torch.int32)
    # Row 4 has fewer generated tokens than context_width, row 5 is empty.
    total_lens = torch.tensor([40, 33, 27, 12, 10, 0], dtype=torch.int32)
    # Row 3 is a padding row (req_idx -1); the batch also visits a row twice.
    req_indices = torch.tensor([0, 1, 2, -1, 4, 5, 2], dtype=torch.int32)
    watermarking = torch.tensor([True, False, True, True, True, True])
    temperatures = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    # Plant a repeat of row 0's context early in its generated history.
    context = all_token_ids[0, 40 - context_width : 40].clone()
    plant_at = 8 if scope != "all" else 2
    all_token_ids[0, plant_at : plant_at + context_width] = context
    return dict(
        all_token_ids=all_token_ids,
        req_indices=req_indices,
        prompt_lens=prompt_lens,
        total_lens=total_lens,
        context_width=context_width,
        watermarking=watermarking,
        temperatures=temperatures,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("scope", ["none", "single_turn", "all"])
@pytest.mark.parametrize("max_history", [None, 5, 8192])
@pytest.mark.parametrize("seed", range(20))
def test_watermark_prep_accelerator_parity(scope: str, max_history, seed: int):
    fixture = _watermark_prep_fixture(seed, scope)
    kwargs = dict(
        max_history=max_history,
        include_prompt=scope == "all",
        scan=scope != "none",
    )

    expected_contexts, expected_skip = _watermark_prep_cpu(
        fixture["all_token_ids"],
        fixture["req_indices"],
        fixture["prompt_lens"],
        fixture["total_lens"],
        fixture["context_width"],
        fixture["watermarking"],
        fixture["temperatures"],
        max_history,
        scope == "all",
        scope != "none",
    )
    contexts, skip = watermark_prep(
        fixture["all_token_ids"].cuda(),
        fixture["req_indices"].cuda(),
        fixture["prompt_lens"].cuda(),
        fixture["total_lens"].cuda(),
        fixture["context_width"],
        watermarking=fixture["watermarking"].cuda(),
        temperatures=fixture["temperatures"].cuda(),
        **kwargs,
    )

    assert torch.equal(contexts.cpu(), expected_contexts)
    assert torch.equal(skip.cpu(), expected_skip)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize(
    "max_history,expected", [(1200 - 680, True), (1200 - 680 - 1, False)]
)
def test_repeated_context_mask_unaligned_max_history_accelerator_parity(
    max_history: int, expected: bool
):
    inputs = _unaligned_max_history_inputs()

    reference = repeated_context_mask(*inputs, max_history=max_history)
    actual = repeated_context_mask(
        *(value.cuda() for value in inputs), max_history=max_history
    ).cpu()

    assert torch.equal(reference, torch.tensor([expected]))
    assert torch.equal(actual, reference)


@pytest.mark.parametrize("max_history", [512, 8192])
@pytest.mark.parametrize("include_prompt", [False, True])
def test_watermark_prep_many_rows_is_deterministic(max_history, include_prompt):
    """128 sampled rows with planted repeats, launched repeatedly.

    The fused kernel stores each row's context from one lane and reloads it
    from every lane of the program; without a barrier between the two the skip
    mask differs from launch to launch. Small fixtures do not open that window,
    so this case runs many concurrent programs several times.
    """
    generator = torch.Generator().manual_seed(901)
    num_reqs, num_rows, context_width, max_len = 64, 128, 4, 3000
    all_token_ids = torch.randint(
        0, 7, (num_reqs, max_len), dtype=torch.int32, generator=generator
    )
    prompt_lens = torch.randint(
        0, 40, (num_reqs,), dtype=torch.int32, generator=generator
    )
    total_lens = prompt_lens + torch.randint(
        context_width + 8, 2100, (num_reqs,), dtype=torch.int32, generator=generator
    )
    for req in range(num_reqs):
        total = int(total_lens[req])
        start = 0 if include_prompt else int(prompt_lens[req])
        all_token_ids[req, start + 3 : start + 3 + context_width] = all_token_ids[
            req, total - context_width : total
        ].clone()
    req_indices = torch.randint(
        -1, num_reqs, (num_rows,), dtype=torch.int32, generator=generator
    )
    watermarking = torch.rand(num_reqs, generator=generator) > 0.2
    temperatures = (torch.rand(num_reqs, generator=generator) > 0.2).float()

    expected_contexts, expected_skip = _watermark_prep_cpu(
        all_token_ids,
        req_indices,
        prompt_lens,
        total_lens,
        context_width,
        watermarking,
        temperatures,
        max_history,
        include_prompt,
        True,
    )
    cuda_inputs = [
        t.cuda() for t in (all_token_ids, req_indices, prompt_lens, total_lens)
    ]
    for _ in range(10):
        contexts, skip = watermark_prep(
            *cuda_inputs,
            context_width,
            watermarking=watermarking.cuda(),
            temperatures=temperatures.cuda(),
            max_history=max_history,
            include_prompt=include_prompt,
            scan=True,
        )
        assert torch.equal(contexts.cpu(), expected_contexts)
        assert torch.equal(skip.cpu(), expected_skip)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("scope", ["single_turn", "all"])
@pytest.mark.parametrize("max_history", [None, 5, 8192])
def test_watermark_prep_matches_separate_context_and_mask_kernels(
    scope: str, max_history
):
    """The fused kernel reproduces `_get_contexts` + `repeated_context_mask`."""
    fixture = _watermark_prep_fixture(3, scope)
    device_inputs = {
        key: value.cuda() if torch.is_tensor(value) else value
        for key, value in fixture.items()
    }

    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = SimpleNamespace(context_width=fixture["context_width"])
    sampler.deduplicate_contexts = scope
    sampler.deduplicate_contexts_max_history = max_history
    sampler.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=device_inputs["all_token_ids"]),
        prompt_len=SimpleNamespace(gpu=device_inputs["prompt_lens"]),
        total_len=SimpleNamespace(gpu=device_inputs["total_lens"]),
    )
    sampler.watermarking = SimpleNamespace(gpu=device_inputs["watermarking"])
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(gpu=device_inputs["temperatures"])
    )
    req_indices = device_inputs["req_indices"]

    # The head formula: contexts from `_get_contexts`, repeats from the mask
    # kernel, skip = not (enabled and temperature != 0 and not repeated).
    reference_contexts = sampler._get_contexts(req_indices)
    repeated = repeated_context_mask(
        device_inputs["all_token_ids"],
        req_indices,
        device_inputs["prompt_lens"],
        device_inputs["total_lens"],
        reference_contexts,
        max_history,
        include_prompt=scope == "all",
    )
    safe = req_indices.to(torch.int64).clamp_min(0)
    watermarking = (
        device_inputs["watermarking"][safe]
        & (device_inputs["temperatures"][safe] != 0)
        & ~repeated
    )
    reference_skip = ~(watermarking & (req_indices.to(torch.int64) >= 0))

    contexts, skip = sampler._prepare_watermark_inputs(req_indices)

    assert torch.equal(contexts, reference_contexts)
    assert torch.equal(skip, reference_skip)


def _repeated_context_inputs(
    context_width: int, device: str = "cpu"
) -> tuple[torch.Tensor, ...]:
    """Build repeated, unique, and padded request rows for mask tests."""
    prompt = [101, 102, 103, 104]
    repeated_output = [*range(1, context_width + 1)] * 2
    unique_output = [*range(1, context_width + 2)]
    max_len = len(prompt) + len(repeated_output)
    rows = [
        prompt + repeated_output,
        prompt + unique_output,
        [],
    ]
    all_token_ids = torch.zeros((3, max_len), dtype=torch.int32, device=device)
    for row, token_ids in enumerate(rows):
        all_token_ids[row, : len(token_ids)] = torch.tensor(
            token_ids, dtype=torch.int32, device=device
        )
    return (
        all_token_ids,
        torch.tensor([0, 1, -1], dtype=torch.int32, device=device),
        torch.tensor([len(prompt), len(prompt), 0], dtype=torch.int32, device=device),
        torch.tensor([len(rows[0]), len(rows[1]), 0], dtype=torch.int32, device=device),
        torch.tensor(
            [
                repeated_output[-context_width:],
                unique_output[-context_width:],
                [-1] * context_width,
            ],
            dtype=torch.int64,
            device=device,
        ),
    )


@pytest.mark.parametrize("context_width", [1, 3, 4, 16, 17])
def test_repeated_context_mask(context_width: int):
    repeated = repeated_context_mask(*_repeated_context_inputs(context_width))

    assert torch.equal(repeated, torch.tensor([True, False, False]))


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("context_width", [1, 3, 4, 16, 17])
def test_repeated_context_mask_accelerator_parity(context_width: int):
    cpu_inputs = _repeated_context_inputs(context_width)
    accelerator_inputs = list(_repeated_context_inputs(context_width, "cuda"))
    contexts = accelerator_inputs[-1]
    storage = torch.empty(
        contexts.shape[0], contexts.shape[1] * 2, dtype=contexts.dtype, device="cuda"
    )
    storage[:, ::2] = contexts
    accelerator_inputs[-1] = storage[:, ::2]

    expected = repeated_context_mask(*cpu_inputs)
    actual = repeated_context_mask(*accelerator_inputs).cpu()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_repeated_context_mask_scans_multiple_blocks():
    context = [1, 2, 3, 4]
    repeated_output = [*range(10_000, 11_030), *context, 99, *context]
    unique_output = list(range(20_000, 21_039))
    prompt = [101, 102, 103, 104]
    rows = [prompt + repeated_output, prompt + unique_output]
    all_token_ids = torch.zeros((2, len(rows[0])), dtype=torch.int32)
    for row, token_ids in enumerate(rows):
        all_token_ids[row, : len(token_ids)] = torch.tensor(token_ids)
    inputs = (
        all_token_ids,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([len(prompt), len(prompt)], dtype=torch.int32),
        torch.tensor([len(rows[0]), len(rows[1])], dtype=torch.int32),
        torch.tensor([context, unique_output[-4:]], dtype=torch.int64),
    )

    expected = repeated_context_mask(*inputs)
    actual = repeated_context_mask(*(value.cuda() for value in inputs)).cpu()

    assert torch.equal(expected, torch.tensor([True, False]))
    assert torch.equal(actual, expected)


def test_gpu_sampler_skips_watermarking_for_greedy_batch(monkeypatch):
    class StubWatermarker:
        context_width = 1

        def sample(self, logits, contexts, random_sample):
            raise AssertionError("watermarker should not run for greedy requests")

    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = StubWatermarker()
    sampler.watermarking = SimpleNamespace(
        np=np.array([True, True]), gpu=torch.tensor([True, True])
    )
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.zeros(2), gpu=torch.zeros(2)),
    )
    expected = (torch.tensor([3, 4]), torch.zeros(2, 8))
    monkeypatch.setattr(
        "vllm.v1.watermarking.gpu_sampler.Sampler._sample_random",
        lambda *args, **kwargs: expected,
    )

    actual = sampler._sample_random(
        torch.zeros(2, 8),
        torch.tensor([0, 1]),
        np.array([0, 1]),
        torch.zeros(2, dtype=torch.int64),
        None,
        None,
        False,
    )

    assert actual is expected
