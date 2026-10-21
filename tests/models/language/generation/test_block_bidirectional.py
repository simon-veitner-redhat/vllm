# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-bidirectional attention (Base Labs mammoth-1, elephant-1) against
the transformers block mask.

A four-layer random Qwen3.5 (GatedDeltaNet and full attention, twice) with
the checkpoints' `modeling_overrides` is served in bf16 (vLLM's chunked
GatedDeltaNet kernel has no fp32 path) with each backend that applies request
attention blocks. With a single full-attention layer at
the top, the last token would see every key under either mask. The oracle
is transformers' `create_causal_mask(block_sequence_ids=...)`, the mask the
reference engine (qev) and training use: causal, or query and key in the
same block.

The cases cover a fresh prefill, a cache hit inside the prefix block (after
the prefix-only warm-up), a full-prompt hit, and two questions whose suffixes
start with the same tokens, which must not share cache.

Needs a GPU. The base config, tokenizer and processor come from
BLOCK_ATTN_BASE (a Hugging Face id or a local checkpoint directory).
"""

import os

import pytest
import torch

from vllm.platforms import current_platform

BASE = os.environ.get("BLOCK_ATTN_BASE", "baselabs/elephant-1")
NUM_SCORED = 256
# vLLM runs in bf16 against an fp32 reference, so a read must be close to the
# block-mask reference and much closer to it than the causal reference is.
ATOL = 0.1
MAX_ERROR_VS_MASK_GAP = 0.2

OVERRIDES = {
    "attention": {
        "full_attention": "block_bidirectional",
        "linear_attention": "causal",
    },
    "readout": "slot_lm",
    "head_dim": 256,
    "n_slots": 256,
}

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="block attention needs a CUDA GPU"
)


def build_tiny_checkpoint(path: str) -> None:
    """A random four-layer Qwen3.5 with the checkpoints' overrides, saved with
    the base checkpoint's tokenizer and processor."""
    from transformers import (
        AutoConfig,
        AutoProcessor,
        Qwen3_5ForConditionalGeneration,
    )

    config = AutoConfig.from_pretrained(BASE)
    text = config.text_config
    text.update(
        {
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "full_attention"] * 2,
            "full_attention_interval": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "mtp_num_hidden_layers": 0,
            "dtype": "float32",
        }
    )
    config.vision_config.update(
        {
            "depth": 1,
            "hidden_size": 64,
            "num_heads": 2,
            "intermediate_size": 128,
            "out_hidden_size": 256,
        }
    )
    config.dtype = "float32"
    config.modeling_overrides = OVERRIDES
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(config).to(torch.float32)
    # Random init is near-uniform; sharper logits make mask errors visible.
    # Stronger full-attention values and outputs make the mask move the last
    # token's logits by about 1.4 instead of 0.05, well above bf16 noise.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() > 1:
                param.normal_(0.0, 0.08)
                if "self_attn.v_proj" in name or "self_attn.o_proj" in name:
                    param.mul_(4.0)
    model.save_pretrained(path)
    AutoProcessor.from_pretrained(BASE).save_pretrained(path)


def hf_logprobs(path: str, prompts, blocks, device: str) -> list[torch.Tensor]:
    """Log-softmax of the last position over the first NUM_SCORED tokens, with
    each prompt's blocks (None: causal)."""
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    original = modeling_qwen3_5.create_causal_mask
    current: dict = {}

    def create_causal_mask(*args, **kwargs):
        if current.get("ids") is not None:
            kwargs["block_sequence_ids"] = current["ids"]
        return original(*args, **kwargs)

    out = []
    modeling_qwen3_5.create_causal_mask = create_causal_mask
    try:
        for prompt, prompt_blocks in zip(prompts, blocks):
            ids = None
            if prompt_blocks is not None:
                ids = torch.full((1, len(prompt)), -1, dtype=torch.long)
                for i, (start, end) in enumerate(prompt_blocks):
                    ids[0, start : end + 1] = i
                ids = ids.to(device)
            current["ids"] = ids
            with torch.inference_mode():
                logits = model(input_ids=torch.tensor([prompt], device=device)).logits[
                    0, -1
                ]
            out.append(torch.log_softmax(logits.float(), -1)[:NUM_SCORED].cpu())
    finally:
        modeling_qwen3_5.create_causal_mask = original
    del model
    torch.accelerator.empty_cache()
    return out


def two_blocks(prompt: list[int], prefix_len: int) -> list[list[int]]:
    return [[0, prefix_len - 1], [prefix_len, len(prompt) - 1]]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("block_bidirectional"))
    build_tiny_checkpoint(path)
    return path


@pytest.fixture(scope="module")
def prompts():
    """Token prompts: a long shared prefix, then questions. q1 and q2 start
    their suffix with the same 40 tokens."""
    g = torch.Generator().manual_seed(1)

    def tokens(n: int) -> list[int]:
        return torch.randint(1000, 100000, (n,), generator=g).tolist()

    prefix = tokens(700)
    shared = tokens(40)
    questions = {
        "q0": prefix + tokens(90),
        "q1": prefix + shared + tokens(50),
        "q2": prefix + shared + tokens(50),
    }
    return prefix, questions


@pytest.mark.parametrize("backend", ["TRITON_ATTN", "FLEX_ATTENTION"])
def test_block_attention_matches_transformers(
    vllm_runner, checkpoint, prompts, backend, monkeypatch
):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    monkeypatch.delenv("VLLM_MODELING_OVERRIDES", raising=False)
    prefix, questions = prompts
    p = len(prefix)
    order = ["q0", "q1", "q2"]
    expected = dict(
        zip(
            order,
            hf_logprobs(
                checkpoint,
                [questions[q] for q in order],
                [two_blocks(questions[q], p) for q in order],
                "cuda",
            ),
        )
    )
    causal = dict(
        zip(
            order,
            hf_logprobs(checkpoint, [questions[q] for q in order], [None] * 3, "cuda"),
        )
    )
    for q in order:
        # The mask must matter, or the comparison proves nothing.
        assert (causal[q] - expected[q]).abs().max() > 5 * ATOL, q

    def check(got: torch.Tensor, name: str) -> None:
        error = (got - expected[name]).abs().max().item()
        gap = (causal[name] - expected[name]).abs().max().item()
        print(f"{backend} {name}: max error {error:.4f}, causal gap {gap:.4f}")
        assert error < ATOL, (name, error)
        assert error < MAX_ERROR_VS_MASK_GAP * gap, (name, error, gap)

    def params(blocks):
        return SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprob_token_ids=list(range(NUM_SCORED)),
            extra_args={"attention_blocks": blocks},
        )

    with vllm_runner(
        checkpoint,
        dtype="bfloat16",
        max_model_len=2048,
        # vllm_runner turns chunked prefill off, which Mamba prefix caching needs.
        enable_chunked_prefill=True,
        max_num_batched_tokens=2048,
        enable_prefix_caching=True,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        attention_config={"backend": backend},
        limit_mm_per_prompt={"image": 0, "video": 0},
    ) as runner:
        llm = runner.llm
        assert llm.llm_engine.model_config.request_attention_blocks

        def run(name: str, request_blocks=None):
            prompt = questions[name]
            out = llm.generate(
                TokensPrompt(prompt_token_ids=prompt),
                params(request_blocks or two_blocks(prompt, p)),
                use_tqdm=False,
            )[0]
            got = torch.tensor(
                [out.outputs[0].logprobs[0][t].logprob for t in range(NUM_SCORED)]
            )
            return got, out.num_cached_tokens or 0

        # Fresh prefill.
        got, cached = run("q0")
        assert cached == 0
        check(got, "q0")

        # The prefix block alone, then a question that hits inside it.
        llm.generate(
            TokensPrompt(prompt_token_ids=prefix),
            params([[0, p - 1]]),
            use_tqdm=False,
        )
        got, cached = run("q1")
        assert 0 < cached <= p, cached
        check(got, "q1")

        # A question whose suffix starts like q1's must not read q1's suffix.
        got, cached = run("q2")
        assert cached <= p, cached
        check(got, "q2")

        # The full prompt again: everything but the last token is cached.
        got, cached = run("q2")
        assert cached > p, cached
        check(got, "q2")


def test_unsupported_backend_fails_at_startup(vllm_runner, checkpoint, monkeypatch):
    monkeypatch.delenv("VLLM_MODELING_OVERRIDES", raising=False)
    # Build the engine in this process so the startup error reaches the test.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with (
        # On GPUs where FLASH_ATTN cannot serve the PrefixLM path at all, the
        # backend selector refuses it before the attention-blocks check does.
        pytest.raises(
            Exception,
            match="does not apply request attention blocks|"
            "is not valid for this configuration",
        ),
        vllm_runner(
            checkpoint,
            dtype="bfloat16",
            max_model_len=2048,
            # vllm_runner turns chunked prefill off, which Mamba prefix caching needs.
            enable_chunked_prefill=True,
            enforce_eager=True,
            gpu_memory_utilization=0.4,
            attention_config={"backend": "FLASH_ATTN"},
            limit_mm_per_prompt={"image": 0, "video": 0},
        ),
    ):
        pass


def test_causal_switch_serves_without_blocks(vllm_runner, checkpoint, monkeypatch):
    """VLLM_MODELING_OVERRIDES=ignore serves the stock causal model, which
    refuses request blocks."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    monkeypatch.setenv("VLLM_MODELING_OVERRIDES", "ignore")
    with vllm_runner(
        checkpoint,
        dtype="bfloat16",
        max_model_len=2048,
        # vllm_runner turns chunked prefill off, which Mamba prefix caching needs.
        enable_chunked_prefill=True,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        limit_mm_per_prompt={"image": 0, "video": 0},
    ) as runner:
        llm = runner.llm
        assert not llm.llm_engine.model_config.request_attention_blocks
        assert llm.llm_engine.model_config.architecture == (
            "Qwen3_5ForConditionalGeneration"
        )
        with pytest.raises(Exception, match="block_bidirectional"):
            llm.generate(
                TokensPrompt(prompt_token_ids=list(range(1000, 1100))),
                SamplingParams(
                    max_tokens=1, extra_args={"attention_blocks": [[0, 49], [50, 99]]}
                ),
                use_tqdm=False,
            )
