# Handoff: fixes for vLLM PR #56122 (dual-key Gumbel watermarking with speculative decoding)

Written 2026-09-11 by Simon Veitner. This document is self-contained: it says what the PR does,
what was found wrong, where the fixes are, and how to check them.

## Context in one paragraph

vLLM's Gumbel-max watermark (PR #54053) adds keyed pseudo-random noise to the logits before an
argmax; a detector that knows the key recomputes the noise from the last four tokens and scores
each output token. PR #56122 makes this work with speculative decoding by deriving two keys from
the configured one: draft tokens are marked with key A, the token the target model supplies after
a rejection (or the bonus token when every draft is accepted) is marked with key B inside the
rejection sampler's kernel. The detector scores both keys. A second PR, #56233, adds context
deduplication to the same sampler and lands first. So the fixes below are built on PR #56122
rebased onto PR #56233, not on the PR head. The review behind them found the PR's core right and three defects
that block the merge.

## Where the fixes are

Fork: `git@github.com:simon-veitner-redhat/vllm.git`. Six branches, all under `patches/`.

```bash
git fetch git@github.com:simon-veitner-redhat/vllm.git 'refs/heads/patches/*:refs/remotes/simon/patches/*'
```

| branch | commit | what it is |
| --- | --- | --- |
| `patches/rebase-on-56233` | `b52af87004` | PR #56122 head `ced66e14c3` rebased onto PR #56233 head `42908fd643`. The base of everything below. |
| `patches/dual-key-sample-signature` | `74a7c8592a` | fixes a `TypeError` that appears after the rebase |
| `patches/dual-key-routing-cache` | `4b5c170666` | removes a per-step host stall, TPOT 4.32 ms to 2.63 ms |
| `patches/config-reject-gumbel-specdec` | `1df37d0614` | rejects an unsupported config at validation instead of in the worker |
| `patches/docs-dual-key` | `98f34ffcb0` | documents the dual-key detector, new fields and limits |
| `patch/minor` | `3673087e0c` | three lint and warning fixes found by the validation pass, one commit on `1db8717152` |
| `patch/warmup` | `1f23fc2da6` | warms the watermark sampler kernel for every specialization the engine launches; fixes a base-feature gap; optional for the merge |
| `patches/all` | `1db8717152` + this file | the four fixes cherry-picked onto the base, in that order, verified together; commits after `1db8717152` only add this document |

### How to pull them in

```bash
git remote add simon git@github.com:simon-veitner-redhat/vllm.git
git fetch simon 'refs/heads/patches/*:refs/remotes/simon/patches/*'
```

Option A, take everything: point the PR branch at `patches/all`, which is the PR rebased onto
PR #56233 plus the four fixes, and force-push it.

```bash
git checkout <pr-branch>
git reset --hard simon/patches/all
git push --force-with-lease origin <pr-branch>
```

Option B, pick and choose: rebase the PR branch onto PR #56233 yourself (or reset to
`simon/patches/rebase-on-56233`, which is that rebase with the nine conflicts resolved), then
cherry-pick the fixes you want, in this order.

```bash
git cherry-pick 74a7c8592a   # dual-key sample signature
git cherry-pick 4b5c170666   # routing cache (conflicts if picked before the previous one)
git cherry-pick 1df37d0614   # config-time rejection
git cherry-pick 98f34ffcb0   # docs
```

To see what a branch changes before taking it: `git diff simon/patches/rebase-on-56233 simon/patches/<name>`.
To check later whether these branches moved: `git fetch simon` and compare the shas in the table.

Each fix branch is exactly one commit on top of `b52af87004`. The commit message body says the
same as the section below. Cherry-pick the four in the table's order; the first two edit the same
method and conflict in the other order. `patches/all` already holds the resolution if you want all
four at once.

Which ones are needed: the first three fix defects and are needed for the merge. The docs branch
is needed for the Detection section (without it the docs point at the wrong detector class); the
rest of it is useful but optional.

## The rebase, `patches/rebase-on-56233`

Nine conflicting hunks in six files, all mechanical except one: both PRs replace the same
paragraph under "Gumbel-max" in `docs/features/watermarking.md`. The dedup PR replaces it with
the deduplication docs, this PR with "the naive single-key implementation is not
production-ready". Since deduplication is the mitigation for the loop that sentence describes,
the resolution keeps the dedup text and appends the caveat with one sentence saying deduplication
mitigates it. The other eight hunks are import blocks, adjacent insertions, and a constructor
call that both PRs extend; both sides were kept. Git also auto-merged one test wrongly with no
conflict marker (`test_watermarker_contract`: this PR widened its parametrize list, the dedup PR
changed its call signature), and that is how the first defect below was found.

## The four fixes

### 1. `patches/dual-key-sample-signature`, commit `74a7c8592a`

Issue. PR #56233 turns `Watermarker.sample` into a template method
`sample(logits, contexts, random_sampler=None, skip_mask=None)`, and the GPU sampler always
passes `skip_mask=` as a keyword. `DualKeyGumbelWatermarker.sample` still had the old signature
`sample(logits, contexts, random_sample)`. Result on the rebased tree: every watermarked step of
`dual_key_gumbel` without speculative decoding raises `TypeError`. With speculative decoding the
bug is invisible, because the model runner replaces the dual-key object with a plain key-B
watermarker before sampling.

Change. The override takes the base signature and forwards `random_sampler` and `skip_mask` to
both keyed draws in all three alpha branches (alpha 0, alpha 1, mixed). `_sample_random` builds
the `RandomSampler` unconditionally, as the PR head did, so the routing draw always has one.
`test_watermarker_contract` is restored to pass a sampler. A new test checks that rows in the
skip mask get the sampler's draw for alpha 0, 0.25 and 1. Files: `vllm/v1/watermarking/gumbel.py`,
`vllm/v1/watermarking/gpu_sampler.py`, `tests/watermarking/test_watermarking.py` (+55/-17).

Verified. 52 CPU tests in `tests/watermarking`, 18 config tests, 142 GPU tests. Ordinary dual-key
generation, the exact path that raised, now runs and detects (dual-key p-values 3.4e-71 and
2.8e-56 on two prompts, control prompt 0.60). Under MTP with deduplication on: p from 3.2e-19 to
1.8e-51 at mean acceptance length 2.10, control 0.30. Bit-identical to the PR head's behaviour for
skipped and unskipped rows across alpha 0, 0.1, 0.5 and 1 on a CPU comparison.

### 2. `patches/dual-key-routing-cache`, commit `4b5c170666`

Issue. Without speculative decoding, `dual_key_gumbel` routes each token to key A or key B by a
categorical draw over `[1 - alpha, alpha]`. `sample` rebuilt that two-element tensor from a
Python list on every call. On CUDA that is an 8-byte pageable host-to-device copy, and it blocks
the host on `cudaStreamSynchronize` once per decode step. Serving measurement (Qwen3.5-2B, one
H200, 128-token prompts, 512 output tokens, concurrency 8): mean time per output token 2.34 ms
unwatermarked, 2.50 ms single-key, 4.32 ms dual-key, so +84% over no watermark and +72% over
single-key. An nsys trace of 128 decode steps showed 128 synchronizations totalling 153 ms and
128 eight-byte copies in the dual-key run, none in the single-key run; the second keyed argmax
itself added 1.25 ms of GPU time.

Change. The log-routing row is built once per `torch.device` and expanded per call. Same
expression, same categorical draw, so outputs are bit-identical. A test asserts the row is cached
and fails with `AttributeError` when the fix is reverted. Files: `gumbel.py`,
`test_watermarking.py` (+36/-6).

Verified. Same serving cell with the fix: mean TPOT 2.63 ms (2.79 ms on `patches/all`, where the
three GPU checks ran concurrently on one host), output throughput 2728 tok/s against 1764 without
the fix, 2998 single-key, 3264 unwatermarked. CPU and GPU test suites pass. Single runs; the
unwatermarked and single-key baselines are earlier single runs on the PR head.

### 3. `patches/config-reject-gumbel-specdec`, commit `1df37d0614`

Issue. Plain `gumbel` does not support speculative decoding unless
`allow_target_only_watermarking` is set. The PR moved that capability check out of
`VllmConfig` into the speculator constructor, which runs inside the EngineCore worker. So
`--watermark-config '{"algorithm":"gumbel","key":42}'` with a valid MTP config passes validation
and fails about 30 s later in the child process; the caller sees only
`RuntimeError: Engine core initialization failed`. The test that was supposed to cover the flag
passed with the flag off.

Change. `_check_watermarking_unsupported` in `vllm/config/vllm.py` builds the configured
watermarker (a function-local import, to keep the config layer from importing the runtime at
module scope; construction is two SHA-256 digests, no tensors) and raises `ValueError` naming the
algorithm and the `allow_target_only_watermarking` escape hatch unless the watermarker supports
speculative decoding. It also logs once when `dual_key_gumbel` is configured with a non-default
`alpha` under speculative decoding, because the protocol picks the key by role and `alpha` is
unused there. Four tests added in `tests/test_config.py`, including the negative case. Files:
`vllm/config/vllm.py`, `tests/test_config.py` (+90/-1).

Verified. 22 config tests pass (18 before); the two new positive tests fail on the unpatched
tree. On GPU, building the engine with the bad config raises `ValueError` in the calling process
after 1.3 s; target-only mode still starts and detects, with no warning at the default alpha.

### 4. `patches/docs-dual-key`, commit `98f34ffcb0`

Issue. The Detection section of `docs/features/watermarking.md` showed only
`GumbelWatermarkDetector`. A reader who generates with `dual_key_gumbel` and follows that section
scores the output with a single-key detector on the master key, which marks nothing, and gets a
null detection with no error. The example detection server had the same class hardcoded.
`alpha`, `allow_target_only_watermarking` and the four speculative constraints were undocumented.

Change. The Detection section shows `DualKeyGumbelWatermarkDetector` and says the detector class
must match the algorithm. The example server takes `--algorithm {gumbel,dual_key_gumbel}`. The
Configuration section documents both fields. The Speculative decoding section lists the four
constraints (`draft_sample_method="probabilistic"`, `rejection_sample_method="standard"`, no
`parallel_drafting` outside dspark, method in dspark/eagle/eagle3/mtp), states the measured
target-only dilution as an example (detection at p <= 0.01 fell from 74.7% to 42.0% over 300
GSM8K generations, MTP with two speculative tokens on a 2B model, acceptance length 2.58), says
`alpha` is unused under speculative decoding, and says `deduplicate_contexts` does not apply to
draft, recovery or bonus tokens. Files: the docs page and
`examples/basic/online_serving/watermark_detection_server.py` (+69/-11).

Verified. The Python snippets and the example compile, `--help` lists `--algorithm`, markdownlint
reports no issues, ruff clean.

### 5. `patch/minor`, commit `3673087e0c`

Issue. Validation of `patches/all` found three small defects. `DraftWatermarker.sample` passed a
lambda where the new template method types `RandomSampler | None`, so mypy and the pre-commit
hook fail; the argument was dead because the draft role is always the plain key-A watermarker.
`watermarker.py` kept an unused `TypeAlias` import from the rebase, so ruff fails. The
repetition-loop warning in `WatermarkConfig` was gated on `algorithm == "gumbel"` and silently
exempted `dual_key_gumbel`, although both of its key streams reuse the keyed vector on a repeated
context.

Change. Drop the lambda, drop the import, widen the gate to both algorithms and remove
"Single-key" from the message. The warning test is parametrized over both algorithms, and the
draft-sampler unit test's stub now matches the real `sample` signature instead of locking in the
old call. Files: `spec_decode.py`, `watermarker.py`, `config/watermarking.py`,
`test_watermarking.py` (+12/-11).

Verified. 60 CPU tests, 22 config tests, 145 GPU tests, ruff and mypy clean on the changed files,
both end-to-end smoke runs detect every watermarked output with negative controls.

Not in this branch: the first watermarked request at temperature 1.0 still JIT-compiles
`_philox_gumbel_kernel` once per engine, because the sampler warmup only exercises the fp32 logits
path. That is in the base feature (PR #54053), harmless under the default JIT monitor mode and
fatal under `--jit-monitor-mode=error`; it is left for a separate follow-up.

### 6. `patch/warmup`, commit `1f23fc2da6`

Issue. The first watermarked request at temperature 1.0 JIT-compiles `_philox_gumbel_kernel`
during inference, once per engine, and kills the engine under `--jit-monitor-mode=error`. The
sampler warmup only ever hands the kernel fp32 logits, because its warmup requests carry
penalties and top-k that force a logits copy; a plain request passes model-dtype logits, a
different Triton specialization. This is in the base feature (PR #54053); PR #56122 only doubles
the compile by adding key B. Optional for the merge.

Change. A new `vllm/model_executor/warmup/watermark_sample_warmup.py`, hooked in
`kernel_warmup.py` on the line after the rejection-sampler warmup, under the same
`enable_jit_warmup` gate. It reads the runner's `GPUWatermarkSampler` after model load and calls
`philox_gumbel_sample` once per configured key, in model dtype and fp32, with a skip mask and,
when `deduplicate_contexts` is `"none"`, also without one, forwarding `use_fp64_gumbel` only on
the masked launch as production does. It returns early without a watermark config, on a
non-CUDA platform, or on a rank whose runner has no watermark sampler; failures log a warning
and return. Plus a CPU test that pins the launch set for single-key, dual-key, dual-key under a
speculative config and both dedup settings. Files: the module (150 lines), four lines in
`kernel_warmup.py`, the test.

Wiring. This is the convention every sampler-side warmup in the tree uses, including the
rejection-sampler warmup this PR extends and the modules that landed in the last two weeks.
The tree also has a shared JIT warmup registry, documented as the target for new warmable
kernels, but no registry activation covers sampler construction in the V2 runner today, and
moving this kernel there means rewriting its runtime launch path. That migration is recorded
as a follow-up for the Philox and rejection-sampler kernels together (`review/warmup/WIRING.md`).

Verified. CPU: 17 warmup tests, 70 watermarking tests, 22 config tests, ruff and mypy clean.
GPU, end to end at `1f23fc2da6` under `jit_monitor_mode=error` (`review/warmup/logs/warmup-jit-probe.log`):
ordinary dual-key generation, single-key with `deduplicate_contexts="none"` and one opted-out
request, and dual-key with MTP all complete their first request, with no JIT compilation
logged after the monitor activated; at head the same probe raised `EngineDeadError`. Engine
init took 17.7 to 18.1 s against 18.0 to 18.5 s before, so the warmup adds no measurable
startup cost. The speculative smoke run detects both keys on every output.

### `patches/all`, commit `1db8717152`

The four commits in order. Only the routing-cache cherry-pick conflicted (the routing block of
`sample` and adjacent test insertions); resolved by keeping the signature branch's calls and
taking the cache's two routing lines. Checks on the combined tree: 148 GPU tests, 58 CPU tests
with the GPU-gated ones deselected, 22 config tests, arg-utils test, ruff clean; ordinary and
speculative end-to-end runs detect every watermarked output at both dedup scopes with negative
controls; the serving cell above at 2.79 ms.

## How to check a branch yourself

```bash
# CPU, any machine with the venv
pytest tests/watermarking tests/model_executor/test_spec_decode_rejection_warmup.py -q -p no:cacheprovider \
  --deselect tests/watermarking/test_prf.py::test_philox_accelerator_matches_cpu \
  --deselect tests/watermarking/test_gumbel.py::test_fused_watermarker_matches_cpu \
  --deselect tests/watermarking/test_gumbel.py::test_fused_watermarker_handles_nan_logits \
  --deselect tests/watermarking/test_gumbel.py::test_fused_watermarker_handles_noncontiguous_inputs \
  --deselect tests/watermarking/test_gumbel.py::test_philox_gumbel_sample_skip_mask_matches_separate_samplers \
  --deselect tests/watermarking/test_watermarking.py::test_repeated_context_mask_max_history_accelerator_parity \
  --deselect tests/watermarking/test_watermarking.py::test_repeated_context_mask_partial_context_accelerator_parity \
  --deselect tests/watermarking/test_watermarking.py::test_repeated_context_mask_unaligned_max_history_accelerator_parity \
  --deselect tests/watermarking/test_watermarking.py::test_repeated_context_mask_accelerator_parity \
  --deselect tests/watermarking/test_watermarking.py::test_repeated_context_mask_scans_multiple_blocks \
  --deselect tests/watermarking/test_watermarking.py::test_repeated_context_mask_last_request_at_capacity \
  --deselect tests/watermarking/test_watermarking.py::test_gpu_sampler_uses_fused_gumbel_for_repeated_contexts
pytest tests/test_config.py -q -p no:cacheprovider -k "watermark or dual_key or target_only or dedup"

# GPU
pytest tests/watermarking tests/v1/spec_decode/test_rejection_sampler_utils.py \
  tests/model_executor/test_spec_decode_rejection_warmup.py -q -p no:cacheprovider
```

Expected on `patches/all`: 58 passed / 37 deselected, 22 passed, 148 passed.

End to end, the smallest check is ordinary generation with
`watermark_config={"algorithm":"dual_key_gumbel","key":42,"alpha":0.1}` on Qwen/Qwen3.5-2B
and detection with `DualKeyGumbelWatermarkDetector(key=42)`; before fix 1 this raises on the
rebased tree. For speculative decoding add
`speculative_config={"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic"}`.
Note for Qwen3.5-2B: build `vllm.LLM` under `if __name__ == "__main__"`, because its multimodal
warmup forces the spawn start method.
