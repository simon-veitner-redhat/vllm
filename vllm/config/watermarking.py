# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from vllm.config.utils import config

WatermarkingAlgorithm = Literal["gumbel", "dual_key_gumbel"]
WatermarkPRFName = Literal["philox"]


def derive_watermark_key(key: int, domain: bytes) -> int:
    digest = hashlib.sha256(domain + key.to_bytes(8, "big")).digest()
    return int.from_bytes(digest[:8], "big")


@config
class WatermarkConfig:
    """Configuration for text watermark generation."""

    key: int = Field(ge=0, repr=False, exclude=True)
    """Secret key used to watermark generated text."""
    algorithm: WatermarkingAlgorithm = "gumbel"
    """Algorithm used to watermark generated text."""
    alpha: float = Field(default=0.1, ge=0, le=1)
    """Probability of selecting key B for dual-key watermarking."""
    context_width: int = Field(default=4, ge=1)
    """Number of prior output tokens used by the watermark PRF."""
    prf: WatermarkPRFName = "philox"
    """Pseudorandom function used by the watermarking algorithm."""
    allow_target_only_watermarking: bool = False
    """Allow speculative decoding without watermarking draft tokens."""

    @model_validator(mode="after")
    def validate_key(self) -> Self:
        if self.key > 2**64 - 1:
            raise ValueError("philox keys must fit in 64 bits")
        return self
