"""Loader for compressed-tensors W8A8 INT8 per-channel checkpoints."""

import torch

from rtp_llm.config.quant_config import (
    CompressedW8A8Int8PerChannelQuantConfig,
    QuantizationConfig,
)
from rtp_llm.model_loader.per_channel_fp8_quant_weight import (
    PerChannelFp8Weight,
    _ckpt_base_matches_quant_exclude,
)
from rtp_llm.model_loader.weight_module import WeightModule


class CompressedW8A8Int8PerChannelWeight(PerChannelFp8Weight):
    """Load INT8 kernels and FP32 channel scales without re-quantization.

    The tensor mappings and TP/EP split rules are identical to the existing
    per-channel FP8 path. Only the checkpoint dtype and device post-processing
    differ: INT8 weights remain byte-exact and do not pass through FP8 layout
    conversion.
    """

    weight_dtype = torch.int8
    convert_fp8_weight_params = False

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, CompressedW8A8Int8PerChannelQuantConfig
        ):
            return False
        if src_weight_info.name not in cls.w8a8_weight_list:
            return False
        if quant_config.exclude_modules and hasattr(src_weight_info, "weights"):
            for ckpt_weight in src_weight_info.weights:
                base_name = ckpt_weight.name.rsplit(".", 1)[0]
                if _ckpt_base_matches_quant_exclude(
                    base_name, quant_config.exclude_modules
                ):
                    return False
        return True
