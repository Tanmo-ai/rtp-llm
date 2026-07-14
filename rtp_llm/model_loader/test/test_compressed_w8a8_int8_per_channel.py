import json
import os
import tempfile
import unittest

import torch

from rtp_llm.config.quant_config import (
    CompressedW8A8Int8PerChannelQuantConfig,
    Fp8PerChannelCompressedQuantConfig,
    QuantizationConfig,
)
from rtp_llm.model_loader.compressed_w8a8_int8_per_channel_weight import (
    CompressedW8A8Int8PerChannelWeight,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity


def _compressed_group(weight_type="int", symmetric=True):
    return {
        "weights": {
            "num_bits": 8,
            "type": weight_type,
            "strategy": "channel",
            "symmetric": symmetric,
            "dynamic": False,
        },
        "input_activations": {
            "num_bits": 8,
            "type": weight_type,
            "strategy": "token" if weight_type == "int" else "tensor",
            "symmetric": symmetric,
            "dynamic": True,
        },
        "targets": ["Linear"],
    }


class CompressedW8A8ConfigTest(unittest.TestCase):
    def _load_config(self, quantization_config):
        with tempfile.TemporaryDirectory() as model_dir:
            with open(os.path.join(model_dir, "config.json"), "w") as output:
                json.dump({"quantization_config": quantization_config}, output)
            return QuantizationConfig.load_from_ckpt(model_dir)

    def test_parses_named_w8a8_group_and_ignore(self):
        config = self._load_config(
            {
                "quant_method": "compressed-tensors",
                "config_groups": {"W8A8": _compressed_group()},
                "ignore": ["model.layers.0.mlp.gate"],
            }
        )
        self.assertIsInstance(config, CompressedW8A8Int8PerChannelQuantConfig)
        self.assertEqual(config.get_algo(), "w8a8_int8_per_channel")
        self.assertEqual(config.bits, 8)
        self.assertEqual(config.group_size(), 0)
        self.assertEqual(config.exclude_modules, {"model.layers.0.mlp.gate"})

    def test_rejects_asymmetric_w8a8(self):
        with self.assertRaisesRegex(ValueError, "asymmetric INT8"):
            self._load_config(
                {
                    "quant_method": "compressed-tensors",
                    "config_groups": {"W8A8": _compressed_group(symmetric=False)},
                }
            )

    def test_fp8_group_name_is_not_hardcoded(self):
        config = self._load_config(
            {
                "quant_method": "compressed-tensors",
                "config_groups": {"FLOAT8": _compressed_group("float")},
            }
        )
        self.assertIsInstance(config, Fp8PerChannelCompressedQuantConfig)


class CompressedW8A8WeightTest(unittest.TestCase):
    def _source(self):
        return AtomicWeight(
            W.attn_gate_w,
            [CkptWeightInfo("model.layers.{i}.self_attn.gate.weight", identity)],
        )

    def test_support_and_checkpoint_tensor_dtypes(self):
        config = CompressedW8A8Int8PerChannelQuantConfig()
        source = self._source()
        self.assertTrue(CompressedW8A8Int8PerChannelWeight.support(config, source))

        weight = WeightModule.create(source, config)
        self.assertIsInstance(weight, CompressedW8A8Int8PerChannelWeight)
        self.assertEqual(weight.kernel.data_type, torch.int8)
        self.assertEqual(weight.scale.data_type, torch.float32)
        self.assertEqual(
            weight.kernel.weights[0].name,
            "model.layers.{i}.self_attn.gate.weight",
        )
        self.assertEqual(
            weight.scale.weights[0].name,
            "model.layers.{i}.self_attn.gate.weight_scale",
        )

    def test_ignore_disables_quantized_loader(self):
        config = CompressedW8A8Int8PerChannelQuantConfig(
            ignore_patterns=["model.layers.7.self_attn.gate"]
        )
        self.assertFalse(
            CompressedW8A8Int8PerChannelWeight.support(config, self._source())
        )


if __name__ == "__main__":
    unittest.main()
