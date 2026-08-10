import inspect
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
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.compressed_w8a8_int8_per_channel_weight import (
    CompressedW8A8Int8PerChannelWeight,
)
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.per_channel_fp8_quant_weight import PerChannelFp8Weight
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.ops import QuantAlgo
from rtp_llm.utils.database import BaseDatabase
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


class _RecordingDevice:
    """Stand-in for the exported device, the one external dependency here.

    Records how often the FP8 layout conversion is invoked and returns tensors
    that cannot be confused with the inputs, so the gate can be asserted on
    behaviour instead of on the class attribute that drives it.
    """

    def __init__(self):
        self.convert_calls = 0
        self.sentinel_kernel = torch.full((1, 1), 7, dtype=torch.int32)
        self.sentinel_scale = torch.full((1, 1), 9, dtype=torch.int32)

    def maybe_rewrite_weight_by_key(self, key, tensor):
        return tensor

    def convert_fp8_weight_params(self, kernel, scale):
        self.convert_calls += 1
        return self.sentinel_kernel, self.sentinel_scale


def _load_config(exported_device):
    return LoadConfig(
        database=BaseDatabase(),
        num_layers=1,
        hidden_size=2,
        head_num=1,
        head_num_kv=1,
        size_per_head=2,
        moe_pure_tp_mode=False,
        align_size=1,
        moe_align_size=1,
        moe_layer_index=[],
        moe_n_group=1,
        expert_num=0,
        enable_eplb=False,
        phy_exp_num=0,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        dp_size=1,
        dp_rank=0,
        lm_head_tp_size=1,
        lm_head_tp_rank=0,
        ffn_tp_size=1,
        ffn_tp_rank=0,
        num_nodes=1,
        exported_device=exported_device,
    )


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

    def test_historical_group_0_name_still_parses(self):
        config = self._load_config(
            {
                "quant_method": "compressed-tensors",
                "config_groups": {"group_0": _compressed_group("float")},
            }
        )
        self.assertIsInstance(config, Fp8PerChannelCompressedQuantConfig)

    def test_group_0_wins_over_a_conflicting_sibling(self):
        # Pre-existing behaviour: a checkpoint carrying group_0 keeps loading as
        # group_0 even when another group declares a different scheme.
        config = self._load_config(
            {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": _compressed_group("float"),
                    "W8A8": _compressed_group("int"),
                },
            }
        )
        self.assertIsInstance(config, Fp8PerChannelCompressedQuantConfig)

    def test_named_groups_with_one_scheme_are_merged(self):
        config = self._load_config(
            {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "W8A8": _compressed_group("int"),
                    "W8A8_MOE": _compressed_group("int"),
                },
            }
        )
        self.assertIsInstance(config, CompressedW8A8Int8PerChannelQuantConfig)

    def test_named_groups_with_conflicting_schemes_raise(self):
        with self.assertRaisesRegex(ValueError, "conflicting schemes") as caught:
            self._load_config(
                {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "W8A8": _compressed_group("int"),
                        "FLOAT8": _compressed_group("float"),
                    },
                }
            )
        message = str(caught.exception)
        self.assertIn("W8A8", message)
        self.assertIn("FLOAT8", message)

    def test_empty_config_groups_raise(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self._load_config(
                {"quant_method": "compressed-tensors", "config_groups": {}}
            )

    def test_null_ignore_is_normalized(self):
        config = self._load_config(
            {
                "quant_method": "compressed-tensors",
                "config_groups": {"W8A8": _compressed_group()},
                "ignore": None,
            }
        )
        self.assertIsInstance(config, CompressedW8A8Int8PerChannelQuantConfig)
        self.assertEqual(config.exclude_modules, set())

    def test_exclude_key_is_read_when_ignore_is_absent(self):
        config = self._load_config(
            {
                "quant_method": "compressed-tensors",
                "config_groups": {"W8A8": _compressed_group()},
                "exclude": ["lm_head"],
            }
        )
        self.assertEqual(config.exclude_modules, {"lm_head"})

    def test_weight_only_int8_is_not_read_as_w8a8(self):
        group = _compressed_group()
        group["input_activations"] = None
        try:
            config = self._load_config(
                {
                    "quant_method": "compressed-tensors",
                    "config_groups": {"W8A8": group},
                }
            )
        except Exception:
            # Falling through to an unsupported method is acceptable; what must
            # not happen is being silently read as dynamic per-token W8A8.
            return
        self.assertNotIsInstance(config, CompressedW8A8Int8PerChannelQuantConfig)

    def test_kv_cache_dtypes_match_the_sibling_w4a8_scheme(self):
        config = CompressedW8A8Int8PerChannelQuantConfig()
        self.assertIn(torch.float8_e4m3fn, config.get_supported_kv_cache_dtypes())

    def test_moe_activation_spec_requires_int8_per_token(self):
        config = CompressedW8A8Int8PerChannelQuantConfig()
        self.assertEqual(config.get_moe_activation_quant_spec(), (torch.int8, True))
        self.assertIsNone(
            Fp8PerChannelCompressedQuantConfig(
                bits=8, is_quanted=True
            ).get_moe_activation_quant_spec()
        )


class CompressedW8A8WeightTest(unittest.TestCase):
    def _source(self, name=W.attn_gate_w, ckpt="model.layers.{i}.self_attn.gate"):
        return AtomicWeight(name, [CkptWeightInfo(ckpt + ".weight", identity)])

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

    def test_non_matching_ignore_keeps_quantized_loader(self):
        # The other direction of the {i} match: a non-empty ignore list that does
        # not cover this weight must leave it quantized.
        config = CompressedW8A8Int8PerChannelQuantConfig(
            ignore_patterns=["model.layers.7.mlp.down_proj"]
        )
        self.assertTrue(
            CompressedW8A8Int8PerChannelWeight.support(config, self._source())
        )

    def test_int8_dtype_applies_to_every_templated_weight(self):
        config = CompressedW8A8Int8PerChannelQuantConfig()
        attn_config = AttnConfig(
            hidden_size=4, size_per_head=2, head_num=2, head_num_kv=2
        )
        cases = [
            (W.attn_gate_w, "model.layers.{i}.self_attn.gate", AtomicWeight, {}),
            (
                W.attn_o_w,
                "model.layers.{i}.self_attn.o_proj",
                AttnAtomicWeight,
                {"config": attn_config},
            ),
        ]
        for name, ckpt, weight_cls, kwargs in cases:
            with self.subTest(weight=name):
                source = weight_cls(
                    name, [CkptWeightInfo(ckpt + ".weight", identity)], **kwargs
                )
                weight = WeightModule.create(source, config)
                self.assertEqual(weight.kernel.data_type, torch.int8)
                self.assertEqual(weight.scale.data_type, torch.float32)
                self.assertTrue(weight.kernel.weights[0].name.endswith(".weight"))
                self.assertTrue(weight.scale.weights[0].name.endswith(".weight_scale"))


class PerChannelFp8PostprocessTest(unittest.TestCase):
    """Assert the _postprocess behaviour, not the attribute that drives it."""

    def _run_postprocess(self, quant_config, kernel_dtype):
        source = AtomicWeight(
            W.attn_gate_w,
            [CkptWeightInfo("model.layers.{i}.self_attn.gate.weight", identity)],
        )
        weight = WeightModule.create(source, quant_config)
        device = _RecordingDevice()
        tensors = {
            weight.kernel.name: torch.arange(4, dtype=torch.int32)
            .reshape(2, 2)
            .to(kernel_dtype),
            weight.scale.name: torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        }
        processed = weight._postprocess(tensors, "cpu", _load_config(device))
        return weight, device, processed

    def test_int8_skips_fp8_conversion_and_stays_byte_exact(self):
        weight, device, processed = self._run_postprocess(
            CompressedW8A8Int8PerChannelQuantConfig(), torch.int8
        )
        self.assertEqual(device.convert_calls, 0)
        kernel = processed[weight.kernel.name]
        self.assertEqual(kernel.dtype, torch.int8)
        torch.testing.assert_close(
            kernel.to(torch.int32),
            torch.arange(4, dtype=torch.int32).reshape(2, 2),
            rtol=0,
            atol=0,
        )
        self.assertEqual(processed[weight.scale.name].dtype, torch.float32)

    def test_fp8_still_runs_the_device_conversion(self):
        weight, device, processed = self._run_postprocess(
            Fp8PerChannelCompressedQuantConfig(bits=8, is_quanted=True),
            torch.float8_e4m3fn,
        )
        self.assertEqual(device.convert_calls, 1)
        torch.testing.assert_close(
            processed[weight.kernel.name], device.sentinel_kernel, rtol=0, atol=0
        )
        torch.testing.assert_close(
            processed[weight.scale.name], device.sentinel_scale, rtol=0, atol=0
        )

    def test_no_hardcoded_fp8_dtype_survives_in_the_template(self):
        # Guards the ten call sites the templating replaced: a missed one would
        # keep loading FP8 kernels for an INT8 checkpoint.
        source = inspect.getsource(PerChannelFp8Weight)
        self.assertEqual(source.count("data_type=torch.float8_e4m3fn"), 0)
        self.assertGreaterEqual(source.count("self.weight_dtype"), 10)


class QuantAlgoBindingTest(unittest.TestCase):
    """Pin the python -> C++ string contract the loader relies on."""

    def test_algo_string_round_trips_to_w8a8_int8_ptpc(self):
        config = CompressedW8A8Int8PerChannelQuantConfig()
        algo = QuantAlgo()
        algo.setQuantAlgo(config.get_algo().lower(), config.bits, config.group_size())
        self.assertTrue(algo.isW8a8Int8PTPC())
        self.assertTrue(algo.isQuant())
        self.assertFalse(algo.isGroupwise())
        self.assertEqual(algo.getWeightBits(), 8)
        self.assertEqual(algo.getActivationBits(), 8)
        self.assertEqual(algo.getGroupSize(), 0)
        # The numeric value matters for mixed-version deployments.
        self.assertEqual(int(algo.getQuantMethod()), 12)
        self.assertIn("W8A8INT8PTPC", str(algo.getQuantMethod()))


if __name__ == "__main__":
    unittest.main()
