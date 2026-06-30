import importlib
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collector" / "layerwise" / "vllm"))

os.environ["LAYERWISE_SKIP_ENABLE"] = "0"
patch = importlib.import_module("vllm_layer_skip_patch")


class VllmLayerSkipPatchTests(unittest.TestCase):
    def test_transformer_block_uses_two_value_decoder_return(self):
        self.assertEqual(patch._RETURN_ARITY["TransformerBlock"], 2)

    def test_moe_noop_patches_target_layer_mlp(self):
        class FakeTensor:
            def dim(self):
                return 2

        class DummyMoe:
            def __init__(self):
                self.router = object()
                self.experts = object()

            def forward(self, x):
                return "original"

        class TransformerBlock:
            def __init__(self):
                self.mlp = DummyMoe()

            def named_modules(self):
                yield "", self
                yield "mlp", self.mlp

        layer = TransformerBlock()
        x = FakeTensor()

        match = patch._patch_first_moe_mlp(layer)

        self.assertEqual(match, ("mlp", "DummyMoe"))
        self.assertIs(layer.mlp.forward(x), x)


if __name__ == "__main__":
    unittest.main()
