"""CPU tests for filtered Ascend expert recovery reloads."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


REPO_ROOT = Path(__file__).resolve().parents[4]
WEIGHT_UPDATER_PATH = (
    REPO_ROOT
    / "python/sglang/srt/model_executor/model_runner_components/weight_updater.py"
)
EPLB_MANAGER_PATH = REPO_ROOT / "python/sglang/srt/eplb/eplb_manager.py"
QWEN3_MOE_PATH = REPO_ROOT / "python/sglang/srt/models/qwen3_moe.py"
FUSED_MOE_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/moe/fused_moe_triton/layer.py"
)


def _load_functions(names, path, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    globals_dict = {
        "Any": Any,
        "Callable": Callable,
        "DefaultModelLoader": object,
        "Optional": Optional,
        "logger": Mock(),
        "torch": torch,
    }
    if namespace is not None:
        globals_dict.update(namespace)
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(path), "exec"), globals_dict)
    return {name: globals_dict[name] for name in names}


def _load_class_method(path, class_name, method_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    method.decorator_list = []
    namespace = {"Dict": dict, "List": list, "torch": torch}
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


class TestNpuPartialWeightReload(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        weight_functions = _load_functions(
            {
                "_should_skip_full_model_postprocess_for_filtered_npu_reload",
                "_load_weights_for_disk_update",
            },
            WEIGHT_UPDATER_PATH,
        )
        cls.should_skip = staticmethod(
            weight_functions[
                "_should_skip_full_model_postprocess_for_filtered_npu_reload"
            ]
        )
        cls.load_weights = staticmethod(
            weight_functions["_load_weights_for_disk_update"]
        )
        cls.validate_reload = staticmethod(
            _load_functions(
                {"_validate_missing_expert_disk_reload"}, EPLB_MANAGER_PATH
            )["_validate_missing_expert_disk_reload"]
        )
        cls.generate_filter = staticmethod(
            _load_class_method(
                QWEN3_MOE_PATH,
                "Qwen3MoeForCausalLM",
                "generate_weight_name_filter",
            )
        )
        cls.load_formatted_expert = staticmethod(
            _load_class_method(
                FUSED_MOE_PATH,
                "FusedMoE",
                "_load_npu_formatted_expert_weight",
            )
        )
        cls.finalize_formatted_expert = staticmethod(
            _load_class_method(
                FUSED_MOE_PATH,
                "FusedMoE",
                "finalize_npu_formatted_expert_reload",
            )
        )

    def test_only_filtered_unquantized_npu_reload_skips_postprocess(self):
        weight_filter = lambda _name: True

        self.assertTrue(
            self.should_skip("npu", SimpleNamespace(quantization=None), weight_filter)
        )
        self.assertFalse(
            self.should_skip("npu", SimpleNamespace(quantization=None), None)
        )
        self.assertFalse(
            self.should_skip("cuda", SimpleNamespace(quantization=None), weight_filter)
        )
        self.assertFalse(
            self.should_skip("npu", SimpleNamespace(quantization="fp8"), weight_filter)
        )

    def test_filtered_reload_writes_in_place_and_finalizes(self):
        loader = SimpleNamespace(load_weights_and_postprocess=Mock())
        model = SimpleNamespace(
            load_weights=Mock(), finalize_ft_filtered_weight_reload=Mock()
        )
        weights = object()

        result = self.load_weights(
            loader,
            model,
            weights,
            object(),
            skip_full_model_postprocess=True,
        )

        self.assertIs(result, model)
        model.load_weights.assert_called_once_with(weights)
        model.finalize_ft_filtered_weight_reload.assert_called_once_with()
        loader.load_weights_and_postprocess.assert_not_called()

    def test_normal_reload_keeps_full_model_postprocess(self):
        loader = SimpleNamespace(load_weights_and_postprocess=Mock())
        model = SimpleNamespace(load_weights=Mock())
        weights = object()
        target_device = object()

        result = self.load_weights(
            loader,
            model,
            weights,
            target_device,
            skip_full_model_postprocess=False,
        )

        self.assertIs(result, model)
        loader.load_weights_and_postprocess.assert_called_once_with(
            model, weights, target_device
        )
        model.load_weights.assert_not_called()

    def test_qwen_filter_records_checkpoint_pair_coverage(self):
        weight_filter = self.generate_filter(None, {2: [64, 65]})

        self.assertTrue(
            weight_filter("model.layers.2.mlp.experts.64.gate_proj.weight")
        )
        self.assertTrue(
            weight_filter("model.layers.2.mlp.experts.65.down_proj.weight")
        )
        self.assertFalse(
            weight_filter("model.layers.3.mlp.experts.64.gate_proj.weight")
        )

        stats = weight_filter._sglang_ft_reload_stats
        self.assertEqual(stats["expected_pairs"], {(2, 64), (2, 65)})
        self.assertEqual(stats["selected_pairs"], {(2, 64), (2, 65)})
        self.assertEqual(stats["selected_weight_names"], 2)

    def test_reload_validation_rejects_loader_failure(self):
        with self.assertRaisesRegex(RuntimeError, "disk read failed"):
            self.validate_reload(
                update_result=(False, "disk read failed"),
                weight_name_filter=None,
                tp_rank=2,
            )

    def test_reload_validation_rejects_unmatched_expert(self):
        weight_filter = self.generate_filter(None, {2: [64, 65]})
        weight_filter("model.layers.2.mlp.experts.64.gate_proj.weight")

        with self.assertRaisesRegex(
            RuntimeError, r"unmatched_pairs=\[\(2, 65\)\]"
        ):
            self.validate_reload(
                update_result=(True, "ok"),
                weight_name_filter=weight_filter,
                tp_rank=2,
            )

        weight_filter("model.layers.2.mlp.experts.65.down_proj.weight")
        self.validate_reload(
            update_result=(True, "ok"),
            weight_name_filter=weight_filter,
            tp_rank=2,
        )

    def test_formatted_w13_commits_only_after_both_halves(self):
        destination = torch.zeros((6, 4), dtype=torch.bfloat16)
        gate = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        up = gate + 20

        def load_w13(*, expert_data, shard_id, loaded_weight, **_kwargs):
            start = 0 if shard_id == "w1" else expert_data.shape[0] // 2
            expert_data[start : start + loaded_weight.shape[0]].copy_(loaded_weight)

        owner = SimpleNamespace(
            _load_w13=load_w13,
            _load_w2=Mock(),
            _commit_npu_formatted_expert_reload=Mock(),
        )
        kwargs = {
            "param": SimpleNamespace(),
            "expert_data": destination,
            "expert_id": 7,
            "shard_dim": 0,
            "tp_rank": 0,
        }

        self.load_formatted_expert(
            owner, shard_id="w1", loaded_weight=gate, **kwargs
        )
        owner._commit_npu_formatted_expert_reload.assert_not_called()
        self.load_formatted_expert(
            owner, shard_id="w3", loaded_weight=up, **kwargs
        )

        owner._commit_npu_formatted_expert_reload.assert_called_once()
        committed_destination, committed_staging = (
            owner._commit_npu_formatted_expert_reload.call_args.args
        )
        self.assertIs(committed_destination, destination)
        self.assertTrue(torch.equal(committed_staging[:3], gate))
        self.assertTrue(torch.equal(committed_staging[3:], up))
        self.assertEqual(owner._npu_formatted_expert_reload_pending, {})

    def test_formatted_reload_rejects_incomplete_w13(self):
        owner = SimpleNamespace(
            _load_w13=lambda **_kwargs: None,
            _load_w2=Mock(),
            _commit_npu_formatted_expert_reload=Mock(),
        )
        self.load_formatted_expert(
            owner,
            param=SimpleNamespace(),
            expert_data=torch.zeros((6, 4), dtype=torch.bfloat16),
            expert_id=3,
            shard_dim=0,
            shard_id="w1",
            loaded_weight=torch.zeros((3, 4), dtype=torch.bfloat16),
            tp_rank=0,
        )

        with self.assertRaisesRegex(RuntimeError, "Incomplete NPU formatted"):
            self.finalize_formatted_expert(owner)


if __name__ == "__main__":
    unittest.main()
