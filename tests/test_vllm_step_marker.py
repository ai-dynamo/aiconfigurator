import importlib
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collector" / "layerwise" / "vllm"))
os.environ["LAYERWISE_STEP_MARKER"] = "0"
sys.modules.setdefault("torch", mock.Mock())
sys.modules.setdefault("torch.cuda", mock.Mock())
sys.modules.setdefault("torch.cuda.nvtx", mock.Mock())
marker = importlib.import_module("vllm_step_marker")


def _scheduler_output(*, req_ids, computed, scheduled, new_reqs=(), scheduled_req_ids=None):
    scheduled_ids = list(req_ids if scheduled_req_ids is None else scheduled_req_ids)
    return SimpleNamespace(
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=list(req_ids),
            num_computed_tokens=list(computed),
        ),
        num_scheduled_tokens=dict(zip(scheduled_ids, scheduled, strict=False)),
    )


class VllmStepMarkerTests(unittest.TestCase):
    def test_marked_step_records_execute_model_wall_time(self):
        events = []

        def fake_write_progress(event, **kwargs):
            events.append((event, kwargs))

        def fake_orig(runner, scheduler_output, intermediate_tensors):
            return "ok"

        with (
            mock.patch.object(marker, "_write_progress", side_effect=fake_write_progress),
            mock.patch.object(marker.torch.cuda, "synchronize") as sync,
            mock.patch.object(marker.time, "perf_counter", side_effect=[10.0, 10.017]),
        ):
            result = marker._run_marked_step(
                fake_orig,
                SimpleNamespace(),
                SimpleNamespace(),
                None,
                step=400,
                batch_size=1,
                past_kv=3696,
                control={
                    "phase": "ctx",
                    "trigger": "ctx_chunk",
                    "run": 3,
                    "live_step_driver": True,
                },
            )

        self.assertEqual(result, "ok")
        sync.assert_called_once_with()
        self.assertEqual(events[0][0], "started")
        self.assertEqual(events[1][0], "completed_execution")
        completed = events[1][1]
        self.assertAlmostEqual(completed["execute_model_wall_time_ms"], 17.0)
        self.assertEqual(completed["run"], 3)
        self.assertEqual(completed["trigger"], "ctx_chunk")
        self.assertTrue(completed["live_step_driver"])

    def test_decode_only_match_requires_target_past_before_step(self):
        runner = SimpleNamespace(
            requests={
                "r0": SimpleNamespace(num_prompt_tokens=4096),
                "r1": SimpleNamespace(num_prompt_tokens=4096),
            }
        )
        control = {"bs": 2, "past": 4096}

        self.assertFalse(
            marker._decode_only_match(
                runner,
                _scheduler_output(
                    req_ids=["r0", "r1"],
                    computed=[2048, 2048],
                    scheduled=[2048, 2048],
                ),
                control,
            )[0]
        )

        matched, step, batch_size, past = marker._decode_only_match(
            runner,
            _scheduler_output(
                req_ids=["r0", "r1"],
                computed=[4096, 4096],
                scheduled=[1, 1],
            ),
            control,
        )

        self.assertTrue(matched)
        self.assertEqual(step, 4097)
        self.assertEqual(batch_size, 2)
        self.assertEqual(past, 4096)

        self.assertFalse(
            marker._decode_only_match(
                runner,
                _scheduler_output(
                    req_ids=["r0", "r1"],
                    computed=[4097, 4097],
                    scheduled=[1, 1],
                ),
                control,
            )[0]
        )

    def test_decode_only_match_rejects_new_requests(self):
        runner = SimpleNamespace(requests={"r0": SimpleNamespace(num_prompt_tokens=16)})

        self.assertFalse(
            marker._decode_only_match(
                runner,
                _scheduler_output(
                    req_ids=["r0"],
                    computed=[0],
                    scheduled=[16],
                    new_reqs=[SimpleNamespace(req_id="r0")],
                ),
                {"bs": 1, "past": 16},
            )[0]
        )

    def test_decode_only_match_accepts_new_prefix_cached_requests_when_enabled(self):
        runner = SimpleNamespace(requests={})
        new_reqs = [
            SimpleNamespace(req_id="r0", prompt_token_ids=[1] * 16, num_computed_tokens=16),
            SimpleNamespace(req_id="r1", prompt_token_ids=[2] * 16, num_computed_tokens=16),
        ]

        matched, step, batch_size, past = marker._decode_only_match(
            runner,
            _scheduler_output(
                req_ids=[],
                computed=[],
                scheduled=[1, 1],
                scheduled_req_ids=["r0", "r1"],
                new_reqs=new_reqs,
            ),
            {"bs": 2, "past": 16, "allow_new_cached": True},
        )

        self.assertTrue(matched)
        self.assertEqual(step, 17)
        self.assertEqual(batch_size, 2)
        self.assertEqual(past, 16)

    def test_decode_only_match_prefers_cached_tokens_for_new_prefix_cache_hit(self):
        runner = SimpleNamespace(requests={})
        new_reqs = [
            SimpleNamespace(req_id="r0", prompt_token_ids=[1] * 16, num_computed_tokens=0),
            SimpleNamespace(req_id="r1", prompt_token_ids=[2] * 16, num_computed_tokens=0),
        ]

        matched, step, batch_size, past = marker._decode_only_match(
            runner,
            _scheduler_output(
                req_ids=["r0", "r1"],
                computed=[16, 16],
                scheduled=[1, 1],
                new_reqs=new_reqs,
            ),
            {"bs": 2, "past": 16, "allow_new_cached": True},
        )

        self.assertTrue(matched)
        self.assertEqual(step, 17)
        self.assertEqual(batch_size, 2)
        self.assertEqual(past, 16)

    def test_decode_only_match_rejects_new_prefix_cache_miss(self):
        runner = SimpleNamespace(requests={})

        self.assertFalse(
            marker._decode_only_match(
                runner,
                _scheduler_output(
                    req_ids=[],
                    computed=[],
                    scheduled=[1],
                    scheduled_req_ids=["r0"],
                    new_reqs=[
                        SimpleNamespace(
                            req_id="r0",
                            prompt_token_ids=[1] * 16,
                            num_computed_tokens=0,
                        )
                    ],
                ),
                {"bs": 1, "past": 16, "allow_new_cached": True},
            )[0]
        )
