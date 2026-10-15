"""CPU-only regression checks for spawned result writers and failure cleanup."""
import argparse
import builtins
import json
import multiprocessing as mp
import os
from pathlib import Path
import symtable
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vwb import native


REASONS = ["wrong_answer", "parse_fail", "request_error", "image_open_fail",
           "image_to_data_url_fail", "worker_exception"]


def mock_worker(rank, gpu_ids, shard, batch_size, task_type, prompt_style,
                err_q, pred_q, out_q, *args):
    for item in shard:
        rec = {"idx": item["idx"], "reason": REASONS[item["idx"] % len(REASONS)],
               "model_output": "x" * 4096}
        err_q.put(rec)
        pred_q.put(rec)
    out_q.put((0, len(shard)))


def crashed_writer(*args):
    os._exit(3)


class ResultWriterTests(unittest.TestCase):
    def test_native_global_dependencies_are_present(self):
        path = Path(native.__file__)
        table = symtable.symtable(path.read_text(), str(path), "exec")
        known = set(table.get_identifiers()) | set(dir(builtins))

        def check(scope):
            for child in scope.get_children():
                for symbol in child.get_symbols():
                    if symbol.is_referenced() and symbol.is_global():
                        self.assertIn(symbol.get_name(), known, child.get_name())
                check(child)
        check(table)

    def run_pipeline(self, root, failed_writer=None):
        questions = root / "mcq.jsonl"
        questions.write_text("".join(json.dumps({"qid": str(i), "image": "unused.jpg",
                              "options": ["a", "b"], "gt_index": 0}) + "\n"
                                     for i in range(600)))
        args = argparse.Namespace(
            questions=str(questions), task_type="mcq", model="unused", model_name="test",
            error_dir=str(root / "errors"), infra_error_dir=str(root / "infra"),
            pred_dir=str(root / "pred"), resume=False, gpu_ids="0,1",
            vllm_gpus_per_worker=1, vllm_cache_mode="disabled", score_path="", metrics_path=None,
            batch_size_per_gpu=258, prompt_style="plain", trust_remote_code=True,
            dtype="auto", gpu_memory_utilization=0.9, max_tokens_vllm=2048, thinking_mode="auto")
        replacements = {"vllm_worker": mock_worker}
        if failed_writer:
            replacements[failed_writer] = crashed_writer
        with patch.multiple(native, **replacements):
            return native.run_vllm(args)

    def test_spawned_writers_route_and_save_all_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.run_pipeline(root), 0)
            read = lambda kind: [json.loads(s) for s in
                                  (root / kind / "test/mcq.jsonl").read_text().splitlines()]
            predictions, errors, infra = read("pred"), read("errors"), read("infra")
            self.assertEqual({r["idx"] for r in predictions}, set(range(600)))
            self.assertEqual(len(predictions), 600)
            self.assertEqual(len(errors), 200)
            self.assertEqual(len(infra), 400)
            self.assertEqual({r["reason"] for r in errors}, set(REASONS[:2]))
            self.assertEqual({r["reason"] for r in infra}, set(REASONS[2:]))

    def test_writer_failure_does_not_hang_on_full_queue(self):
        for target in ["error_router_writer", "pred_writer"]:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                before = {p.pid for p in mp.active_children()}
                start = time.monotonic()
                with self.assertRaisesRegex(RuntimeError, "Result writer exited unexpectedly"):
                    self.run_pipeline(Path(directory), failed_writer=target)
                self.assertLess(time.monotonic() - start, 20)
                self.assertEqual({p.pid for p in mp.active_children()}, before)


if __name__ == "__main__":
    unittest.main()
