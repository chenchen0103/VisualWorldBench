"""Check the local Open-QA route without loading a model or calling an API."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vwb import evaluate
from vwb.eval_protocol import build_prompt_for_task


class OpenQARunnerTests(unittest.TestCase):
    def test_open_prompt_and_native_settings(self):
        args = argparse.Namespace(data_dir=Path('data'), model=Path('model'),
                                  model_name='Qwen3-VL-8B', batch_size=258, max_tokens=2048)
        cfg = evaluate.native_args(args, 'open_qa', list('01234567'), Path('results'))
        self.assertEqual(cfg.task_type, 'open')
        self.assertEqual(cfg.prompt_style, 'open_reason')
        self.assertEqual(cfg.vllm_gpus_per_worker, 1)
        self.assertEqual(cfg.batch_size_per_gpu, 258)
        self.assertEqual(cfg.max_tokens_vllm, 2048)
        self.assertEqual(evaluate.worker_count('8B', 'open_qa'), 8)
        q = {'prompt': 'What is the object?'}
        prompt = evaluate.native.build_prompt_for_task(q, cfg.task_type, cfg.prompt_style)
        self.assertEqual(prompt, build_prompt_for_task(q, 'open', 'open_reason'))
        self.assertIn('Answer: <category name>', prompt)
        self.assertNotIn('open_qa', evaluate.DEFAULT_TASKS)
        self.assertEqual(evaluate.native_args(args, 'mcq', ['0'], Path('results')).prompt_style, 'plain')

    def test_unscored_open_answers_are_complete_but_failures_are_rejected(self):
        questions = [{'qid': 'q0'}, {'qid': 'q1'}]
        rows = [{'idx': i, 'qid': q['qid'], 'reason': 'ok', 'counted': False,
                 'model_output': 'Answer: cat'} for i, q in enumerate(questions)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'open_qa.jsonl'
            def write(records):
                path.write_text(''.join(json.dumps(r) + '\n' for r in records))
            write(rows)
            evaluate.validate_predictions(path, questions, 'open_qa')
            write(rows[:1])
            with self.assertRaises(RuntimeError):
                evaluate.validate_predictions(path, questions, 'open_qa')
            rows[1]['reason'] = 'worker_exception: unavailable'
            write(rows)
            with self.assertRaises(RuntimeError):
                evaluate.validate_predictions(path, questions, 'open_qa')


if __name__ == '__main__':
    unittest.main()
