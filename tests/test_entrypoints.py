"""Ensure public entrypoints and moved shell examples resolve repository paths."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def test_cli_help_from_another_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            for entry in ('evaluate.py', 'infer_api.py', 'judge.py', 'score.py'):
                with self.subTest(entry=entry):
                    result = subprocess.run([sys.executable, str(ROOT / entry), '--help'],
                                            cwd=tmp, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('usage:', result.stdout)

    def test_shell_examples_resolve_entries_from_repository_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            log = tmp / 'calls.jsonl'
            # Stub Python records invocations; no GPU/model/API is used.
            stub = tmp / 'python-stub'
            stub.write_text('#!' + sys.executable + '\n' +
                            'import json, os, pathlib, sys\n' +
                            'assert pathlib.Path(sys.argv[1]).is_file(), sys.argv[1]\n' +
                            'with open(os.environ["TEST_CALL_LOG"], "a") as f:\n' +
                            ' f.write(json.dumps({"cwd": os.getcwd(), "args": sys.argv[1:]}) + "\\n")\n')
            stub.chmod(0o755)
            for size in ('2B', '4B', '8B'):
                model = tmp / 'models' / ('Qwen3-VL-' + size + '-Instruct')
                model.mkdir(parents=True)
                (model / 'config.json').write_text('{}')
            (tmp / 'data').mkdir()
            env = dict(os.environ, PYTHON=str(stub), MODEL_ROOT=str(tmp / 'models'),
                       MODEL_SIZES='2B 4B 8B', DATA_DIR=str(tmp / 'data'),
                       TEST_CALL_LOG=str(log), SKIP_DOWNLOAD='1', CHECK_ONLY='1')
            for name, expected in (
                ('run_qwen3_vl.sh', ['evaluate.py'] * 3),
                ('run_openqa_qwen3_vl_8b.sh', ['evaluate.py', 'judge.py']),
                ('download_and_evaluate.sh', ['evaluate.py'] * 3),
            ):
                with self.subTest(script=name):
                    log.write_text('')
                    env['OUTPUT_ROOT'] = str(tmp / name)
                    result = subprocess.run(['bash', str(ROOT / 'scripts' / name)],
                                            cwd=tmp, env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                    calls = [json.loads(line) for line in log.read_text().splitlines()]
                    self.assertEqual([call['args'][0] for call in calls], expected)
                    self.assertTrue(all(call['cwd'] == str(ROOT) for call in calls))
                    if name == 'run_openqa_qwen3_vl_8b.sh':
                        self.assertIn('--dry-run', calls[-1]['args'])
