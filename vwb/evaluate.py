#!/usr/bin/env python3
"""Evaluate Qwen3-VL locally; Open-QA is available via --tasks open_qa."""
import argparse
from datetime import datetime
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from vwb import native
from .prediction_validation import infrastructure_failure

TASKS = {'mcq': 'mcq', 'sel_2x2': 'grid_loc', 'sel_3x3': 'grid_loc',
         'loc_2x2': 'grid_box', 'loc_3x3': 'grid_box', 'open_qa': 'open'}
DEFAULT_TASKS = 'mcq,sel_2x2,sel_3x3,loc_2x2,loc_3x3'


def prompt_style(task):
    return 'open_reason' if task == 'open_qa' else 'plain'


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_data(data, tasks):
    """Fail before inference if normalization drops records or images are missing."""
    result = {}
    for task in tasks:
        path = data / (task + '.jsonl')
        rows = read_rows(path)
        qids = [str(q['qid']) for q in rows]
        if not rows or len(qids) != len(set(qids)):
            raise ValueError(f'{task}: empty data or duplicate qids; check the task JSONL file')
        questions = native.load_questions(str(path), task_type=TASKS[task])
        if len(rows) != len(questions):
            raise ValueError(f'{task}: invalid records were rejected by the data loader')
        for raw, q in zip(rows, questions):
            if str(raw['qid']) != str(q['qid']):
                raise ValueError(f'{task}: row identity changed during loading')
            if not Path(q['image_path']).is_file():
                raise FileNotFoundError(q['image_path'])
            native.build_prompt_for_task(q, TASKS[task], prompt_style(task))
        result[task] = questions
        print(f'{task}: {len(questions)} samples checked', flush=True)
    return result


def validate_predictions(path, questions, task):
    rows = read_rows(path)
    seen = set()
    for row in rows:
        idx = row.get('idx')
        if not isinstance(idx, int) or idx in seen or not 0 <= idx < len(questions):
            raise ValueError(f'{task}: invalid or duplicate prediction index {idx}')
        seen.add(idx)
        if str(row.get('qid')) != str(questions[idx]['qid']):
            raise ValueError(f'{task}: prediction qid mismatch at {idx}')
        reason = str(row.get('reason', ''))
        if infrastructure_failure(row):
            raise RuntimeError(f'{task}: infrastructure error at {idx}: {reason}')
        if TASKS[task] in {'mcq', 'grid_loc'} and not row.get('counted'):
            raise RuntimeError(f'{task}: uncounted response at {idx}')
    if len(seen) != len(questions):
        raise RuntimeError(f'{task}: only {len(seen)}/{len(questions)} predictions')


def worker_count(model_size, task):
    if task in {'mcq', 'open_qa'}:
        return 8
    if task.startswith('sel_'):
        return 1 if model_size == '2B' else 8
    return 4


def native_args(args, task, gpu_ids, output):
    return argparse.Namespace(
        questions=str(args.data_dir / (task + '.jsonl')), task_type=TASKS[task],
        model=str(args.model), model_name=args.model_name,
        dataset_stem_override=task, pred_dir=str(output / 'predictions' / task),
        error_dir=str(output / 'errors' / task), infra_error_dir=str(output / 'errors-infra' / task),
        score_path=str(output / 'discrete_scores.tsv'), metrics_path=None,
        resume=False, shard_rank=0, num_shards=1, gpu_ids=','.join(gpu_ids),
        vllm_gpus_per_worker=1, batch_size_per_gpu=args.batch_size,
        vllm_cache_mode='isolated', prompt_style=prompt_style(task), thinking_mode='auto',
        trust_remote_code=True, dtype='auto', gpu_memory_utilization=0.9,
        max_tokens_vllm=args.max_tokens,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-size', choices=['2B', '4B', '8B'], required=True)
    parser.add_argument('--model-name')
    parser.add_argument('--tasks', default=DEFAULT_TASKS)
    parser.add_argument('--gpu-ids', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--batch-size', type=int, default=258)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--check-only', action='store_true', help='Validate inputs/configuration without loading a model')
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    args.model = args.model.resolve()
    args.model_name = args.model_name or f'Qwen3-VL-{args.model_size}'
    if not args.model_name or any(x in args.model_name for x in ['/', '\\']) or args.model_name in ['.', '..']:
        parser.error('model-name must be a simple directory name')
    tasks = args.tasks.split(',')
    if not tasks or len(set(tasks)) != len(tasks) or any(t not in TASKS for t in tasks):
        parser.error('tasks must be distinct keys from ' + ','.join(TASKS))
    if args.batch_size < 1 or args.max_tokens < 1:
        parser.error('batch-size and max-tokens must be positive')
    if not (args.model / 'config.json').is_file():
        parser.error('model must be a local checkpoint directory containing config.json')
    config = json.loads((args.model / 'config.json').read_text())
    if config.get('model_type') != 'qwen3_vl':
        parser.error('this runner supports Qwen3-VL checkpoints')
    gpus = args.gpu_ids.split(',')
    if any(not g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus):
        parser.error('gpu-ids must contain distinct numeric physical GPU IDs')
    if len(gpus) < max(worker_count(args.model_size, task) for task in tasks):
        parser.error('not enough GPU IDs for the selected task layout (up to 8 required)')
    questions = validate_data(args.data_dir, tasks)
    for task in tasks:
        print(f'{task}: GPUs {gpus[:worker_count(args.model_size, task)]}, TP=1, batch={args.batch_size}, max_tokens={args.max_tokens}')
    if args.check_only:
        return 0
    # Set inference environment variables for this runner.
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    for key in ['VLLM_USE_V1', 'VLLM_ENABLE_V1_MULTIPROCESSING', 'VLLM_BATCH_INVARIANT',
                'VLLM_ATTENTION_BACKEND', 'VLLM_USE_FLASHINFER_SAMPLER']:
        os.environ.pop(key, None)
    output = (args.output or Path('results') / (args.model_name + '-' + datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Save the run configuration alongside predictions and scores.
    run_config = {'model': str(args.model), 'model_name': args.model_name,
                  'data_dir': str(args.data_dir), 'tasks': tasks,
                  'prompt_styles': {task: prompt_style(task) for task in tasks},
                  'batch_size': args.batch_size, 'max_tokens': args.max_tokens,
                  'gpu_ids': gpus, 'temperature': 0, 'top_p': 1, 'tensor_parallel_size': 1,
                  'python': sys.version.split()[0],
                  'packages': {p: metadata.version(p) for p in ['vllm', 'torch', 'transformers', 'tokenizers', 'pillow']}}
    (output / 'config.json').write_text(json.dumps(run_config, indent=2) + '\n')
    for task in tasks:
        print(f'\nStarting {args.model_name} / {task}', flush=True)
        native.run_vllm(native_args(args, task, gpus[:worker_count(args.model_size, task)], output))
        pred = output / 'predictions' / task / args.model_name / (task + '.jsonl')
        validate_predictions(pred, questions[task], task)
        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'score.py'),
                        '--bench-dir', str(args.data_dir), '--pred-dir', str(output / 'predictions'),
                        '--output', str(output / 'scores.tsv')], check=True)
    print(f'Finished. Scores: {output / "scores.tsv"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
