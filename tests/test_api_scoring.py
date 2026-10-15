"""CPU-only HTTP/CLI regression tests; no external model or API is called."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vwb.judge_protocol import parse_judge_response
from PIL import Image


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def read_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


class ApiScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bench = self.root / 'data'
        self.pred = self.root / 'pred'
        self.responses = deque()
        self.calls = []
        self.call_paths = []
        self.bench.mkdir()
        Image.new('RGB', (16, 16), 'red').save(self.bench / 'image.png')
        rows = [{'qid':str(i), 'image':'image.png', 'options':['cat','dog'], 'gt_index':0}
                for i in range(2)]
        write_rows(self.bench/'mcq.jsonl', rows)
        write_rows(self.bench/'open_qa.jsonl', [
            {'qid':str(i), 'image':'image.png', 'question':'Name the animal.',
             'gt_name':'cat', 'gt_aliases_base':['cat']} for i in range(2)])
        outer = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                outer.calls.append(payload)
                outer.call_paths.append(self.path)
                status, text = outer.responses.popleft() if outer.responses else (500, 'Unexpected call')
                self.send_response(status)
                self.send_header('Content-Type','application/json')
                self.end_headers()
                body = text if isinstance(text, dict) else {'choices':[{'message':{'content':text}}]}
                self.wfile.write(json.dumps(body).encode())
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def cli(self, script, *args, success=True):
        env=os.environ.copy()
        env.pop('JUDGE_MODEL', None)
        env.update(OPENAI_API_KEY='local-test-only', JUDGE_API_KEY='local-test-only',
                   PYTHONDONTWRITEBYTECODE='1')
        r=subprocess.run([sys.executable, str(ROOT/script), *map(str,args)],
                         env=env, capture_output=True, text=True, timeout=30)
        if success:
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        else:
            self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        return r

    def infer(self, success=True):
        return self.cli('infer_api.py', '--bench-dir', self.bench, '--pred-dir', self.pred,
                        '--model','Mock','--base-url',self.url, '--tasks','mcq',
                        '--num-workers','1','--max-retries','0', success=success)

    def judge(self, retries=0, success=True):
        return self.cli('judge.py','--bench-jsonl',self.bench/'open_qa.jsonl',
                        '--pred-file',self.pred/'open_qa/Mock/open_qa.jsonl',
                        '--judge-model','Mock','--base-url',self.url,
                        '--api-format','chat-completions',
                        '--max-retries',retries, success=success)

    def test_explicit_gemini_judge_uses_text_only_responses_and_saves_usage(self):
        self.open_predictions()
        self.responses.append((200, {
            'candidates':[{'finishReason':'STOP','content':{'parts':[
                {'thought':True,'text':'private reasoning'},
                {'text':'{"judgment":"wrong","reason":"different species"}'}]}}],
            'usageMetadata':{'promptTokenCount':177,'candidatesTokenCount':26}}))
        result = self.cli('judge.py','--bench-jsonl',self.bench/'open_qa.jsonl',
                         '--pred-file',self.pred/'open_qa/Mock/open_qa.jsonl',
                         '--base-url',self.url,'--judge-model','MockGemini','--max-retries','0')
        self.assertEqual(self.call_paths, ['/v1/responses'])
        self.assertEqual(self.calls[0]['model'], 'MockGemini')
        self.assertNotIn('messages', self.calls[0])
        self.assertNotIn('input', self.calls[0])
        self.assertEqual(set(self.calls[0]['contents'][0]['parts'][0]), {'text'})
        rows = read_rows(self.pred/'open_qa/Mock/open_qa.judge.jsonl')
        self.assertEqual(rows[0]['judgment'],'wrong')
        self.assertEqual(rows[0]['usage_attempts'][0]['promptTokenCount'],177)
        self.assertNotIn('private reasoning', rows[0]['raw_response'])

    def test_judge_requires_model_before_api_call_but_not_for_dry_run(self):
        self.open_predictions()
        args = ('--bench-jsonl', self.bench/'open_qa.jsonl',
                '--pred-file', self.pred/'open_qa/Mock/open_qa.jsonl',
                '--base-url', self.url)
        result = self.cli('judge.py', *args, success=False)
        self.assertIn('Set JUDGE_MODEL or pass --judge-model', result.stderr)
        result = self.cli('judge.py', *args, '--dry-run')
        self.assertIn('pending judge calls: 1', result.stdout)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.pred/'open_qa/Mock/open_qa.judge.run.json').exists())

    def score(self, *args, success=True):
        return self.cli('score.py','--bench-dir',self.bench,'--pred-dir',self.pred,
                        '--output',self.root/'scores.tsv',*args,success=success)

    def scores(self):
        with (self.root/'scores.tsv').open() as f:
            return next(csv.DictReader(f, delimiter='\t'))

    def open_predictions(self):
        write_rows(self.pred/'open_qa/Mock/open_qa.jsonl', [
            {'qid':'0','model_output':'cat','gt_aliases_base':['cat'],'reason':'ok'},
            {'qid':'1','model_output':'lion','gt_aliases_base':['cat'],'reason':'ok'}])

    def test_api_resume_retries_failed_requests_only(self):
        self.responses.extend([(200,'2'),(500,'unavailable')])
        self.infer(success=False)
        self.assertEqual(len(self.calls),2)
        path=self.pred/'mcq/Mock/mcq.jsonl'
        self.assertEqual(read_rows(path)[1]['reason'],'request_error')
        self.responses.append((200,'1'))
        self.infer()
        rows=read_rows(path)
        self.assertEqual(len(self.calls),3)
        self.assertEqual([r['qid'] for r in rows],['0','1'])
        self.assertEqual(rows[0]['model_output'],'2')  # Wrong answers are valid responses, not retries.
        self.assertFalse(rows[0]['correct'])
        self.assertTrue(rows[1]['correct'])
        self.infer()
        self.assertEqual(len(self.calls),3)
        self.score('--tasks','mcq')
        self.assertEqual(self.scores()['mcq'],'0.5000')

    def test_api_resume_compacts_duplicates_and_truncated_tail(self):
        self.responses.extend([(200,'1'),(200,'1')])
        self.infer()
        path=self.pred/'mcq/Mock/mcq.jsonl'
        first=path.read_text().splitlines()[0]
        with path.open('a') as f: f.write(first+'\n{"qid":')
        self.infer()
        self.assertEqual(len(self.calls),2)
        self.assertEqual(len(read_rows(path)),2)
        self.score('--tasks','mcq')

    def test_judge_parser_never_converts_invalid_text_to_wrong(self):
        for text in ['', 'wrong', 'correct', 'Service unavailable', '{bad}',
                     '{"judgment":"unknown"}', '[{"judgment":"wrong"}]']:
            self.assertEqual(parse_judge_response(text)['judgment'],'error',text)
        for value in ['correct','wrong']:
            text=json.dumps({'judgment':value,'reason':'a {brace} in explanation'})
            self.assertEqual(parse_judge_response(text)['judgment'],value)
            self.assertEqual(parse_judge_response('```json\n'+text+'\n```')['judgment'],value)

    def test_judge_retries_invalid_format(self):
        self.open_predictions()
        self.responses.extend([(200,'Service unavailable'),(200,'{"judgment":"correct"}')])
        self.judge(retries=1)
        self.assertEqual(len(self.calls),2)
        self.score('--tasks','open_qa')
        self.assertEqual(self.scores()['open_qa_corrected'],'1.0000')

    def test_judge_error_remains_pending_then_can_resume(self):
        self.open_predictions()
        self.responses.append((200,'invalid'))
        self.judge(success=False)
        self.score('--tasks','open_qa')
        self.assertEqual(self.scores()['open_qa_corrected'],'')
        self.assertEqual(self.scores()['avg'],'')
        self.responses.append((200,'{"judgment":"wrong"}'))
        self.judge()
        self.score('--tasks','open_qa')
        self.assertEqual(len(self.calls),2)
        self.assertEqual(self.scores()['open_qa_corrected'],'0.5000')

    def test_judge_retries_http_failure_with_same_budget(self):
        self.open_predictions()
        self.responses.extend([(503,'unavailable'),(200,'{"judgment":"correct"}')])
        self.judge(retries=1)
        self.assertEqual(len(self.calls),2)

    def test_scoring_rejects_missing_duplicate_unknown_failed_and_malformed(self):
        a={'qid':'0','correct':True,'counted':True,'reason':'ok'}
        b={**a,'qid':'1'}
        path=self.pred/'mcq/Mock/mcq.jsonl'
        cases=[[a], [a,a,b], [a,b,{**a,'qid':'extra'}],
               [a,{**b,'reason':'request_error','counted':False}]]
        for rows in cases:
            with self.subTest(rows=rows):
                write_rows(path,rows)
                self.score(success=False)
                self.assertFalse((self.root/'scores.tsv').exists())
        write_rows(path,[a,b])
        with path.open('a') as f:f.write('{bad}\n')
        self.score(success=False)
        self.assertFalse((self.root/'scores.tsv').exists())

    def test_explicit_tasks_reject_missing_task_and_auto_has_no_total(self):
        write_rows(self.pred/'mcq/Mock/mcq.jsonl',[
            {'qid':str(i),'correct':True,'counted':True} for i in range(2)])
        self.score('--tasks','all',success=False)
        self.score()
        self.assertEqual(self.scores()['mcq'],'1.0000')
        self.assertEqual(self.scores()['avg'],'')

    def test_complete_all_task_scores_preserved(self):
        for task in ['mcq','sel_2x2','sel_3x3','loc_2x2','loc_3x3']:
            bench=[];pred=[]
            for i in range(2):
                bench.append({'qid':str(i),'bbox_norm':[0,0,500,500],
                              'meta':{'image_size':[16,16]}})
                pred.append({'qid':str(i),'counted':True,'correct':i==0,
                             'model_output':'[0,0,500,500]' if i==0 else '',
                             'reason':'ok' if i==0 else 'parse_fail'})
            write_rows(self.bench/(task+'.jsonl'),bench)
            write_rows(self.pred/task/'Mock'/(task+'.jsonl'),pred)
        self.open_predictions()
        write_rows(self.pred/'open_qa/Mock/open_qa.judge.jsonl',[
            {'qid':'1','judgment':'correct'}])
        self.score('--tasks','all')
        row=self.scores()
        for task in ['mcq','sel_2x2','sel_3x3','loc_2x2','loc_3x3']:
            self.assertEqual(row[task],'0.5000')
            self.assertEqual(row[task+'_n'],'2')
        self.assertEqual(row['open_qa'],'0.5000')
        self.assertEqual(row['open_qa_corrected'],'1.0000')
        self.assertEqual(row['avg'],'0.6250')


if __name__=='__main__': unittest.main()
