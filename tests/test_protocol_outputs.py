"""Regression checks for discrete answers and score-table column alignment."""
import csv
from pathlib import Path
import tempfile
import unittest

from vwb import eval_protocol, native
from vwb.scoreboard import update_score_cells_tsv


class ProtocolOutputTests(unittest.TestCase):
    def test_discrete_answer_formats(self):
        cases = {
            "2": 1,
            "Answer: 2": 1,
            "最终答案：2": 1,
            "选项：2": 1,
            "Reasoning mentions 1.\n<answer>2</answer>": 1,
            "0": None,
            "5": None,
            "No answer": None,
        }
        for module in (native, eval_protocol):
            for text, expected in cases.items():
                with self.subTest(module=module.__name__, text=text):
                    self.assertEqual(module.parse_index(text, 4), expected)

    def test_score_columns_preserve_values_across_updates(self):
        tasks = ["mcq", "sel_2x2", "sel_3x3", "loc_2x2", "loc_3x3", "open_qa"]
        scores = {task: f"{i / 10:.4f}" for i, task in enumerate(tasks, 1)}
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "scores.tsv")
            for task in reversed(tasks):
                update_score_cells_tsv(path, model_name="example", updates={task: scores[task]})
            with open(path, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                self.assertEqual(reader.fieldnames, ["model", *tasks, "Average"])
                row = next(reader)
            self.assertEqual({task: row[task] for task in tasks}, scores)
            self.assertEqual(row["Average"], "0.3500")


if __name__ == "__main__":
    unittest.main()
