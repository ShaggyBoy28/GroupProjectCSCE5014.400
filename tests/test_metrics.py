"""Known-answer tests; these fixtures are not model performance results."""

import ast
import json
from pathlib import Path
import unittest
import warnings

import numpy as np

from src.metrics import ACTIVITY_LABELS, calculate_classification_metrics


class ClassificationMetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        result = calculate_classification_metrics(range(6), range(6))
        for key in (
            "accuracy", "macroPrecision", "macroRecall", "macroF1",
            "weightedPrecision", "weightedRecall", "weightedF1",
        ):
            self.assertEqual(result[key], 1.0)
        self.assertEqual(result["sampleCount"], 6)
        self.assertEqual(result["confusionMatrix"], np.eye(6, dtype=int).tolist())
        self.assertTrue(all(row["support"] == 1 for row in result["perClass"]))

    def test_partial_predictions_manual_values(self):
        # Actual/predicted counts: [[1,1,0], [0,1,0], [1,0,0]].
        # Class 0: P=1/2, R=1/2, F1=1/2; class 1: P=1/2, R=1, F1=2/3.
        result = calculate_classification_metrics([0, 0, 1, 2], [0, 1, 1, 0])
        self.assertEqual(result["accuracy"], 1 / 2)
        self.assertEqual(result["sampleCount"], 4)
        self.assertEqual([row["support"] for row in result["perClass"]], [2, 1, 1, 0, 0, 0])
        self.assertAlmostEqual(result["macroPrecision"], 1 / 6)
        self.assertAlmostEqual(result["macroRecall"], 1 / 4)
        self.assertAlmostEqual(result["macroF1"], 7 / 36)
        self.assertAlmostEqual(result["weightedPrecision"], 3 / 8)
        self.assertAlmostEqual(result["weightedRecall"], 1 / 2)
        self.assertAlmostEqual(result["weightedF1"], 5 / 12)
        self.assertEqual(result["perClass"][1]["precision"], 1 / 2)
        self.assertEqual(result["perClass"][1]["recall"], 1)
        self.assertAlmostEqual(result["perClass"][1]["f1"], 2 / 3)

    def test_completely_incorrect_predictions(self):
        result = calculate_classification_metrics(range(6), [1, 2, 3, 4, 5, 0])
        for key, value in result.items():
            if key not in ("perClass", "confusionMatrix", "sampleCount"):
                self.assertEqual(value, 0.0)
        self.assertTrue(all(row["f1"] == 0 for row in result["perClass"]))

    def test_missing_predicted_class(self):
        result = calculate_classification_metrics([0, 1], [0, 0])
        self.assertEqual(result["perClass"][1], {
            "label": "JUMPING_JACKS", "precision": 0.0,
            "recall": 0.0, "f1": 0.0, "support": 1,
        })

    def test_missing_ground_truth_class(self):
        result = calculate_classification_metrics([0, 0], [0, 1])
        self.assertEqual(result["perClass"][1]["support"], 0)
        self.assertEqual(result["perClass"][1]["recall"], 0.0)
        self.assertEqual(result["perClass"][1]["precision"], 0.0)
        self.assertAlmostEqual(result["weightedPrecision"], 1.0)
        self.assertAlmostEqual(result["weightedRecall"], 0.5)

    def test_zero_division_and_fixed_six_class_macro(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = calculate_classification_metrics([0], [0])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["macroF1"], 1 / 6)
        self.assertAlmostEqual(result["macroPrecision"], 1 / 6)
        self.assertAlmostEqual(result["macroRecall"], 1 / 6)
        self.assertEqual(result["weightedF1"], 1.0)
        for row in result["perClass"][1:]:
            self.assertEqual([row[k] for k in ("precision", "recall", "f1", "support")], [0, 0, 0, 0])

    def test_empty_inputs(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            calculate_classification_metrics([], [])

    def test_mismatched_lengths(self):
        for actual, predicted in [([0], []), ([], [0]), ([0, 1], [0])]:
            with self.subTest(actual=actual, predicted=predicted):
                with self.assertRaisesRegex(ValueError, "identical lengths"):
                    calculate_classification_metrics(actual, predicted)

    def test_invalid_class_labels_in_either_input(self):
        for invalid in [-1, 6, "UNKNOWN", "jumping", "0", 0.9, 0.0, True, np.bool_(False), None, float("nan")]:
            for side in ("y_true", "y_pred"):
                with self.subTest(invalid=invalid, side=side):
                    args = ([invalid], [0]) if side == "y_true" else ([0], [invalid])
                    with self.assertRaisesRegex(ValueError, side + r"\[0\]"):
                        calculate_classification_metrics(*args)

    def test_rejects_scalars_nested_arrays_and_confidence_scores(self):
        for invalid in [None, 0, "JUMPING", b"JUMPING", [[0]], np.array([[0]]), [[0.8, 0.2, 0, 0, 0, 0]]]:
            for args in [(invalid, [0]), ([0], invalid)]:
                with self.subTest(invalid=invalid, args=args):
                    with self.assertRaises(ValueError):
                        calculate_classification_metrics(*args)

    def test_fixed_confusion_matrix_order_and_orientation(self):
        result = calculate_classification_metrics([5, 0, 2, 5], [0, 2, 5, 5])
        expected = [[0] * 6 for _ in range(6)]
        expected[5][0] = expected[0][2] = expected[2][5] = expected[5][5] = 1
        self.assertEqual(result["confusionMatrix"], expected)
        self.assertEqual([row["label"] for row in result["perClass"]], list(ACTIVITY_LABELS))

    def test_names_ids_numpy_arrays_and_generators(self):
        expected = calculate_classification_metrics([0, 1, 5], [0, 5, 5])
        actual = calculate_classification_metrics(
            (name for name in ["JUMPING", "JUMPING_JACKS", "CLAPPING_HANDS"]),
            np.array([0, 5, 5], dtype=np.int64),
        )
        self.assertEqual(actual, expected)
        self.assertEqual(calculate_classification_metrics([0, "CLAPPING_HANDS"], ["JUMPING", 5])["accuracy"], 1)

    def test_json_compatibility_and_score_bounds(self):
        result = calculate_classification_metrics([0, 0, 1, 5], [0, 1, 2, 5])
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        for key, value in result.items():
            if key not in ("perClass", "confusionMatrix", "sampleCount"):
                self.assertTrue(0 <= value <= 1, key)
        for row in result["perClass"]:
            for key in ("precision", "recall", "f1"):
                self.assertTrue(0 <= row[key] <= 1)
        self.assertEqual(sum(map(sum, result["confusionMatrix"])), result["sampleCount"])

    def test_class_mapping_matches_pipeline_without_importing_models(self):
        path = Path(__file__).resolve().parents[1] / "src" / "video_analyzer.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        mapping = next(
            ast.literal_eval(node.value)
            for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "LABELS" for target in node.targets)
        )
        self.assertEqual(mapping, dict(enumerate(ACTIVITY_LABELS)))


if __name__ == "__main__":
    unittest.main()
