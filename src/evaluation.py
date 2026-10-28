"""Read an explicitly configured, reviewed offline evaluation artifact.

Artifacts contain calculate_classification_metrics() output plus dataset name.
No evaluation, model loading, or training occurs during API requests. Structural
and arithmetic checks cannot prove dataset provenance; only configure an artifact
after reviewing its labeled dataset, preprocessing, split and checkpoint.
"""

import hashlib
import json
import math
from pathlib import Path


MAX_COUNT = 2**53 - 1  # Exact integer range shared by JSON/JavaScript consumers.
_REQUIRED = {"dataset", "sampleCount", "accuracy", "macroPrecision", "macroRecall", "macroF1", "perClass", "confusionMatrix"}
_WEIGHTED = {"weightedPrecision", "weightedRecall", "weightedF1"}


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - set(optional):
        raise ValueError("Missing or unsupported evaluation fields")


def _integer(value):
    return type(value) is int and 0 <= value <= MAX_COUNT


def validate_metrics(metrics, labels):
    """Validate known fields and return a new, explicitly whitelisted payload.

    Count/score consistency does not establish genuine ground-truth provenance.
    """
    _keys(metrics, _REQUIRED, _WEIGHTED)
    if not isinstance(metrics.get("dataset"), str) or not metrics["dataset"].strip():
        raise ValueError("Evaluation dataset name is required")
    count = metrics["sampleCount"]
    if not _integer(count) or count <= 0:
        raise ValueError("Evaluation sampleCount must be a positive integer")
    size = len(labels)
    matrix = metrics["confusionMatrix"]
    if (not isinstance(matrix, list) or len(matrix) != size
            or any(not isinstance(row, list) or len(row) != size for row in matrix)
            or any(not _integer(n) for row in matrix for n in row)
            or sum(map(sum, matrix)) != count):
        raise ValueError("Invalid evaluation confusion matrix")
    rows = metrics["perClass"]
    if not isinstance(rows, list) or len(rows) != size:
        raise ValueError("Evaluation perClass size does not match the model")
    for row in rows:
        _keys(row, {"label", "precision", "recall", "f1", "support"})
    if [row["label"] for row in rows] != list(labels):
        raise ValueError("Evaluation perClass order does not match the model")

    def check(value, expected):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0 <= value <= 1 or not math.isfinite(value)
                or not math.isclose(value, expected, rel_tol=1e-7, abs_tol=1e-8)):
            raise ValueError("Evaluation score is invalid or inconsistent with counts")

    expected = []
    for i, row in enumerate(rows):
        support = sum(matrix[i])
        predicted = sum(r[i] for r in matrix)
        precision = matrix[i][i] / predicted if predicted else 0.0
        recall = matrix[i][i] / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if not _integer(row["support"]) or row["support"] != support:
            raise ValueError("Evaluation support does not match counts")
        expected.append((precision, recall, f1, support))
        for key, value in zip(("precision", "recall", "f1"), expected[-1]):
            check(row[key], value)
    check(metrics["accuracy"], sum(matrix[i][i] for i in range(size)) / count)
    for i, suffix in enumerate(("Precision", "Recall", "F1")):
        check(metrics["macro" + suffix], sum(row[i] for row in expected) / size)
        if "weighted" + suffix in metrics:
            check(metrics["weighted" + suffix], sum(row[i] * row[3] for row in expected) / count)
    result = {key: metrics[key] for key in _REQUIRED | (_WEIGHTED & metrics.keys())}
    result["perClass"] = [{key: row[key] for key in ("label", "precision", "recall", "f1", "support")} for row in rows]
    result["confusionMatrix"] = [list(row) for row in matrix]
    json.dumps(result, allow_nan=False)
    return result


def _reject_constant(value):
    raise ValueError("Non-finite evaluation values are not supported")


def _parse_integer(value):
    if len(value.lstrip("-")) > 16:
        raise ValueError("Evaluation integer is excessively large")
    number = int(value)
    if abs(number) > MAX_COUNT:
        raise ValueError("Evaluation integer exceeds exact JSON range")
    return number


def load_evaluation(path, checkpoint_path, labels, window_size):
    """Read a reviewed artifact; None if unconfigured, ValueError if invalid."""
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    if len(text) > 1_000_000:
        raise ValueError("Evaluation artifact is too large")
    artifact = json.loads(text, parse_constant=_reject_constant, parse_int=_parse_integer)
    _keys(artifact, {"checkpointSha256", "labels", "windowSize", "split", "provenance", "metrics"})
    digest = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    if (artifact["checkpointSha256"] != digest
            or artifact["labels"] != list(labels)
            or type(artifact["windowSize"]) is not int or artifact["windowSize"] != window_size
            or not isinstance(artifact["split"], str) or not artifact["split"].strip()
            or not isinstance(artifact["provenance"], str) or not artifact["provenance"].strip()):
        raise ValueError("Evaluation identity or provenance does not match the running model")
    return validate_metrics(artifact["metrics"], labels)
