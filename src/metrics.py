"""Standalone evaluation of discrete six-class activity predictions.

No models are imported or executed. scikit-learn is required by this module.
"""

from numbers import Integral

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


# Match src/video_analyzer.py's LSTM output order without importing its models.
ACTIVITY_LABELS = (
    "JUMPING",
    "JUMPING_JACKS",
    "BOXING",
    "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS",
)
_LABEL_TO_ID = {label: index for index, label in enumerate(ACTIVITY_LABELS)}
_CLASS_IDS = tuple(range(len(ACTIVITY_LABELS)))


def _class_ids(values, name):
    """Validate a one-dimensional iterable and normalize labels to IDs."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a one-dimensional iterable of class labels")
    try:
        values = list(values)
    except TypeError as error:
        raise ValueError(
            f"{name} must be a one-dimensional iterable of class labels"
        ) from error

    result = []
    for index, value in enumerate(values):
        if isinstance(value, str) and value in _LABEL_TO_ID:
            result.append(_LABEL_TO_ID[value])
        elif isinstance(value, Integral) and not isinstance(value, bool) and value in _CLASS_IDS:
            result.append(int(value))
        else:
            raise ValueError(
                f"{name}[{index}] must be an integer class ID 0–5 or an exact "
                f"activity name from {ACTIVITY_LABELS}; got {value!r}. "
                "Supply discrete class labels, not confidence scores."
            )
    return result


def calculate_classification_metrics(y_true, y_pred):
    """Return JSON-compatible metrics from ground truth and actual predictions.

    Inputs are nonempty, equally sized, one-dimensional iterables containing
    integer IDs 0–5 or exact names in ACTIVITY_LABELS. Names and IDs may be mixed;
    floats, booleans, unknown labels, and probability arrays are rejected.

    Confusion matrix rows are actual classes and columns are predicted classes.
    Matrix axes and perClass rows always follow ACTIVITY_LABELS. Macro averages
    include all six classes, even absent ones. Weighted averages use ground-truth
    support. Undefined precision/recall/F1 values are zero. All scores are 0–1.

    This evaluates the supplied labeled samples only; it does not establish
    that a dataset is independent of training or select a prediction granularity.

    Raises:
        ValueError: Inputs are empty, differ in length, or contain invalid labels.
    """
    actual = _class_ids(y_true, "y_true")
    predicted = _class_ids(y_pred, "y_pred")
    if len(actual) != len(predicted):
        raise ValueError(
            "y_true and y_pred must have identical lengths "
            f"(got {len(actual)} and {len(predicted)})"
        )
    if not actual:
        raise ValueError("Evaluation inputs must not be empty")

    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=_CLASS_IDS, average=None, zero_division=0
    )
    count = len(actual)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macroPrecision": float(precision.mean()),
        "macroRecall": float(recall.mean()),
        "macroF1": float(f1.mean()),
        "weightedPrecision": float((precision * support).sum() / count),
        "weightedRecall": float((recall * support).sum() / count),
        "weightedF1": float((f1 * support).sum() / count),
        "perClass": [
            {
                "label": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(ACTIVITY_LABELS)
        ],
        "confusionMatrix": confusion_matrix(
            actual, predicted, labels=_CLASS_IDS
        ).tolist(),
        "sampleCount": count,
    }
