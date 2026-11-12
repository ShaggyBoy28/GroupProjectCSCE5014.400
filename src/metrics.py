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


def _class_ids(values, name, labels=ACTIVITY_LABELS):
    """Validate a one-dimensional iterable and normalize labels to IDs."""
    label_to_id = {label: index for index, label in enumerate(labels)}
    class_ids = range(len(labels))
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
        if isinstance(value, str) and value in label_to_id:
            result.append(label_to_id[value])
        elif isinstance(value, Integral) and not isinstance(value, bool) and value in class_ids:
            result.append(int(value))
        else:
            raise ValueError(
                f"{name}[{index}] must be an integer class ID 0–{len(labels) - 1} or an exact "
                f"activity name from {tuple(labels)}; got {value!r}. "
                "Supply discrete class labels, not confidence scores."
            )
    return result


def calculate_classification_metrics(y_true, y_pred, labels=ACTIVITY_LABELS):
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

    `labels` defaults to the app's six classes; pass a longer tuple to score a
    model trained with added classes (scripts/train_model.py does this).

    Raises:
        ValueError: Inputs are empty, differ in length, or contain invalid labels.
    """
    labels = tuple(labels)
    ids = tuple(range(len(labels)))
    actual = _class_ids(y_true, "y_true", labels)
    predicted = _class_ids(y_pred, "y_pred", labels)
    if len(actual) != len(predicted):
        raise ValueError(
            "y_true and y_pred must have identical lengths "
            f"(got {len(actual)} and {len(predicted)})"
        )
    if not actual:
        raise ValueError("Evaluation inputs must not be empty")

    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=ids, average=None, zero_division=0
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
            for index, label in enumerate(labels)
        ],
        "confusionMatrix": confusion_matrix(
            actual, predicted, labels=ids
        ).tolist(),
        "sampleCount": count,
    }
