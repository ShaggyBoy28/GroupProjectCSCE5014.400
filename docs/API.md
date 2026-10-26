# API reference

The React frontend talks to Flask through a small JSON API under `/api`, defined in [`src/api.py`](../src/api.py). Video names are plain file names in the project folder (e.g. `sample_video.mp4`); anything that isn't a safe `.mp4` name returns 404.

Errors are JSON: `{"error": "message"}` with the matching HTTP status.

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | `{"status": "ok"}` |
| GET | `/api/sample` | Video info for `sample_video.mp4` |
| POST | `/api/videos` | Upload: multipart field `video` (`.mp4`, max `MAX_UPLOAD_MB`, default 200). `201 {"name": "<saved name>"}` |
| GET | `/api/videos/<name>` | Video info |
| GET | `/api/videos/<name>/file` | The original video |
| GET | `/api/videos/<name>/analyze` | Runs the analysis; server-sent events (see below) |
| GET | `/api/videos/<name>/result` | Annotated result video (`?download=1` to download) |
| GET | `/api/videos/<name>/breakdown` | Activity Breakdown data |

`/result` and `/breakdown` return 404 until the video has been analyzed.

## Video info

```json
{ "name": "sample_video.mp4", "fps": 29.97, "frameCount": 304, "width": 1280, "height": 720, "durationS": 10.143, "analyzed": false }
```

## Analysis events

`/analyze` streams `text/event-stream`. Each message is one JSON object:

```json
{ "type": "progress", "percent": 50, "frame": 152, "totalFrames": 304, "label": "CLAPPING_HANDS" }
{ "type": "done" }
{ "type": "error", "message": "..." }
```

`label` is `null` until the LSTM has its first 32-frame window. When the stream ends with `done`, two files exist next to the video:

- `res_<name>.mp4`: annotated video, re-encoded to H.264 when ffmpeg (`imageio-ffmpeg`) is available
- `res_<name-without-.mp4>.json`: every prediction (frame number and all 6 class probabilities)

Opening `/analyze` again re-runs the analysis and overwrites both files.

## Activity Breakdown

Built in [`src/api.py`](../src/api.py) from the saved predictions. Class order everywhere is `labels`, the LSTM's output order.

```json
{
  "video": { "name": "...", "fps": 29.97, "frameCount": 304, "width": 1280, "height": 720, "durationS": 10.143 },
  "windowSize": 32,
  "labels": ["JUMPING", "JUMPING_JACKS", "BOXING", "WAVING_2HANDS", "WAVING_1HAND", "CLAPPING_HANDS"],
  "summary": {
    "mainActivity": "WAVING_1HAND",
    "mainActivityShare": 0.4,
    "labelChanges": 8,
    "averageConfidence": 0.81,
    "predictionCount": 272,
    "firstPredictionS": 1.068
  },
  "segments": [
    { "label": "JUMPING", "startFrame": 32, "endFrame": 66, "startS": 1.068, "endS": 2.202, "durationS": 1.134, "avgConfidence": 0.85, "peakConfidence": 0.9 }
  ],
  "timePerActivity": [{ "label": "WAVING_1HAND", "seconds": 3.6, "share": 0.4 }],
  "predictions": [
    { "frame": 32, "timeS": 1.068, "label": "JUMPING", "confidence": 0.85, "probabilities": [0.85, 0.03, 0.03, 0.03, 0.03, 0.03] }
  ]
}
```

Definitions:

- **Segment**: a run of consecutive predictions with the same top label. It lasts from its first predicted frame to the first frame of the next segment (the last one ends one frame after the final prediction). `avgConfidence` and `peakConfidence` are the mean and max top-class probability inside it.
- **labelChanges**: number of segments minus 1.
- **averageConfidence**: mean top-class probability over all predictions.
- **timePerActivity**: seconds per class summed over segments, with its share of the analyzed time. All 6 classes are listed, largest first.
- **mainActivity**: the class with the most time.

## Planned fields (Phase 2 extensions)

The frontend already displays these. Every field is optional: until the backend sends it, the matching part of the UI stays hidden. To finish an extension, the backend only has to start sending the fields below; no frontend changes are needed.

### Extension 1: live classification (threshold, smoothing, majority vote)

Add to each `progress` event:

```json
{ "smoothedLabel": "WAVING_1HAND", "confidence": 0.84 }
```

- `smoothedLabel`: label after smoothing / majority vote, or `"UNCERTAIN"` when `confidence` is below the threshold. The Analyzing page shows it instead of `label`, with the raw label underneath when they differ.
- `confidence`: confidence of the shown label, 0–1.

Add to the breakdown response, and build `segments`, `timePerActivity` and `summary.labelChanges` from the **smoothed** labels (segments may use the label `"UNCERTAIN"`):

```json
"smoothing": { "method": "majority vote", "window": 5, "threshold": 0.6, "rawLabelChanges": 15, "uncertainS": 0.4 }
```

This adds a "Prediction smoothing" card, a threshold line on the confidence chart, an "Uncertain" legend entry, and "After smoothing (raw: N)" under Label changes.

### Extension 2a: live performance metrics

Add to each `progress` event (shown in a "Live performance" card on the Analyzing page):

```json
"metrics": { "frameMs": 412.5, "poseMs": 405.1, "lstmMs": 1.8, "fps": 2.4, "memoryMb": 1843 }
```

`lstmMs` is `null` before the first 32-frame window. Add the run's averages to the breakdown response (shown in a "Performance" card):

```json
"performance": { "totalS": 125.4, "avgFrameMs": 412.5, "avgPoseMs": 405.1, "avgLstmMs": 1.8, "avgFps": 2.4, "peakMemoryMb": 1912 }
```

### Extension 2b: model accuracy · Extension 3: new classes

New endpoint `GET /api/model`:

```json
{
  "labels": ["JUMPING", "JUMPING_JACKS", "BOXING", "WAVING_2HANDS", "WAVING_1HAND", "CLAPPING_HANDS"],
  "windowSize": 32,
  "metrics": {
    "dataset": "Berkeley MHAD test split",
    "sampleCount": 5751,
    "accuracy": 0.912,
    "macroPrecision": 0.85, "macroRecall": 0.892, "macroF1": 0.865,
    "perClass": [{ "label": "JUMPING", "precision": 0.9, "recall": 0.93, "f1": 0.91, "support": 940 }],
    "confusionMatrix": [[900, 30, 10, 0, 0, 0]]
  }
}
```

- `metrics` (2b): test-set results from running the model on `X_test.txt` / `Y_test.txt`. `confusionMatrix[actual][predicted]` is in `labels` order. Shown as a "Model accuracy" card (summary, per-class table, confusion matrix) on the breakdown page. Send `null` until the evaluation exists.
- `labels` / `windowSize` (3): the model's classes in output order. The home page legend and "How it works" text follow this list. In the breakdown response, `labels` drives the confidence chart and timeline legend. A retrained model with new classes needs no frontend change: known classes keep their names and colors, and new ones get a readable name (`SIT_DOWN` → "Sit down") and a color automatically.

Until `/api/model` exists, the frontend assumes the current 6 classes and a 32-frame window.
