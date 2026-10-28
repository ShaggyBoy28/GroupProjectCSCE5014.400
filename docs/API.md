# API reference

The React frontend talks to Flask through a small JSON API under `/api`, defined in [`src/api.py`](../src/api.py). Video names are plain file names in the project folder (e.g. `sample_video.mp4`); anything that isn't a safe `.mp4` name returns 404.

All numerical examples below use synthetic/mock data, not measured model performance.

Errors from the video API are JSON: `{"error": "message"}` with the matching HTTP status.

| Method | Path | Returns |
|---|---|---|
| GET | `/api/model` | Model labels, window size and verified evaluation metrics (otherwise `metrics: null`) |
| GET | `/api/health` | `{"status": "ok"}` |
| POST | `/api/live/sessions` | Start a live classification session; `201 {"sessionId": "...", "windowSize": 32}` |
| POST | `/api/live/sessions/<sessionId>/frames` | Classify one JPEG camera frame |
| DELETE | `/api/live/sessions/<sessionId>` | Stop and release a live classification session |
| GET | `/api/sample` | Video info for `sample_video.mp4` |
| POST | `/api/videos` | Upload: multipart field `video` (`.mp4`, max `MAX_UPLOAD_MB`, default 200). `201 {"name": "<saved name>"}` |
| GET | `/api/videos/<name>` | Video info |
| GET | `/api/videos/<name>/file` | The original video |
| GET | `/api/videos/<name>/analyze` | Runs the analysis; server-sent events (see below) |
| GET | `/api/videos/<name>/result` | Annotated result video (`?download=1` to download) |
| GET | `/api/videos/<name>/breakdown` | Activity Breakdown data |

## Live camera classification

The React **Live activity classification** page requests camera access only after the user starts it. It scales frames to at most 640 pixels wide, sends them sequentially as `image/jpeg` requests, and does not create or save a recording. The Flask server performs Detectron2 pose detection and LSTM inference. The initial result appears after 32 person detections have filled the pose window and another pose arrives; frames without a detected person do not fill the window.

Start a session with `POST /api/live/sessions`. Send JPEG bytes with `Content-Type: image/jpeg` to `POST /api/live/sessions/<sessionId>/frames`; each response contains `framesSeen`, `posesBuffered`, `personDetected`, and nullable `label`/`confidence`. Stop with `DELETE /api/live/sessions/<sessionId>`. Frames are limited to 2 MB and 1920 pixels per dimension; the server allows at most four active sessions, expiring idle sessions after ten minutes. Use localhost or HTTPS for browser camera permissions.

`/result` and `/breakdown` return 404 until a complete result for the current input is available. Both endpoints use the same completion record. Replacing an upload invalidates that record before replacing the input, so old predictions and performance cannot be reused for the new video. The success response remains `201 {"name": "..."}`. A conflicting upload returns JSON status 409; a filesystem upload failure returns generic JSON status 500.

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

`label` is `null` while the first 32 accepted poses fill the buffer; the first prediction uses those poses on the next accepted pose frame. `percent: 100` is progress only: wait for `done` before requesting results. When the stream ends with `done`, two files exist next to the video:

- `res_<name>.mp4`: annotated video, re-encoded to H.264 when ffmpeg (`imageio-ffmpeg`) is available
- `res_<name-without-.mp4>.json`: every prediction (frame number and all 6 class probabilities), plus final `performance`. Each run stages unique files. The completed video is published first and the sidecar is atomically published last as the completion record. The sidecar includes a `result` object with `runId`, `inputSha256` and `videoSha256`; API lookups verify these bindings. Sidecar serialization, hashing during publication, and publication are excluded from processing time.

Opening `/analyze` again starts a new run and invalidates the old completion record. A per-video filesystem lock serializes uploads, analysis, and result verification. If video publication or metadata publication fails, no complete result is exposed and no `done` event is emitted. Cleanup addresses only the current run's staging files; previous output videos are not broadly deleted. Old sidecars without content bindings remain readable only if neither output predates the input.

A lock left by a killed process fails closed: stop/check the owner before manually recovering that specific lock. Automatic stale-lock deletion is intentionally not performed. Hash verification reads video content, so metadata/result lookup cost scales with video size. On Windows, an active download may prevent replacement of its open video; the new run then fails without publishing a completion record.

Processing exceptions produce an SSE `error` event, not `done`. A disconnected stream closes its underlying analysis iterator. Capture and writer resources are released when analysis stops. Corrupt or unreadable breakdown sidecars return a JSON error with status 500. `GET /api/videos/<name>` probing failures return generic JSON: 422 for invalid/unsupported video metadata, 404 when the input disappears, and 500 for I/O or decoder failures. Internal exception details are not included. Successful metadata fields are unchanged.

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

## Optional classification smoothing (not implemented here)

The following fields remain planned. Processing and model metadata support are described separately below.

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

## Processing performance (implemented)

Existing `progress`, `done`, and `error` event types and existing fields are preserved. New analyses add a `metrics` object to each progress event. Older services may omit the object. Each measurement is a nonnegative finite number or `null`; unavailable values are never substituted with zero.

Synthetic/mock progress example (not a benchmark):

```json
{
  "type": "progress", "percent": 50, "frame": 17, "totalFrames": 34,
  "label": null,
  "metrics": { "frameMs": 10.0, "poseMs": 7.0, "lstmMs": null, "fps": 50.0, "memoryMb": 123.0 }
}
```

| Field | Unit and definition |
|---|---|
| `frameMs` | Milliseconds from successful frame read/decode through output write, including drawing and tensor preparation |
| `poseMs` | Milliseconds for Detectron2 detection, filtering and keypoint transfer to CPU; null for skipped pose frames |
| `lstmMs` | Milliseconds for forward pass, softmax, probability extraction and class selection; excludes tensor preparation; null when no prediction runs |
| `fps` | Cumulative processed frames / active elapsed seconds, including setup; not the source video's FPS and not a rolling average |
| `memoryMb` | Sampled RSS of the backend process in decimal MB (1 MB = 1,000,000 bytes); null when psutil or the measurement is unavailable |

The existing `/api/videos/<name>/breakdown` response adds the stored `performance` object. Older sidecars without it remain supported and omit the object. Incomplete stored measurements are normalized to null.

Synthetic/mock final summary example (not a benchmark):

```json
{
  "totalS": 1.0, "avgFrameMs": 20.0, "avgPoseMs": 15.0,
  "avgLstmMs": 1.0, "avgFps": 34.0, "peakMemoryMb": 123.0,
  "framesProcessed": 34, "predictionCount": 2
}
```

This object appears under `performance` beside `summary`, `predictions`, and the other existing breakdown fields.

| Field | Definition |
|---|---|
| `totalS` | Active seconds including video metadata probing, capture/writer setup, decoding, processing, writing/encoding, resource release and H.264 transcoding if used |
| `avgFrameMs` | Sum of successful frame times / processed frame count |
| `avgPoseMs` | Sum of pose stage times / actual pose calls |
| `avgLstmMs` | Sum of LSTM stage times / actual predictions |
| `avgFps` | Processed frames / `totalS`, including encoding finalization |
| `peakMemoryMb` | Highest observed process RSS sample; not an exact peak, GPU memory, or per-video allocation |
| `framesProcessed` | Number of successfully decoded and written frames, including frames with no person or skipped inference |
| `predictionCount` | Number of LSTM classifications generated |

Elapsed measurements use `time.perf_counter()`. CUDA stages synchronize on the model's device. Callback execution and time suspended at a yielded event are excluded. JSON sidecar serialization/publication and model initialization before analysis are excluded. Therefore `avgFps` need not equal `1000 / avgFrameMs`. Transcoding child-process memory is not included in backend RSS.

No predictions means `predictionCount: 0` and `avgLstmMs: null`, not a zero inference time. An empty decoded stream has null frame/stage averages. A positive elapsed duration with zero frames has zero effective FPS; zero elapsed duration has null FPS. Invalid capture or writer initialization raises an error. Optional callbacks on `analyse_video_events(..., on_progress=..., on_complete=...)` receive these dictionaries; completion runs only after successful result publication. The legacy `analyse_video()` percentage SSE and callbacks retain their existing behavior.

Related React types and formatters accept nullable measurements and display `—`. Performance and model information remain visible even if no predictions exist. No smoothing behavior was changed.

## Model information (implemented)

`GET /api/model` returns the six labels in LSTM output order, the window size, and `metrics`. With no valid offline evaluation configured, its response is:

```json
{
  "labels": ["JUMPING", "JUMPING_JACKS", "BOXING", "WAVING_2HANDS", "WAVING_1HAND", "CLAPPING_HANDS"],
  "windowSize": 32,
  "metrics": null
}
```

No evaluation or training runs inside this request. Uploaded-video confidence is never treated as accuracy. Genuine labeled evaluation is still required; no evaluation artifact is shipped or created by this integration.

When an independently reviewed artifact is available, set `MODEL_EVALUATION_PATH` to its JSON path before starting Flask. Its schema is:

| Artifact field | Required content |
|---|---|
| `checkpointSha256` | SHA-256 of `models/saved_model.ckpt` |
| `labels` | Exact six labels in the order above |
| `windowSize` | 32 |
| `split` | Nonempty description identifying the evaluated split |
| `provenance` | Nonempty record of labeled data source, preprocessing and evaluation procedure |
| `metrics` | `calculate_classification_metrics(y_true, y_pred)` output, augmented with a nonempty `dataset` name |

The loader checks checkpoint identity, ordering, dimensions, sample counts, supports, finite scores, and consistency with the confusion matrix. Unknown artifact, metrics, and per-class fields are rejected. Counts must be integers in 0–9,007,199,254,740,991 (sampleCount must be positive); scores must be finite fractions in 0–1. Artifact text is limited to 1,000,000 characters. API model output is independently validated and contains only explicitly allowed fields. Keep full precision when saving scores. Missing, unreadable, mismatched or invalid artifacts produce `metrics: null` and a server warning. Structural validation cannot prove that ground truth or provenance is genuine: configure only a reviewed offline artifact. A split used for checkpoint selection must be described as validation, not an independent test set.

The non-null `metrics` response contains:

- `dataset`: dataset/split name.
- `sampleCount`: positive number of labeled sequences evaluated.
- `accuracy`: correct predictions / total samples.
- `macroPrecision`, `macroRecall`, `macroF1`: arithmetic means across all six classes, including absent classes.
- `weightedPrecision`, `weightedRecall`, `weightedF1`: averages weighted by actual class support (additional fields ignored by older frontend clients).
- `perClass`: six entries in label order, each containing `label`, `precision`, `recall`, `f1`, and integer `support`.
- `confusionMatrix`: 6 × 6 integer counts; rows are actual classes and columns predicted classes, in `labels` order. Counts sum to `sampleCount`.

All classification scores are fractions from 0 to 1, not percentages. Undefined precision, recall and F1 are zero. Confidence charts continue to use each prediction's softmax probabilities and are independent of labeled evaluation metrics.
