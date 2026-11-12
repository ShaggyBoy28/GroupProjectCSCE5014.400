# Expanding the dataset

Phase 2 extension: add action classes, replace the single train/test split with cross-validation, and
split the data by quality and environment. This page covers the tooling and how to use it.

## What's in the code

| File | Purpose |
|---|---|
| `src/lstm.py` | The model, ported to PyTorch Lightning 2.x. `num_classes` and `class_names` are saved in the checkpoint. The shipped `models/saved_model.ckpt` still loads. |
| `src/pose_dataset.py` | Loads the dataset windows and recovers the 1,438 source videos (consecutive windows share 26 of 32 frames). Builds block folds and the quality/size measures, and loads extracted clips. |
| `scripts/train_model.py` | `standard`: retrain on the original split. `cv`: k-fold cross-validation with reports. `final`: train on all data for the app. |
| `src/pose_extraction.py`, `src/dataset_api.py`, `frontend/src/pages/DatasetPage.tsx` | The Dataset page: upload recordings in the browser, with pose extraction in a background thread (`/api/dataset`). |
| `scripts/extract_poses.py` | Runs the app's Detectron2 pose detector over videos in `data/recordings/<person>/<activity>/` and writes pose sequences plus an index. |
| `app.py` | `MODEL_CHECKPOINT` selects the model. Class names come from the checkpoint, so a model with new classes shows them in the UI. |

Reports and checkpoints go to `models/training/` (git-ignored).

## Cross-validation and quality/environment splits

```bash
python scripts/train_model.py cv --folds 5 --epochs 60 --lr 1e-3 --compare-random
```

**Block CV.** The dataset has no subject or camera IDs. Within each class, videos appear in recording
order, and neighbouring videos are repetitions by the same person and view. Folds are therefore
contiguous blocks of videos per class, which approximates leaving out whole people and views.

**Random CV (`--compare-random`).** This is shown for contrast. Overlapping windows from one video land
on both sides, so its scores are inflated.

The report breaks the original dataset into three buckets two ways:

- **Quality:** the share of keypoints OpenPose missed in the window.
- **Person size:** shoulder-to-ankle height in pixels, i.e. how near the camera the person is.

Added recordings are folded by their real subject, so a fold's test people are unseen. They are also
reported per `source: environment`.

## Adding classes (group recordings)

New classes come from the group's own videos: sit down / stand up, walking, throwing. Record the six
existing activities as well. Otherwise the new classes would be the only home-recorded, Detectron2-extracted
data, and the model could learn "home vs lab" instead of the activity. Recording the six also gives the
environment comparison (Berkeley MHAD lab vs home).

### Recording tips
- One person in frame, whole body visible head to feet, camera still (tripod or propped phone).
- 3+ takes per activity, 15–20 s of continuous activity each, one activity per clip.
- Vary rooms, distance and angle across people.
- At least 3–4 different people, so leave-one-person-out cross-validation is meaningful.

### Ingesting from the app (easiest)
Start the app (`bash run.sh`, or start.command) and open **Dataset** in the header (`/dataset`).

1. Choose one or more videos.
2. Enter who is in them and pick the activity. "Other" creates a new class.
3. Optionally, enter the room.
4. Click **Add**.

Each video is saved to `data/recordings/<person>/[<room>/]<activity>/` and processed in the background,
one at a time, with a live progress bar. You can keep uploading or close the page while it runs.

The page shows totals per activity (videos, people, training windows) and flags activities with fewer
than 3 people. Each video gets a preview frame, so you can check the right person was found, and it
flags clips too short to train on. Remove deletes the video and its pose data. Training is still a
command; the page shows it.

### Ingesting from the command line: drop videos into folders
The folders say who and what; no spreadsheet is needed.

```
data/recordings/
  josh/
    walking/take1.mp4
    sit_stand/take1.mp4
    jumping_jacks/IMG_0412.MOV
  amy/
    garage/                 ← optional room level: <person>/<environment>/<activity>/
      throwing/clip.mp4
```

Folder names are forgiving (case, spaces and dashes don't matter). Recognised names:

| Class | Folder names |
|---|---|
| WALKING | walking, walk |
| THROWING | throwing, throw |
| SIT_DOWN_STAND_UP | sit_stand, sit_down_stand_up, sit_down, stand_up |
| JUMPING | jumping, jump |
| JUMPING_JACKS | jumping_jacks, jacks |
| BOXING | boxing, punching |
| WAVING_2HANDS | waving_2hands, waving_two_hands |
| WAVING_1HAND | waving_1hand, waving_one_hand |
| CLAPPING_HANDS | clapping, clap |

Any other folder name becomes a new class with that name.

```bash
python scripts/extract_poses.py                     # processes every new video under data/recordings/
python scripts/extract_poses.py ~/Downloads/test.mov --person josh --activity walking   # one video
```

Each video prints its pose frames and how many 32-frame training windows it gives. A clip with fewer
than 32 pose frames gives none. Re-running only processes new videos. Check `data/extracted/previews/`
to confirm the right person was picked up.

### Train and evaluate

```bash
python scripts/train_model.py cv --extra data/extracted/index.csv --epochs 60 --lr 1e-3
python scripts/train_model.py final --extra data/extracted/index.csv --epochs 60 --lr 1e-3
MODEL_CHECKPOINT=models/training/final.ckpt bash run.sh
```

Pose extraction runs Detectron2 on the Mac's CPU, at roughly a second per frame. A 15 s clip at 22 fps
is about 330 frames.

The `final` model is trained on every window, including `X_test`, so judge it by the `cv` report, not
by `scripts/evaluate_model.py`.

## Known limits

- **Frame rate.** MHAD windows are 22 Hz. The extraction resamples to 22 fps to match, but the app
  analyses uploads at their native rate, so 32 frames covers about 1.07 s at 30 fps.
- **Coordinates.** Keypoints are raw pixels, as in the app and training data. A larger or closer
  person changes the input scale, which the person-size breakdown shows.
- **Missing joints.** OpenPose stored missing joints as 0,0, while Detectron2 always predicts all 17
  keypoints.
