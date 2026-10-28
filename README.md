# Human Activity Detection: CSCE 5214 Group Project

Selected open-source project: <https://github.com/shsarv/Machine-Learning-Projects/tree/main/Human%20Activity%20Detection>

This repository is **Group 7's** working copy of that project for CSCE 5214 (Fall 2026, UNT). It contains the original application plus the fixes needed to run it on current macOS, Linux and Windows machines.

**Group 7:** Joshua Yao · Hannah Clethan · Joshua Rithaler · Frank Torralva

---

## What the app does

A web app (React frontend, Flask backend) that classifies human actions in a video:

1. **Pose detection:** Detectron2's Keypoint R-CNN finds 17 body keypoints in every frame.
2. **Buffering:** the (x, y) keypoints from 32 consecutive frames form one input window (32 × 34 values).
3. **Classification:** an LSTM (PyTorch Lightning, 1 layer, 50 hidden units) predicts one of 6 actions: `JUMPING`, `JUMPING_JACKS`, `BOXING`, `WAVING_2HANDS`, `WAVING_1HAND`, `CLAPPING_HANDS`.
4. **Output:** a copy of the video with the skeleton and predicted action drawn on each frame, playable in the browser and available to download.
5. **Activity breakdown:** a summary of how the model decided: main activity, label changes, average confidence, a timeline of activities, each class's probability over time, time per activity and a segments table.

> The original project README ([docs/ORIGINAL_README.md](docs/ORIGINAL_README.md)) describes a TensorFlow 1.x, 2-layer LSTM trained on OpenPose keypoints. That model lives in `Human_Activity_Recogination.ipynb`. The web app runs the **PyTorch** model in `src/lstm.py` with the checkpoint `models/saved_model.ckpt`.

---

## Requirements

| | macOS | Windows (WSL2, recommended) | Windows (native) | Linux |
|---|---|---|---|---|
| OS | macOS 13+ on Apple Silicon (M1 or newer) | Windows 10 (2004+) or 11 | Windows 10/11, 64-bit | Ubuntu 22.04+ / Debian 12+ |
| Build tools | Xcode Command Line Tools | `build-essential` in Ubuntu | Visual Studio 2022 Build Tools (C++) | `build-essential` |
| Node.js (frontend) | 20.19+ (`brew install node`) | 20.19+ in Ubuntu | 20.19+ (`winget install OpenJS.NodeJS.LTS`) | 20.19+ |
| Disk space | ~3 GB | ~3 GB | ~6 GB (incl. build tools) | ~3 GB |
| GPU | Not needed | Not needed | Not needed | Not needed |

The setup scripts install **Python 3.11** automatically (via [uv](https://docs.astral.sh/uv/)), so you don't need to install Python yourself. They also build the React frontend if Node.js is installed; without Node.js the app falls back to the original page.

Why so much setup? Detectron2 isn't published as a ready-made package, so it's compiled from source on your machine. That takes 3–15 minutes the first time.

---

## Setup on macOS

Open **Terminal** and run:

```bash
git clone https://github.com/ShaggyBoy28/GroupProjectCSCE5014.400.git
cd GroupProjectCSCE5014.400
bash setup.sh
```

- If a dialog asks to install the **Command Line Tools**, click Install, wait for it to finish, then run `bash setup.sh` again.
- When it prints `Setup complete`, start the app:

```bash
bash run.sh
```

Open **<http://127.0.0.1:5001>** in your browser. The first start downloads the Detectron2 model weights (~240 MB) before the page is available.

> Intel Macs aren't supported: current PyTorch releases only ship Apple Silicon builds for macOS.

---

## Setup on Windows

There are two ways to run it on Windows. **Option A (WSL2) is recommended**: it uses the same Linux route we tested, and Detectron2 compiles much more reliably there than natively on Windows.

### Option A: WSL2 (recommended)

WSL2 runs a real Ubuntu Linux inside Windows. You'll still use your normal Windows browser to open the app.

1. **Install WSL2 and Ubuntu (one time).** Open **PowerShell as Administrator** and run:
   ```powershell
   wsl --install -d Ubuntu
   ```
   Restart the computer when asked. After the restart, an Ubuntu window opens and asks you to create a Linux username and password.
2. **Install build tools inside Ubuntu.** In the Ubuntu window:
   ```bash
   sudo apt update && sudo apt install -y build-essential git curl
   ```
3. **Clone into your Linux home folder** (not `/mnt/c/...`, which is much slower) and run setup:
   ```bash
   cd ~
   git clone https://github.com/ShaggyBoy28/GroupProjectCSCE5014.400.git
   cd GroupProjectCSCE5014.400
   bash setup.sh
   ```
4. **Start the app:**
   ```bash
   bash run.sh
   ```
5. Open **<http://localhost:5001>** in your Windows browser (Edge, Chrome, …).

To open the project files from Windows (e.g., in VS Code), run `explorer.exe .` from the project folder in Ubuntu, or use VS Code's *WSL* extension.

### Option B: Native Windows

> **Not tested by the group yet.** If the Detectron2 build fails, use Option A.

1. **Install Git and the C++ Build Tools (one time).** In **PowerShell**:
   ```powershell
   winget install --id Git.Git -e
   winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```
   Close and reopen PowerShell afterwards so the new tools are on your PATH.
2. **Clone and run setup:**
   ```powershell
   git clone https://github.com/ShaggyBoy28/GroupProjectCSCE5014.400.git
   cd GroupProjectCSCE5014.400
   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
   ```
3. **Start the app:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run_windows.ps1
   ```
4. Open **<http://127.0.0.1:5001>** in your browser.

`-ExecutionPolicy Bypass` only applies to that one command; it doesn't change your system settings.

---

## Setup on Linux

```bash
sudo apt update && sudo apt install -y build-essential git curl
git clone https://github.com/ShaggyBoy28/GroupProjectCSCE5014.400.git
cd GroupProjectCSCE5014.400
bash setup.sh
bash run.sh
```

Then open <http://127.0.0.1:5001>. On Linux, `setup.sh` installs the CPU-only PyTorch build (~200 MB instead of several GB of CUDA libraries).

---

## Using the app

1. **Home:** choose an `.mp4` and click **Analyze**, click **Analyze sample**, or choose **Start live camera** for predictions from your webcam.
2. **Analyzing:** shows the video, the percentage, frames analyzed and the current label. It runs on the CPU, so expect a few minutes for the 10-second sample. Keep the page open; the results open automatically.
3. **Result video:** the annotated video with the skeleton and `Action: …` label, plus length, frames, frame rate and resolution. **Download video** saves it.
4. **Activity breakdown** (tab next to Result video): summary cards, timeline, confidence over time, time per activity and segments.

The original page is still available at <http://127.0.0.1:5001/legacy>.

Stop the app with **Ctrl+C** in the terminal.

Live camera classification requires camera permission and a secure browser context (localhost or HTTPS). Camera frames are sent to the app server for inference; the live flow does not save a recording.

**What to expect from the output:**
- **No label for the first ~1 second:** the model needs 32 frames of poses before its first prediction.
- **Labels lag ~1 second behind the action:** each prediction covers the previous 32 frames.
- **Only the first person detected** in each frame is analyzed.
- **Results** are saved as `res_<name>.mp4` (video) and `res_<name>.json` (predictions) in the project folder. They're ignored by git.

---

## Project structure

```
.
├── app.py                      # Flask app: model loading, serves the React build and the legacy page
├── frontend/                   # React + TypeScript + Tailwind UI
├── src/
│   ├── api.py                  # JSON API used by the frontend + Activity Breakdown math (docs/API.md)
│   ├── video_analyzer.py       # frame loop: pose → 32-frame buffer → LSTM → annotated video
│   ├── lstm.py                 # LSTM model, dataset and data module (PyTorch Lightning)
│   ├── train.py                # training entry point
│   └── utils.py                # keypoint filtering and skeleton drawing
├── templates/index.html        # original web page (Bootstrap + jQuery), served at /legacy
├── models/saved_model.ckpt     # trained LSTM checkpoint used by the app
├── sample_video.mp4            # demo video
├── images/                     # validation accuracy/loss curves from training
├── lstm_train.ipynb            # PyTorch training notebook
├── Human_Activity_Recogination.ipynb   # original TensorFlow research notebook
├── Human_Action_Classification_deployment_with_ngrok.ipynb
├── requirements-modern.txt     # dependencies that install today (used by the setup scripts)
├── requirements.txt            # original 2021 pins, kept for reference
├── setup.sh / run.sh           # macOS, Linux, WSL2
├── setup_windows.ps1 / run_windows.ps1   # native Windows
└── docs/
    ├── API.md                  # API reference
    └── ORIGINAL_README.md      # upstream README
```

## Training the LSTM

After running `bash setup.sh`, activate the project environment and start training from the repository root:

```bash
source .venv/bin/activate
python -m src.train -d ./RNN-HAR-2D-Pose-database
```

The script trains for up to 400 epochs by default, stopping early when validation loss stops improving. It saves the best checkpoint to `models/saved_model.ckpt`, which is the checkpoint loaded by the app. You can adjust the run with `--epochs`, `--batch_size`, and `--learning_rate`.

The current dataset provides `X_train.txt`/`Y_train.txt` and `X_test.txt`/`Y_test.txt`; the training script uses the latter pair as its per-epoch validation data and to select the best checkpoint. It does not provide a separate final test run, so treat those validation scores as model-selection results rather than an independent test estimate.

---

## Development

**Frontend:** with the backend running (`bash run.sh`), start the dev server for hot reload:

```bash
cd frontend
npm ci
npm run dev        # http://localhost:5173, API calls are proxied to Flask on port 5001
```

Run `npm run build` afterwards so Flask serves your changes at port 5001. Building needs Node.js 20.19+.

Frontend layout (`frontend/src/`): `api.ts` (API types and calls), `lib.ts` (activity names/colors, formatting, data hooks), `components.tsx` (shared UI), `pages/` (Home, Analyzing, Results, Breakdown), `index.css` (Tailwind and the Figma colors).

---

## Changes from the original project

The upstream app doesn't run on current machines as-is. We made these changes:

| Problem | Fix | File |
|---|---|---|
| `requirements.txt` pins 2021 packages (e.g., `torch 1.13`, `opencv 4.2`) that don't install on Apple Silicon or Python 3.10+; its Flask and Werkzeug pins also conflict with each other | New `requirements-modern.txt`; the original file is kept for reference | `requirements-modern.txt` |
| Detectron2 isn't listed at all | Setup scripts build it from source at a pinned, tested commit | `setup.sh`, `setup_windows.ps1` |
| Detectron2 still imports `pkg_resources`, which `setuptools` 81 removed | Pin `setuptools<81` | `requirements-modern.txt` |
| PyTorch Lightning 2 removed APIs used by the original trainer | Use Lightning 2 trainer options and epoch-level metric logging | `src/train.py`, `src/lstm.py` |
| Detectron2 defaults to an NVIDIA GPU (CUDA) | Use the CPU when no GPU is available | `app.py` |
| PyTorch 2.6+ refuses to load the 2021 checkpoint by default | Load it with `weights_only=False`, `map_location="cpu"` | `app.py` |
| Newer OpenCV rejects tensor coordinates when drawing the skeleton | Convert keypoints to plain integers | `src/utils.py` |
| macOS uses port 5000 for AirPlay Receiver | Default to port 5001, configurable with `PORT` / `HOST` | `app.py` |
| Auto-reloader loaded both models twice | Reloader off | `app.py` |
| OpenCV's `mp4v` result videos don't play in browsers | Re-encode results to H.264 with the ffmpeg bundled in `imageio-ffmpeg` | `src/video_analyzer.py` |
| Result videos were always written at 30 fps | Use the input video's frame rate | `src/video_analyzer.py` |

### New in this repo

- **React frontend** (`frontend/`) implementing the Phase 2 design: Home, Analyzing, Result video and the new **Activity breakdown** page.
- **JSON API** (`src/api.py`, [docs/API.md](docs/API.md)) with live progress over server-sent events.
- **Breakdown data** (`src/api.py`): the analysis now keeps all 6 class probabilities for every prediction and turns them into segments and summary statistics.

The original research notebooks and legacy template are retained. The Python training entry point and LSTM hooks have been updated for PyTorch Lightning 2.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Address already in use` / port 5001 busy | Run on another port: `PORT=5050 bash run.sh` (Windows: `$env:PORT="5050"` first, then the usual `run_windows.ps1` command) |
| Page returns **403** at port 5000 on a Mac | That's AirPlay Receiver. Use port 5001 (the default here). |
| Detectron2 build fails on macOS | Make sure `xcode-select -p` prints a path; if not, run `xcode-select --install`, then `bash setup.sh` again |
| Detectron2 build fails on Linux/WSL2 | Run `sudo apt install -y build-essential`, then `bash setup.sh` again |
| Detectron2 build fails on native Windows | Check the Build Tools install includes "Desktop development with C++"; otherwise use WSL2 (Option A) |
| `ModuleNotFoundError: No module named 'pkg_resources'` | `setuptools` got upgraded. Run `uv pip install "setuptools<81"` inside the venv. |
| First start seems stuck | It's downloading the ~240 MB Detectron2 weights (cached afterwards in `~/.torch/iopath_cache`) |
| Analysis is slow | Expected: pose detection runs on the CPU, so the 10-second sample takes several minutes. Shorter or lower-resolution videos finish faster. |
| Want to start over | Delete the `.venv` folder and run setup again |

Run `bash setup.sh 2>&1 | tee setup.log` to save the full setup output when asking for help.

---

## Credits

- Original project: [shsarv/Machine-Learning-Projects: Human Activity Detection](https://github.com/shsarv/Machine-Learning-Projects/tree/main/Human%20Activity%20Detection)
- Pose estimation: [Detectron2](https://github.com/facebookresearch/detectron2) (Meta AI Research)
- Training data: [Berkeley MHAD](http://tele-immersion.citris-uc.org/berkeley_mhad), with 2D poses extracted by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- LSTM approach based on Guillaume Chevalier's [LSTMs for Human Activity Recognition](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition) (see the original README's references)
