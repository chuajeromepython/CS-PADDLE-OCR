<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*qklqV9yl7bw8tzkyhCALNw.png" alt="CS-PADDLE-OCR" height="200" width="500">
</p>

<p align="center">Handwriting OCR: PaddleOCR (PP-OCRv6) with a fine-tuned recognizer for journal handwriting</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square" alt="python version">
  <img src="https://img.shields.io/badge/model-PP--OCRv6__medium__rec-yellow?style=flat-square" alt="model">
  <img src="https://img.shields.io/badge/framework-PaddleOCR%203.x-informational?style=flat-square" alt="framework">
</p>

---

A command-line tool that reads handwriting from a photo and writes the recognized text to a `.txt` file. Under the hood it's PaddleOCR, with two additions:

1. **A fine-tuned recognition model** (`journal_v7_infer/`) trained on handwritten journal lines plus samples from other writers, so it copes with messy handwriting better than the stock model.
2. **A reading-order fix** (`ocr_utils.py`) that regroups detected words into visual rows. Handwritten words on one line sit at slightly different heights, so plain top-to-bottom sorting can output "Chua / Jerome / S." instead of "Jerome S. Chua".

It runs entirely on your own computer, in a terminal — no web app, no API key, no per-request cost. You give it a photo, it prints each line with a confidence score and saves the text.

---

## 1. Install Python

You need Python **3.9 or newer**. Python 3.12 is a safe choice, since PaddlePaddle wheels are most reliable on slightly older Python versions and brand-new releases may not be supported yet.

1. Go to [python.org/downloads](https://www.python.org/downloads/) and download the installer for your OS.
2. **On Windows**, tick "Add python.exe to PATH" if this is your only Python. If you have several versions installed, leave it unticked and use the `py` launcher in Section 4 to pick the right one.

To check it worked, open a terminal (PowerShell / Command Prompt / Terminal) and run:

```
py -3.12 --version        # Windows
python3.12 --version      # macOS / Linux
```

You should see `Python 3.12.x`. On Windows, `py -0p` lists every Python you have installed.

## 2. Install Git

Check Git is installed:

```
git --version
```

If it's not found, download it from [git-scm.com](https://git-scm.com/downloads) and install with default options. Git LFS is **not** needed for this project — the fine-tuned model (~74 MB) is stored as normal files.

## 3. Download the project

Open a terminal, navigate to wherever you want the folder to live (e.g. your Desktop), and run:

```
git clone https://github.com/chuajeromepython/CS-PADDLE-OCR.git
cd CS-PADDLE-OCR
```

The repo includes the training crops and sample images, so the clone is larger than a typical code-only repo.

## 4. Set up a virtual environment and install packages

A **virtual environment** (venv) is a private folder of Python packages just for this project. It keeps this project's `paddlepaddle` and `opencv` from clashing with anything else on your computer.

Open a terminal **inside the `CS-PADDLE-OCR` folder** (in VS Code: open the folder, then press `` Ctrl+` ``).

**Step 1: Create the venv.** The repo's `.gitignore` excludes `paddle-env/`, so that's the recommended name:

| OS | Command |
|---|---|
| Windows | `py -3.12 -m venv paddle-env` |
| macOS / Linux | `python3.12 -m venv paddle-env` |

**Step 2: Activate it.**

| Shell | Command |
|---|---|
| Windows PowerShell | `paddle-env\Scripts\activate` |
| Windows Command Prompt | `paddle-env\Scripts\activate.bat` |
| macOS / Linux | `source paddle-env/bin/activate` |

Your prompt should now start with `(paddle-env)`. If PowerShell says "running scripts is disabled", run this once and try again:

```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Step 3: Install the packages.**

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs `opencv-python`, `paddleocr` (3.0 or newer), `paddlepaddle` and `jiwer` (used for scoring). It may take a few minutes, since `paddlepaddle` is a large download.

**Step 4 (VS Code only): Select the interpreter.** Press `Ctrl+Shift+P`, run **Python: Select Interpreter**, and choose the one inside `paddle-env`.

**To leave the venv**, run `deactivate`.

> **Git note:** `paddle-env/`, `__pycache__/` and `debug_output/` are listed in `.gitignore`, so they are never committed. If you name your venv something else, add it to `.gitignore` too, or Git will try to commit it. Run `git status` before committing: if it lists a huge folder you don't recognize, don't add it.

---

## 5. Run the script

Make sure your venv is active (you should see `(paddle-env)` in your prompt), then run it on a photo:

```
python CS-PADDLE-OCR.py path/to/photo.jpg
```

By default this uses the fine-tuned model in `journal_v7_infer/`. To try it on a sample image that ships with the repo:

```
python CS-PADDLE-OCR.py test_images/test_img1.jpg
```

**An internet connection is only needed the first time.** PaddleOCR downloads its small detection and orientation models on first run (roughly 10–15 MB for the English defaults) and caches them. After that everything runs offline. The fine-tuned recognizer is already in the repo.

You'll see loading messages first:

```
Loading PaddleOCR (models already cached from your last run)...
Using fine-tuned recognition model: .../journal_v7_infer
Running OCR on ...
```

### Options

| Option | What it does |
|---|---|
| `--stock` | Use the stock PaddleOCR recognition model instead of the fine-tuned one. Handy for side-by-side comparison. |
| `--rec-model-dir PATH` | Use a different fine-tuned recognition model folder. |
| `--rec-thresh 0.5` | Drop recognized regions below this confidence. Default `0` keeps everything. |
| `--edge-margin 0.08` | Drop short (3 characters or fewer) text within this fraction of the left/right image edge, which removes stray marks from notebook margins. Default `0` is off. |

Examples:

```
python CS-PADDLE-OCR.py photo.jpg --stock
python CS-PADDLE-OCR.py photo.jpg --rec-thresh 0.4 --edge-margin 0.08
python CS-PADDLE-OCR.py photo.jpg --rec-model-dir ./some_other_model
```

## 6. Reading the output

The script prints one line per detected text row, in reading order, with its confidence:

```
Detected 12 line(s)/text region(s).

Line 1/12 (conf 0.97): Jerome S. Chua
Line 2/12 (conf 0.91): Today I went to the ...
...
Saved: .../debug_output/photo_extracted_text_paddle_finetuned.txt
```

Low confidence on a line usually means messy handwriting or a bad crop — that's the line to double-check by eye.

## 7. Where the results are saved

A `debug_output/` folder is created **next to your photo**. It contains:

- `<photo>_extracted_text_paddle_finetuned.txt` — the recognized text, one line per row (`..._paddle.txt` without `_finetuned` if you used `--stock`).
- `<photo>_resized_for_ocr.jpg` — only if your photo was downscaled (see below).

**About the downscale:** photos larger than 2000 px on the longest side are shrunk to 2000 px before OCR. A full-resolution phone photo (e.g. 3456x4608) can take around 3 minutes on CPU, while the downscaled version is much faster and still legible. If tight handwriting is being misread, you can raise `MAX_SIDE` at the top of `CS-PADDLE-OCR.py`.

---

## 8. Measuring accuracy (CER / WER)

To turn "it feels better" into an actual number, use `ocr_training/evaluate_ocr.py`. It scores a model against ground-truth transcriptions using **Character Error Rate** and **Word Error Rate**.

1. Make a folder of image / transcription pairs. Each `.txt` holds the correct text, one line per line of handwriting, in reading order:
   ```
   eval_set/
     test_img1.jpg
     test_img1.txt
     test_img3.jpg
     test_img3.txt
   ```
   The repo already includes `eval_set/` (your own handwriting) and `eval_others/` (held-out pages from other writers).

2. Run it:
   ```
   python ocr_training/evaluate_ocr.py eval_set --rec-model-dir journal_v7_infer
   python ocr_training/evaluate_ocr.py eval_set --rec-model-dir journal_v7_infer --compare-stock
   python ocr_training/evaluate_ocr.py eval_others --rec-model-dir journal_v7_infer --compare-stock
   ```
   `--compare-stock` runs the stock model too, so you can see the fine-tuned model's gain (or loss) directly. Lower CER/WER is better.

3. Make **one change at a time** (a threshold, a model, new training data), then re-run so you know which change moved the score.

---

## 9. Fine-tuning on your own handwriting (optional)

The model in `journal_v7_infer/` is `PP-OCRv6_medium_rec` fine-tuned on a mix of data. Training is done on a free GPU in Google Colab, not on your PC.

**What it was trained on:**

| Data | Where |
|---|---|
| Your own journal line crops (plus augmented copies) | `journal_crops/` (`train_list.txt`, `val_list.txt`, `test_list.txt`) |
| Pages from other writers (Student Messy Handwritten Dataset, SMHD) | `other_writers/` |
| An IAM line subset (optional, you build and upload it) | made in the notebook, step 5 |

Training uses a low learning rate and trains the CTC head only, so the model adapts to handwriting without forgetting what the stock model already knows. The best checkpoint is picked on a validation set that includes other writers, not just your own hand.

**Steps:**

1. **Add your photos.** Put your handwriting photos in `journal_photos/` (this folder is git-ignored, so your photos stay private).
2. **Make line crops.** Cut photos into line crops that match what the recognizer sees at inference:
   ```
   python reexport_crops_polygon.py journal_photos journal_crops_poly
   ```
   Check a few crops by eye. If they look right, move `journal_crops_poly/*.png` over `journal_crops/` (keep the `*_list.txt` files), and write the matching transcription for each crop in the list files (`path<TAB>text`, paths relative to the repo root).
3. **(Optional) Add other writers.** If you have the SMHD dataset in an `SMHD/` folder:
   ```
   python ocr_training/prepare_smhd_eval.py --smhd SMHD --out eval_others
   python ocr_training/prepare_smhd_train.py --smhd SMHD --manifest eval_others/split_manifest.json --out other_writers
   ```
   The first command builds a held-out test set and keeps those pages out of training. The second builds training crops from the remaining pages. It can be stopped (`Ctrl+C`) and restarted, and `--limit 10` tries it on 10 pages first.
4. **Commit and push** your new crops and lists so Colab can pull them.
5. **Open `colab_finetune_journal_v7.ipynb` in Google Colab**, set the runtime to GPU, and run the cells top to bottom. It clones this repo, installs PaddlePaddle and PaddleOCR, builds the data mix, trains, and exports an inference model.
6. **Drop the export in.** Replace the contents of `journal_v7_infer/` (or add a new folder and pass `--rec-model-dir`). The script uses `journal_v7_infer/` automatically on its next run — no code changes needed.
7. **Re-run the accuracy test** (Section 8) to confirm the new model actually beats the old one.

---

## Project layout

| Path | What it is |
|---|---|
| `CS-PADDLE-OCR.py` | Main script: photo in, text out |
| `ocr_utils.py` | Reading-order fix, confidence filter and edge-fragment filter |
| `journal_v7_infer/` | The fine-tuned recognition model (inference format) |
| `ocr_training/` | Evaluation (`evaluate_ocr.py`) and SMHD data-prep scripts |
| `reexport_crops_polygon.py` | Re-cuts training crops to match inference-time cropping |
| `colab_finetune_journal_v7.ipynb` | Colab notebook that fine-tunes the model |
| `journal_crops/`, `other_writers/` | Training crops and label lists |
| `eval_set/`, `eval_others/` | Held-out test images with ground-truth text |
| `test_images/` | Sample photos to try the script on |

## Running it again later

You don't need to reinstall anything or recreate the venv. Open a terminal, go back into the folder, activate the venv, and run:

```
cd CS-PADDLE-OCR
paddle-env\Scripts\activate          # Windows PowerShell  (macOS/Linux: source paddle-env/bin/activate)
python CS-PADDLE-OCR.py path/to/photo.jpg
```

If you pull updates that change `requirements.txt`, refresh your packages with `python -m pip install -r requirements.txt`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` / `pip` not recognized | On Windows use `py -3.12 ...` to create the venv, then activate it (Section 4); inside the venv, `python` always works. On macOS/Linux try `python3` / `pip3` |
| No `(paddle-env)` in the prompt, or activation fails with "running scripts is disabled" (PowerShell) | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again. Or use Command Prompt and `paddle-env\Scripts\activate.bat` |
| `ModuleNotFoundError` (e.g. `No module named 'paddleocr'` or `'cv2'`) | The venv isn't active, or packages went into a different Python. Activate it, then run `python -m pip install -r requirements.txt` |
| Crash mentioning `ConvertPirAttribute2RuntimeAttribute` | This is a CPU/oneDNN bug in some PaddlePaddle + PaddleOCR combos. The script already disables it with `enable_mkldnn=False`. If you edited that line, put it back |
| Install takes a long time | Normal — `paddlepaddle` is a large package |
| First run needs internet | Expected — PaddleOCR downloads and caches its small detection models once. After that it works offline |
| Script is slow on a big photo | Photos are downscaled to 2000 px first. Lower `MAX_SIDE` in `CS-PADDLE-OCR.py` for more speed, or raise it if small handwriting is misread |
| Output has extra junk like stray marks from the margin | Try `--edge-margin 0.08` and/or `--rec-thresh 0.4` |
| Lines come out in the wrong order or words are split across lines | Reading order is regrouped by `ocr_utils.py`. Very slanted lines can still confuse it; try a flatter, more upright photo |
| Results are worse than expected | Compare against the stock model with `--stock`. If the fine-tuned model isn't better on your handwriting, fine-tune it on your own crops (Section 9) |
| `evaluate_ocr.py` says it needs jiwer | Run `python -m pip install jiwer` (it's already in `requirements.txt`) |
| Git shows hundreds of untracked files (or a huge Source Control badge in VS Code) | Your venv folder isn't ignored. `paddle-env/` is already in `.gitignore`; if you named it differently, add that name |
