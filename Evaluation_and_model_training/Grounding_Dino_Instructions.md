# Grounding DINO – Building Detection Test

This script tests **building detection in street-view images**, using the pretrained Grounding DINO model and detection settings described in the NHRA *Final report on computer vision training for building detection and classification* (July 2026), **Building Detection** section and **Appendix 1**.

It detects buildings and draws bounding boxes. **It does not classify buildings** as Legacy URM, New URM, or non-masonry; those are later stages of the report's pipeline.

## 1. Setup

You need Python 3.11 and a working internet connection the first time you run the script, to download the pretrained model from Hugging Face. You can use **either** Conda **or** `venv` (not both).

### Option A — Conda

```bash
conda create -n grounding-dino python=3.11 -y
conda activate grounding-dino
```

### Option B — Python venv (Linux/macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate using `.venv\Scripts\activate` instead.

### Install packages (either environment)

**For an NVIDIA GPU with a compatible driver**, install a CUDA-enabled PyTorch build. For example, for drivers that support CUDA 12.1:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

If you have **no compatible NVIDIA GPU**, install CPU-only PyTorch instead:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

Then, for either choice:

```bash
python -m pip install 'transformers==4.52.3' 'tokenizers==0.21.4' pillow matplotlib
```

> On a different machine, the appropriate PyTorch/CUDA build may differ. Check `nvidia-smi` and the [PyTorch installation instructions](https://pytorch.org/get-started/locally/). The CUDA build above is an example, not a universal requirement. Avoid installing a CUDA build newer than the installed NVIDIA driver supports.

Check installation:

```bash
python -c "from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor; print('Transformers OK')"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available: False` is normal for CPU-only installation, but inference will be slower.

## 2. Run the script

Run commands from the folder containing `test_grounding_dino_buildings.py`, or provide the full path to that script.

**Test every supported image in a folder:**

```bash
python test_grounding_dino_buildings.py --input-folder ./images
```

**Display bounding boxes in a window:**

```bash
python test_grounding_dino_buildings.py --input-folder ./images --visualise
```

The script displays **one image at a time**. Close the plot window to move to the next image.

**Save detections to a JSON file:**

```bash
python test_grounding_dino_buildings.py --input-folder ./images --output-json results.json
```

**Visualise and save results:**

```bash
python test_grounding_dino_buildings.py \
    --input-folder ./images \
    --visualise \
    --output-json results.json
```

Use quotation marks around paths containing spaces, for example `--input-folder "/home/student/street view images"`.

The script supports `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, and `.tiff` files **directly inside the specified folder** (it does not search subfolders). It loads the model once and processes the images in filename order.

### Parameters you may change

| Argument | Purpose | Example |
|---|---|---|
| `--input-folder` | **Required.** Folder containing images | `--input-folder ./images` |
| `--visualise` | Optional. Open a plot of each image and its detections | `--visualise` |
| `--output-json` | Optional. Write all detection results to JSON | `--output-json results.json` |

The JSON contains each image filename, image dimensions, and detections with bounding box coordinates (`[x1, y1, x2, y2]` in pixels), scores and predicted text labels. Images that fail processing are recorded with an `error` field. Omitting `--output-json` means no results file is saved.

## 3. Detection parameters: keep the report defaults

**Do not change these for the standard experiment.** They are already set in the script to the values given in the report's building-detection appendix.

| Script argument | Default | Meaning |
|---|---:|---|
| `--box-threshold` | `0.25` | Initial Grounding DINO box threshold |
| `--text-threshold` | `0.20` | Text matching threshold |
| `--score-threshold` | `0.30` | Additional detection score filter |
| `--max-raw` | `30` | Maximum raw candidates retained before duplicate filtering |
| `--iou-threshold` | `0.35` | IoU threshold for suppressing duplicate boxes |
| `--containment-threshold` | `0.75` | Suppression threshold for boxes largely contained in another box |
| `--max-detections` | `3` | Maximum retained detections per image |
| `--min-area` | `0.04` | Minimum box area, as a fraction of the image area (4%) |

The script also fixes the model to **`IDEA-Research/grounding-dino-base`** and the prompts to **`building`, `house`, `apartment building`, `facade`**, as described in the report.

The optional `--device` setting defaults to `auto` (CUDA if available, otherwise MPS if available, otherwise CPU). You should normally leave it alone; use `--device cpu` only if GPU execution has problems.

## 4. Troubleshooting

- **First run is slow:** The Grounding DINO model weights are downloaded on first use, then cached. Subsequent runs reuse the cache.
- **`CUDA available: False` despite having an NVIDIA GPU:** Your NVIDIA driver may be too old for the installed PyTorch CUDA build. Check `nvidia-smi`; use a compatible build of PyTorch or temporarily run with `--device cpu`.
- **Expired Hugging Face token / HTTP 401:** The model is public. If an expired token is being sent, try `export HF_HUB_DISABLE_IMPLICIT_TOKEN=1` and `unset HF_TOKEN` in the terminal before running the script.
- **Plots do not appear:** `--visualise` requires a graphical desktop session; it will not work in a typical headless SSH session.
- **Import/version errors:** Check that the intended environment is active, then run `python -m pip check`. The package versions specified above avoid the Transformers/tokenizers mismatch encountered with newer releases.
