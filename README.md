# 3D Spleen Segmentation from CT with MONAI + UNet

[![MONAI](https://img.shields.io/badge/MONAI-1.6.0-blue)](https://monai.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-orange)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-MSD%20Task09-green)](http://medicaldecathlon.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end project that trains a 3D UNet to segment the spleen in
abdominal CT scans, built with [MONAI](https://monai.io/) (Medical Open
Network for AI) and PyTorch. It follows MONAI's official spleen segmentation
tutorial, rewritten as a clean, commented Python project you can read, run,
and extend.

> **Learning project.** I am a radiologic technologist, and I built this to
> learn MONAI and 3D medical image segmentation hands-on: how a model reads
> the CT volumes I acquire every day, and what it takes to turn them into a
> reliable measurement.

## Why spleen segmentation?

The spleen is one of the largest organs in the abdomen, and its size and
shape carry clinical meaning: an enlarged spleen (splenomegaly) can point to
liver disease, infection, or blood disorders; spleen volume is tracked after
trauma and before/after surgery; and in oncology it is a reference organ for
treatment response. Measuring it by hand, tracing the spleen slice by slice
in a CT volume, takes a radiologist 10-20 minutes per case and varies
between readers. An automatic segmentation does it in seconds and gives a
reproducible volume. That is what this project does: given a CT volume, it
produces a 3D mask of the spleen (1 = spleen, 0 = background).

## Dataset

| | |
|---|---|
| **Task** | Medical Segmentation Decathlon (MSD), Task09: Spleen |
| **Modality** | Portal venous phase CT |
| **Size** | 61 3D volumes: **41 training** (with expert masks) + **20 test** (no masks) |
| **Source** | Memorial Sloan Kettering Cancer Center |
| **Format** | NIfTI (`.nii.gz`), 512×512 axial slices |
| **Licence** | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| **Download** | [medicaldecathlon.com](http://medicaldecathlon.com/), or the MONAI mirror used by `download_data.py` (no registration needed) |

Split used here (same as MONAI's official tutorial): 32 cases for training,
9 cases for validation. Please cite the dataset as:
Antonelli, M. et al. *The Medical Segmentation Decathlon.* Nature
Communications 13, 4128 (2022), https://arxiv.org/abs/1902.09063

## Method

- **Model:** 3D UNet: `channels=(16, 32, 64, 128, 256)`, `strides=(2, 2, 2, 2)`,
  `num_res_units=2`, batch normalisation. One input channel (CT), two output
  channels (background / spleen).
- **Preprocessing:** Hounsfield-unit window `[-57, 164]` scaled to `[0, 1]`,
  foreground cropping, RAS orientation, resampling to `1.5 × 1.5 × 2.0 mm`
  voxels, and balanced random `96³` patches (half centred on the spleen).
- **Loss:** Dice loss. **Optimiser:** Adam, learning rate `1e-4`.
- **Metric:** mean Dice score (background excluded) on the 9 validation cases.
- **Validation:** sliding-window inference (`160³` window), because a whole CT
  volume does not fit in GPU memory at once.
- **Training:** 100 epochs by default (`train.py --epochs`). MONAI's official
  tutorial trains 600 epochs; see *Results* below.

## Results

The Dice score measures overlap between the predicted mask and the expert
mask: 0 = no overlap, 1 = perfect overlap. For spleen segmentation, a Dice
above 0.9 is considered excellent.

| Run | Epochs | Validation Dice (mean, 9 cases) |
|---|---|---|
| MONAI official tutorial (`spleen_segmentation_3d.ipynb`) | 600 | **0.9510** (best, epoch 598) |
| My run | 100 | Unstable: 0.03-0.13 at most epochs, one spike to 0.58 at epoch 94 |

> The 0.9510 figure is the result reported in MONAI's official tutorial, not
> a result of this repo.

My 100-epoch run did not produce a stable validation score: Dice sat
between 0.03 and 0.13 at almost every epoch, with a single spike to 0.58
at epoch 94. A diagnostic run on one validation case traced the problem
to a mismatch between the training patch size (96) and the validation
window (160): the same case scored 0.66 with a 96 window and 0.17 with a
160 window. I am now running the diagnostic over all 9 validation cases
to get a trustworthy mean Dice before I report a number here.

## Quickstart

### Option A: Google Colab (free GPU, recommended)

```python
# 1. In Colab: Runtime -> Change runtime type -> T4 GPU
# 2. Clone the repo and install (one cell):
!git clone https://github.com/FaezeHajeb/MONAI-spleen-segmentation.git
%cd MONAI-spleen-segmentation
!pip install -r requirements.txt

# 3. Download the data (~1.6 GB):
!python download_data.py

# 4. Smoke test (1 epoch, 2 cases - proves everything works):
!python train.py --quick

# 5. Real training (about 2-3 hours on a free T4 GPU):
!python train.py --data-dir data/Task09_Spleen --output-dir runs/spleen_unet
```

### Option B: Local machine (needs a GPU for real training)

```bash
git clone https://github.com/FaezeHajeb/MONAI-spleen-segmentation.git
cd MONAI-spleen-segmentation
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_data.py
python train.py --quick          # smoke test
python train.py                  # full training (100 epochs)
```

### Segment a new CT volume

```bash
python predict.py --checkpoint runs/spleen_unet/best_model.pth \
    --input data/Task09_Spleen/imagesTr/spleen_1.nii.gz
# -> outputs/spleen_1_seg.nii.gz  (0 = background, 1 = spleen)
```

Open the input CT and the `_seg.nii.gz` mask together in
[3D Slicer](https://www.slicer.org/) or ITK-SNAP to see the result.

## Repository structure

```
MONAI-spleen-segmentation/
├── train.py            # training script (UNet + Dice loss + validation)
├── predict.py          # inference: segment a CT volume, save the mask
├── download_data.py    # downloads MSD Task09 Spleen (~1.6 GB)
├── requirements.txt    # pinned versions (MONAI 1.6.0, PyTorch 2.8.0, ...)
├── README.md           # this file
├── .gitignore          # keeps data/, runs/ and checkpoints out of git
├── data/               # dataset (downloaded, not committed)
├── runs/               # checkpoints + metrics.csv + TensorBoard logs
└── outputs/            # predicted masks
```

## What I learned

- Loading and preprocessing 3D medical images with MONAI transforms
  (`LoadImaged`, `Spacingd`, `ScaleIntensityRanged`, `RandCropByPosNegLabeld`).
- Why Dice loss (not accuracy) is the right objective when the target organ
  is a small part of the volume.
- Sliding-window inference for volumes that do not fit in GPU memory.
- The full loop: data → model → metric → checkpoint → inference → visual
  check in 3D Slicer.

## Licence

Code in this repository is released under the MIT licence. The MSD Spleen
dataset is released under CC BY-SA 4.0. Please respect its licence and cite
the Decathlon paper (see *Dataset* above) if you use it.
