#!/usr/bin/env python3
"""
3D spleen segmentation from CT with MONAI + UNet.

This script trains a 3D UNet to segment the spleen in abdominal CT scans,
using the Medical Segmentation Decathlon (MSD) Task09 Spleen dataset.
It follows MONAI's official spleen segmentation tutorial
(https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb)
but is rewritten as a plain, well-commented Python script so you can read
it top to bottom and understand every step.

What you will learn from this script
--------------------------------------
1. How to load 3D medical images (NIfTI files) with MONAI.
2. How to preprocess CT volumes: intensity scaling, resampling to a
   common voxel spacing, orientation, and cropping.
3. How to train a 3D UNet with Dice loss and track the Dice score.
4. How to validate with sliding-window inference (needed because a whole
   CT volume does not fit in GPU memory at once).
5. How to save the best model and log the training history.

Quick start
-----------
    # 1. Download the data (about 1.6 GB, see download_data.py)
    python download_data.py

    # 2. Smoke test: 1 epoch on 2 cases, just to prove everything runs
    python train.py --quick

    # 3. Real training (default: 100 epochs, needs a GPU - see README.md)
    python train.py --data-dir data/Task09_Spleen --output-dir runs/spleen_unet

    # 4. Segment a new CT volume with the trained model
    python predict.py --checkpoint runs/spleen_unet/best_model.pth \
        --input data/Task09_Spleen/imagesTr/spleen_1.nii.gz
"""

import argparse
import csv
import glob
import json
import os
import time

import torch
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.utils import set_determinism

# ---------------------------------------------------------------------------
# Dataset download (MONAI's public mirror of the MSD Spleen data).
# No registration is needed for this link; the data is released under
# the CC BY-SA 4.0 licence (see README.md).
# ---------------------------------------------------------------------------
MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
MSD_MD5 = "410d4a301da4e5b2f6f86ec3ddba524e"


def parse_args():
    """Read the command-line options (run `python train.py --help` to see them)."""
    parser = argparse.ArgumentParser(
        description="Train a 3D UNet for spleen segmentation on MSD Task09."
    )
    parser.add_argument(
        "--data-dir",
        default="data/Task09_Spleen",
        help="Folder with imagesTr/ and labelsTr/ (default: data/Task09_Spleen).",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/spleen_unet",
        help="Folder for checkpoints, logs and plots (default: runs/spleen_unet).",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--val-interval", type=int, default=2,
        help="Validate every N epochs (default: 2).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate for the Adam optimiser (default: 0.0001).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Cases per training step (default: 2).",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download the MSD Spleen data automatically if the folder is missing.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke test: 1 epoch on 2 training cases + 1 validation case.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed, so results are reproducible (default: 0).",
    )
    return parser.parse_args()


def get_transforms(patch_size=(96, 96, 96), num_samples=4):
    """Build the preprocessing pipelines for training and validation.

    A "transform" is one small step applied to every image. MONAI chains
    them with Compose, and the trailing "d" means the transform works on
    dictionaries, e.g. {"image": ..., "label": ...}.

    Training transforms (in order):
      LoadImaged          read the NIfTI files (image + label)
      EnsureChannelFirstd make the shape (channels, x, y, z)
      ScaleIntensityRanged map CT numbers from [-57, 164] Hounsfield units
                          to [0, 1]. The spleen and soft tissue live in
                          this window; values outside it are clipped.
      CropForegroundd     cut away the empty (zero) borders around the body
      Orientationd        rotate every volume into the same RAS orientation
      Spacingd            resample every volume to the same voxel size
                          (1.5 x 1.5 x 2.0 mm), so the network always sees
                          the same physical scale
      RandCropByPosNegLabeld cut random 96x96x96 patches; half of them are
                          centred on the spleen (positive) and half on
                          background (negative), so the network does not
                          only see empty background

    Validation transforms are the same, minus the random crop: validation
    runs on the whole volume with sliding-window inference.
    """
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),  # smooth for image, exact for label
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1, neg=1,          # 1 positive patch per 1 negative patch
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
        ]
    )
    return train_transforms, val_transforms


def main():
    args = parse_args()
    set_determinism(seed=args.seed)  # same seed -> same random crops -> same result
    print_config()  # prints MONAI / PyTorch / GPU versions (useful for debugging)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Get the data (download it if asked, otherwise expect it on disk)
    # ------------------------------------------------------------------
    if not os.path.isdir(args.data_dir):
        if args.download:
            print(f"Downloading MSD Spleen data (~1.6 GB) to {args.data_dir} ...")
            download_and_extract(
                MSD_URL,
                os.path.join(os.path.dirname(args.data_dir) or ".", "Task09_Spleen.tar"),
                os.path.dirname(args.data_dir) or ".",
                MSD_MD5, "md5",
            )
        else:
            raise FileNotFoundError(
                f"Data folder not found: {args.data_dir}\n"
                "Run `python download_data.py` first, or re-run with --download."
            )

    # Pair every training image with its label (same file name in both folders).
    train_images = sorted(glob.glob(os.path.join(args.data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(args.data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [
        {"image": img, "label": lab}
        for img, lab in zip(train_images, train_labels)
    ]
    # Same split as MONAI's official tutorial: last 9 cases are validation.
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]
    if args.quick:
        # Smoke test: tiny subset so one epoch finishes in ~1 minute on CPU.
        train_files, val_files = train_files[:2], val_files[:1]
        args.epochs = 1
        print("QUICK MODE: 1 epoch, 2 training cases, 1 validation case.")
    print(f"Training cases: {len(train_files)}, validation cases: {len(val_files)}")

    # ------------------------------------------------------------------
    # 2. Datasets and loaders
    #    CacheDataset loads each volume once, applies the deterministic
    #    transforms (everything except the random crop) and keeps the
    #    result in RAM, so later epochs are much faster than a plain
    #    Dataset. Use a lower cache_rate (e.g. 0.5) if RAM is limited.
    # ------------------------------------------------------------------
    train_transforms, val_transforms = get_transforms()
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=1.0, num_workers=0 if args.quick else 4,
    )
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms,
        cache_rate=1.0, num_workers=0 if args.quick else 4,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0 if args.quick else 4,
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0 if args.quick else 4)

    # ------------------------------------------------------------------
    # 3. Model, loss, optimiser, metric
    #    UNet: the classic encoder-decoder for medical segmentation.
    #      - spatial_dims=3      : 3D volumes (not 2D slices)
    #      - in_channels=1       : one CT volume per case
    #      - out_channels=2      : background + spleen (2 classes)
    #      - channels=(16,32,64,128,256): feature maps per level
    #      - strides=(2,2,2,2)   : downsampling between levels
    #      - num_res_units=2     : residual blocks (helps training)
    #    DiceLoss: 1 - Dice score. It directly optimises the overlap
    #      between prediction and ground truth, which is what we care
    #      about (the spleen is small, so plain accuracy would be
    #      misleading: predicting "all background" is already ~99%
    #      accurate but useless).
    #    DiceMetric: the evaluation score, background class excluded.
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Post-processing: turn the network's 2-channel output into a label map.
    #   argmax picks the winning class per voxel; to_onehot=2 keeps the
    #   shape (2, x, y, z) so the metric can compare it with the label.
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    # TensorBoard is optional: if it is installed we log there too,
    # otherwise we only write metrics.csv (which is always written).
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    except Exception as exc:  # TensorBoard not installed -> CSV only
        print(f"TensorBoard unavailable ({exc}); logging to CSV only.")
        writer = None

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    max_epochs = args.epochs
    best_metric = -1.0
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    roi_size = (96, 96, 96) if args.quick else (160, 160, 160)
    sw_batch_size = 1 if args.quick else 4

    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "train_loss", "val_dice", "best_dice"])
        start_time = time.time()

        for epoch in range(max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()  # training mode: dropout/batchnorm behave accordingly
            epoch_loss = 0.0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                optimizer.zero_grad()          # clear gradients from last step
                outputs = model(inputs)        # forward pass
                loss = loss_function(outputs, labels)
                loss.backward()                # backward pass
                optimizer.step()               # update the weights
                epoch_loss += loss.item()
                print(f"  step {step}/{len(train_loader)}, train_loss: {loss.item():.4f}")
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            if writer is not None:
                writer.add_scalar("train/loss", epoch_loss, epoch + 1)

            # ---- Validation every val_interval epochs -----------------
            # A whole CT volume is too big for the GPU, so validation
            # uses sliding_window_inference: the volume is cut into
            # overlapping 160x160x160 windows, each is segmented, and
            # the results are stitched back together.
            val_dice = None
            if (epoch + 1) % args.val_interval == 0 or epoch == max_epochs - 1:
                model.eval()  # evaluation mode (no dropout, frozen batchnorm)
                with torch.no_grad():  # no gradients needed -> saves memory
                    for val_data in val_loader:
                        val_inputs = val_data["image"].to(device)
                        val_labels = val_data["label"].to(device)
                        val_outputs = sliding_window_inference(
                            val_inputs, roi_size, sw_batch_size, model
                        )
                        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                        val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                        dice_metric(y_pred=val_outputs, y=val_labels)
                    val_dice = dice_metric.aggregate().item()
                    dice_metric.reset()
                metric_values.append(val_dice)
                if val_dice > best_metric:
                    best_metric = val_dice
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.output_dir, "best_model.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {val_dice:.4f}\n"
                    f"best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )
                if writer is not None:
                    writer.add_scalar("val/dice", val_dice, epoch + 1)

            csv_writer.writerow(
                [epoch + 1, f"{epoch_loss:.4f}",
                 f"{val_dice:.4f}" if val_dice is not None else "",
                 f"{best_metric:.4f}" if best_metric > 0 else ""]
            )
            csv_file.flush()

    # Save the final model too (even if it is not the best one).
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))
    if writer is not None:
        writer.close()

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    print(f"total time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"results saved in: {args.output_dir}")

    # Remember the run settings, so the run can be reproduced later.
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)


if __name__ == "__main__":
    main()
