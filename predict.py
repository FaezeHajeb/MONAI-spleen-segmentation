#!/usr/bin/env python3
"""
Segment one CT volume with a trained spleen model and save the result.

The prediction is written back into the ORIGINAL image space (same size,
voxel spacing and orientation as the input file), so you can open the input
CT and the predicted mask side by side in 3D Slicer or ITK-SNAP and compare
them. This is done with MONAI's Invertd transform, which undoes the
resampling/orientation steps applied before inference.

Usage
-----
    python predict.py --checkpoint runs/spleen_unet/best_model.pth \
        --input data/Task09_Spleen/imagesTr/spleen_1.nii.gz

    # output: outputs/spleen_1_seg.nii.gz  (0 = background, 1 = spleen)

    # segment every case in a folder:
    python predict.py --checkpoint runs/spleen_unet/best_model.pth \
        --input data/Task09_Spleen/imagesTs
"""

import argparse
import glob
import os

import torch
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.utils import convert_to_tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment CT volumes with a trained spleen UNet."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a trained model, e.g. runs/spleen_unet/best_model.pth",
    )
    parser.add_argument(
        "--input", required=True,
        help="One NIfTI file (.nii.gz) or a folder of NIfTI files.",
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Folder for the predicted masks (default: outputs).",
    )
    parser.add_argument(
        "--roi-size", type=int, nargs=3, default=[160, 160, 160],
        help="Sliding-window size, e.g. --roi-size 160 160 160 (default).",
    )
    parser.add_argument(
        "--sw-batch-size", type=int, default=4,
        help="Windows processed at once (default: 4; lower it if you run "
        "out of GPU memory).",
    )
    parser.add_argument(
        "--device", default=None,
        help="cuda or cpu (default: cuda if available, else cpu).",
    )
    return parser.parse_args()


def build_model(checkpoint, device):
    """Create the same UNet as in train.py and load the trained weights."""
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    # weights_only=True is safe here: a checkpoint is just a dict of tensors.
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Same preprocessing as validation in train.py, but WITHOUT the label:
    #   load -> channel-first -> scale HU [-57, 164] to [0, 1]
    #   -> RAS orientation -> resample to 1.5 x 1.5 x 2.0 mm voxels.
    infer_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            Orientationd(keys=["image"], axcodes="RAS", labels=None),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        ]
    )

    # After inference, Invertd undoes the transforms above, so the mask
    # comes back to the original voxel grid; SaveImaged writes it as NIfTI.
    # NOTE (MONAI 1.6): Invertd reads the transform trace and the metadata
    # from the "image" MetaTensor itself, so meta_keys/orig_meta_keys are
    # left as None instead of pointing at "image_meta_dict" (a dict entry
    # that no longer exists by default in MONAI 1.6).
    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=infer_transforms,
                orig_keys="image",
                meta_keys=None,
                orig_meta_keys=None,
                meta_key_postfix="meta_dict",
                nearest_interp=True,  # nearest neighbour: keep labels 0/1 exact
                to_tensor=True,
            ),
            # NOTE (MONAI 1.6): the metadata travels ON the MetaTensor itself
            # (pred.meta), not in a separate "pred_meta_dict" dict entry,
            # so we pass meta_keys=None and let SaveImage read pred.meta.
            SaveImaged(
                keys="pred",
                meta_keys=None,
                meta_key_postfix=None,
                output_dir=args.output_dir,
                output_postfix="seg",
                output_ext=".nii.gz",
                resample=False,
                separate_folder=False,  # save as outputs/<name>_seg.nii.gz
            ),
        ]
    )

    # argmax over the 2 output channels -> one label map (0 background, 1 spleen)
    post_pred = AsDiscrete(argmax=True)

    model = build_model(args.checkpoint, device)

    if os.path.isdir(args.input):
        input_files = sorted(glob.glob(os.path.join(args.input, "*.nii.gz")))
    else:
        input_files = [args.input]
    print(f"Found {len(input_files)} volume(s) to segment.")

    with torch.no_grad():  # no gradients needed for inference
        for input_file in input_files:
            name = os.path.splitext(os.path.splitext(os.path.basename(input_file))[0])[0]
            print(f"Segmenting {input_file} ...")
            data = infer_transforms({"image": input_file})
            image = data["image"].unsqueeze(0).to(device)  # add batch dimension
            outputs = sliding_window_inference(
                image, tuple(args.roi_size), args.sw_batch_size, model
            )
            pred = post_pred(outputs[0])  # drop batch dim, take winning class
            # MONAI 1.6 keeps image metadata (affine, spacing, the list of
            # applied transforms) ON the MetaTensor itself, not in a separate
            # "image_meta_dict" dict entry. Invertd finds the transform trace
            # through the "image" MetaTensor, but only if the prediction is
            # a MetaTensor too -- so wrap it before inverting.
            pred = convert_to_tensor(pred, track_meta=True)
            data["pred"] = pred
            post_transforms(data)
            print(f"  saved: {os.path.join(args.output_dir, name + '_seg.nii.gz')}")

    print("Done.")


if __name__ == "__main__":
    main()
