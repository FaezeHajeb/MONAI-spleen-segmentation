#!/usr/bin/env python3
"""
Download the MSD Task09 Spleen dataset (~1.6 GB).

The download comes from MONAI's public mirror of the Medical Segmentation
Decathlon data. No account or registration is needed. The data is released
under the CC BY-SA 4.0 licence.

Layout after the download (inside the data folder)::

    data/Task09_Spleen/
        imagesTr/     41 training CT volumes  (spleen_1.nii.gz, ...)
        labelsTr/     41 expert spleen masks  (same file names)
        imagesTs/     20 test CT volumes (no labels - for the challenge)
        dataset.json  task description and metadata

Usage
-----
    python download_data.py                # downloads into data/
    python download_data.py --out my_data  # downloads into my_data/
"""

import argparse
import os

from monai.apps import download_and_extract

# MONAI's public mirror of the MSD Spleen archive, with its MD5 checksum
# (the checksum proves the download was not corrupted).
MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
MSD_MD5 = "410d4a301da4e5b2f6f86ec3ddba524e"


def main():
    parser = argparse.ArgumentParser(description="Download MSD Task09 Spleen data.")
    parser.add_argument(
        "--out", default="data",
        help="Folder to download into (default: data).",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, "Task09_Spleen")
    if os.path.isdir(target):
        print(f"Already downloaded: {target}")
        print("Delete that folder first if you want to download it again.")
        return

    print("Downloading Task09_Spleen.tar (~1.6 GB). This takes a few minutes ...")
    download_and_extract(
        MSD_URL,
        os.path.join(args.out, "Task09_Spleen.tar"),
        args.out,
        MSD_MD5, "md5",
    )
    print(f"Done. Data is in: {target}")
    print("Next step: python train.py --quick   (smoke test)")


if __name__ == "__main__":
    main()
