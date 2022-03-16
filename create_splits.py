import argparse
import glob
import os
import random

# import shutil
from typing import List

import numpy as np

from utils import get_module_logger


def split(data_dir: str) -> None:
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    # TODO: Implement function
    files = glob.glob("/home/workspace/data/waymo/training_and_validation/*.tfrecord")
    np.random.shuffle(files)
    train_test_vals = dict(
        zip(
            ["train", "test", "val"],
            np.split(files, [int(0.75 * len(files)), int(0.9 * len(files))]),
        )
    )
    for k, values in train_test_vals.items():
        os.makedirs(os.path.join(data_dir, k), exist_ok=True)
        for v in values:
            # shutil.move(v, os.path.join(data_dir, k))
            os.system(f"mv {v} {os.path.join(data_dir, k)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split data into training / validation / testing"
    )
    # parser.add_argument("--temp_dir", required=True, help="source data directory")
    parser.add_argument(
        "--data_dir",
        required=True,
        help="destination data directory",
    )
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info("Creating splits...")
    split(args.data_dir)