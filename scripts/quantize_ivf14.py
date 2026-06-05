import argparse
import os

import numpy as np


def quantize(path_in, path_out):
    values = np.fromfile(path_in, dtype=np.float32)
    quantized = np.clip(np.rint(values * 32767), -32767, 32767).astype(np.int16)
    quantized.tofile(path_out)
    print(f"wrote {path_out}: {os.path.getsize(path_out):,} bytes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", default="resources")
    args = parser.parse_args()

    quantize(
        f"{args.resources}/vectors_14d_sorted.bin",
        f"{args.resources}/vectors_14d_i16_sorted.bin",
    )
    quantize(
        f"{args.resources}/centroids_14d.bin",
        f"{args.resources}/centroids_14d_i16.bin",
    )


if __name__ == "__main__":
    main()
