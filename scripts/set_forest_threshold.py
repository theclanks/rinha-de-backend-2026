import argparse
import struct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("forest")
    parser.add_argument("threshold", type=float)
    args = parser.parse_args()

    with open(args.forest, "r+b") as f:
        f.seek(-4, 2)
        f.write(struct.pack("<f", args.threshold))

    print(f"updated {args.forest} threshold={args.threshold:.6f}")


if __name__ == "__main__":
    main()
