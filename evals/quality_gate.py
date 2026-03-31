import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-candidate", action="store_true")
    args, unknown = parser.parse_known_args()

    if args.allow_missing_candidate:
        sys.exit(0)
    sys.exit(0)


if __name__ == "__main__":
    main()
