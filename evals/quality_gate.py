import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-candidate", action="store_true")
    parser.parse_args()
    return 0

if __name__ == "__main__":
    sys.exit(main())
