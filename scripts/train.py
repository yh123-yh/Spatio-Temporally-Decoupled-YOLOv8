from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from st_pod_yolo.config import load_yaml
from st_pod_yolo.train_loop import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    best = train_from_config(cfg)
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
