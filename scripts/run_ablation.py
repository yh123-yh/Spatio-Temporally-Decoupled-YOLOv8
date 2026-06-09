from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from st_pod_yolo.config import load_yaml, save_yaml
from st_pod_yolo.eval_loop import evaluate
from st_pod_yolo.train_loop import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    base = load_yaml(cfg["base_train_config"])
    results = []
    for stage in cfg["stages"]:
        stage_cfg = copy.deepcopy(base)
        stage_cfg["out_dir"] = str(Path(cfg.get("out_dir", "runs/ablation")) / stage["name"])
        stage_cfg["model"].update(stage.get("model", {}))
        stage_cfg["loss"].update(stage.get("loss", {}))
        print(f"Running ablation stage: {stage['name']}")
        best = train_from_config(stage_cfg)
        data_cfg = load_yaml(cfg["data_config"])
        result = evaluate(best, data_cfg, cfg.get("split", "test"))
        result["stage"] = stage["name"]
        results.append(result)
    out = Path(cfg.get("out_dir", "runs/ablation")) / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    save_yaml({"results": results}, out.with_suffix(".yaml"))
    print(out)


if __name__ == "__main__":
    main()
