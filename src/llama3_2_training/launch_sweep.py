import argparse
import math
import os
import subprocess

import wandb
import yaml

DEFAULT_CONFIG = "src/llama3_2_training/sweeps/lrs.yaml"
DEFAULT_SCRIPT = "src/llama3_2_training/sbatch/sweep.sh"


def get_args():
    p = argparse.ArgumentParser(
        description="Create a wandb sweep and submit a Slurm array of agents that run it in parallel")
    p.add_argument("config", nargs="?", default=DEFAULT_CONFIG, help=f"sweep yaml (default: {DEFAULT_CONFIG})")
    p.add_argument("--count", type=int, default=None,
                   help="number of array tasks (= runs); default: grid size, required for random/bayes")
    p.add_argument("--parallel", type=int, default=None,
                   help="max array tasks running at the same time (sbatch --array=...%%N)")
    p.add_argument("--sweep_id", default=None,
                   help="reuse an existing sweep (entity/project/id) instead of creating a new one")
    p.add_argument("--script", default=DEFAULT_SCRIPT, help=f"sbatch worker script (default: {DEFAULT_SCRIPT})")
    p.add_argument("--dry_run", action="store_true", help="print what would happen without creating or submitting")
    return p.parse_args()


def grid_size(cfg):
    if cfg.get("method") != "grid":
        return None
    return math.prod(len(v["values"]) for v in cfg["parameters"].values() if "values" in v)


def main():
    args = get_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    count = args.count or grid_size(cfg)
    if not count:
        raise SystemExit(f"--count is required for method '{cfg.get('method')}'")
    array = f"0-{count - 1}" + (f"%{args.parallel}" if args.parallel else "")
    print(f"config: {args.config} | method: {cfg.get('method')} | runs: {count} | array: {array}", flush=True)

    if args.sweep_id:
        sweep_path = args.sweep_id
    elif args.dry_run:
        sweep_path = "<entity>/<project>/<new sweep id>"
    else:
        entity = os.environ.get("WANDB_ENTITY") or wandb.Api().default_entity
        sweep_id = wandb.sweep(cfg, entity=entity, project=cfg["project"])
        sweep_path = f"{entity}/{cfg['project']}/{sweep_id}"
    print(f"sweep: {sweep_path}", flush=True)

    cmd = ["sbatch", f"--array={array}", args.script, sweep_path]
    print("submit: " + " ".join(cmd), flush=True)
    if args.dry_run:
        return
    # wandb.sweep() 會在 login node 起 wandb service 並設 WANDB_SERVICE（socket 路徑）；
    # sbatch 預設 --export=ALL 會把它帶進 job，compute node 上的 agent 就會連不到而失敗
    env = {k: v for k, v in os.environ.items() if k != "WANDB_SERVICE"}
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env).stdout.strip()
    print(out, flush=True)


if __name__ == "__main__":
    main()
