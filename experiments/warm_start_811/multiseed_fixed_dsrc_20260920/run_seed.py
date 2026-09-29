"""Resume-safe orchestrator for one fixed-protocol warm-start seed."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path("/data1/ztx/MyModel-MDTA")
HERE = PROJECT / "experiments/warm_start_811/multiseed_fixed_dsrc_20260920"
SEED = int(os.environ["WARM_SEED"])
GPU = int(os.environ["WARM_GPU"])
ROOT = PROJECT / f"outputs/Refine_experiment/davis/warm_start/random_pair_811_seed42/multiseed_fixed_dsrc_20260920/seed_{SEED}"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_stage(name, command, log_path, environment):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        ROOT / "status.json",
        {
            "state": "running",
            "seed": SEED,
            "gpu": GPU,
            "stage": name,
            "worker_pid": os.getpid(),
            "log": str(log_path),
            "updated": time.time(),
        },
    )
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        exit_code = process.wait()
    if exit_code:
        dump(
            ROOT / "status.json",
            {
                "state": "failed",
                "seed": SEED,
                "gpu": GPU,
                "stage": name,
                "exit_code": exit_code,
                "worker_pid": os.getpid(),
                "child_pid": process.pid,
                "log": str(log_path),
                "updated": time.time(),
            },
        )
        raise RuntimeError(f"{name} failed with exit code {exit_code}; see {log_path}")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "worker.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ANOTHER_SEED_WORKER_IS_ACTIVE", flush=True)
        return

    environment = os.environ.copy()
    environment.update(
        WARM_SEED=str(SEED),
        WARM_GPU=str(GPU),
        CUDA_VISIBLE_DEVICES=str(GPU),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
    )
    python = sys.executable
    if not (ROOT / "p13d/result.json").exists():
        run_stage(
            "p13d_training",
            [python, "-B", "-u", str(HERE / "train_p13d_seed.py")],
            ROOT / "p13d/train.log",
            environment,
        )
    if not (ROOT / "pcim_dsrc_fixed/preflight.json").exists():
        run_stage(
            "pcim_dsrc_preflight",
            [python, "-B", "-u", str(HERE / "run_fixed_pipeline.py"), "preflight"],
            ROOT / "pcim_dsrc_fixed/logs/preflight.log",
            environment,
        )
    if not (ROOT / "pcim_dsrc_fixed/result.json").exists():
        run_stage(
            "pcim_dsrc_worker",
            [python, "-B", "-u", str(HERE / "run_fixed_pipeline.py"), "worker"],
            ROOT / "pcim_dsrc_fixed/logs/worker.log",
            environment,
        )

    result = json.loads((ROOT / "pcim_dsrc_fixed/result.json").read_text(encoding="utf-8"))
    dump(
        ROOT / "status.json",
        {
            "state": "complete",
            "seed": SEED,
            "gpu": GPU,
            "result": result,
            "finished": time.time(),
        },
    )
    print("MULTISEED_RUN_COMPLETE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
