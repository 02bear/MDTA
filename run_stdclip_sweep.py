
# -*- coding: utf-8 -*-
"""
Sweep runner for train_p13d_to_drug_stdclip.py

What it does:
1. Runs a series of experiments automatically (default: batch_size = 2, 8, 16, 32)
2. Writes one combined console log file so you can inspect CLIP loss / training prints
3. Writes one summary CSV + JSON with best metrics for each experiment
4. Supports extending the search to other parameters such as lr / lambda_clip / temperature_init
"""

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    # fixed base command args
    parser.add_argument("--train_script", type=str, default="train_p13d_to_drug_stdclip.py")
    parser.add_argument("--pairs_csv", type=str, default="data/raw/davis/pairs.csv")
    parser.add_argument("--drug_1d_dir", type=str, default="data/processed/davis/drug_1d_chemberta2")
    parser.add_argument("--protein_1d_dir", type=str, default="data/processed/davis/protein_1d_esm2")
    parser.add_argument("--protein_3d_dir", type=str, default="data/processed/davis/protein_3d_min")
    parser.add_argument("--split_json", type=str, default="data/splits/davis_fixed_split_size2.json")
    parser.add_argument("--drug_1d_in_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--contrastive_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)

    # sweep params
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 8, 16, 32])
    parser.add_argument("--lrs", type=float, nargs="+", default=[3e-4])
    parser.add_argument("--lambda_clips", type=float, nargs="+", default=[0.1])
    parser.add_argument("--temperature_inits", type=float, nargs="+", default=[0.07])

    # output
    parser.add_argument("--sweep_name", type=str, default="")
    parser.add_argument("--output_root", type=str, default="outputs/stdclip_sweeps")
    parser.add_argument("--skip_finished", action="store_true", default=True)
    parser.add_argument("--stop_on_error", action="store_true")

    return parser.parse_args()


def make_sweep_dir(args):
    if args.sweep_name.strip():
        sweep_name = args.sweep_name.strip()
    else:
        sweep_name = "stdclip_sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_root) / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    return sweep_dir


def build_experiments(args):
    exps = []
    for bs, lr, lam, temp in itertools.product(
        args.batch_sizes, args.lrs, args.lambda_clips, args.temperature_inits
    ):
        tag = f"bs{bs}_lr{lr:g}_lc{lam:g}_temp{temp:g}"
        exps.append({
            "batch_size": bs,
            "lr": lr,
            "lambda_clip": lam,
            "temperature_init": temp,
            "tag": tag,
        })
    return exps


def make_command(args, exp, output_dir):
    return [
        sys.executable,
        args.train_script,
        "--pairs_csv", args.pairs_csv,
        "--drug_1d_dir", args.drug_1d_dir,
        "--protein_1d_dir", args.protein_1d_dir,
        "--protein_3d_dir", args.protein_3d_dir,
        "--split_json", args.split_json,
        "--output_dir", str(output_dir),
        "--drug_1d_in_dim", str(args.drug_1d_in_dim),
        "--hidden_dim", str(args.hidden_dim),
        "--contrastive_dim", str(args.contrastive_dim),
        "--dropout", str(args.dropout),
        "--epochs", str(args.epochs),
        "--batch_size", str(exp["batch_size"]),
        "--lr", str(exp["lr"]),
        "--lambda_clip", str(exp["lambda_clip"]),
        "--temperature_init", str(exp["temperature_init"]),
        "--num_workers", str(args.num_workers),
        "--weight_decay", str(args.weight_decay),
        "--seed", str(args.seed),
    ]


def load_best_metrics(exp_output_dir):
    summary_json = exp_output_dir / "best_summary.json"
    if not summary_json.exists():
        return None

    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    best_epoch = data.get("best_epoch", None)
    val = data.get("best_val_metrics", {})
    train = data.get("best_train_metrics", {})

    return {
        "best_epoch": best_epoch,
        "val_loss": val.get("loss"),
        "val_mse": val.get("mse"),
        "val_rmse": val.get("rmse"),
        "val_mae": val.get("mae"),
        "val_ci": val.get("ci"),
        "val_rm2": val.get("rm2"),
        "train_loss": train.get("loss"),
        "train_mse": train.get("mse"),
        "train_rmse": train.get("rmse"),
        "train_mae": train.get("mae"),
        "train_ci": train.get("ci"),
        "train_rm2": train.get("rm2"),
    }


def write_summary_csv(csv_path, rows):
    fieldnames = [
        "tag", "status", "return_code", "elapsed_sec",
        "batch_size", "lr", "lambda_clip", "temperature_init",
        "best_epoch",
        "val_loss", "val_mse", "val_rmse", "val_mae", "val_ci", "val_rm2",
        "train_loss", "train_mse", "train_rmse", "train_mae", "train_ci", "train_rm2",
        "output_dir",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    sweep_dir = make_sweep_dir(args)
    sweep_name = sweep_dir.name

    combined_log_path = sweep_dir / "combined_console.log"
    summary_csv_path = sweep_dir / "summary.csv"
    summary_json_path = sweep_dir / "summary.json"
    commands_json_path = sweep_dir / "commands.json"

    exps = build_experiments(args)
    summary_rows = []
    commands_dump = []

    with open(combined_log_path, "a", encoding="utf-8") as combined_log:
        combined_log.write(f"SWEEP START: {datetime.now().isoformat()}\n")
        combined_log.write(f"SWEEP DIR: {sweep_dir}\n\n")

        for idx, exp in enumerate(exps, start=1):
            exp_output_dir = sweep_dir / exp["tag"]
            exp_output_dir.mkdir(parents=True, exist_ok=True)

            cmd = make_command(args, exp, exp_output_dir)
            cmd_str = " ".join(cmd)

            commands_dump.append({
                "index": idx,
                "tag": exp["tag"],
                "command": cmd,
                "command_str": cmd_str,
                "output_dir": str(exp_output_dir),
            })

            best_summary_file = exp_output_dir / "best_summary.json"
            if args.skip_finished and best_summary_file.exists():
                metrics = load_best_metrics(exp_output_dir)
                row = {
                    "tag": exp["tag"],
                    "status": "skipped_finished",
                    "return_code": 0,
                    "elapsed_sec": 0.0,
                    "batch_size": exp["batch_size"],
                    "lr": exp["lr"],
                    "lambda_clip": exp["lambda_clip"],
                    "temperature_init": exp["temperature_init"],
                    "output_dir": str(exp_output_dir),
                }
                if metrics:
                    row.update(metrics)
                summary_rows.append(row)

                msg = f"\n[SKIP {idx}/{len(exps)}] {exp['tag']} already finished\n"
                print(msg, end="")
                combined_log.write(msg)
                combined_log.flush()
                continue

            header = (
                f"\n{'=' * 100}\n"
                f"[RUN {idx}/{len(exps)}] {exp['tag']}\n"
                f"START: {datetime.now().isoformat()}\n"
                f"OUTPUT_DIR: {exp_output_dir}\n"
                f"COMMAND: {cmd_str}\n"
                f"{'=' * 100}\n"
            )
            print(header, end="")
            combined_log.write(header)
            combined_log.flush()

            run_log_path = exp_output_dir / "console.log"
            start_time = time.time()
            return_code = None

            with open(run_log_path, "w", encoding="utf-8") as run_log:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                try:
                    for line in process.stdout:
                        sys.stdout.write(line)
                        combined_log.write(line)
                        run_log.write(line)
                    process.wait()
                    return_code = process.returncode
                except KeyboardInterrupt:
                    process.kill()
                    raise

            elapsed = time.time() - start_time
            metrics = load_best_metrics(exp_output_dir)

            row = {
                "tag": exp["tag"],
                "status": "ok" if return_code == 0 else "failed",
                "return_code": return_code,
                "elapsed_sec": round(elapsed, 2),
                "batch_size": exp["batch_size"],
                "lr": exp["lr"],
                "lambda_clip": exp["lambda_clip"],
                "temperature_init": exp["temperature_init"],
                "output_dir": str(exp_output_dir),
            }
            if metrics:
                row.update(metrics)
            summary_rows.append(row)

            footer = (
                f"\n[END {idx}/{len(exps)}] {exp['tag']} | "
                f"return_code={return_code} | elapsed_sec={elapsed:.2f}\n"
            )
            print(footer, end="")
            combined_log.write(footer)
            combined_log.flush()

            write_summary_csv(summary_csv_path, summary_rows)
            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump(summary_rows, f, indent=2, ensure_ascii=False)
            with open(commands_json_path, "w", encoding="utf-8") as f:
                json.dump(commands_dump, f, indent=2, ensure_ascii=False)

            if return_code != 0 and args.stop_on_error:
                raise RuntimeError(f"Experiment failed: {exp['tag']}")

        combined_log.write(f"\nSWEEP END: {datetime.now().isoformat()}\n")

    print("\nSweep finished.")
    print(f"Combined console log: {combined_log_path}")
    print(f"Summary CSV:          {summary_csv_path}")
    print(f"Summary JSON:         {summary_json_path}")
    print(f"Commands JSON:        {commands_json_path}")


if __name__ == "__main__":
    main()
