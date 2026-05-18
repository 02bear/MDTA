# -*- coding: utf-8 -*-
import argparse, csv, itertools, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_script", type=str, default="train_p13d_stdclip_pcl.py")
    p.add_argument("--pairs_csv", type=str, default="data/raw/davis/pairs.csv")
    p.add_argument("--drug_1d_dir", type=str, default="data/processed/davis/drug_1d_chemberta2")
    p.add_argument("--protein_1d_dir", type=str, default="data/processed/davis/protein_1d_esm2")
    p.add_argument("--protein_3d_dir", type=str, default="data/processed/davis/protein_3d_min")
    p.add_argument("--split_json", type=str, default="data/splits/davis_fixed_split_size2.json")
    p.add_argument("--drug_1d_in_dim", type=int, default=768)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--contrastive_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[2,4,8,16,32])
    p.add_argument("--lrs", type=float, nargs="+", default=[3e-4])
    p.add_argument("--lambda_clips", type=float, nargs="+", default=[0.1])
    p.add_argument("--lambda_pcls", type=float, nargs="+", default=[0.1])
    p.add_argument("--temperature_inits", type=float, nargs="+", default=[0.07])
    p.add_argument("--pcl_temperature_inits", type=float, nargs="+", default=[0.07])
    p.add_argument("--sweep_name", type=str, default="")
    p.add_argument("--output_root", type=str, default="outputs/stdclip_pcl_sweeps")
    p.add_argument("--skip_finished", action="store_true", default=True)
    p.add_argument("--stop_on_error", action="store_true")
    return p.parse_args()


def make_sweep_dir(args):
    name = args.sweep_name.strip() or ("stdclip_pcl_sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    d = Path(args.output_root) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_experiments(args):
    exps = []
    for bs, lr, lc, lp, temp, pcl_temp in itertools.product(args.batch_sizes, args.lrs, args.lambda_clips, args.lambda_pcls, args.temperature_inits, args.pcl_temperature_inits):
        tag = f"bs{bs}_lr{lr:g}_lc{lc:g}_lp{lp:g}_temp{temp:g}_pt{pcl_temp:g}"
        exps.append({"batch_size": bs, "lr": lr, "lambda_clip": lc, "lambda_pcl": lp, "temperature_init": temp, "pcl_temperature_init": pcl_temp, "tag": tag})
    return exps


def make_command(args, exp, output_dir):
    return [sys.executable, "-u", args.train_script,
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
            "--lambda_pcl", str(exp["lambda_pcl"]),
            "--temperature_init", str(exp["temperature_init"]),
            "--pcl_temperature_init", str(exp["pcl_temperature_init"]),
            "--num_workers", str(args.num_workers),
            "--weight_decay", str(args.weight_decay),
            "--seed", str(args.seed)]


def load_best_metrics(exp_output_dir):
    p = exp_output_dir / "best_summary.json"
    if not p.exists():
        return None
    data = json.load(open(p, "r", encoding="utf-8"))
    best_epoch = data.get("best_epoch")
    val = data.get("best_val_metrics", {})
    train = data.get("best_train_metrics", {})
    return {
        "best_epoch": best_epoch,
        "val_loss": val.get("loss"), "val_reg_loss": val.get("reg_loss"), "val_clip_loss": val.get("clip_loss"), "val_pcl_loss": val.get("pcl_loss"),
        "val_mse": val.get("mse"), "val_rmse": val.get("rmse"), "val_mae": val.get("mae"), "val_ci": val.get("ci"), "val_rm2": val.get("rm2"),
        "train_loss": train.get("loss"), "train_reg_loss": train.get("reg_loss"), "train_clip_loss": train.get("clip_loss"), "train_pcl_loss": train.get("pcl_loss"),
        "train_mse": train.get("mse"), "train_rmse": train.get("rmse"), "train_mae": train.get("mae"), "train_ci": train.get("ci"), "train_rm2": train.get("rm2"),
    }


def write_summary_csv(csv_path, rows):
    fields = ["tag","status","return_code","elapsed_sec","batch_size","lr","lambda_clip","lambda_pcl","temperature_init","pcl_temperature_init","best_epoch",
              "val_loss","val_reg_loss","val_clip_loss","val_pcl_loss","val_mse","val_rmse","val_mae","val_ci","val_rm2",
              "train_loss","train_reg_loss","train_clip_loss","train_pcl_loss","train_mse","train_rmse","train_mae","train_ci","train_rm2","output_dir"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    args = parse_args()
    sweep_dir = make_sweep_dir(args)
    combined_log_path = sweep_dir / "combined_console.log"
    summary_csv_path = sweep_dir / "summary.csv"
    summary_json_path = sweep_dir / "summary.json"
    commands_json_path = sweep_dir / "commands.json"
    exps = build_experiments(args)
    summary_rows, commands_dump = [], []
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
    with open(combined_log_path, "a", encoding="utf-8") as combined_log:
        combined_log.write(f"SWEEP START: {datetime.now().isoformat()}\\nSWEEP DIR: {sweep_dir}\\n\\n")
        for idx, exp in enumerate(exps, start=1):
            exp_output_dir = sweep_dir / exp["tag"]
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            cmd = make_command(args, exp, exp_output_dir)
            cmd_str = " ".join(cmd)
            commands_dump.append({"index": idx, "tag": exp["tag"], "command": cmd, "command_str": cmd_str, "output_dir": str(exp_output_dir)})
            best_summary_file = exp_output_dir / "best_summary.json"
            if args.skip_finished and best_summary_file.exists():
                metrics = load_best_metrics(exp_output_dir)
                row = {"tag": exp["tag"], "status": "skipped_finished", "return_code": 0, "elapsed_sec": 0.0, "batch_size": exp["batch_size"], "lr": exp["lr"], "lambda_clip": exp["lambda_clip"], "lambda_pcl": exp["lambda_pcl"], "temperature_init": exp["temperature_init"], "pcl_temperature_init": exp["pcl_temperature_init"], "output_dir": str(exp_output_dir)}
                if metrics: row.update(metrics)
                summary_rows.append(row)
                msg = f"\\n[SKIP {idx}/{len(exps)}] {exp['tag']} already finished\\n"
                print(msg, end=""); combined_log.write(msg); combined_log.flush(); continue
            header = f"\\n{'='*100}\\n[RUN {idx}/{len(exps)}] {exp['tag']}\\nSTART: {datetime.now().isoformat()}\\nOUTPUT_DIR: {exp_output_dir}\\nCOMMAND: {cmd_str}\\n{'='*100}\\n"
            print(header, end=""); combined_log.write(header); combined_log.flush()
            run_log_path = exp_output_dir / "console.log"
            start_time = time.time(); return_code = None
            with open(run_log_path, "w", encoding="utf-8") as run_log:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True, env=env)
                try:
                    for line in process.stdout:
                        sys.stdout.write(line); sys.stdout.flush(); combined_log.write(line); combined_log.flush(); run_log.write(line); run_log.flush()
                    process.wait(); return_code = process.returncode
                except KeyboardInterrupt:
                    process.kill(); raise
            elapsed = time.time() - start_time
            metrics = load_best_metrics(exp_output_dir)
            row = {"tag": exp["tag"], "status": "ok" if return_code == 0 else "failed", "return_code": return_code, "elapsed_sec": round(elapsed,2), "batch_size": exp["batch_size"], "lr": exp["lr"], "lambda_clip": exp["lambda_clip"], "lambda_pcl": exp["lambda_pcl"], "temperature_init": exp["temperature_init"], "pcl_temperature_init": exp["pcl_temperature_init"], "output_dir": str(exp_output_dir)}
            if metrics: row.update(metrics)
            summary_rows.append(row)
            footer = f"\\n[END {idx}/{len(exps)}] {exp['tag']} | return_code={return_code} | elapsed_sec={elapsed:.2f}\\n"
            print(footer, end=""); combined_log.write(footer); combined_log.flush()
            write_summary_csv(summary_csv_path, summary_rows)
            json.dump(summary_rows, open(summary_json_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            json.dump(commands_dump, open(commands_json_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            if return_code != 0 and args.stop_on_error:
                raise RuntimeError(f"Experiment failed: {exp['tag']}")
        combined_log.write(f"\\nSWEEP END: {datetime.now().isoformat()}\\n")
    print("\\nSweep finished.")
    print(f"Combined console log: {combined_log_path}")
    print(f"Summary CSV:          {summary_csv_path}")
    print(f"Summary JSON:         {summary_json_path}")
    print(f"Commands JSON:        {commands_json_path}")


if __name__ == "__main__":
    main()
