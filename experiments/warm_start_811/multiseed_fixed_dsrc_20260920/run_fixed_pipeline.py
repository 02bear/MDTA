"""Fixed DAVIS warm-start P13D + PCIM + standard RNC 0.01 + DSRC.

The P13D training seed varies, while every architecture/training hyperparameter
and the final DSRC calibration configuration are frozen to the seed-42 optimum.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch


PROJECT = Path("/data1/ztx/MyModel-MDTA")
SEED = int(os.environ["WARM_SEED"])
GPU = int(os.environ.get("WARM_GPU", "0"))
RUN_ROOT = PROJECT / f"outputs/Refine_experiment/davis/warm_start/random_pair_811_seed42/multiseed_fixed_dsrc_20260920/seed_{SEED}"
P13D_EXPERIMENT = PROJECT / "experiments/warm_start_811/p13d_pure_20260918"
P13D_OUTPUT = RUN_ROOT / "p13d"
TUNE_EXPERIMENT = PROJECT / "experiments/warm_start_811/rank16_pcim_rnc_lowcost_tune_20260917"
OUTPUT = RUN_ROOT / "pcim_dsrc_fixed"
SPLIT = PROJECT / "data/splits/davis_fixed_split_811_full.json"
CHECKPOINT = P13D_OUTPUT / "best_model.pt"
SIMILARITY_PATH = TUNE_EXPERIMENT / "entity_similarities.npz"

sys.path.insert(0, str(TUNE_EXPERIMENT))
sys.path.insert(0, str(P13D_EXPERIMENT))
import tune_experiment as tune  # noqa: E402
import run_experiment as p13d_runner  # noqa: E402

base = p13d_runner.base


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def protocol():
    checkpoint_hash = sha(CHECKPOINT) if CHECKPOINT.exists() else None
    return {
        "name": f"warm_start_p13d_pcim_rnc_fixed_dsrc_seed_{SEED}",
        "gpu": GPU,
        "fixed_backbone": {
            "model": "pure_p13d_original_mlp_head",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": checkpoint_hash,
            "training_seed": SEED,
        },
        "adapter": {
            "variant": "bidirectional",
            "width": 32,
            "top_k": 128,
            "coverage_budget": 16,
            "max_delta": 0.5,
            "cross_max_scale": 0.25,
            "residual_l2": 0.001,
            "rnc_weight": 0.01,
            "temperature": 2.0,
            "projection_dim": 16,
            "rnc_batch_size": 32,
            "microbatch_size": 16,
            "interval": 2,
            "high_count": 8,
            "mid_count": 8,
            "high_threshold": 7.0,
            "mid_threshold": 5.0,
            "warmup_epochs": 1.0,
            "seed": SEED,
        },
        "screening": {
            "max_epochs": 60,
            "patience": 10,
            "min_delta": 0.00001,
            "batch_size": 16,
            "eval_batch_size": 64,
            "lr": 0.0003,
            "weight_decay": 0.00001,
            "grad_clip": 1.0,
            "screen_lambdas": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5],
        },
        "fixed_dsrc": {
            "drug_config": {"gamma": 2.0, "k": 8, "minimum": 0.1, "tau": 0.02, "beta": 0.5, "clip": 1.0, "scale": 1.0},
            "protein_config": {"gamma": 7.0, "k": 12, "minimum": 0.5, "tau": 0.0025, "beta": 0.375, "clip": 1.25, "scale": 0.75},
            "adapter_lambda": 0.85,
            "rho_drug": 0.5,
            "rho_protein": 0.5,
            "fusion_scale": 1.9,
        },
        "reference_seed42_test_mse": 0.18321482603472625,
        "test_policy": "all hyperparameters fixed before new-seed test evaluation",
    }


# Reuse the audited PCIM/RNC/calibration implementation with this experiment's
# output and protocol. These globals are resolved dynamically by its functions.
tune.OUT = OUTPUT
tune.SOURCE_OUT = OUTPUT
tune.protocol = protocol


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_cache():
    destination = OUTPUT / "cache/warm_811.pt"
    audit_path = OUTPUT / "cache/audit.json"
    if destination.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if sha(destination) != audit["cache_sha256"]:
            raise RuntimeError("Existing P13D cache checksum mismatch")
        print("CACHE_REUSED", flush=True)
        return

    if sha(CHECKPOINT) != protocol()["fixed_backbone"]["checkpoint_sha256"]:
        raise RuntimeError("P13D checkpoint checksum changed")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    arguments = p13d_runner.arguments()
    seed_all(protocol()["adapter"]["seed"])
    model = base.build_model(arguments, torch.device("cuda"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    dataset, train_set, validation_set, _, _ = base.build_dataloaders(arguments)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    if list(train_set.indices) != split["train_indices"]:
        raise RuntimeError("P13D train split mismatch")
    if list(validation_set.indices) != split["val_indices"]:
        raise RuntimeError("P13D validation split mismatch")

    frame = dataset.df.copy()
    raw = pd.read_csv(PROJECT / arguments.pairs_csv, dtype={"drug_id": str, "protein_id": str})
    frame["drug_id"] = frame.drug_id.astype(str)
    frame["protein_id"] = frame.protein_id.astype(str)
    if len(frame) != 30056 or not frame[["drug_id", "protein_id"]].equals(raw[["drug_id", "protein_id"]]):
        raise RuntimeError("P13D dataset row order differs from pairs.csv")

    similarity_data = np.load(TUNE_EXPERIMENT / "entity_similarities.npz", allow_pickle=True)
    drugs = list(map(str, similarity_data["drug_ids"].tolist()))
    proteins = list(map(str, similarity_data["protein_ids"].tolist()))
    drug_lookup = {value: index for index, value in enumerate(drugs)}
    protein_lookup = {value: index for index, value in enumerate(proteins)}
    pair_indices = np.full((len(drugs), len(proteins)), -1, dtype=np.int64)
    row_drug = np.empty(len(frame), dtype=np.int64)
    row_protein = np.empty(len(frame), dtype=np.int64)
    for row_index, row in enumerate(frame.itertuples(index=False)):
        drug = drug_lookup[row.drug_id]
        protein = protein_lookup[row.protein_id]
        pair_indices[drug, protein] = row_index
        row_drug[row_index] = drug
        row_protein[row_index] = protein
    if not (pair_indices >= 0).all() or np.unique(pair_indices).size != len(frame):
        raise RuntimeError("Davis pair grid is incomplete or duplicated")
    labels = np.empty_like(pair_indices, dtype=np.float64)
    labels[row_drug, row_protein] = frame.label.to_numpy(dtype=np.float64)

    amino_acids = [
        "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
        "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",
    ]
    amino_lookup = {name: index for index, name in enumerate(amino_acids)}
    amino_lookup["MSE"] = amino_lookup["MET"]
    atoms, residues, amino_types, drug_global, protein_global = [], [], [], [], []
    with torch.inference_mode():
        for index, drug_id in enumerate(drugs):
            graph_data = torch.load(
                PROJECT / arguments.drug_3d_dir / f"{drug_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
            one_d = torch.load(
                PROJECT / arguments.drug_1d_dir / f"{drug_id}.pt",
                map_location="cpu",
                weights_only=False,
            )["mean"].float()[None].cuda()
            graph = {key: graph_data[key].cuda() for key in ["x", "pos", "edge_index"]}
            graph["x"] = graph["x"].float()
            graph["pos"] = graph["pos"].float()
            graph["batch"] = torch.zeros(len(graph_data["x"]), dtype=torch.long, device="cuda")
            encoded = model.drug_3d_encoder(graph, return_node=True)
            atoms.append(encoded["node_feat"].cpu().clone())
            drug_global.append(
                model.drug_fusion([model.drug_1d_encoder(one_d), encoded["graph_feat"]]).cpu().clone()[0]
            )
            if index % 20 == 0:
                print(f"CACHE_DRUG {index}/{len(drugs)}", flush=True)

        for index, protein_id in enumerate(proteins):
            graph_data = torch.load(
                PROJECT / arguments.protein_3d_dir / f"{protein_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
            types = torch.tensor(
                [amino_lookup.get(str(meta["resname"]).upper(), 20) for meta in graph_data["residue_meta"]],
                dtype=torch.long,
            )
            amino_types.append(types)
            one_d = torch.load(
                PROJECT / arguments.protein_1d_dir / f"{protein_id}.pt",
                map_location="cpu",
                weights_only=False,
            )["mean"].float()[None].cuda()
            graph = {
                key: graph_data[key].cuda()
                for key in ["node_s", "node_v", "coords", "edge_index", "edge_s", "edge_v"]
            }
            graph["batch"] = torch.zeros(len(types), dtype=torch.long, device="cuda")
            encoded = model.protein_3d_encoder(graph, return_node=True)
            residues.append(encoded["node_feat"].cpu().clone())
            protein_global.append(
                model.protein_fusion([model.protein_1d_encoder(one_d), encoded["graph_feat"]]).cpu().clone()[0]
            )
            if index % 50 == 0:
                print(f"CACHE_PROTEIN {index}/{len(proteins)}", flush=True)

        drug_global = torch.stack(drug_global)
        protein_global = torch.stack(protein_global)
        base_prediction = np.empty_like(labels)
        for start in range(0, len(frame), 64):
            drugs_batch = row_drug[start : start + 64]
            proteins_batch = row_protein[start : start + 64]
            prediction = model.decoder(
                torch.cat(
                    [drug_global[drugs_batch].cuda(), protein_global[proteins_batch].cuda()], dim=-1
                )
            ).reshape(-1).cpu().numpy()
            base_prediction[drugs_batch, proteins_batch] = prediction

    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    validation_indices = np.asarray(split["val_indices"], dtype=np.int64)
    if not np.isfinite(base_prediction).all():
        raise RuntimeError("Non-finite P13D predictions in the feature cache")

    split_pairs = {}
    for part in ["train", "val", "test"]:
        indices = np.asarray(split[f"{part}_indices"], dtype=np.int64)
        split_pairs[part] = {
            "index": indices,
            "drug": row_drug[indices],
            "protein": row_protein[indices],
        }
    data = {
        "drugs": drugs,
        "proteins": proteins,
        "pair_indices": pair_indices,
        "row_drug": row_drug,
        "row_protein": row_protein,
        "labels": labels,
        "base": base_prediction,
        "atoms": atoms,
        "residues": residues,
        "aa": amino_types,
        "drug_global": drug_global,
        "protein_global": protein_global,
        "split_pairs": split_pairs,
        "similarity": similarity_data["drug_similarity"].astype(np.float64),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha(CHECKPOINT),
        "split_sha256": sha(SPLIT),
    }
    save(destination, data)
    dump(
        audit_path,
        {
            "passed": True,
            "cache_sha256": sha(destination),
            "checkpoint_sha256": sha(CHECKPOINT),
            "split_sha256": sha(SPLIT),
            "training_seed": SEED,
            "base_prediction_finite": True,
            "train": len(split_pairs["train"]["index"]),
            "validation": len(split_pairs["val"]["index"]),
            "test_sealed": len(split_pairs["test"]["index"]),
            "finished": time.time(),
        },
    )
    print(f"CACHE_FINISHED seed={SEED}", flush=True)


def metric(prediction, label):
    return base.compute_regression_metrics(
        torch.as_tensor(prediction, dtype=torch.float64),
        torch.as_tensor(label, dtype=torch.float64),
    )


def neighbor_stats(query, residual, mask, similarity, config, axis):
    count = len(query["drug"])
    mean = np.zeros(count, dtype=np.float64)
    variance = np.zeros(count, dtype=np.float64)
    support = np.zeros(count, dtype=np.float64)
    for index, (drug, protein) in enumerate(zip(query["drug"], query["protein"])):
        if axis == "drug":
            references = np.flatnonzero(mask[:, protein])
            similarities = similarity[drug, references]
            values = residual[references, protein]
        elif axis == "protein":
            references = np.flatnonzero(mask[drug, :])
            similarities = similarity[protein, references]
            values = residual[drug, references]
        else:
            raise ValueError(axis)
        limit = min(config["k"], len(references))
        order = np.argsort(similarities)[::-1][:limit]
        similarities = similarities[order]
        values = values[order]
        keep = similarities >= config["minimum"]
        similarities = similarities[keep]
        values = values[keep]
        if not len(values):
            continue
        weights = similarities ** config["gamma"]
        total = weights.sum()
        if total <= 1e-12:
            continue
        mean[index] = np.sum(weights * values) / total
        variance[index] = np.sum(weights * (values - mean[index]) ** 2) / total
        support[index] = total / config["k"]
    return mean, variance, support


def branch_correction(mean, variance, support, config):
    confidence = support / (support + config["tau"]) * np.exp(-config["beta"] * variance)
    correction = config["scale"] * confidence * np.clip(mean, -config["clip"], config["clip"])
    return correction, confidence


def residual_grid(cache, train_prediction, labels):
    train = cache["split_pairs"]["train"]
    residual = np.zeros_like(cache["labels"], dtype=np.float64)
    mask = np.zeros_like(cache["labels"], dtype=bool)
    mask[train["drug"], train["protein"]] = True
    residual[train["drug"], train["protein"]] = labels[train["drug"], train["protein"]] - train_prediction
    return residual, mask


def fuse(drug_delta, protein_delta, drug_support, protein_support, fixed):
    drug_available = (drug_support > 0).astype(np.float64)
    protein_available = (protein_support > 0).astype(np.float64)
    drug_weight = fixed["rho_drug"] * drug_available
    protein_weight = fixed["rho_protein"] * protein_available
    denominator = drug_weight + protein_weight
    correction = np.zeros_like(drug_delta)
    valid = denominator > 0
    correction[valid] = fixed["fusion_scale"] * (
        drug_weight[valid] * drug_delta[valid]
        + protein_weight[valid] * protein_delta[valid]
    ) / denominator[valid]
    return correction


def apply_fixed_dsrc(cache, similarities, train_delta, query_delta, query, labels, fixed):
    train = cache["split_pairs"]["train"]
    adapter_lambda = fixed["adapter_lambda"]
    train_prediction = cache["base"][train["drug"], train["protein"]] + adapter_lambda * train_delta
    query_prediction = cache["base"][query["drug"], query["protein"]] + adapter_lambda * query_delta
    residual, mask = residual_grid(cache, train_prediction, labels)
    drug_stats = neighbor_stats(
        query, residual, mask, similarities["drug_similarity"].astype(np.float64),
        fixed["drug_config"], "drug",
    )
    protein_stats = neighbor_stats(
        query, residual, mask, similarities["protein_similarity"].astype(np.float64),
        fixed["protein_config"], "protein",
    )
    drug_delta, drug_confidence = branch_correction(*drug_stats, fixed["drug_config"])
    protein_delta, protein_confidence = branch_correction(*protein_stats, fixed["protein_config"])
    correction = fuse(drug_delta, protein_delta, drug_stats[2], protein_stats[2], fixed)
    return {
        "calibrated": query_prediction + correction,
        "raw": query_prediction,
        "correction": correction,
        "drug_delta": drug_delta,
        "protein_delta": protein_delta,
        "drug_confidence": drug_confidence,
        "protein_confidence": protein_confidence,
        "drug_support": drug_stats[2],
        "protein_support": protein_stats[2],
    }


def train_and_evaluate():
    result_path = OUTPUT / "result.json"
    if result_path.exists():
        print("RESULT_REUSED", flush=True)
        return
    cache = tune.load_cache()
    store = tune.FeatureStore(cache)
    config = protocol()["adapter"]
    candidate = tune.train_candidate("p13d_pcim_rnc", config, store, cache)
    _, development_delta = tune.load_meta_deltas(candidate, store, cache, parts=("train", "val"))
    fixed = protocol()["fixed_dsrc"]
    similarities = np.load(SIMILARITY_PATH, allow_pickle=True)
    validation = cache["split_pairs"]["val"]
    labels = cache["labels"]
    validation_outputs = apply_fixed_dsrc(
        cache, similarities, development_delta["train"], development_delta["val"],
        validation, labels, fixed,
    )
    validation_target = labels[validation["drug"], validation["protein"]]
    selection = {
        "adapter": candidate,
        "fixed_dsrc": fixed,
        "raw_validation_selection": candidate["best"],
        "fixed_dsrc_validation_metrics": metric(validation_outputs["calibrated"], validation_target),
        "selection_metric": "adapter early stopping on validation; DSRC hyperparameters fixed from seed 42",
        "selection_data": "no seed-specific DSRC tuning; test labels sealed",
        "backbone_checkpoint": str(CHECKPOINT),
        "backbone_sha256": sha(CHECKPOINT),
        "locked_at": time.time(),
    }
    dump(OUTPUT / "selection_locked.json", selection)
    print("SELECTION_LOCKED " + json.dumps(selection), flush=True)

    # Test deltas and labels are accessed only after the fixed configuration is persisted.
    _, test_delta = tune.load_meta_deltas(candidate, store, cache, parts=("train", "test"))
    test = cache["split_pairs"]["test"]
    outputs = apply_fixed_dsrc(
        cache, similarities, test_delta["train"], test_delta["test"], test, labels, fixed,
    )
    poisoned = labels.copy()
    poisoned[validation["drug"], validation["protein"]] = np.nan
    poisoned[test["drug"], test["protein"]] = np.nan
    poisoned_outputs = apply_fixed_dsrc(
        cache, similarities, test_delta["train"], test_delta["test"], test, poisoned, fixed,
    )
    if not np.array_equal(outputs["calibrated"], poisoned_outputs["calibrated"]):
        raise RuntimeError("Prediction changed after validation/test label poisoning")

    target = labels[test["drug"], test["protein"]]
    base_prediction = cache["base"][test["drug"], test["protein"]]
    raw_base_metrics = metric(base_prediction, target)
    original = json.loads((P13D_OUTPUT / "result.json").read_text(encoding="utf-8"))
    if abs(raw_base_metrics["mse"] - original["test_metrics"]["mse"]) > 1e-5:
        raise RuntimeError("P13D raw test replay does not match the completed baseline")
    high = target >= config["high_threshold"]
    result = {
        "complete": True,
        "training_seed": SEED,
        "model": "pure_p13d_plus_pcim_standard_rnc_0.01_plus_fixed_DSRC",
        "selection": selection,
        "test": {
            "p13d_raw": raw_base_metrics,
            "p13d_pcim_rnc_raw": metric(outputs["raw"], target),
            "DSRC": metric(outputs["calibrated"], target),
        },
        "high_affinity_count": int(high.sum()),
        "high_affinity": {
            "p13d_pcim_rnc_raw": metric(outputs["raw"][high], target[high]),
            "DSRC": metric(outputs["calibrated"][high], target[high]),
        },
        "correction": {
            "absolute_mean": float(np.abs(outputs["correction"]).mean()),
            "absolute_max": float(np.abs(outputs["correction"]).max()),
            "drug_confidence_mean": float(outputs["drug_confidence"].mean()),
            "protein_confidence_mean": float(outputs["protein_confidence"].mean()),
        },
        "raw_replay_matches_original": True,
        "label_poisoning_invariance": True,
        "test_policy": "test evaluated once after fixed DSRC configuration and adapter checkpoint were locked",
        "finished": time.time(),
    }
    save_npz(
        OUTPUT / "test_predictions.npz",
        indices=test["index"],
        y_true=target,
        p13d_raw=base_prediction,
        pcim_rnc_raw=outputs["raw"],
        DSRC=outputs["calibrated"],
        correction=outputs["correction"],
        drug_correction=outputs["drug_delta"],
        protein_correction=outputs["protein_delta"],
    )
    dump(result_path, result)
    dump(OUTPUT / "status.json", {"state": "complete", **result})
    print("FINAL_RESULT " + json.dumps(result), flush=True)


def preflight():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    sets = [set(split[f"{part}_indices"]) for part in ["train", "val", "test"]]
    if [len(values) for values in sets] != [24044, 3005, 3007]:
        raise RuntimeError("Unexpected 8:1:1 split sizes")
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("Split overlap detected")
    if sha(CHECKPOINT) != protocol()["fixed_backbone"]["checkpoint_sha256"]:
        raise RuntimeError("P13D checkpoint checksum mismatch")
    config = protocol()["adapter"]
    pair = tune.PairResidual(
        config["variant"],
        width=config["width"],
        top_k=config["top_k"],
        max_delta=config["max_delta"],
        selection="coverage",
        coverage_budget=config["coverage_budget"],
        cross_max_scale=config["cross_max_scale"],
        enable_contrast=True,
        contrast_dim=config["projection_dim"],
    )
    report = {
        "passed": True,
        "train": len(sets[0]),
        "validation": len(sets[1]),
        "test_sealed": len(sets[2]),
        "split_sha256": sha(SPLIT),
        "checkpoint_sha256": sha(CHECKPOINT),
        "adapter_parameters": sum(parameter.numel() for parameter in pair.parameters()),
        "adapter": config,
        "screening": protocol()["screening"],
        "test_policy": protocol()["test_policy"],
        "finished": time.time(),
    }
    dump(OUTPUT / "preflight.json", report)
    dump(OUTPUT / "protocol.json", protocol())
    print("PREFLIGHT_PASSED " + json.dumps(report), flush=True)


def worker():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(exist_ok=True)
    lock = (OUTPUT / "worker.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ANOTHER_WORKER_IS_ACTIVE", flush=True)
        return
    stages = ["cache", "run"]
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(protocol()["gpu"]),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
    )
    for position, action in enumerate(stages, start=1):
        free = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(protocol()["gpu"]),
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        if free < 30000:
            raise RuntimeError(f"GPU {protocol()['gpu']} has only {free} MiB free")
        log_path = OUTPUT / "logs" / f"{action}.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-B", "-u", str(Path(__file__).resolve()), action],
                cwd=PROJECT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            dump(
                OUTPUT / "status.json",
                {
                    "state": "running",
                    "stage": action,
                    "task_index": position,
                    "total_tasks": len(stages),
                    "worker_pid": os.getpid(),
                    "child_pid": process.pid,
                    "gpu": protocol()["gpu"],
                    "log": str(log_path),
                    "updated": time.time(),
                },
            )
            exit_code = process.wait()
        if exit_code:
            dump(
                OUTPUT / "status.json",
                {
                    "state": "failed",
                    "stage": action,
                    "exit_code": exit_code,
                    "worker_pid": os.getpid(),
                    "child_pid": process.pid,
                    "gpu": protocol()["gpu"],
                    "log": str(log_path),
                    "updated": time.time(),
                },
            )
            raise RuntimeError(f"{action} failed with exit code {exit_code}; see {log_path}")
    final_status = {
        "state": "complete",
        "worker_pid": os.getpid(),
        "gpu": protocol()["gpu"],
        "finished": time.time(),
    }
    if (OUTPUT / "result.json").exists():
        final_status.update(json.loads((OUTPUT / "result.json").read_text(encoding="utf-8")))
    dump(OUTPUT / "status.json", final_status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "cache", "run", "worker"])
    arguments = parser.parse_args()
    {
        "preflight": preflight,
        "cache": build_cache,
        "run": train_and_evaluate,
        "worker": worker,
    }[arguments.action]()
