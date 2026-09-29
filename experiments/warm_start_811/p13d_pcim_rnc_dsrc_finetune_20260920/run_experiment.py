"""Leakage-safe fine tuning of DSRC for DAVIS warm-start.

DSRC = Drug-Protein Dual-Similarity-Guided Residual Calibration.
The P13D backbone and PCIM/RNC adapter are fixed. Only validation-selected
residual calibration parameters are fitted; test labels stay sealed until the
selection file is persisted.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch


PROJECT = Path("/data1/ztx/MyModel-MDTA")
SOURCE_EXPERIMENT = PROJECT / "experiments/warm_start_811/rank16_pcim_rnc_lowcost_tune_20260917"
SOURCE_OUTPUT = PROJECT / "outputs/Refine_experiment/davis/warm_start/random_pair_811_seed42/p13d_pcim_rnc_r1_20260920"
OUTPUT = PROJECT / "outputs/Refine_experiment/davis/warm_start/random_pair_811_seed42/p13d_pcim_rnc_dsrc_finetune_20260920"
CACHE_PATH = SOURCE_OUTPUT / "cache/warm_811.pt"
ADAPTER_RESULT = SOURCE_OUTPUT / "candidates/p13d_pcim_rnc/result.json"
SIMILARITY_PATH = SOURCE_EXPERIMENT / "entity_similarities.npz"

sys.path.insert(0, str(SOURCE_EXPERIMENT))
import tune_experiment as tune  # noqa: E402


DRUG_CONFIG = {
    "gamma": 2.0,
    "k": 8,
    "minimum": 0.1,
    "tau": 0.02,
    "beta": 0.5,
    "clip": 1.0,
    "scale": 1.0,
}

PROTEIN_SEARCH = {
    # Expand the old upper-bound optimum (gamma=4, minimum=0.5), while
    # retaining the exact incumbent configuration as a non-regression anchor.
    "gamma": [3.0, 4.0, 5.0, 6.0, 8.0],
    "k": [6, 8, 10, 12],
    "minimum": [0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
    "tau": [0.01, 0.015, 0.02, 0.03, 0.05],
    "beta": [0.0, 0.125, 0.25, 0.5, 0.75],
    "clip": [0.75, 1.0, 1.25, 1.5],
    "scale": [0.75, 0.9, 1.0, 1.1, 1.25],
}

FUSION_SEARCH = {
    # The incumbent selected both the lower adapter boundary (0.75) and the
    # upper fusion boundary (1.25), so search beyond both and refine rho.
    "adapter_lambda": [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1],
    "rho_drug": [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
    "fusion_scale": [1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.75],
    "protein_candidates": 100,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
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


def metric(prediction, label):
    return tune.base_module.compute_regression_metrics(
        torch.as_tensor(prediction, dtype=torch.float64),
        torch.as_tensor(label, dtype=torch.float64),
    )


def development_labels(cache):
    labels = cache["labels"].copy()
    validation = cache["split_pairs"]["val"]
    test = cache["split_pairs"]["test"]
    # Validation labels remain available for selection targets but are never
    # reference residuals; test labels are fully masked before selection.
    labels[test["drug"], test["protein"]] = np.nan
    if not np.isfinite(labels[validation["drug"], validation["protein"]]).all():
        raise RuntimeError("Validation labels were unexpectedly masked")
    return labels


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
    correction = config.get("scale", 1.0) * confidence * np.clip(
        mean, -config["clip"], config["clip"]
    )
    return correction, confidence


def residual_grid(cache, train_prediction, labels):
    train = cache["split_pairs"]["train"]
    residual = np.zeros_like(cache["labels"], dtype=np.float64)
    mask = np.zeros_like(cache["labels"], dtype=bool)
    mask[train["drug"], train["protein"]] = True
    residual[train["drug"], train["protein"]] = (
        labels[train["drug"], train["protein"]] - train_prediction
    )
    return residual, mask


def fuse(drug_delta, protein_delta, drug_support, protein_support, rho_drug, scale):
    drug_available = (drug_support > 0).astype(np.float64)
    protein_available = (protein_support > 0).astype(np.float64)
    drug_weight = rho_drug * drug_available
    protein_weight = (1.0 - rho_drug) * protein_available
    denominator = drug_weight + protein_weight
    correction = np.zeros_like(drug_delta)
    valid = denominator > 0
    correction[valid] = scale * (
        drug_weight[valid] * drug_delta[valid]
        + protein_weight[valid] * protein_delta[valid]
    ) / denominator[valid]
    return correction


def load_inputs():
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    adapter = json.loads(ADAPTER_RESULT.read_text(encoding="utf-8"))
    similarities = np.load(SIMILARITY_PATH, allow_pickle=True)
    if list(map(str, similarities["drug_ids"].tolist())) != list(cache["drugs"]):
        raise RuntimeError("Drug similarity order differs from cache")
    if list(map(str, similarities["protein_ids"].tolist())) != list(cache["proteins"]):
        raise RuntimeError("Protein similarity order differs from cache")
    if cache["checkpoint_sha256"] != "f89963e638610696689e0770a23fc062b70efcc52b56e4dd9d8ea4cb795234db":
        raise RuntimeError("Unexpected P13D checkpoint in source cache")
    return cache, adapter, similarities


def development_predictions(cache, adapter):
    path = OUTPUT / "development_predictions.npz"
    if path.exists():
        data = np.load(path)
        print("DEVELOPMENT_PREDICTIONS_REUSED", flush=True)
        return {
            "train_delta": data["train_delta"].astype(np.float64),
            "validation_delta": data["validation_delta"].astype(np.float64),
        }
    store = tune.FeatureStore(cache)
    _, predictions = tune.load_meta_deltas(
        adapter, store, cache, parts=("train", "val")
    )
    save_npz(
        path,
        train_indices=cache["split_pairs"]["train"]["index"],
        validation_indices=cache["split_pairs"]["val"]["index"],
        train_delta=predictions["train"],
        validation_delta=predictions["val"],
    )
    return {
        "train_delta": predictions["train"],
        "validation_delta": predictions["val"],
    }


def select_dsrc(cache, similarities, predictions):
    train = cache["split_pairs"]["train"]
    validation = cache["split_pairs"]["val"]
    labels = development_labels(cache)
    target = labels[validation["drug"], validation["protein"]]
    drug_similarity = similarities["drug_similarity"].astype(np.float64)
    protein_similarity = similarities["protein_similarity"].astype(np.float64)

    # Reproduce the current one-sided R1 boundary at adapter lambda 1.0.
    base_train = cache["base"][train["drug"], train["protein"]] + predictions["train_delta"]
    base_validation = cache["base"][validation["drug"], validation["protein"]] + predictions["validation_delta"]
    residual, mask = residual_grid(cache, base_train, labels)
    drug_stats = neighbor_stats(validation, residual, mask, drug_similarity, DRUG_CONFIG, "drug")
    drug_delta, drug_confidence = branch_correction(*drug_stats, DRUG_CONFIG)
    drug_only_prediction = base_validation + drug_delta
    drug_only_mse = float(np.mean((drug_only_prediction - target) ** 2))

    # Stage 1: select protein-side configurations independently at lambda=1.
    protein_ranking = []
    for gamma in PROTEIN_SEARCH["gamma"]:
        for k in PROTEIN_SEARCH["k"]:
            for minimum in PROTEIN_SEARCH["minimum"]:
                kernel = {"gamma": gamma, "k": k, "minimum": minimum}
                mean, variance, support = neighbor_stats(
                    validation, residual, mask, protein_similarity, kernel, "protein"
                )
                for tau in PROTEIN_SEARCH["tau"]:
                    for beta in PROTEIN_SEARCH["beta"]:
                        for clip in PROTEIN_SEARCH["clip"]:
                            branch = {**kernel, "tau": tau, "beta": beta, "clip": clip}
                            unit_delta, confidence = branch_correction(
                                mean, variance, support, branch
                            )
                            for scale in PROTEIN_SEARCH["scale"]:
                                prediction = base_validation + scale * unit_delta
                                protein_ranking.append(
                                    {
                                        **branch,
                                        "scale": scale,
                                        "mse": float(np.mean((prediction - target) ** 2)),
                                        "support_fraction": float(np.mean(support > 0)),
                                        "confidence_mean": float(confidence.mean()),
                                    }
                                )
    protein_ranking.sort(key=lambda item: item["mse"])

    # Stage 2: jointly select adapter scale, dual fusion, and global scale using
    # only the most promising protein-side configurations from stage 1.
    fusion_ranking = []
    seen_protein_configs = []
    seen_keys = set()
    for item in protein_ranking:
        key = tuple(item[name] for name in ["gamma", "k", "minimum", "tau", "beta", "clip", "scale"])
        if key not in seen_keys:
            seen_keys.add(key)
            seen_protein_configs.append({name: item[name] for name in [
                "gamma", "k", "minimum", "tau", "beta", "clip", "scale"
            ]})
        if len(seen_protein_configs) >= FUSION_SEARCH["protein_candidates"]:
            break

    for adapter_lambda in FUSION_SEARCH["adapter_lambda"]:
        train_prediction = (
            cache["base"][train["drug"], train["protein"]]
            + adapter_lambda * predictions["train_delta"]
        )
        validation_prediction = (
            cache["base"][validation["drug"], validation["protein"]]
            + adapter_lambda * predictions["validation_delta"]
        )
        residual, mask = residual_grid(cache, train_prediction, labels)
        d_mean, d_variance, d_support = neighbor_stats(
            validation, residual, mask, drug_similarity, DRUG_CONFIG, "drug"
        )
        d_delta, d_confidence = branch_correction(
            d_mean, d_variance, d_support, DRUG_CONFIG
        )
        for protein_config in seen_protein_configs:
            p_mean, p_variance, p_support = neighbor_stats(
                validation,
                residual,
                mask,
                protein_similarity,
                protein_config,
                "protein",
            )
            p_delta, p_confidence = branch_correction(
                p_mean, p_variance, p_support, protein_config
            )
            for rho_drug in FUSION_SEARCH["rho_drug"]:
                for fusion_scale in FUSION_SEARCH["fusion_scale"]:
                    correction = fuse(
                        d_delta,
                        p_delta,
                        d_support,
                        p_support,
                        rho_drug,
                        fusion_scale,
                    )
                    prediction = validation_prediction + correction
                    fusion_ranking.append(
                        {
                            "adapter_lambda": adapter_lambda,
                            "rho_drug": rho_drug,
                            "fusion_scale": fusion_scale,
                            "protein": protein_config,
                            "mse": float(np.mean((prediction - target) ** 2)),
                            "drug_support_fraction": float(np.mean(d_support > 0)),
                            "protein_support_fraction": float(np.mean(p_support > 0)),
                            "drug_confidence_mean": float(d_confidence.mean()),
                            "protein_confidence_mean": float(p_confidence.mean()),
                        }
                    )
    fusion_ranking.sort(key=lambda item: item["mse"])
    best = fusion_ranking[0]
    return {
        "method": "Drug-Protein Dual-Similarity-Guided Residual Calibration",
        "acronym": "DSRC",
        "drug_config": DRUG_CONFIG,
        "protein_config": best["protein"],
        "adapter_lambda": best["adapter_lambda"],
        "rho_drug": best["rho_drug"],
        "rho_protein": 1.0 - best["rho_drug"],
        "fusion_scale": best["fusion_scale"],
        "validation": {
            "pcim_rnc_raw_mse": float(np.mean((base_validation - target) ** 2)),
            "drug_only_r1_mse": drug_only_mse,
            "protein_only_best_mse": protein_ranking[0]["mse"],
            "dsrc_mse": best["mse"],
            "gain_vs_drug_only": drug_only_mse - best["mse"],
        },
        "search": {
            "protein_grid": PROTEIN_SEARCH,
            "fusion_grid": FUSION_SEARCH,
            "protein_top": protein_ranking[:50],
            "fusion_top": fusion_ranking[:50],
        },
        "selection_metric": "validation_mse",
        "selection_data": "training residual bank and validation targets only; test labels sealed",
        "locked_at": time.time(),
    }


def apply_selected(cache, similarities, train_delta, query_delta, query, labels, selection):
    train = cache["split_pairs"]["train"]
    adapter_lambda = selection["adapter_lambda"]
    train_prediction = (
        cache["base"][train["drug"], train["protein"]]
        + adapter_lambda * train_delta
    )
    query_prediction = (
        cache["base"][query["drug"], query["protein"]]
        + adapter_lambda * query_delta
    )
    residual, mask = residual_grid(cache, train_prediction, labels)
    drug_stats = neighbor_stats(
        query,
        residual,
        mask,
        similarities["drug_similarity"].astype(np.float64),
        selection["drug_config"],
        "drug",
    )
    protein_stats = neighbor_stats(
        query,
        residual,
        mask,
        similarities["protein_similarity"].astype(np.float64),
        selection["protein_config"],
        "protein",
    )
    drug_delta, drug_confidence = branch_correction(
        *drug_stats, selection["drug_config"]
    )
    protein_delta, protein_confidence = branch_correction(
        *protein_stats, selection["protein_config"]
    )
    correction = fuse(
        drug_delta,
        protein_delta,
        drug_stats[2],
        protein_stats[2],
        selection["rho_drug"],
        selection["fusion_scale"],
    )
    return (
        query_prediction + correction,
        query_prediction,
        correction,
        drug_delta,
        protein_delta,
        drug_confidence,
        protein_confidence,
        drug_stats[2],
        protein_stats[2],
    )


def preflight():
    cache, adapter, similarities = load_inputs()
    report = {
        "passed": True,
        "method": "DSRC",
        "full_name": "Drug-Protein Dual-Similarity-Guided Residual Calibration",
        "source_cache_sha256": sha256(CACHE_PATH),
        "adapter_checkpoint": adapter["checkpoint"],
        "adapter_checkpoint_sha256": sha256(adapter["checkpoint"]),
        "similarity_sha256": sha256(SIMILARITY_PATH),
        "drug_similarity_shape": list(similarities["drug_similarity"].shape),
        "protein_similarity_shape": list(similarities["protein_similarity"].shape),
        "train": len(cache["split_pairs"]["train"]["index"]),
        "validation": len(cache["split_pairs"]["val"]["index"]),
        "test_sealed": len(cache["split_pairs"]["test"]["index"]),
        "drug_config": DRUG_CONFIG,
        "protein_search": PROTEIN_SEARCH,
        "fusion_search": FUSION_SEARCH,
        "test_policy": "selection_locked.json persisted before test inference/evaluation",
        "finished": time.time(),
    }
    dump_json(OUTPUT / "preflight.json", report)
    print("PREFLIGHT_PASSED " + json.dumps(report), flush=True)


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock_handle = (OUTPUT / "worker.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ANOTHER_WORKER_IS_ACTIVE", flush=True)
        return
    if (OUTPUT / "result.json").exists():
        print("RESULT_ALREADY_COMPLETE", flush=True)
        return

    cache, adapter, similarities = load_inputs()
    dump_json(
        OUTPUT / "status.json",
        {"state": "predicting_development", "test_sealed": 3007, "updated": time.time()},
    )
    predictions = development_predictions(cache, adapter)
    selection_path = OUTPUT / "selection_locked.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        print("SELECTION_REUSED", flush=True)
    else:
        dump_json(
            OUTPUT / "status.json",
            {"state": "selecting_dsrc_on_validation", "test_sealed": 3007, "updated": time.time()},
        )
        selection = select_dsrc(cache, similarities, predictions)
        ranking = selection.pop("search")
        dump_json(OUTPUT / "validation_ranking.json", ranking)
        dump_json(selection_path, selection)
        print("SELECTION_LOCKED " + json.dumps(selection), flush=True)

    # Test inference and test-label access happen only after selection is locked.
    dump_json(
        OUTPUT / "status.json",
        {"state": "evaluating_test", "selection": selection, "updated": time.time()},
    )
    store = tune.FeatureStore(cache)
    _, test_predictions = tune.load_meta_deltas(
        adapter, store, cache, parts=("train", "test")
    )
    test = cache["split_pairs"]["test"]
    validation = cache["split_pairs"]["val"]
    labels = cache["labels"]
    outputs = apply_selected(
        cache,
        similarities,
        test_predictions["train"],
        test_predictions["test"],
        test,
        labels,
        selection,
    )
    calibrated, raw, correction, drug_delta, protein_delta = outputs[:5]
    poisoned = labels.copy()
    poisoned[validation["drug"], validation["protein"]] = np.nan
    poisoned[test["drug"], test["protein"]] = np.nan
    poisoned_calibrated = apply_selected(
        cache,
        similarities,
        test_predictions["train"],
        test_predictions["test"],
        test,
        poisoned,
        selection,
    )[0]
    if not np.array_equal(calibrated, poisoned_calibrated):
        raise RuntimeError("DSRC changed after validation/test label poisoning")

    target = labels[test["drug"], test["protein"]]
    reference = json.loads((SOURCE_OUTPUT / "result.json").read_text(encoding="utf-8"))
    high = target >= 7.0
    result = {
        "complete": True,
        "method": "DSRC",
        "full_name": "Drug-Protein Dual-Similarity-Guided Residual Calibration",
        "base_model": "P13D + PCIM + standard RNC 0.01",
        "selection": selection,
        "test": {
            "pcim_rnc_raw": metric(raw, target),
            "drug_only_R1_reference": reference["test"]["p13d_pcim_rnc_R1"],
            "DSRC": metric(calibrated, target),
        },
        "high_affinity_count": int(high.sum()),
        "high_affinity": {
            "pcim_rnc_raw": metric(raw[high], target[high]),
            "DSRC": metric(calibrated[high], target[high]),
        },
        "correction": {
            "absolute_mean": float(np.abs(correction).mean()),
            "absolute_max": float(np.abs(correction).max()),
            "drug_absolute_mean": float(np.abs(drug_delta).mean()),
            "protein_absolute_mean": float(np.abs(protein_delta).mean()),
            "drug_confidence_mean": float(outputs[5].mean()),
            "protein_confidence_mean": float(outputs[6].mean()),
            "drug_support_fraction": float(np.mean(outputs[7] > 0)),
            "protein_support_fraction": float(np.mean(outputs[8] > 0)),
        },
        "label_poisoning_invariance": True,
        "test_policy": "test evaluated once after validation selection was locked",
        "finished": time.time(),
    }
    save_npz(
        OUTPUT / "test_predictions.npz",
        indices=test["index"],
        y_true=target,
        pcim_rnc_raw=raw,
        DSRC=calibrated,
        correction=correction,
        drug_correction=drug_delta,
        protein_correction=protein_delta,
        drug_confidence=outputs[5],
        protein_confidence=outputs[6],
    )
    dump_json(OUTPUT / "result.json", result)
    dump_json(OUTPUT / "status.json", {"state": "complete", **result})
    print("DSRC_COMPLETE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "run"])
    arguments = parser.parse_args()
    try:
        {"preflight": preflight, "run": run}[arguments.action]()
    except Exception as error:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        dump_json(
            OUTPUT / "status.json",
            {
                "state": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "failed_at": time.time(),
            },
        )
        raise
