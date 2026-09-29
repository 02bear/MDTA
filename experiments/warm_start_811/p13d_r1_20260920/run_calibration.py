"""Leakage-safe R1 calibration for the completed DAVIS 8:1:1 pure P13D run."""

import fcntl
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT = Path("/data1/ztx/MyModel-MDTA")
P13D_EXPERIMENT = PROJECT / "experiments/warm_start_811/p13d_pure_20260918"
P13D_OUTPUT = PROJECT / "outputs/Refine_experiment/davis/warm_start/random_pair_811_seed42/p13d_pure_20260918"
RANK16_OUTPUT = PROJECT / "outputs/Refine_experiment/davis/warm_start/random_pair_811_seed42/rank16_pcim_pair_rnc_20260917"
TUNE_PROTOCOL = PROJECT / "experiments/warm_start_811/rank16_pcim_rnc_lowcost_tune_20260917/tune_protocol.json"
SPLIT_PATH = PROJECT / "data/splits/davis_fixed_split_811_full.json"
OUTPUT = PROJECT / "outputs/Refine_experiment/davis/warm_start/random_pair_811_seed42/p13d_r1_20260920"
CACHE_PATH = RANK16_OUTPUT / "cache/warm_811.pt"
CHECKPOINT_PATH = P13D_OUTPUT / "best_model.pt"

sys.path.insert(0, str(P13D_EXPERIMENT))
import run_experiment as p13d_runner  # noqa: E402

base = p13d_runner.base


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
    return base.compute_regression_metrics(
        torch.as_tensor(prediction, dtype=torch.float64),
        torch.as_tensor(label, dtype=torch.float64),
    )


def masked_stats(query_drug, query_protein, residual, mask, similarity, config):
    mean = np.zeros(len(query_drug), dtype=np.float64)
    variance = np.zeros(len(query_drug), dtype=np.float64)
    support = np.zeros(len(query_drug), dtype=np.float64)
    for index, (drug, protein) in enumerate(zip(query_drug, query_protein)):
        references = np.flatnonzero(mask[:, protein])
        similarities = similarity[drug, references]
        limit = min(config["k_drug"], len(references))
        order = np.argsort(similarities)[::-1][:limit]
        references = references[order]
        similarities = similarities[order]
        keep = similarities >= config["min_drug_similarity"]
        references = references[keep]
        similarities = similarities[keep]
        if not len(references):
            continue
        weights = similarities ** config["gamma"]
        total = weights.sum()
        if total <= 1e-12:
            continue
        values = residual[references, protein]
        mean[index] = np.sum(weights * values) / total
        variance[index] = np.sum(weights * (values - mean[index]) ** 2) / total
        support[index] = total / config["k_drug"]
    return mean, variance, support


def apply_r1(raw, mean, variance, support, config):
    alpha = (
        config["scale"]
        * support
        / (support + config["tau"])
        * np.exp(-config["beta"] * variance)
    )
    correction = alpha * np.clip(mean, -config["clip"], config["clip"])
    return raw + correction, correction, alpha


def r1_correct(train_prediction, query_prediction, labels, train, query, similarity, config):
    shape = labels.shape
    residual = np.zeros(shape, dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)
    mask[train["drug"], train["protein"]] = True
    residual[train["drug"], train["protein"]] = (
        labels[train["drug"], train["protein"]] - train_prediction
    )
    mean, variance, support = masked_stats(
        query["drug"], query["protein"], residual, mask, similarity, config
    )
    return apply_r1(query_prediction, mean, variance, support, config)


@torch.no_grad()
def predict_indices(model, dataset, indices, arguments, device, label):
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        collate_fn=base.mdta_collate_fn_p13d,
        pin_memory=True,
    )
    model.eval()
    predictions = []
    started = time.time()
    for step, batch in enumerate(loader, start=1):
        batch = base.move_batch_to_device(batch, device)
        predictions.append(model(batch).view(-1).detach().cpu().numpy())
        if step % 250 == 0:
            print(f"PREDICT {label} {step}/{len(loader)}", flush=True)
    result = np.concatenate(predictions).astype(np.float64)
    if len(result) != len(indices) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid {label} predictions: {len(result)} for {len(indices)} rows")
    print(
        f"PREDICT_DONE {label} rows={len(result)} seconds={time.time() - started:.1f}",
        flush=True,
    )
    return result


def select_calibration(cache, train_prediction, validation_prediction, protocol):
    train = cache["split_pairs"]["train"]
    validation = cache["split_pairs"]["val"]
    test = cache["split_pairs"]["test"]
    labels = cache["labels"].copy()
    labels[test["drug"], test["protein"]] = np.nan
    target = labels[validation["drug"], validation["protein"]]
    if not np.isfinite(target).all():
        raise RuntimeError("Validation labels unexpectedly contain non-finite values")

    shape = cache["labels"].shape
    residual = np.zeros(shape, dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)
    mask[train["drug"], train["protein"]] = True
    residual[train["drug"], train["protein"]] = (
        labels[train["drug"], train["protein"]] - train_prediction
    )

    search = protocol["calibration"]
    coarse = []
    for gamma in search["gamma"]:
        for k_drug in search["k_drug"]:
            for minimum in search["min_drug_similarity"]:
                kernel = {
                    "gamma": gamma,
                    "k_drug": k_drug,
                    "min_drug_similarity": minimum,
                }
                mean, variance, support = masked_stats(
                    validation["drug"],
                    validation["protein"],
                    residual,
                    mask,
                    cache["similarity"],
                    kernel,
                )
                for tau in search["tau"]:
                    for beta in search["beta"]:
                        config = {
                            **kernel,
                            "tau": tau,
                            "beta": beta,
                            "clip": 1.0,
                            "scale": 1.0,
                        }
                        prediction, _, _ = apply_r1(
                            validation_prediction, mean, variance, support, config
                        )
                        coarse.append(
                            {**config, "mse": float(np.mean((prediction - target) ** 2))}
                        )
    coarse.sort(key=lambda item: item["mse"])

    fine = []
    for seed_config in coarse[: search["coarse_keep"]]:
        kernel = {
            key: seed_config[key]
            for key in ["gamma", "k_drug", "min_drug_similarity"]
        }
        mean, variance, support = masked_stats(
            validation["drug"],
            validation["protein"],
            residual,
            mask,
            cache["similarity"],
            kernel,
        )
        for clip in search["clip"]:
            for scale in search["scale"]:
                config = {
                    **kernel,
                    "tau": seed_config["tau"],
                    "beta": seed_config["beta"],
                    "clip": clip,
                    "scale": scale,
                }
                prediction, _, _ = apply_r1(
                    validation_prediction, mean, variance, support, config
                )
                fine.append({**config, "mse": float(np.mean((prediction - target) ** 2))})
    fine.sort(key=lambda item: item["mse"])
    best = fine[0]
    r1_config = {key: best[key] for key in [
        "gamma", "k_drug", "min_drug_similarity", "tau", "beta", "clip", "scale"
    ]}
    calibrated, correction, alpha = r1_correct(
        train_prediction,
        validation_prediction,
        labels,
        train,
        validation,
        cache["similarity"],
        r1_config,
    )
    return {
        "calibration": best,
        "raw_validation": metric(validation_prediction, target),
        "r1_validation": metric(calibrated, target),
        "validation_gain_mse": float(
            np.mean((validation_prediction - target) ** 2)
            - np.mean((calibrated - target) ** 2)
        ),
        "correction_abs_mean": float(np.abs(correction).mean()),
        "alpha_mean": float(alpha.mean()),
        "coarse_top": coarse[: search["coarse_keep"]],
        "fine_top": fine[:30],
    }


def load_inputs():
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(TUNE_PROTOCOL.read_text(encoding="utf-8"))
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    for part, expected in [("train", 24044), ("val", 3005), ("test", 3007)]:
        indices = np.asarray(split[f"{part}_indices"], dtype=np.int64)
        cached = np.asarray(cache["split_pairs"][part]["index"], dtype=np.int64)
        if len(indices) != expected or not np.array_equal(indices, cached):
            raise RuntimeError(f"Split/cache mismatch for {part}")
    return split, protocol, cache


def load_model_and_dataset(split):
    arguments = p13d_runner.arguments()
    device = torch.device("cuda")
    dataset, train_set, validation_set, _, _ = base.build_dataloaders(arguments)
    if len(dataset) != 30056:
        raise RuntimeError(f"Unexpected dataset size: {len(dataset)}")
    if list(train_set.indices) != split["train_indices"]:
        raise RuntimeError("P13D training indices differ from the locked split")
    if list(validation_set.indices) != split["val_indices"]:
        raise RuntimeError("P13D validation indices differ from the locked split")
    model = base.build_model(arguments, device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return arguments, device, dataset, model, checkpoint


def main():
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

    split, protocol, cache = load_inputs()
    arguments, device, dataset, model, checkpoint = load_model_and_dataset(split)
    preflight = {
        "passed": True,
        "model": "pure_p13d_original_mlp_head_plus_R1",
        "physical_gpu": 1,
        "train": len(split["train_indices"]),
        "validation": len(split["val_indices"]),
        "test_sealed": len(split["test_indices"]),
        "p13d_best_epoch": checkpoint["best_epoch"],
        "checkpoint_sha256": sha256(CHECKPOINT_PATH),
        "split_sha256": sha256(SPLIT_PATH),
        "cache_sha256": sha256(CACHE_PATH),
        "test_policy": "selection_locked.json persisted before test inference/evaluation",
        "started": time.time(),
    }
    dump_json(OUTPUT / "preflight.json", preflight)
    dump_json(OUTPUT / "status.json", {"state": "predicting_development", **preflight})
    print("P13D_R1_START " + json.dumps(preflight), flush=True)

    development_path = OUTPUT / "development_predictions.npz"
    if development_path.exists():
        development = np.load(development_path)
        train_prediction = development["train_prediction"].astype(np.float64)
        validation_prediction = development["validation_prediction"].astype(np.float64)
        print("DEVELOPMENT_PREDICTIONS_REUSED", flush=True)
    else:
        train_prediction = predict_indices(
            model, dataset, split["train_indices"], arguments, device, "train"
        )
        validation_prediction = predict_indices(
            model, dataset, split["val_indices"], arguments, device, "validation"
        )
        save_npz(
            development_path,
            train_indices=np.asarray(split["train_indices"], dtype=np.int64),
            validation_indices=np.asarray(split["val_indices"], dtype=np.int64),
            train_prediction=train_prediction,
            validation_prediction=validation_prediction,
        )

    selection_path = OUTPUT / "selection_locked.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        print("SELECTION_REUSED " + json.dumps(selection["calibration"]), flush=True)
    else:
        dump_json(OUTPUT / "status.json", {"state": "calibrating_validation", **preflight})
        selected = select_calibration(
            cache, train_prediction, validation_prediction, protocol
        )
        dump_json(
            OUTPUT / "calibration_ranking.json",
            {
                "coarse_top": selected.pop("coarse_top"),
                "fine_top": selected.pop("fine_top"),
            },
        )
        selection = {
            **selected,
            "selection_metric": "validation_mse",
            "selection_data": "train labels for residuals; validation labels for calibration; test sealed",
            "checkpoint_sha256": preflight["checkpoint_sha256"],
            "locked_at": time.time(),
        }
        dump_json(selection_path, selection)
        print("SELECTION_LOCKED " + json.dumps(selection), flush=True)

    # Test inference and all test-label access occur only after selection is persisted.
    dump_json(OUTPUT / "status.json", {"state": "evaluating_test", "selection": selection, **preflight})
    test_prediction = predict_indices(
        model, dataset, split["test_indices"], arguments, device, "test"
    )
    train = cache["split_pairs"]["train"]
    validation = cache["split_pairs"]["val"]
    test = cache["split_pairs"]["test"]
    labels = cache["labels"]
    target = labels[test["drug"], test["protein"]]
    config = {
        key: selection["calibration"][key]
        for key in ["gamma", "k_drug", "min_drug_similarity", "tau", "beta", "clip", "scale"]
    }
    calibrated, correction, alpha = r1_correct(
        train_prediction,
        test_prediction,
        labels,
        train,
        test,
        cache["similarity"],
        config,
    )
    poisoned = labels.copy()
    poisoned[validation["drug"], validation["protein"]] = np.nan
    poisoned[test["drug"], test["protein"]] = np.nan
    calibrated_poisoned = r1_correct(
        train_prediction,
        test_prediction,
        poisoned,
        train,
        test,
        cache["similarity"],
        config,
    )[0]
    if not np.array_equal(calibrated, calibrated_poisoned):
        raise RuntimeError("R1 prediction changed after validation/test label poisoning")

    raw_metrics = metric(test_prediction, target)
    reference = json.loads((P13D_OUTPUT / "result.json").read_text(encoding="utf-8"))
    if abs(raw_metrics["mse"] - reference["test_metrics"]["mse"]) > 1e-5:
        raise RuntimeError(
            f"Raw P13D replay mismatch: {raw_metrics['mse']} vs {reference['test_metrics']['mse']}"
        )
    high = target >= 7.0
    result = {
        "complete": True,
        "model": "pure_p13d_original_mlp_head_plus_R1",
        "p13d_best_epoch": checkpoint["best_epoch"],
        "selection": selection,
        "test": {
            "p13d_raw": raw_metrics,
            "p13d_R1": metric(calibrated, target),
        },
        "test_mse_gain": float(raw_metrics["mse"] - np.mean((calibrated - target) ** 2)),
        "high_affinity_count": int(high.sum()),
        "high_affinity": {
            "p13d_raw": metric(test_prediction[high], target[high]),
            "p13d_R1": metric(calibrated[high], target[high]),
        },
        "correction": {
            "absolute_mean": float(np.abs(correction).mean()),
            "absolute_max": float(np.abs(correction).max()),
            "alpha_mean": float(alpha.mean()),
            "alpha_nonzero_fraction": float(np.mean(alpha != 0.0)),
        },
        "raw_replay_matches_original": True,
        "label_poisoning_invariance": True,
        "test_policy": "test inferred/evaluated once after validation selection was locked",
        "finished": time.time(),
    }
    save_npz(
        OUTPUT / "test_predictions.npz",
        indices=np.asarray(split["test_indices"], dtype=np.int64),
        y_true=target,
        p13d_raw=test_prediction,
        p13d_R1=calibrated,
        correction=correction,
        alpha=alpha,
    )
    dump_json(OUTPUT / "result.json", result)
    dump_json(OUTPUT / "status.json", {"state": "complete", **result})
    print("P13D_R1_COMPLETE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
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
