#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

P = Path("/data1/ztx/MyModel-MDTA")

sys.path[:0] = [
    str(P),
    str(P / "experiments/klifs85_interaction"),
]

import train_p13d_earlystop as t

from experiments.klifs85_interaction import run_residual_kernel as r1
from experiments.klifs85_interaction.train_klifs_interact import metrics


SPLITS = (
    P
    / "data/splits/"
      "davis_drug_cold_5fold_seed42"
)

ROOT = (
    P
    / "outputs/Refine_experiment/davis/cold_start/"
      "drug_cold_5fold_seed42"
)

BASE = ROOT / "baseline"

BASE_TEST = ROOT / "evaluation/baseline_e2_5fold"

SIM = (
    P
    / "experiments/klifs85_interaction/data/"
      "similarity_audit_fold1/entity_similarities.npz"
)

OUT = (
    P
    / "experiments/klifs85_interaction/outputs/"
      "r1_original_5fold_outer_test_20260908"
)


def sha(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


@torch.inference_mode()
def infer(model, loader, device):
    preds = []
    labels = []

    model.eval()

    for i, batch in enumerate(loader):
        batch = t.move_batch_to_device(batch, device)

        pred = model(batch)

        preds.append(
            pred.reshape(-1).detach().cpu()
        )
        labels.append(
            batch["label"].reshape(-1).detach().cpu()
        )

        if i % 100 == 0:
            print(
                f"  infer {i}/{len(loader)}",
                flush=True,
            )

    return (
        torch.cat(preds).numpy(),
        torch.cat(labels).numpy(),
    )


def make_full_loader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=t.mdta_collate_fn_p13d,
        pin_memory=True,
    )


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # similarity cache
    # --------------------------------------------------

    sim = np.load(
        SIM,
        allow_pickle=True,
    )

    drug_ids = [
        str(x)
        for x in sim["drug_ids"].tolist()
    ]

    protein_ids = [
        str(x)
        for x in sim["protein_ids"].tolist()
    ]

    drug_lookup = {
        x: i
        for i, x in enumerate(drug_ids)
    }

    protein_lookup = {
        x: i
        for i, x in enumerate(protein_ids)
    }

    drug_similarity = (
        sim["drug_similarity"]
        .astype(np.float64)
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "DEVICE:",
        device,
        flush=True,
    )

    fold_results = []
    test_frames = []
    seen_test_drugs = set()

    # ==================================================
    # 5 folds
    # ==================================================

    for fold in range(1, 6):

        print(
            f"\n===== FOLD {fold} =====",
            flush=True,
        )

        fold_out = OUT / f"fold_{fold}"

        result_file = (
            fold_out / "results.json"
        )

        pred_file = (
            fold_out
            / "test_predictions.csv"
        )

        # resume support
        if (
            result_file.exists()
            and pred_file.exists()
        ):
            print(
                f"FOLD {fold}: "
                f"reuse completed result",
                flush=True,
            )

            result = json.loads(
                result_file.read_text()
            )

            frame = pd.read_csv(
                pred_file,
                dtype={
                    "drug_id": str,
                    "protein_id": str,
                },
            )

            fold_results.append(result)
            test_frames.append(frame)

            seen_test_drugs.update(
                frame["drug_id"].unique()
            )

            continue

        fold_out.mkdir(
            parents=True,
            exist_ok=True,
        )

        split_path = (
            SPLITS
            / f"fold_{fold}"
            / "split.json"
        )

        checkpoint_path = (
            BASE
            / f"fold_{fold}"
            / "best_model.pt"
        )

        split = json.loads(
            split_path.read_text()
        )

        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        cfg = SimpleNamespace(
            **ckpt["args"]
        )

        # checkpoint must belong to this split
        assert (
            Path(cfg.split_json).resolve()
            ==
            split_path.resolve()
        ), (
            cfg.split_json,
            split_path,
        )

        t.set_seed(cfg.seed)

        (
            dataset,
            train_set,
            val_set,
            _,
            val_loader,
        ) = t.build_dataloaders(cfg)

        raw = pd.read_csv(
            cfg.pairs_csv,
            dtype={
                "drug_id": str,
                "protein_id": str,
            },
        )

        assert raw[
            ["drug_id", "protein_id"]
        ].equals(
            dataset.df[
                ["drug_id", "protein_id"]
            ]
        )

        assert np.allclose(
            raw["label"].to_numpy(),
            dataset.df["label"].to_numpy(),
        )

        train_names = list(
            map(
                str,
                split["train_drugs"],
            )
        )

        val_names = list(
            map(
                str,
                split["val_drugs"],
            )
        )

        test_names = list(
            map(
                str,
                split["test_drugs"],
            )
        )

        # ------------------------------------------------
        # split guardrails
        # ------------------------------------------------

        assert set(
            train_names
        ).isdisjoint(
            val_names
        )

        assert set(
            train_names
        ).isdisjoint(
            test_names
        )

        assert set(
            val_names
        ).isdisjoint(
            test_names
        )

        assert not (
            seen_test_drugs
            &
            set(test_names)
        )

        seen_test_drugs.update(
            test_names
        )

        # ------------------------------------------------
        # P13D
        # ------------------------------------------------

        model = t.build_model(
            cfg,
            device,
        )

        model.load_state_dict(
            ckpt["model_state_dict"],
            strict=True,
        )

        # reproduce validation
        val_pred, val_y = infer(
            model,
            val_loader,
            device,
        )

        val_metrics = (
            t.compute_regression_metrics(
                torch.tensor(val_pred),
                torch.tensor(val_y),
            )
        )

        val_diff = {
            k: abs(
                val_metrics[k]
                -
                ckpt["val_metrics"][k]
            )
            for k in val_metrics
        }

        assert (
            max(val_diff.values())
            < 1e-3
        ), val_diff

        # ------------------------------------------------
        # infer all 30056 Davis pairs
        # ------------------------------------------------

        full_pred, full_y = infer(
            model,
            make_full_loader(
                dataset,
                cfg.batch_size,
            ),
            device,
        )

        assert len(full_pred) == 30056
        assert len(full_y) == 30056

        assert np.allclose(
            full_y,
            raw["label"].to_numpy(),
            atol=1e-6,
        )

        # ------------------------------------------------
        # convert to 68 x 442 grid
        # ------------------------------------------------

        row_index = np.full(
            (
                len(drug_ids),
                len(protein_ids),
            ),
            -1,
            dtype=int,
        )

        for i, (
            drug,
            protein,
        ) in enumerate(
            zip(
                raw["drug_id"],
                raw["protein_id"],
            )
        ):
            row_index[
                drug_lookup[str(drug)],
                protein_lookup[str(protein)],
            ] = i

        assert (
            row_index >= 0
        ).all()

        labels = full_y[
            row_index
        ].astype(
            np.float64
        )

        baseline = full_pred[
            row_index
        ].astype(
            np.float64
        )

        residual = (
            labels
            -
            baseline
        )

        train_drugs = np.asarray(
            [
                drug_lookup[x]
                for x in train_names
            ],
            dtype=int,
        )

        val_drugs = np.asarray(
            [
                drug_lookup[x]
                for x in val_names
            ],
            dtype=int,
        )

        test_drugs = np.asarray(
            [
                drug_lookup[x]
                for x in test_names
            ],
            dtype=int,
        )

        # ------------------------------------------------
        # save full-grid prediction cache
        # ------------------------------------------------

        cache_path = (
            fold_out
            / "global_predictions.pt"
        )

        torch.save(
            {
                "drug_id":
                    raw["drug_id"]
                    .astype(str)
                    .tolist(),

                "protein_id":
                    raw["protein_id"]
                    .astype(str)
                    .tolist(),

                "label":
                    torch.tensor(
                        full_y,
                        dtype=torch.float32,
                    ),

                "prediction":
                    torch.tensor(
                        full_pred,
                        dtype=torch.float32,
                    ),

                "checkpoint":
                    str(checkpoint_path),

                "checkpoint_sha256":
                    sha(checkpoint_path),

                "split":
                    str(split_path),

                "split_sha256":
                    sha(split_path),
            },
            cache_path,
        )

        # ==================================================
        # R1 parameter selection
        #
        # IMPORTANT:
        # train_drugs ONLY
        # ==================================================

        inner_folds = (
            r1.build_inner_folds(
                train_drugs,
                seed=42,
                folds=5,
            )
        )

        (
            best,
            grid,
            inner_r0_mse,
        ) = r1.select_r1(
            train_drugs,
            inner_folds,
            residual,
            baseline,
            labels,
            drug_similarity,
        )

        grid.head(
            200
        ).to_csv(
            fold_out
            / "r1_inner_top200.csv",
            index=False,
        )

        best_config = (
            r1.clean_config(
                best
            )
        )

        # ------------------------------------------------
        # validation:
        # descriptive only
        # NOT used to select R1
        # ------------------------------------------------

        val_r0 = baseline[
            val_drugs
        ]

        (
            val_r1,
            _,
            _,
        ) = r1.outer_predictions(
            val_drugs,
            train_drugs,
            residual,
            baseline,
            drug_similarity,
            best,
        )

        val_result = {
            "P13D":
                metrics(
                    labels[
                        val_drugs
                    ].ravel(),
                    val_r0.ravel(),
                ),

            "R1":
                metrics(
                    labels[
                        val_drugs
                    ].ravel(),
                    val_r1.ravel(),
                ),
        }

        # ==================================================
        # OUTER TEST
        # ==================================================

        test_r0 = baseline[
            test_drugs
        ]

        (
            test_r1,
            correction,
            alpha,
        ) = r1.outer_predictions(
            test_drugs,
            train_drugs,
            residual,
            baseline,
            drug_similarity,
            best,
        )

        y_test = labels[
            test_drugs
        ]

        test_result = {
            "P13D":
                metrics(
                    y_test.ravel(),
                    test_r0.ravel(),
                ),

            "R1":
                metrics(
                    y_test.ravel(),
                    test_r1.ravel(),
                ),
        }

        # ------------------------------------------------
        # reproduce existing P13D test result
        # ------------------------------------------------

        baseline_ref = (
            BASE_TEST
            / f"fold_{fold}"
            / "test_metrics.json"
        )

        test_repro = None

        if baseline_ref.exists():

            ref = json.loads(
                baseline_ref.read_text()
            )["baseline"]["test_global_metrics"]

            test_repro = {
                k:
                    abs(
                        test_result["P13D"][k]
                        -
                        ref[k]
                    )
                for k in (
                    "mse",
                    "ci",
                    "rm2",
                )
            }

            assert (
                max(
                    test_repro.values()
                )
                < 1e-3
            ), test_repro

        # ------------------------------------------------
        # per-fold drug bootstrap
        # ------------------------------------------------

        bootstrap = (
            r1.bootstrap_by_drug(
                y_test,
                test_r1,
                test_r0,
                n=20000,
                seed=20260901,
            )
        )

        # ------------------------------------------------
        # predictions
        # ------------------------------------------------

        frame = pd.DataFrame(
            {
                "fold":
                    fold,

                "pair_index":
                    row_index[
                        test_drugs
                    ].ravel(),

                "drug_id":
                    np.repeat(
                        test_names,
                        len(protein_ids),
                    ),

                "protein_id":
                    np.tile(
                        protein_ids,
                        len(test_drugs),
                    ),

                "label":
                    y_test.ravel(),

                "P13D":
                    test_r0.ravel(),

                "R1":
                    test_r1.ravel(),

                "R1_correction":
                    correction.ravel(),

                "R1_alpha":
                    alpha.ravel(),
            }
        )

        frame.to_csv(
            pred_file,
            index=False,
        )

        test_frames.append(
            frame
        )

        # ------------------------------------------------
        # per-drug
        # ------------------------------------------------

        per_drug = []

        for j, drug in enumerate(
            test_names
        ):

            m0 = metrics(
                y_test[j],
                test_r0[j],
            )

            m1 = metrics(
                y_test[j],
                test_r1[j],
            )

            per_drug.append(
                {
                    "drug_id":
                        drug,

                    "n_pairs":
                        442,

                    "P13D_mse":
                        m0["mse"],

                    "R1_mse":
                        m1["mse"],

                    "delta_mse_R1_minus_P13D":
                        m1["mse"]
                        -
                        m0["mse"],

                    "P13D_ci":
                        m0["ci"],

                    "R1_ci":
                        m1["ci"],

                    "P13D_rm2":
                        m0["rm2"],

                    "R1_rm2":
                        m1["rm2"],
                }
            )

        pd.DataFrame(
            per_drug
        ).to_csv(
            fold_out
            / "per_drug.csv",
            index=False,
        )

        # ------------------------------------------------
        # fold result
        # ------------------------------------------------

        result = {
            "fold":
                fold,

            "checkpoint":
                str(
                    checkpoint_path
                ),

            "checkpoint_sha256":
                sha(
                    checkpoint_path
                ),

            "split":
                str(
                    split_path
                ),

            "split_sha256":
                sha(
                    split_path
                ),

            "train_drugs":
                len(
                    train_drugs
                ),

            "validation_drugs":
                len(
                    val_drugs
                ),

            "test_drugs":
                len(
                    test_drugs
                ),

            "inner_fold_sizes":
                [
                    len(x)
                    for x
                    in inner_folds
                ],

            "inner_P13D_mse":
                float(
                    inner_r0_mse
                ),

            "R1_best":
                best_config,

            "checkpoint_validation_reproduction_abs_diff":
                val_diff,

            "checkpoint_validation_reproduction_tolerance":
                1e-3,

            "P13D_outer_test_reproduction_abs_diff":
                test_repro,

            "P13D_outer_test_reproduction_tolerance":
                1e-3,

            "validation_descriptive":
                val_result,

            "test":
                test_result,

            "R1_test_bootstrap_vs_P13D":
                bootstrap,

            "R1_correction_abs_mean":
                float(
                    np.abs(
                        correction
                    ).mean()
                ),

            "R1_alpha_mean":
                float(
                    alpha.mean()
                ),

            "R1_selection_data":
                "train_drugs only",

            "R1_reference_data":
                "train_drugs only",

            "validation_used_for_R1_selection":
                False,

            "test_used_for_R1_selection":
                False,

            "P13D_refit_performed":
                False,
        }

        dump(
            result_file,
            result,
        )

        fold_results.append(
            result
        )

        print(
            json.dumps(
                {
                    "fold":
                        fold,

                    "R1_best":
                        best_config,

                    "P13D_test_mse":
                        test_result[
                            "P13D"
                        ]["mse"],

                    "R1_test_mse":
                        test_result[
                            "R1"
                        ]["mse"],

                    "delta":
                        test_result[
                            "R1"
                        ]["mse"]
                        -
                        test_result[
                            "P13D"
                        ]["mse"],

                    "improved_drugs":
                        bootstrap[
                            "drugs_improved"
                        ],

                    "worsened_drugs":
                        bootstrap[
                            "drugs_worsened"
                        ],
                },
                indent=2,
            ),
            flush=True,
        )

        del model
        del ckpt
        del dataset

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==================================================
    # 5-fold pooled
    # ==================================================

    assert (
        len(
            seen_test_drugs
        )
        ==
        68
    )

    all_test = pd.concat(
        test_frames,
        ignore_index=True,
    )

    assert (
        len(
            all_test
        )
        ==
        30056
    )

    assert (
        sorted(
            all_test[
                "pair_index"
            ].tolist()
        )
        ==
        list(
            range(
                30056
            )
        )
    )

    all_test.to_csv(
        OUT
        / "test_predictions_allfolds.csv",
        index=False,
    )

    summary = {}

    for name in (
        "P13D",
        "R1",
    ):

        summary[name] = {

            "macro": {

                metric_name: {

                    "mean":
                        float(
                            np.mean(
                                [
                                    x["test"]
                                     [name]
                                     [metric_name]
                                    for x
                                    in fold_results
                                ]
                            )
                        ),

                    "sample_std":
                        float(
                            np.std(
                                [
                                    x["test"]
                                     [name]
                                     [metric_name]
                                    for x
                                    in fold_results
                                ],
                                ddof=1,
                            )
                        ),
                }

                for metric_name
                in (
                    "mse",
                    "ci",
                    "rm2",
                )
            },

            "pooled":
                metrics(
                    all_test[
                        "label"
                    ].to_numpy(),

                    all_test[
                        name
                    ].to_numpy(),
                ),
        }

    # ==================================================
    # 68-drug cluster bootstrap
    # ==================================================

    per_drug = (
        all_test
        .assign(
            P13D_sq=(
                all_test["P13D"]
                -
                all_test["label"]
            ) ** 2,

            R1_sq=(
                all_test["R1"]
                -
                all_test["label"]
            ) ** 2,
        )
        .groupby(
            [
                "fold",
                "drug_id",
            ],
            as_index=False,
        )[
            [
                "P13D_sq",
                "R1_sq",
            ]
        ]
        .mean()
    )

    assert (
        len(
            per_drug
        )
        ==
        68
    )

    delta = (
        per_drug[
            "R1_sq"
        ]
        -
        per_drug[
            "P13D_sq"
        ]
    ).to_numpy()

    rng = np.random.default_rng(
        42
    )

    bootstrap_samples = (
        delta[
            rng.integers(
                0,
                len(delta),
                size=(
                    10000,
                    len(delta),
                ),
            )
        ]
        .mean(
            axis=1
        )
    )

    bootstrap = {

        "delta_definition":
            "MSE(R1)-MSE(P13D)",

        "mean_delta_mse":
            float(
                delta.mean()
            ),

        "95_ci":
            [
                float(
                    np.quantile(
                        bootstrap_samples,
                        0.025,
                    )
                ),

                float(
                    np.quantile(
                        bootstrap_samples,
                        0.975,
                    )
                ),
            ],

        "P_R1_better":
            float(
                np.mean(
                    bootstrap_samples
                    <
                    0
                )
            ),

        "improved_drugs":
            int(
                (
                    delta
                    <
                    0
                ).sum()
            ),

        "worsened_drugs":
            int(
                (
                    delta
                    >
                    0
                ).sum()
            ),

        "n_drugs":
            68,

        "cluster":
            "outer-test drug; all 442 protein pairs",
    }

    per_drug.to_csv(
        OUT
        / "per_drug_allfolds.csv",
        index=False,
    )

    dump(
        OUT
        / "SUMMARY.json",

        {
            "protocol":
                "original drug-cold 5-fold seed42; frozen P13D; R1 selected only on train_drugs; final disjoint outer-test evaluation",

            "split_disclosure":
                "historical original drug-cold fivefold split; no split reconstruction or modification was performed",

            "folds":
                fold_results,

            "summary":
                summary,

            "bootstrap_R1_vs_P13D":
                bootstrap,
        },
    )

    print(
        "\nFINAL="
        +
        json.dumps(
            {
                "summary":
                    summary,

                "bootstrap":
                    bootstrap,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
