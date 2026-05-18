from pathlib import Path
import json
import pickle
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data


def safe_filename(name: str):
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for ch in bad:
        name = name.replace(ch, "_")
    return name


def target2graph_kanpm_style(distance_map, protein_features_esm, threshold=0.5):
    """
    按 KANPM-DTA 的 target2graph 思路构图，但适配你当前的数据格式：
    - protein_features_esm 已经是 [L, D]，不再做 [1:-1]
    - distance_map 是 [L, L]
    - 连边规则：distance_map >= threshold
    - 强制加入：
        1) self-loop
        2) i -> i+1 顺序边
    - 输出：
        target_size, target_feature, target_edge_index, target_edge_weight
    """
    # 兼容 numpy / torch
    if isinstance(protein_features_esm, torch.Tensor):
        protein_features_esm = protein_features_esm.detach().cpu().numpy()
    if isinstance(distance_map, torch.Tensor):
        distance_map = distance_map.detach().cpu().numpy()

    protein_features_esm = np.asarray(protein_features_esm, dtype=np.float32)
    distance_map = np.asarray(distance_map, dtype=np.float32).copy()

    target_size = protein_features_esm.shape[0]

    # 长度对齐：如果不一致，取两者最小长度
    L = min(target_size, distance_map.shape[0], distance_map.shape[1])
    protein_features_esm = protein_features_esm[:L, :]
    distance_map = distance_map[:L, :L]
    target_size = L

    # 按 KANPM-DTA 补自环和顺序边
    for i in range(target_size):
        distance_map[i, i] = 1.0
        if i + 1 < target_size:
            distance_map[i, i + 1] = 1.0

    index_row, index_col = np.where(distance_map >= threshold)

    target_edge_index = []
    target_edge_weight = []

    for i, j in zip(index_row.tolist(), index_col.tolist()):
        target_edge_index.append([i, j])
        target_edge_weight.append(float(distance_map[i, j]))

    target_feature = torch.tensor(protein_features_esm, dtype=torch.float32)

    if len(target_edge_index) == 0:
        target_edge_index = torch.empty((2, 0), dtype=torch.long)
        target_edge_weight = torch.empty((0,), dtype=torch.float32)
    else:
        target_edge_index = torch.tensor(target_edge_index, dtype=torch.long).t().contiguous()
        target_edge_weight = torch.tensor(target_edge_weight, dtype=torch.float32)

    return target_size, target_feature, target_edge_index, target_edge_weight


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--proteins_csv", type=str, required=True)
    parser.add_argument("--contact_pkl", type=str, required=True)
    parser.add_argument("--esm_feature_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--id_col", type=str, default="protein_id")
    parser.add_argument("--seq_col", type=str, default="sequence")
    args = parser.parse_args()

    proteins_csv = Path(args.proteins_csv)
    contact_pkl = Path(args.contact_pkl)
    esm_feature_dir = Path(args.esm_feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(proteins_csv)
    df = df[[args.id_col, args.seq_col]].drop_duplicates().reset_index(drop=True)

    with open(contact_pkl, "rb") as f:
        contact_obj = pickle.load(f)

    contact_map_dict = contact_obj["contact_map"]

    failures = []
    success_count = 0
    first_x_dim = None

    for idx, row in df.iterrows():
        protein_id = str(row[args.id_col])
        seq = str(row[args.seq_col])

        feat_path = esm_feature_dir / f"{safe_filename(protein_id)}.pt"
        if not feat_path.exists():
            failures.append({
                "protein_id": protein_id,
                "error": f"missing feature file: {feat_path.name}"
            })
            continue

        if protein_id not in contact_map_dict:
            failures.append({
                "protein_id": protein_id,
                "error": "protein_id not found in contact_map pkl"
            })
            continue

        try:
            feat_obj = torch.load(feat_path)
            protein_features_esm = feat_obj["per_tok"]   # [L, 1280]
            distance_map = contact_map_dict[protein_id]  # [L, L]

            target_size, target_feature, target_edge_index, target_edge_weight = target2graph_kanpm_style(
                distance_map=distance_map,
                protein_features_esm=protein_features_esm,
                threshold=args.threshold
            )

            protein_graph = Data(
                x=target_feature,                # [L, 1280]
                edge_index=target_edge_index,    # [2, E]
                edge_weight=target_edge_weight,  # [E]
                num_nodes=target_size
            )

            # 顺手把一些元信息也存进去
            protein_graph.protein_id = protein_id
            protein_graph.orig_seq_len = len(seq)
            protein_graph.used_len = target_size

            save_path = output_dir / f"{safe_filename(protein_id)}.pt"
            torch.save(protein_graph, save_path)

            success_count += 1

            if first_x_dim is None:
                first_x_dim = int(target_feature.shape[1])

        except Exception as e:
            failures.append({
                "protein_id": protein_id,
                "error": str(e)
            })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
            print(f"[{idx + 1}/{len(df)}] processed")

    pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)

    meta = {
        "num_input_proteins": int(len(df)),
        "num_success": int(success_count),
        "num_failed": int(len(failures)),
        "threshold": float(args.threshold),
        "node_feature_dim": first_x_dim,
        "edge_weight_used": True,
        "graph_style": "KANPM-DTA-like"
    }

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Saved graphs to: {output_dir}")


if __name__ == "__main__":
    main()