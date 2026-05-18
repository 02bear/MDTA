# -*- coding: utf-8 -*-
import os
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def unit_vector(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return v / n


def dihedral(p0, p1, p2, p3):
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    p3 = np.asarray(p3, dtype=np.float32)

    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_u = unit_vector(b1)
    if np.linalg.norm(b1_u) < 1e-8:
        return None

    v = b0 - np.dot(b0, b1_u) * b1_u
    w = b2 - np.dot(b2, b1_u) * b1_u

    nv = np.linalg.norm(v)
    nw = np.linalg.norm(w)
    if nv < 1e-8 or nw < 1e-8:
        return None

    x = np.dot(v, w)
    y = np.dot(np.cross(b1_u, v), w)
    return float(np.arctan2(y, x))


def parse_pdb_backbone(pdb_path):
    residues = {}
    order = []

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            if atom_name not in {"N", "CA", "C", "CB"}:
                continue

            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue

            resname = line[17:20].strip()
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            key = (chain, resseq, icode)
            if key not in residues:
                residues[key] = {
                    "resname": resname,
                    "chain": chain,
                    "resseq": resseq,
                    "icode": icode,
                    "atoms": {},
                }
                order.append(key)

            if atom_name not in residues[key]["atoms"]:
                residues[key]["atoms"][atom_name] = np.array([x, y, z], dtype=np.float32)

    residue_list = [residues[k] for k in order]
    return residue_list


def build_node_features(residue_list):
    kept = []
    for r in residue_list:
        atoms = r["atoms"]
        if "N" in atoms and "CA" in atoms and "C" in atoms:
            kept.append(r)

    L = len(kept)
    if L == 0:
        return None, None, None, None

    node_s = np.zeros((L, 6), dtype=np.float32)
    node_v = np.zeros((L, 3, 3), dtype=np.float32)
    coords = np.zeros((L, 3), dtype=np.float32)
    residue_meta = []

    for i, r in enumerate(kept):
        atoms = r["atoms"]
        N = atoms["N"]
        CA = atoms["CA"]
        C = atoms["C"]
        CB = atoms.get("CB", None)
        coords[i] = CA

        if i < L - 1:
            CA_next = kept[i + 1]["atoms"]["CA"]
            forward = unit_vector(CA_next - CA)
        else:
            forward = np.zeros(3, dtype=np.float32)

        if i > 0:
            CA_prev = kept[i - 1]["atoms"]["CA"]
            backward = unit_vector(CA - CA_prev)
        else:
            backward = np.zeros(3, dtype=np.float32)

        if CB is not None:
            sidechain = unit_vector(CB - CA)
        else:
            sidechain = np.zeros(3, dtype=np.float32)

        node_v[i, 0] = forward
        node_v[i, 1] = backward
        node_v[i, 2] = sidechain

        phi = None
        psi = None
        omega = None

        if i > 0:
            prev_atoms = kept[i - 1]["atoms"]
            C_prev = prev_atoms["C"]
            CA_prev = prev_atoms["CA"]
            phi = dihedral(C_prev, N, CA, C)
            omega = dihedral(CA_prev, C_prev, N, CA)

        if i < L - 1:
            next_atoms = kept[i + 1]["atoms"]
            N_next = next_atoms["N"]
            psi = dihedral(N, CA, C, N_next)

        angles = [phi, psi, omega]
        feats = []
        for a in angles:
            if a is None:
                feats.extend([0.0, 0.0])
            else:
                feats.extend([math.sin(a), math.cos(a)])
        node_s[i] = np.array(feats, dtype=np.float32)

        residue_meta.append({
            "resname": r["resname"],
            "chain": r["chain"],
            "resseq": r["resseq"],
            "icode": r["icode"],
        })

    return node_s, node_v, coords, residue_meta


def rbf_features(distances, num_basis=32, d_min=0.0, d_max=20.0):
    centers = np.linspace(d_min, d_max, num_basis, dtype=np.float32)
    width = (d_max - d_min) / max(1, (num_basis - 1))
    gamma = 1.0 / (width ** 2 + 1e-8)
    diff = distances[:, None] - centers[None, :]
    return np.exp(-gamma * (diff ** 2)).astype(np.float32)


def build_knn_graph(coords, knn_k=30, rbf_dim=32):
    coords = np.asarray(coords, dtype=np.float32)
    L = coords.shape[0]
    if L == 0:
        raise ValueError("Empty coords")

    dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    edge_src = []
    edge_dst = []

    for i in range(L):
        k_eff = min(knn_k, max(0, L - 1))
        if k_eff == 0:
            continue
        nn_idx = np.argsort(dmat[i])[1 : 1 + k_eff]
        for j in nn_idx:
            edge_src.append(i)
            edge_dst.append(int(j))

    if len(edge_src) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_s = np.zeros((0, rbf_dim), dtype=np.float32)
        edge_v = np.zeros((0, 1, 3), dtype=np.float32)
        return edge_index, edge_s, edge_v

    edge_src = np.asarray(edge_src, dtype=np.int64)
    edge_dst = np.asarray(edge_dst, dtype=np.int64)

    vec = coords[edge_dst] - coords[edge_src]
    dist = np.linalg.norm(vec, axis=-1)
    unit = vec / np.clip(dist[:, None], 1e-8, None)

    edge_index = np.stack([edge_src, edge_dst], axis=0).astype(np.int64)
    edge_s = rbf_features(dist, num_basis=rbf_dim)
    edge_v = unit[:, None, :].astype(np.float32)
    return edge_index, edge_s, edge_v


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", type=str, default="data/raw/davis/afdb_download_manifest.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed/davis/protein_3d_gvp")
    parser.add_argument("--knn_k", type=int, default=30)
    parser.add_argument("--rbf_dim", type=int, default=32)
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_csv)
    df = df[df["download_status"] == "ok"].copy()

    success = 0
    failures = []
    example_shape = None

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Building protein 3D features"):
        protein_id = row.protein_id
        pdb_path = row.pdb_path

        try:
            residue_list = parse_pdb_backbone(pdb_path)
            node_s, node_v, coords, residue_meta = build_node_features(residue_list)

            if node_s is None or len(node_s) == 0:
                raise ValueError("No valid residues with N/CA/C found")

            edge_index, edge_s, edge_v = build_knn_graph(coords, knn_k=args.knn_k, rbf_dim=args.rbf_dim)

            obj = {
                "protein_id": protein_id,
                "node_s": torch.tensor(node_s, dtype=torch.float32),
                "node_v": torch.tensor(node_v, dtype=torch.float32),
                "coords": torch.tensor(coords, dtype=torch.float32),
                "edge_index": torch.tensor(edge_index, dtype=torch.long),
                "edge_s": torch.tensor(edge_s, dtype=torch.float32),
                "edge_v": torch.tensor(edge_v, dtype=torch.float32),
                "length": int(node_s.shape[0]),
                "residue_meta": residue_meta,
                "source_pdb": pdb_path,
            }

            save_path = output_dir / f"{protein_id}.pt"
            torch.save(obj, save_path)

            success += 1
            if example_shape is None:
                example_shape = {
                    "node_s": list(node_s.shape),
                    "node_v": list(node_v.shape),
                    "coords": list(coords.shape),
                    "edge_index": list(edge_index.shape),
                    "edge_s": list(edge_s.shape),
                    "edge_v": list(edge_v.shape),
                }

        except Exception as e:
            failures.append({"protein_id": protein_id, "pdb_path": pdb_path, "error": str(e)})

    pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)

    meta = {
        "num_input_proteins": int(len(df)),
        "num_success": int(success),
        "num_failed": int(len(failures)),
        "node_s_dim": 6,
        "node_v_dim": 3,
        "edge_s_dim": int(args.rbf_dim),
        "edge_v_dim": 1,
        "knn_k": int(args.knn_k),
        "example_shape": example_shape,
        "feature_type": "protein_3d_gvp_graph",
    }

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Saved to: {output_dir}")
    print(f"Failures: {output_dir / 'failures.csv'}")


if __name__ == "__main__":
    main()
