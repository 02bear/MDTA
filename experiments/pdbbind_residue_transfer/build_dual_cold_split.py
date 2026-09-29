#!/usr/bin/env python3
"""Build an auditable protein-homology + ligand-scaffold disjoint split.

Run ``prepare`` before an MMseqs2 all-vs-all search, then run ``build`` on the
resulting TSV.  Samples connected through either a qualifying protein hit or an
identical Bemis-Murcko scaffold are kept in the same split.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def parse_sequences(path: Path):
    lines = [line.strip() for line in path.open() if line.strip()]
    return {lines[i][1:].lower(): lines[i + 1] for i in range(0, len(lines), 3)}


def scaffold_key(sdf: Path):
    molecule = Chem.MolFromMolFile(str(sdf), removeHs=True, sanitize=True)
    if molecule is None:
        raise ValueError("RDKit failed to parse ligand")
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    smiles = Chem.MolToSmiles(scaffold, canonical=True)
    return smiles or f"ACYCLIC:{Chem.MolToSmiles(molecule, canonical=True)}"


def prepare(args):
    pairs = json.loads(args.pairs.read_text())
    audit = json.loads(args.alignment_audit.read_text())
    aligned = {
        row["pdb_id"]
        for row in audit["rows"]
        if row["atom_count_plip"] == row["atom_count_graph"]
        and row["residue_count_plip"] == row["residue_count_graph"]
    }
    records = {
        record["pdb_id"]: record
        for record in pairs["records"]
        if record["pdb_id"] in aligned and record["pairs"]
    }
    sequences = parse_sequences(args.site_labels)
    samples, failures = {}, []
    for pdb_id in sorted(records):
        try:
            sequence = sequences[pdb_id]
            scaffold = scaffold_key(
                args.refined_root / pdb_id / f"{pdb_id}_ligand.sdf"
            )
            samples[pdb_id] = {
                "sequence": sequence,
                "sequence_length": len(sequence),
                "scaffold": scaffold,
            }
        except Exception as error:
            failures.append({"pdb_id": pdb_id, "error": repr(error)})

    args.fasta.parent.mkdir(parents=True, exist_ok=True)
    args.fasta.write_text(
        "".join(f">{pdb_id}\n{row['sequence']}\n" for pdb_id, row in samples.items()),
        encoding="utf-8",
    )
    payload = {
        "sample_count": len(samples),
        "failures": failures,
        "samples": samples,
    }
    args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "sample_count": len(samples),
        "failure_count": len(failures),
        "unique_sequences": len({row["sequence"] for row in samples.values()}),
        "unique_scaffolds": len({row["scaffold"] for row in samples.values()}),
        "fasta": str(args.fasta),
        "manifest": str(args.manifest),
    }, indent=2))


def balanced_group_split(groups, seed):
    rng = random.Random(seed)
    groups = [sorted(group) for group in groups]
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    total = sum(map(len, groups))
    targets = {"train": 0.8 * total, "val": 0.1 * total, "test": 0.1 * total}
    splits = {name: [] for name in targets}
    group_assignments = {}
    for group_index, group in enumerate(groups):
        destination = min(
            splits,
            key=lambda name: len(splits[name]) / max(targets[name], 1),
        )
        splits[destination].extend(group)
        group_assignments[group_index] = destination
    return {name: sorted(ids) for name, ids in splits.items()}, group_assignments


def overlap_report(splits, keys):
    key_sets = {name: {keys[pdb_id] for pdb_id in ids} for name, ids in splits.items()}
    return {
        "train_val": len(key_sets["train"] & key_sets["val"]),
        "train_test": len(key_sets["train"] & key_sets["test"]),
        "val_test": len(key_sets["val"] & key_sets["test"]),
    }


def build(args):
    manifest = json.loads(args.manifest.read_text())
    samples = manifest["samples"]
    ids = sorted(samples)
    protein_uf = UnionFind(ids)
    qualifying_edges, ignored_edges = [], 0
    with args.hits.open() as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                raise ValueError(f"Malformed MMseqs row {line_number}: {line!r}")
            query, target = fields[:2]
            if query not in samples or target not in samples:
                ignored_edges += 1
                continue
            if query != target:
                protein_uf.union(query, target)
                qualifying_edges.append((query, target))

    protein_keys = {pdb_id: protein_uf.find(pdb_id) for pdb_id in ids}
    scaffold_keys = {pdb_id: samples[pdb_id]["scaffold"] for pdb_id in ids}

    # A second union-find forms connected components in the bipartite relation
    # induced by protein-homology groups and ligand scaffolds.
    dual_uf = UnionFind(ids)
    first_by_protein, first_by_scaffold = {}, {}
    for pdb_id in ids:
        protein_key, scaffold = protein_keys[pdb_id], scaffold_keys[pdb_id]
        if protein_key in first_by_protein:
            dual_uf.union(pdb_id, first_by_protein[protein_key])
        else:
            first_by_protein[protein_key] = pdb_id
        if scaffold in first_by_scaffold:
            dual_uf.union(pdb_id, first_by_scaffold[scaffold])
        else:
            first_by_scaffold[scaffold] = pdb_id

    dual_groups = defaultdict(list)
    for pdb_id in ids:
        dual_groups[dual_uf.find(pdb_id)].append(pdb_id)
    groups = list(dual_groups.values())
    splits, _ = balanced_group_split(groups, args.seed)
    split_of = {pdb_id: name for name, values in splits.items() for pdb_id in values}
    cross_hit_edges = sum(
        split_of[left] != split_of[right] for left, right in qualifying_edges
    )
    group_sizes = sorted((len(group) for group in groups), reverse=True)
    protein_group_sizes = Counter(protein_keys.values())
    report = {
        "parameters": {
            "seed": args.seed,
            "min_sequence_identity": args.min_sequence_identity,
            "minimum_bidirectional_coverage": args.minimum_coverage,
            "mmseqs_version": args.mmseqs_version,
            "mmseqs_command": args.mmseqs_command,
        },
        "sample_count": len(ids),
        "split_ids": splits,
        "split_counts": {name: len(values) for name, values in splits.items()},
        "protein_homology": {
            "component_count": len(protein_group_sizes),
            "largest_component_size": max(protein_group_sizes.values()),
            "qualifying_directed_nonself_hits": len(qualifying_edges),
            "ignored_hit_rows": ignored_edges,
            "cross_split_directed_hits": cross_hit_edges,
            "component_overlap": overlap_report(splits, protein_keys),
        },
        "ligand_scaffold": {
            "group_count": len(set(scaffold_keys.values())),
            "group_overlap": overlap_report(splits, scaffold_keys),
        },
        "dual_components": {
            "component_count": len(groups),
            "largest_component_size": group_sizes[0],
            "top_20_component_sizes": group_sizes[:20],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "split_ids"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--pairs", type=Path, required=True)
    prepare_parser.add_argument("--alignment-audit", type=Path, required=True)
    prepare_parser.add_argument("--site-labels", type=Path, required=True)
    prepare_parser.add_argument("--refined-root", type=Path, required=True)
    prepare_parser.add_argument("--fasta", type=Path, required=True)
    prepare_parser.add_argument("--manifest", type=Path, required=True)
    prepare_parser.set_defaults(func=prepare)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--manifest", type=Path, required=True)
    build_parser.add_argument("--hits", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--seed", type=int, default=42)
    build_parser.add_argument("--min-sequence-identity", type=float, default=0.30)
    build_parser.add_argument("--minimum-coverage", type=float, default=0.80)
    build_parser.add_argument("--mmseqs-version", required=True)
    build_parser.add_argument("--mmseqs-command", required=True)
    build_parser.set_defaults(func=build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
