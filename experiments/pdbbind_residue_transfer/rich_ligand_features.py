"""Consistent ligand graph featurization for PDBbind pretraining and Davis transfer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from torch_geometric.data import Data


ELEMENTS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
    "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag",
    "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni",
    "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "W", "Ru", "Nb",
    "Re", "Te", "Rh", "Tc", "Ba", "Bi", "Hf", "Mo", "U", "Sm", "Os",
    "Ir", "Ce", "Gd", "Ga", "Cs", "unknown",
]
FORMAL_CHARGES = (-2, -1, 0, 1, 2, "other")
HYBRIDIZATIONS = (
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    "other",
)
BASE_DIM = 82
FORMAL_CHARGE_SLICE = slice(82, 88)
HYBRIDIZATION_SLICE = slice(88, 94)
DONOR_INDEX = 94
ACCEPTOR_INDEX = 95
RING_INDEX = 96
ATOM_DIM = 97
EDGE_DIM = 6

_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
)


def one_hot_unknown(value, allowable):
    if value not in allowable:
        value = allowable[-1]
    return [value == candidate for candidate in allowable]


def load_heavy_molecule(path: Path):
    molecule = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
    if molecule is None:
        raise ValueError("RDKit failed to parse SDF")
    return Chem.RemoveHs(molecule)


def donor_acceptor_atoms(molecule):
    donors, acceptors = set(), set()
    for feature in _FEATURE_FACTORY.GetFeaturesForMol(molecule):
        if feature.GetFamily() == "Donor":
            donors.update(feature.GetAtomIds())
        elif feature.GetFamily() == "Acceptor":
            acceptors.update(feature.GetAtomIds())
    return donors, acceptors


def atom_features(molecule):
    donors, acceptors = donor_acceptor_atoms(molecule)
    rows = []
    for atom in molecule.GetAtoms():
        formal_charge = atom.GetFormalCharge()
        hybridization = atom.GetHybridization()
        rows.append(
            one_hot_unknown(atom.GetSymbol(), ELEMENTS)
            + one_hot_unknown(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
            + one_hot_unknown(atom.GetExplicitValence(), [1, 2, 3, 4, 5, 6])
            + one_hot_unknown(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
            + [atom.GetIsAromatic()]
            + one_hot_unknown(formal_charge, FORMAL_CHARGES)
            + one_hot_unknown(hybridization, HYBRIDIZATIONS)
            + [atom.GetIdx() in donors, atom.GetIdx() in acceptors, atom.IsInRing()]
        )
    result = torch.tensor(np.asarray(rows, dtype=np.float32))
    if result.shape[1] != ATOM_DIM:
        raise AssertionError(f"unexpected atom dimension {result.shape}")
    return result


def bond_features(bond):
    bond_type = bond.GetBondType()
    return [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]


def molecule_to_graph(molecule):
    edges, attributes = [], []
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = bond_features(bond)
        edges.extend(((begin, end), (end, begin)))
        attributes.extend((features, features))
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attributes, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_DIM), dtype=torch.float32)
    return Data(x=atom_features(molecule), edge_index=edge_index, edge_attr=edge_attr)


def sdf_to_graph(path: Path):
    return molecule_to_graph(load_heavy_molecule(path))
