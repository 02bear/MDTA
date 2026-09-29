"""Coarse, auditable chemistry masks for typed PLIP interaction heads.

The cached DTBind atom feature layout is reproduced from
DTBind/data_process/graph_construction/{drug_gra,construct_graph_hetero}.py.
Masks are candidate filters, not interaction labels.  Training code must always
union a mask with observed positives so a coarse rule can never erase a label.
"""

from __future__ import annotations

import torch

from rich_ligand_features import (
    ACCEPTOR_INDEX,
    DONOR_INDEX,
    FORMAL_CHARGE_SLICE,
    RING_INDEX,
)


ELEMENTS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
    "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag",
    "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni",
    "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "W", "Ru", "Nb",
    "Re", "Te", "Rh", "Tc", "Ba", "Bi", "Hf", "Mo", "U", "Sm", "Os",
    "Ir", "Ce", "Gd", "Ga", "Cs", "unknown",
]
ELEMENT_INDEX = {element: index for index, element in enumerate(ELEMENTS)}
AROMATIC_INDEX = 81

INTERACTION_TYPES = (
    "hbond",
    "hydrophobic",
    "waterbridge",
    "saltbridge",
    "pistacking",
    "pication",
    "halogenbond",
)

ATOM_ELEMENTS = {
    "hbond": {"N", "O", "S", "P"},
    "hydrophobic": {"C", "S", "F", "Cl", "Br", "I"},
    "waterbridge": {"N", "O", "S", "P"},
    "saltbridge": {"N", "O", "S", "P"},
    "pistacking": set(),  # aromatic flag below
    "pication": {"N", "P", "S"},  # plus aromatic atoms below
    "halogenbond": {"F", "Cl", "Br", "I"},
}

# H-bond/water/halogen interactions can involve backbone atoms from nearly any
# residue.  Restrict only interactions whose protein group identity is reliable.
RESIDUE_CANDIDATES = {
    "hbond": None,
    # PLIP also finds hydrophobic contacts on the carbon portions of nominally
    # polar/charged side chains.  Atom filtering is safer than a residue hard mask.
    "hydrophobic": None,
    "waterbridge": None,
    "saltbridge": set("DEKRH"),
    "pistacking": set("FYW H".replace(" ", "")),
    "pication": set("FYWKRH"),
    "halogenbond": None,
}


def parse_site_sequences(path):
    lines = [line.strip() for line in path.open() if line.strip()]
    return {lines[index][1:].lower(): lines[index + 1] for index in range(0, len(lines), 3)}


def atom_candidate_mask(atom_x: torch.Tensor, interaction_type: str) -> torch.Tensor:
    mask = torch.zeros(atom_x.shape[0], dtype=torch.bool, device=atom_x.device)
    for element in ATOM_ELEMENTS[interaction_type]:
        mask |= atom_x[:, ELEMENT_INDEX[element]] > 0.5
    if interaction_type in {"pistacking", "pication"}:
        mask |= atom_x[:, AROMATIC_INDEX] > 0.5
    if atom_x.shape[1] > RING_INDEX:
        if interaction_type in {"hbond", "waterbridge"}:
            # Keep the element fallback: PLIP can assign a group atom that the
            # RDKit feature factory does not classify as the direct donor/acceptor.
            mask |= (atom_x[:, DONOR_INDEX] > 0.5) | (atom_x[:, ACCEPTOR_INDEX] > 0.5)
        elif interaction_type == "pistacking":
            mask = (atom_x[:, AROMATIC_INDEX] > 0.5) | (atom_x[:, RING_INDEX] > 0.5)
        elif interaction_type == "pication":
            positive_charge = atom_x[:, FORMAL_CHARGE_SLICE.start + 3] > 0.5
            strong_positive_charge = atom_x[:, FORMAL_CHARGE_SLICE.start + 4] > 0.5
            mask |= (
                (atom_x[:, AROMATIC_INDEX] > 0.5)
                | (atom_x[:, RING_INDEX] > 0.5)
                | positive_charge
                | strong_positive_charge
            )
    return mask


def residue_candidate_mask(sequence: str, interaction_type: str, device=None) -> torch.Tensor:
    allowed = RESIDUE_CANDIDATES[interaction_type]
    if allowed is None:
        return torch.ones(len(sequence), dtype=torch.bool, device=device)
    return torch.tensor([residue in allowed for residue in sequence], dtype=torch.bool, device=device)


def pair_candidate_mask(atom_x: torch.Tensor, sequence: str, interaction_type: str) -> torch.Tensor:
    residues = residue_candidate_mask(sequence, interaction_type, atom_x.device)
    atoms = atom_candidate_mask(atom_x, interaction_type)
    return residues[:, None] & atoms[None, :]
