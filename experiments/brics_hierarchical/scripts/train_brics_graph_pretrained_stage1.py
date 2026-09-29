#!/usr/bin/env python3
"""Affinity stage for the pretrained clean BRICS chemistry graph."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_brics_hierarchical_stage1 as base
from train_brics_chem_stage1 import ChemStore
from train_brics_graph_stage1 import BRICSChemGraphP13D


def pop_argument(name):
    if name not in sys.argv:
        raise SystemExit(f"required argument missing: {name}")
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise SystemExit(f"missing value for {name}")
    value = Path(sys.argv[index + 1])
    del sys.argv[index:index + 2]
    return value


PRETRAINED = pop_argument("--pretrained-hierarchy")


class PretrainedBRICSGraph(BRICSChemGraphP13D):
    def __init__(self, project, checkpoint, hidden=128, dropout=0.1, mask_rate=0.15):
        super().__init__(project, checkpoint, hidden, dropout, mask_rate)
        payload = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
        incompatible = self.load_state_dict(payload["hierarchy_state"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected pretrained keys: {incompatible.unexpected_keys}")
        print(
            f"loaded clean BRICS graph pretraining: {PRETRAINED}; "
            f"condition={payload['result']['condition']} seed={payload['result']['seed']}",
            flush=True,
        )


if __name__ == "__main__":
    base.Store = ChemStore
    base.BRICSHierarchicalP13D = PretrainedBRICSGraph
    base.main()
