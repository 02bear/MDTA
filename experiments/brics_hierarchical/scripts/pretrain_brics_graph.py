#!/usr/bin/env python3
"""Training-drug-only masked chemistry pretraining for the clean BRICS graph."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pretrain_brics_chem as pretrain
from train_brics_graph_stage1 import BRICSChemGraphP13D


if __name__ == "__main__":
    pretrain.BRICSChemHierarchicalP13D = BRICSChemGraphP13D
    pretrain.main()
