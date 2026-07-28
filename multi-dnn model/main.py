"""PyTorch single- and multi-DNN training for the PIMA-COPF configuration.

Partitioned training is the default and reads the editable bus-cluster JSON.
The original single-DNN case118 setup follows Table III of the paper:
    [237, 500, 500, 236, 200, 344]
where 237 = Pd(118) + Qd(118) + scalar contingency,
236 = Vm(118) + Va(118), and 344 = Vm/Va(236) + Pg/Qg(108).

The MATLAB generators save one-hot contingencies. To reproduce
Table III exactly, this script converts the outage id to one scalar feature by
default. Use --contingency onehot if you want to train directly on X_xi.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from COPF.core import CASES, LossConfig
from COPF.multi_training import train_partitioned
from COPF.training import train


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=sorted(CASES), default="case118")
    p.add_argument("--data", type=str, default=None, help="Path to generated .mat file")
    p.add_argument(
        "--training-mode",
        choices=("partitioned", "single"),
        default="partitioned",
        help="Train one DNN per bus cluster or the original global DNN",
    )
    p.add_argument(
        "--cluster-file",
        type=str,
        default=None,
        help="JSON bus partition; defaults to COPF/<case>_clusters.json",
    )
    p.add_argument("--contingency", choices=("scalar", "onehot"), default="scalar")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument(
        "--loss-candidate",
        type=LossConfig.parse,
        action="append",
        default=None,
        metavar="CY,CV,CP,CQ,RHO_V,RHO_PG,RHO_QG",
        help="Repeat to tune multiple loss configurations on validation data",
    )
    p.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Fraction of train_idx reserved for validation",
    )
    p.add_argument(
        "--feasibility-target",
        type=float,
        default=99.9,
        help="Minimum eta_V/eta_Pg/eta_Qg percentage preferred in validation",
    )
    p.add_argument(
        "--feasibility-tolerance",
        type=float,
        default=1e-4,
        help="Per-unit tolerance used to classify a constraint as feasible",
    )
    p.add_argument(
        "--selection-warmup",
        type=int,
        default=50,
        help="Initial epochs excluded from validation model selection",
    )
    p.add_argument(
        "--selection-quality-tolerance",
        type=float,
        default=0.05,
        help="Relative quality tolerance before feasibility breaks ties",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="runs")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.training_mode == "partitioned":
        train_partitioned(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
