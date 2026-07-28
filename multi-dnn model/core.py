"""Model, configuration, normalization, and loss definitions for PIMA."""

from __future__ import annotations

import argparse
import dataclasses
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class CaseConfig:
    name: str
    nbus: int
    nbranch: int
    ngen: int
    table_arch: Tuple[int, ...]
    lr: float
    batch_size: int
    train_size: int

    @property
    def state_dim(self) -> int:
        return 2 * self.nbus

    @property
    def gen_dim(self) -> int:
        return 2 * self.ngen

    @property
    def scalar_input_dim(self) -> int:
        return 2 * self.nbus + 1

    @property
    def onehot_input_dim(self) -> int:
        return 2 * self.nbus + self.nbranch

    @property
    def output_dim(self) -> int:
        return self.state_dim + self.gen_dim


@dataclasses.dataclass(frozen=True)
class LossConfig:
    cy: float = 1.0
    cv: float = 1.0
    cp: float = 1.0
    cq: float = 1.0
    rho_v: float = 1e-3
    rho_pg: float = 1e-3
    rho_qg: float = 1e-3

    @classmethod
    def parse(cls, value: str) -> "LossConfig":
        try:
            numbers = [float(item) for item in value.split(",")]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "loss candidate must contain numeric coefficients"
            ) from exc
        if len(numbers) == 5:
            cy, cv, cp, cq, rho = numbers
            return cls(cy, cv, cp, cq, rho, rho, rho)
        if len(numbers) != 7:
            raise argparse.ArgumentTypeError(
                "loss candidate must be cy,cv,cp,cq,rho_v,rho_pg,rho_qg"
            )
        return cls(*numbers)


CASES: Dict[str, CaseConfig] = {
    "case118": CaseConfig(
        name="case118",
        nbus=118,
        nbranch=186,
        ngen=54,
        table_arch=(237, 500, 500, 236, 200, 344),
        lr=5e-4,
        batch_size=200,
        train_size=20_000,
    ),
    "case300": CaseConfig(
        name="case300",
        nbus=300,
        nbranch=411,
        ngen=69,
        table_arch=(601, 500, 500, 600, 400, 738),
        lr=1e-4,
        batch_size=200,
        train_size=80_000,
    ),
}


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.std[self.std < 1e-12] = 1.0

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        return cls(x.mean(axis=0), x.std(axis=0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}


@dataclasses.dataclass(frozen=True)
class PhysicsData:
    """Static MATPOWER quantities required by the physics-informed loss."""

    base_mva: float
    fbus: np.ndarray
    tbus: np.ndarray
    yff: np.ndarray
    yft: np.ndarray
    vmin: np.ndarray
    vmax: np.ndarray
    pmin: np.ndarray
    pmax: np.ndarray
    qmin: np.ndarray
    qmax: np.ndarray
    gencost: Optional[np.ndarray]


class PIMALoss:
    """Equation (21): label, virtual-bus, flow, and constraint losses."""

    def __init__(
        self,
        case: CaseConfig,
        y_scaler: Standardizer,
        physics: PhysicsData,
        device: torch.device,
        cy: float,
        cv: float,
        cp: float,
        cq: float,
        rho_v: float,
        rho_pg: float,
        rho_qg: float,
        virtual_bus_pairs: Sequence[Tuple[int, int]] = (),
    ) -> None:
        self.case = case
        self.cy, self.cv, self.cp, self.cq = cy, cv, cp, cq
        self.rhos = (rho_v, rho_v, rho_pg, rho_pg, rho_qg, rho_qg)
        self.base_mva = physics.base_mva
        self.y_mean = torch.as_tensor(y_scaler.mean, device=device)
        self.y_std = torch.as_tensor(y_scaler.std, device=device)
        self.fbus = torch.as_tensor(physics.fbus, dtype=torch.long, device=device)
        self.tbus = torch.as_tensor(physics.tbus, dtype=torch.long, device=device)
        self.yff = torch.as_tensor(physics.yff, dtype=torch.complex64, device=device)
        self.yft = torch.as_tensor(physics.yft, dtype=torch.complex64, device=device)
        self.vmin = torch.as_tensor(physics.vmin, dtype=torch.float32, device=device)
        self.vmax = torch.as_tensor(physics.vmax, dtype=torch.float32, device=device)
        self.pmin = torch.as_tensor(physics.pmin, dtype=torch.float32, device=device)
        self.pmax = torch.as_tensor(physics.pmax, dtype=torch.float32, device=device)
        self.qmin = torch.as_tensor(physics.qmin, dtype=torch.float32, device=device)
        self.qmax = torch.as_tensor(physics.qmax, dtype=torch.float32, device=device)
        self.virtual_bus_pairs = tuple(virtual_bus_pairs)

        sizes = (case.nbus, case.nbus, case.ngen, case.ngen, case.ngen, case.ngen)
        self.lambdas = [torch.zeros(n, device=device) for n in sizes]

    def _branch_flow(
        self, vm: torch.Tensor, va: torch.Tensor, outage_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        voltage = torch.polar(vm, va)
        vf, vt = voltage[:, self.fbus], voltage[:, self.tbus]
        current_f = vf * self.yff + vt * self.yft
        branch_idx = torch.arange(self.case.nbranch, device=vm.device)
        in_service = branch_idx.unsqueeze(0) != outage_idx.unsqueeze(1)
        sf_pu = vf * current_f.conj() * in_service
        return sf_pu.real, sf_pu.imag

    def __call__(
        self,
        pred_norm: torch.Tensor,
        true_norm: torch.Tensor,
        true_pf: torch.Tensor,
        true_qf: torch.Tensor,
        outage_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        loss_y = self.cy * ((pred_norm - true_norm) ** 2).sum(dim=1).mean()
        pred = pred_norm * self.y_std + self.y_mean
        true = true_norm * self.y_std + self.y_mean
        nbus, ngen = self.case.nbus, self.case.ngen
        vm, va = pred[:, :nbus], pred[:, nbus : 2 * nbus]
        pg = pred[:, 2 * nbus : 2 * nbus + ngen]
        qg = pred[:, 2 * nbus + ngen : 2 * nbus + 2 * ngen]

        loss_vt = pred_norm.new_zeros(())
        if self.virtual_bus_pairs:
            virtual_idx, real_idx = zip(*self.virtual_bus_pairs)
            virtual_idx = torch.as_tensor(virtual_idx, dtype=torch.long, device=pred.device)
            real_idx = torch.as_tensor(real_idx, dtype=torch.long, device=pred.device)
            virtual_error = (vm[:, virtual_idx] - true[:, real_idx]) ** 2
            virtual_error += (va[:, virtual_idx] - true[:, nbus + real_idx]) ** 2
            loss_vt = self.cv * virtual_error.sum(dim=1).mean()

        pf_hat, qf_hat = self._branch_flow(vm, va, outage_idx)
        pf_error = (pf_hat - true_pf / self.base_mva) ** 2
        qf_error = (qf_hat - true_qf / self.base_mva) ** 2
        loss_flow = (self.cp * pf_error + self.cq * qf_error).sum(dim=1).mean()

        violations = (
            F.relu(self.vmin - vm),
            F.relu(vm - self.vmax),
            F.relu((self.pmin - pg) / self.base_mva),
            F.relu((pg - self.pmax) / self.base_mva),
            F.relu((self.qmin - qg) / self.base_mva),
            F.relu((qg - self.qmax) / self.base_mva),
        )
        loss_constraint = sum(
            (lagrange * violation).sum(dim=1).mean()
            for lagrange, violation in zip(self.lambdas, violations)
        )
        parts = {
            "y": loss_y,
            "vt": loss_vt,
            "flow": loss_flow,
            "constraint": loss_constraint,
        }
        return sum(parts.values()), parts, tuple(v.detach() for v in violations)

    @torch.no_grad()
    def update_lambdas(
        self, violations: Sequence[torch.Tensor], batch_fraction: float
    ) -> None:
        for lagrange, violation, rho in zip(self.lambdas, violations, self.rhos):
            lagrange.add_(rho * batch_fraction * violation.mean(dim=0))

    def state_dict(self) -> Dict[str, object]:
        return {
            "rho_v": self.rhos[0],
            "rho_pg": self.rhos[2],
            "rho_qg": self.rhos[4],
            "lambdas": [value.detach().cpu() for value in self.lambdas],
        }


class TwoStageDNN(nn.Module):
    """Two-stage form from Equations (17)-(18)."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        gen_dim: int,
        stage1_hidden: Sequence[int],
        stage2_hidden: Sequence[int],
    ) -> None:
        super().__init__()
        self.stage1 = _make_mlp([input_dim, *stage1_hidden, state_dim])
        self.stage2 = (
            None
            if gen_dim == 0
            else _make_mlp([input_dim + state_dim, *stage2_hidden, gen_dim])
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_pred = self.stage1(x)
        if self.stage2 is None:
            return state_pred, state_pred
        gen_pred = self.stage2(torch.cat([state_pred, x], dim=1))
        return torch.cat([state_pred, gen_pred], dim=1), state_pred


def _make_mlp(sizes: Sequence[int]) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i != len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def describe_dimensions(case: CaseConfig) -> Dict[str, int]:
    return {
        "Pd_dim": case.nbus,
        "Qd_dim": case.nbus,
        "xi_scalar_dim_for_paper_table": 1,
        "xi_onehot_dim_in_matlab_generator": case.nbranch,
        "paper_input_dim": case.scalar_input_dim,
        "onehot_input_dim": case.onehot_input_dim,
        "Vm_dim": case.nbus,
        "Va_dim": case.nbus,
        "stage1_state_output_dim": case.state_dim,
        "Pg_dim": case.ngen,
        "Qg_dim": case.ngen,
        "generator_output_dim": case.gen_dim,
        "full_output_dim_formula": case.output_dim,
        "table_output_dim": case.table_arch[-1],
    }


def architecture_for(case: CaseConfig, contingency: str) -> Tuple[int, ...]:
    if contingency == "scalar":
        return case.table_arch
    arch = list(case.table_arch)
    arch[0] = case.onehot_input_dim
    return tuple(arch)
