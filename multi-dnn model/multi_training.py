"""Partitioned multi-DNN training for clustered corrective OPF."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from COPF.clusters import load_bus_clusters
from COPF.core import (
    CASES,
    CaseConfig,
    LossConfig,
    PhysicsData,
    Standardizer,
    TwoStageDNN,
)
from COPF.data import (
    load_case_matrices,
    load_mat_dataset,
    load_physics_dataset,
    make_loader,
    split_train_validation,
)
from COPF.training import set_random_seed


@dataclasses.dataclass(frozen=True)
class PartitionSpec:
    """Global-to-local indices needed by one partition agent."""

    number: int
    bus_numbers: Tuple[int, ...]
    owned_bus_idx: np.ndarray
    state_bus_idx: np.ndarray
    virtual_bus_idx: np.ndarray
    gen_idx: np.ndarray
    branch_idx: np.ndarray
    fbus_local: np.ndarray
    tbus_local: np.ndarray

    @property
    def n_owned(self) -> int:
        return len(self.owned_bus_idx)

    @property
    def n_state_bus(self) -> int:
        return len(self.state_bus_idx)

    @property
    def n_gen(self) -> int:
        return len(self.gen_idx)

    @property
    def state_dim(self) -> int:
        return 2 * self.n_state_bus

    @property
    def gen_dim(self) -> int:
        return 2 * self.n_gen

    @property
    def output_dim(self) -> int:
        return self.state_dim + self.gen_dim


def build_partition_specs(
    clusters: Sequence[Sequence[int]],
    bus: np.ndarray,
    gen: np.ndarray,
    physics: PhysicsData,
) -> Tuple[PartitionSpec, ...]:
    """Map editable 1-based bus groups to local model and physics indices."""
    bus_rows = {int(number): row for row, number in enumerate(bus[:, 0])}
    specs: List[PartitionSpec] = []
    for number, bus_numbers in enumerate(clusters, start=1):
        owned = np.asarray([bus_rows[bus_id] for bus_id in bus_numbers], dtype=np.int64)
        owned_set = set(owned.tolist())
        incident = np.flatnonzero(
            np.isin(physics.fbus, owned) | np.isin(physics.tbus, owned)
        ).astype(np.int64)
        endpoints = np.unique(
            np.concatenate([physics.fbus[incident], physics.tbus[incident]])
        )
        virtual = np.asarray(
            sorted(endpoint for endpoint in endpoints if endpoint not in owned_set),
            dtype=np.int64,
        )
        state_buses = np.concatenate([owned, virtual])
        local_row = {global_row: row for row, global_row in enumerate(state_buses)}
        fbus_local = np.asarray(
            [local_row[int(row)] for row in physics.fbus[incident]], dtype=np.int64
        )
        tbus_local = np.asarray(
            [local_row[int(row)] for row in physics.tbus[incident]], dtype=np.int64
        )
        gen_idx = np.flatnonzero(np.isin(gen[:, 0].astype(np.int64), bus_numbers))
        specs.append(
            PartitionSpec(
                number=number,
                bus_numbers=tuple(bus_numbers),
                owned_bus_idx=owned,
                state_bus_idx=state_buses,
                virtual_bus_idx=virtual,
                gen_idx=gen_idx.astype(np.int64),
                branch_idx=incident,
                fbus_local=fbus_local,
                tbus_local=tbus_local,
            )
        )

    assigned_generators = np.concatenate([spec.gen_idx for spec in specs])
    if sorted(assigned_generators.tolist()) != list(range(gen.shape[0])):
        raise ValueError("The bus clusters do not assign every generator exactly once")
    return tuple(specs)


def build_partition_dataset(
    x: np.ndarray,
    y: np.ndarray,
    case: CaseConfig,
    spec: PartitionSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice columns while retaining every sample and the global contingency."""
    contingency = x[:, 2 * case.nbus :]
    x_local = np.concatenate(
        [
            x[:, spec.owned_bus_idx],
            x[:, case.nbus + spec.owned_bus_idx],
            contingency,
        ],
        axis=1,
    )
    pg_start = 2 * case.nbus
    qg_start = pg_start + case.ngen
    y_local = np.concatenate(
        [
            y[:, spec.state_bus_idx],
            y[:, case.nbus + spec.state_bus_idx],
            y[:, pg_start + spec.gen_idx],
            y[:, qg_start + spec.gen_idx],
        ],
        axis=1,
    )
    return x_local.astype(np.float32), y_local.astype(np.float32)


class PartitionPIMALoss:
    """Equation (21) evaluated on owned buses plus adjacent virtual buses."""

    def __init__(
        self,
        spec: PartitionSpec,
        y_scaler: Standardizer,
        physics: PhysicsData,
        device: torch.device,
        config: LossConfig,
    ) -> None:
        self.spec = spec
        self.cy, self.cv = config.cy, config.cv
        self.cp, self.cq = config.cp, config.cq
        self.rhos = (
            config.rho_v,
            config.rho_v,
            config.rho_pg,
            config.rho_pg,
            config.rho_qg,
            config.rho_qg,
        )
        self.base_mva = physics.base_mva
        self.y_mean = torch.as_tensor(y_scaler.mean, device=device)
        self.y_std = torch.as_tensor(y_scaler.std, device=device)
        self.fbus = torch.as_tensor(spec.fbus_local, dtype=torch.long, device=device)
        self.tbus = torch.as_tensor(spec.tbus_local, dtype=torch.long, device=device)
        self.branch_idx = torch.as_tensor(
            spec.branch_idx, dtype=torch.long, device=device
        )
        self.yff = torch.as_tensor(
            physics.yff[spec.branch_idx], dtype=torch.complex64, device=device
        )
        self.yft = torch.as_tensor(
            physics.yft[spec.branch_idx], dtype=torch.complex64, device=device
        )
        self.vmin = torch.as_tensor(
            physics.vmin[spec.owned_bus_idx], dtype=torch.float32, device=device
        )
        self.vmax = torch.as_tensor(
            physics.vmax[spec.owned_bus_idx], dtype=torch.float32, device=device
        )
        self.pmin = torch.as_tensor(
            physics.pmin[spec.gen_idx], dtype=torch.float32, device=device
        )
        self.pmax = torch.as_tensor(
            physics.pmax[spec.gen_idx], dtype=torch.float32, device=device
        )
        self.qmin = torch.as_tensor(
            physics.qmin[spec.gen_idx], dtype=torch.float32, device=device
        )
        self.qmax = torch.as_tensor(
            physics.qmax[spec.gen_idx], dtype=torch.float32, device=device
        )

        nstate, nowned = spec.n_state_bus, spec.n_owned
        label_indices = np.concatenate(
            [
                np.arange(nowned),
                nstate + np.arange(nowned),
                np.arange(2 * nstate, spec.output_dim),
            ]
        )
        self.label_idx = torch.as_tensor(
            label_indices, dtype=torch.long, device=device
        )
        sizes = (nowned, nowned, spec.n_gen, spec.n_gen, spec.n_gen, spec.n_gen)
        self.lambdas = [torch.zeros(size, device=device) for size in sizes]

    def branch_flow(
        self, vm: torch.Tensor, va: torch.Tensor, outage_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        voltage = torch.polar(vm, va)
        vf, vt = voltage[:, self.fbus], voltage[:, self.tbus]
        current_f = vf * self.yff + vt * self.yft
        in_service = self.branch_idx.unsqueeze(0) != outage_idx.unsqueeze(1)
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
        label_error = pred_norm[:, self.label_idx] - true_norm[:, self.label_idx]
        loss_y = self.cy * (label_error**2).sum(dim=1).mean()
        pred = pred_norm * self.y_std + self.y_mean
        true = true_norm * self.y_std + self.y_mean
        nstate, nowned, ngen = (
            self.spec.n_state_bus,
            self.spec.n_owned,
            self.spec.n_gen,
        )
        vm, va = pred[:, :nstate], pred[:, nstate : 2 * nstate]
        pg = pred[:, 2 * nstate : 2 * nstate + ngen]
        qg = pred[:, 2 * nstate + ngen :]

        if nstate > nowned:
            virtual_error = (vm[:, nowned:] - true[:, nowned:nstate]) ** 2
            virtual_error += (
                va[:, nowned:] - true[:, nstate + nowned : 2 * nstate]
            ) ** 2
            loss_virtual = self.cv * virtual_error.sum(dim=1).mean()
        else:
            loss_virtual = pred_norm.new_zeros(())

        pf_hat, qf_hat = self.branch_flow(vm, va, outage_idx)
        pf_error = (pf_hat - true_pf / self.base_mva) ** 2
        qf_error = (qf_hat - true_qf / self.base_mva) ** 2
        loss_flow = (self.cp * pf_error + self.cq * qf_error).sum(dim=1).mean()

        owned_vm = vm[:, :nowned]
        violations = (
            F.relu(self.vmin - owned_vm),
            F.relu(owned_vm - self.vmax),
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
            "virtual": loss_virtual,
            "flow": loss_flow,
            "constraint": loss_constraint,
        }
        return sum(parts.values()), parts, tuple(v.detach() for v in violations)

    @torch.no_grad()
    def update_lambdas(
        self, violations: Sequence[torch.Tensor], batch_fraction: float
    ) -> None:
        for lagrange, violation, rho in zip(
            self.lambdas, violations, self.rhos
        ):
            if violation.shape[1] > 0:
                lagrange.add_(rho * batch_fraction * violation.mean(dim=0))

    def state_dict(self) -> Dict[str, object]:
        return {
            "rho_v": self.rhos[0],
            "rho_pg": self.rhos[2],
            "rho_qg": self.rhos[4],
            "lambdas": [value.detach().cpu() for value in self.lambdas],
        }


def _partition_architecture(
    args: argparse.Namespace, case: CaseConfig, spec: PartitionSpec, input_dim: int
) -> Tuple[int, ...]:
    return (
        input_dim,
        case.table_arch[1],
        case.table_arch[2],
        spec.state_dim,
        case.table_arch[-2],
        spec.output_dim,
    )


def _build_partition_model(
    spec: PartitionSpec, arch: Sequence[int]
) -> nn.Module:
    return TwoStageDNN(
        input_dim=arch[0],
        state_dim=spec.state_dim,
        gen_dim=spec.gen_dim,
        stage1_hidden=arch[1:3],
        stage2_hidden=(arch[-2],),
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: PartitionPIMALoss,
    device: torch.device,
) -> None:
    model.train()
    sample_count = len(loader.dataset)
    for xb, yb, pfb, qfb, outage_b in loader:
        xb, yb = xb.to(device), yb.to(device)
        pfb, qfb = pfb.to(device), qfb.to(device)
        outage_b = outage_b.to(device)
        pred, _ = model(xb)
        loss, _, violations = criterion(pred, yb, pfb, qfb, outage_b)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        criterion.update_lambdas(violations, xb.size(0) / sample_count)


def _constraint_metrics(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    base: float,
    tolerance: float,
) -> Tuple[float, float]:
    if values.shape[1] == 0:
        return 100.0, 0.0
    violation = np.maximum(lower - values, 0.0) + np.maximum(values - upper, 0.0)
    return (
        float(np.mean(violation / base <= tolerance) * 100.0),
        float(np.mean(violation)),
    )

def _paper_delta(errors: np.ndarray, nbus: int) -> float:
    """Average the paper's per-sample L2 error divided by system bus count."""
    return float(np.mean(np.sqrt(np.sum(errors**2, axis=1)) / nbus))


def _evaluate_partition(
    model: nn.Module,
    loader: DataLoader,
    criterion: PartitionPIMALoss,
    physics: PhysicsData,
    device: torch.device,
    tolerance: float,
) -> Dict[str, float]:
    model.eval()
    pred_norm_parts, true_norm_parts = [], []
    pred_pf_parts, pred_qf_parts = [], []
    true_pf_parts, true_qf_parts = [], []
    with torch.no_grad():
        for xb, yb, pfb, qfb, outage_b in loader:
            pred_norm, _ = model(xb.to(device))
            pred = pred_norm * criterion.y_std + criterion.y_mean
            nstate = criterion.spec.n_state_bus
            pf_hat, qf_hat = criterion.branch_flow(
                pred[:, :nstate], pred[:, nstate : 2 * nstate], outage_b.to(device)
            )
            pred_norm_parts.append(pred_norm.cpu().numpy())
            true_norm_parts.append(yb.numpy())
            pred_pf_parts.append((pf_hat * physics.base_mva).cpu().numpy())
            pred_qf_parts.append((qf_hat * physics.base_mva).cpu().numpy())
            true_pf_parts.append(pfb.numpy())
            true_qf_parts.append(qfb.numpy())

    pred_norm = np.concatenate(pred_norm_parts)
    true_norm = np.concatenate(true_norm_parts)
    label_idx = criterion.label_idx.cpu().numpy()
    pred = pred_norm * criterion.y_std.cpu().numpy() + criterion.y_mean.cpu().numpy()
    pf_error = np.concatenate(pred_pf_parts) - np.concatenate(true_pf_parts)
    qf_error = np.concatenate(pred_qf_parts) - np.concatenate(true_qf_parts)
    nstate, nowned, ngen = (
        criterion.spec.n_state_bus,
        criterion.spec.n_owned,
        criterion.spec.n_gen,
    )
    vm = pred[:, :nowned]
    pg = pred[:, 2 * nstate : 2 * nstate + ngen]
    qg = pred[:, 2 * nstate + ngen :]
    eta_v, mean_v = _constraint_metrics(
        vm,
        physics.vmin[criterion.spec.owned_bus_idx],
        physics.vmax[criterion.spec.owned_bus_idx],
        1.0,
        tolerance,
    )
    eta_pg, mean_pg = _constraint_metrics(
        pg,
        physics.pmin[criterion.spec.gen_idx],
        physics.pmax[criterion.spec.gen_idx],
        physics.base_mva,
        tolerance,
    )
    eta_qg, mean_qg = _constraint_metrics(
        qg,
        physics.qmin[criterion.spec.gen_idx],
        physics.qmax[criterion.spec.gen_idx],
        physics.base_mva,
        tolerance,
    )
    return {
        "normalized_mse": float(
            np.mean((pred_norm[:, label_idx] - true_norm[:, label_idx]) ** 2)
        ),
        "pflow_mse_pu": float(np.mean((pf_error / physics.base_mva) ** 2)),
        "qflow_mse_pu": float(np.mean((qf_error / physics.base_mva) ** 2)),
        "eta_v_pct": eta_v,
        "eta_pg_pct": eta_pg,
        "eta_qg_pct": eta_qg,
        "constraint_feasibility_min_pct": min(eta_v, eta_pg, eta_qg),
        "mean_constraint_violation_pu": float(
            np.mean([mean_v, mean_pg / physics.base_mva, mean_qg / physics.base_mva])
        ),
    }


def _quality(metrics: Dict[str, float]) -> float:
    return (
        metrics["normalized_mse"]
        + metrics["pflow_mse_pu"]
        + metrics["qflow_mse_pu"]
    )


def _select_result(
    results: Sequence[Tuple[object, Dict[str, float]]],
    quality_tolerance: float,
    feasibility_target: float,
) -> Tuple[object, Dict[str, float]]:
    best_quality = min(_quality(metrics) for _, metrics in results)
    quality_limit = best_quality * (1.0 + quality_tolerance)
    candidates = [item for item in results if _quality(item[1]) <= quality_limit]

    def key(item: Tuple[object, Dict[str, float]]) -> Tuple[float, ...]:
        metrics = item[1]
        feasibility = metrics["constraint_feasibility_min_pct"]
        return (
            0.0 if feasibility >= feasibility_target else 1.0,
            -feasibility,
            metrics["mean_constraint_violation_pu"],
            _quality(metrics),
        )

    return min(candidates, key=key)


def _fit_partition(
    args: argparse.Namespace,
    case: CaseConfig,
    spec: PartitionSpec,
    arch: Sequence[int],
    physics: PhysicsData,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    y_scaler: Standardizer,
    flow_p: np.ndarray,
    flow_q: np.ndarray,
    outage_idx: np.ndarray,
    fit_idx: np.ndarray,
    device: torch.device,
    config: LossConfig,
    epochs: int,
    label: str,
    validation_idx: Optional[np.ndarray] = None,
) -> Tuple[nn.Module, PartitionPIMALoss, int, Optional[Dict[str, float]]]:
    seed = (args.seed if args.seed is not None else 0) + spec.number
    set_random_seed(seed)
    batch_size = args.batch_size or case.batch_size
    fit_loader = make_loader(
        x_norm, y_norm, flow_p, flow_q, outage_idx, fit_idx, batch_size, True
    )
    validation_loader = None
    if validation_idx is not None:
        validation_loader = make_loader(
            x_norm,
            y_norm,
            flow_p,
            flow_q,
            outage_idx,
            validation_idx,
            batch_size,
            False,
        )
    model = _build_partition_model(spec, arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr or case.lr)
    criterion = PartitionPIMALoss(spec, y_scaler, physics, device, config)
    history: List[Tuple[int, Dict[str, float]]] = []
    for epoch in range(1, epochs + 1):
        _train_epoch(model, fit_loader, optimizer, criterion, device)
        metrics = None
        if validation_loader is not None:
            metrics = _evaluate_partition(
                model,
                validation_loader,
                criterion,
                physics,
                device,
                args.feasibility_tolerance,
            )
            history.append((epoch, copy.deepcopy(metrics)))
        if epoch == 1 or epoch % args.log_every == 0 or epoch == epochs:
            if metrics is None:
                print(f"[{label}] epoch {epoch:04d}/{epochs:04d}")
            else:
                print(
                    f"[{label}] epoch {epoch:04d} "
                    f"eta_V={metrics['eta_v_pct']:.3f}% "
                    f"eta_Pg={metrics['eta_pg_pct']:.3f}% "
                    f"eta_Qg={metrics['eta_qg_pct']:.3f}%"
                )

    if not history:
        return model, criterion, epochs, None
    selectable = [item for item in history if item[0] > args.selection_warmup]
    if not selectable:
        selectable = [history[-1]]
    selected_epoch, selected_metrics = _select_result(
        selectable, args.selection_quality_tolerance, args.feasibility_target
    )
    return model, criterion, int(selected_epoch), selected_metrics


def _generation_cost(pg: np.ndarray, gencost: np.ndarray) -> np.ndarray:
    costs = np.zeros(pg.shape[0], dtype=np.float64)
    for generator in range(pg.shape[1]):
        row = gencost[generator]
        model, ncost = int(row[0]), int(row[3])
        if model == 2:
            costs += np.polyval(row[4 : 4 + ncost], pg[:, generator])
        elif model == 1:
            points = row[4 : 4 + 2 * ncost].reshape(-1, 2)
            costs += np.interp(pg[:, generator], points[:, 0], points[:, 1])
        else:
            raise ValueError(f"Unsupported MATPOWER gencost model {model}")
    return costs


def _predict_global(
    artifacts: Sequence[Dict[str, object]],
    x: np.ndarray,
    y: np.ndarray,
    test_idx: np.ndarray,
    case: CaseConfig,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    prediction = np.full((len(test_idx), case.output_dim), np.nan, dtype=np.float32)
    for artifact in artifacts:
        spec = artifact["spec"]
        model = artifact["model"]
        x_scaler = artifact["x_scaler"]
        y_scaler = artifact["y_scaler"]
        x_local, _ = build_partition_dataset(x, y, case, spec)
        x_norm = x_scaler.transform(x_local[test_idx])
        parts = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(test_idx), batch_size):
                xb = torch.from_numpy(x_norm[start : start + batch_size]).to(device)
                pred_norm, _ = model(xb)
                parts.append(pred_norm.cpu().numpy())
        pred_norm = np.concatenate(parts)
        pred = pred_norm * y_scaler.std + y_scaler.mean
        nstate, nowned, ngen = spec.n_state_bus, spec.n_owned, spec.n_gen
        prediction[:, spec.owned_bus_idx] = pred[:, :nowned]
        prediction[:, case.nbus + spec.owned_bus_idx] = pred[
            :, nstate : nstate + nowned
        ]
        prediction[:, 2 * case.nbus + spec.gen_idx] = pred[
            :, 2 * nstate : 2 * nstate + ngen
        ]
        prediction[:, 2 * case.nbus + case.ngen + spec.gen_idx] = pred[
            :, 2 * nstate + ngen :
        ]
    if not np.isfinite(prediction).all():
        raise RuntimeError("Partition outputs did not cover every global output column")
    return prediction, y[test_idx]


def _evaluate_global(
    prediction: np.ndarray,
    truth: np.ndarray,
    train_truth: np.ndarray,
    flow_p: np.ndarray,
    flow_q: np.ndarray,
    objective: Optional[np.ndarray],
    test_idx: np.ndarray,
    outage_idx: np.ndarray,
    case: CaseConfig,
    physics: PhysicsData,
    tolerance: float,
) -> Dict[str, float]:
    scaler = Standardizer.fit(train_truth)
    normalized_error = (prediction - truth) / scaler.std
    vm = prediction[:, : case.nbus]
    va = prediction[:, case.nbus : 2 * case.nbus]
    pg = prediction[:, 2 * case.nbus : 2 * case.nbus + case.ngen]
    qg = prediction[:, 2 * case.nbus + case.ngen :]
    true_vm = truth[:, : case.nbus]
    true_va = truth[:, case.nbus : 2 * case.nbus]
    true_pg = truth[:, 2 * case.nbus : 2 * case.nbus + case.ngen]
    true_qg = truth[:, 2 * case.nbus + case.ngen :]
    
    voltage = vm * np.exp(1j * va)
    vf, vt = voltage[:, physics.fbus], voltage[:, physics.tbus]
    current_f = vf * physics.yff + vt * physics.yft
    sf = vf * np.conj(current_f)
    sf[np.arange(len(test_idx)), outage_idx[test_idx]] = 0.0
    pf_error = sf.real * physics.base_mva - flow_p[test_idx]
    qf_error = sf.imag * physics.base_mva - flow_q[test_idx]
    eta_v, mean_v = _constraint_metrics(
        vm, physics.vmin, physics.vmax, 1.0, tolerance
    )
    eta_pg, mean_pg = _constraint_metrics(
        pg, physics.pmin, physics.pmax, physics.base_mva, tolerance
    )
    eta_qg, mean_qg = _constraint_metrics(
        qg, physics.qmin, physics.qmax, physics.base_mva, tolerance
    )
    metrics = {
        "normalized_mse": float(np.mean(normalized_error**2)),
        "pflow_mse_pu": float(np.mean((pf_error / physics.base_mva) ** 2)),
        "qflow_mse_pu": float(np.mean((qf_error / physics.base_mva) ** 2)),
        "vm_rmse": float(np.sqrt(np.mean((vm - truth[:, : case.nbus]) ** 2))),
        "va_rmse_rad": float(
            np.sqrt(
                np.mean((va - truth[:, case.nbus : 2 * case.nbus]) ** 2)
            )
        ),
        "pg_rmse_mw": float(
            np.sqrt(
                np.mean(
                    (pg - truth[:, 2 * case.nbus : 2 * case.nbus + case.ngen])
                    ** 2
                )
            )
        ),
        "qg_rmse_mvar": float(
            np.sqrt(
                np.mean((qg - truth[:, 2 * case.nbus + case.ngen :]) ** 2)
            )
        ),
        "delta_p_paper_mw": _paper_delta(pg - true_pg, case.nbus),
        "delta_q_paper_mvar": _paper_delta(qg - true_qg, case.nbus),
        "delta_v_paper_pu": _paper_delta(vm - true_vm, case.nbus),
        "delta_theta_paper_deg": _paper_delta(np.rad2deg(va - true_va), case.nbus),

        "eta_v_pct": eta_v,
        "eta_pg_pct": eta_pg,
        "eta_qg_pct": eta_qg,
        "constraint_feasibility_min_pct": min(eta_v, eta_pg, eta_qg),
        "mean_constraint_violation_pu": float(
            np.mean([mean_v, mean_pg / physics.base_mva, mean_qg / physics.base_mva])
        ),
    }
    if objective is not None and physics.gencost is not None:
        predicted_objective = _generation_cost(pg, physics.gencost[: case.ngen])
        denominator = np.maximum(np.abs(objective[test_idx]), 1e-12)
        gap = 100.0 * (predicted_objective - objective[test_idx]) / denominator
        metrics.update(
            {
                "optimality_gap_mean_pct": float(np.mean(gap)),
                "optimality_gap_abs_mean_pct": float(np.mean(np.abs(gap))),
                "optimality_gap_max_abs_pct": float(np.max(np.abs(gap))),
            }
        )
    return metrics


def _outage_indices(x: np.ndarray, case: CaseConfig, contingency: str) -> np.ndarray:
    if contingency == "scalar":
        return np.rint(x[:, -1] * case.nbranch).astype(np.int64) - 1
    return np.argmax(x[:, -case.nbranch :], axis=1).astype(np.int64)


def _partition_dict(spec: PartitionSpec, bus: np.ndarray) -> Dict[str, object]:
    return {
        "cluster": spec.number,
        "owned_buses": list(spec.bus_numbers),
        "virtual_buses": [int(bus[row, 0]) for row in spec.virtual_bus_idx],
        "generator_indices_zero_based": spec.gen_idx.tolist(),
        "branch_indices_zero_based": spec.branch_idx.tolist(),
    }


def train_partitioned(args: argparse.Namespace) -> None:
    """Train one independently normalized DNN per configured bus partition."""
    case = CASES[args.case]
    if args.data is None:
        raise ValueError("Partitioned training requires --data")
    data_path = Path(args.data)
    x, y, train_idx, test_idx = load_mat_dataset(data_path, case, args.contingency)
    flow_p, flow_q, objective, physics = load_physics_dataset(data_path, case)
    bus, gen, _ = load_case_matrices(data_path, case)
    clusters = load_bus_clusters(args.case, case.nbus, args.cluster_file)
    specs = build_partition_specs(clusters, bus, gen, physics)
    outage_idx = _outage_indices(x, case, args.contingency)
    seed = args.seed if args.seed is not None else 0
    fit_idx, validation_idx = split_train_validation(
        train_idx, outage_idx, args.validation_ratio, seed
    )
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"partitioned split: clusters={len(specs)} fit={len(fit_idx)} "
        f"validation={len(validation_idx)} test={len(test_idx)}"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = args.loss_candidate or [LossConfig()]
    artifacts: List[Dict[str, object]] = []
    partition_reports = []
    for spec in specs:
        x_local, y_local = build_partition_dataset(x, y, case, spec)
        local_pf = flow_p[:, spec.branch_idx]
        local_qf = flow_q[:, spec.branch_idx]
        arch = _partition_architecture(args, case, spec, x_local.shape[1])
        print(
            f"cluster {spec.number}: owned_buses={spec.n_owned} "
            f"virtual_buses={len(spec.virtual_bus_idx)} generators={spec.n_gen} "
            f"branches={len(spec.branch_idx)} architecture={arch}"
        )

        tune_x_scaler = Standardizer.fit(x_local[fit_idx])
        tune_y_scaler = Standardizer.fit(y_local[fit_idx])
        tune_x_norm = tune_x_scaler.transform(x_local)
        tune_y_norm = tune_y_scaler.transform(y_local)
        choices = []
        tuning_results = []
        for candidate_number, candidate in enumerate(candidates, start=1):
            label = (
                f"cluster {spec.number} candidate "
                f"{candidate_number}/{len(candidates)}"
            )
            candidate_model, candidate_criterion, best_epoch, metrics = _fit_partition(
                args,
                case,
                spec,
                arch,
                physics,
                tune_x_norm,
                tune_y_norm,
                tune_y_scaler,
                local_pf,
                local_qf,
                outage_idx,
                fit_idx,
                device,
                candidate,
                args.epochs,
                label,
                validation_idx,
            )
            assert metrics is not None
            choices.append((candidate, best_epoch, metrics))
            tuning_results.append(
                {
                    "coefficients": dataclasses.asdict(candidate),
                    "best_epoch": best_epoch,
                    "validation_metrics": metrics,
                }
            )
            del candidate_model, candidate_criterion

        selected_number, _ = _select_result(
            [(number, choice[2]) for number, choice in enumerate(choices)],
            args.selection_quality_tolerance,
            args.feasibility_target,
        )
        best_config, selected_epoch, selected_metrics = choices[int(selected_number)]
        final_x_scaler = Standardizer.fit(x_local[train_idx])
        final_y_scaler = Standardizer.fit(y_local[train_idx])
        final_x_norm = final_x_scaler.transform(x_local)
        final_y_norm = final_y_scaler.transform(y_local)
        model, criterion, _, _ = _fit_partition(
            args,
            case,
            spec,
            arch,
            physics,
            final_x_norm,
            final_y_norm,
            final_y_scaler,
            local_pf,
            local_qf,
            outage_idx,
            train_idx,
            device,
            best_config,
            selected_epoch,
            f"cluster {spec.number} final refit",
        )
        checkpoint_path = out_dir / (
            f"{case.name}_cluster_{spec.number:02d}_{args.contingency}.pt"
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "case": dataclasses.asdict(case),
                "architecture": tuple(arch),
                "contingency": args.contingency,
                "training_mode": "partitioned",
                "partition": _partition_dict(spec, bus),
                "x_scaler": final_x_scaler.state_dict(),
                "y_scaler": final_y_scaler.state_dict(),
                "pima_loss": criterion.state_dict(),
                "loss_coefficients": dataclasses.asdict(best_config),
                "selected_epoch": selected_epoch,
            },
            checkpoint_path,
        )
        print(f"Saved cluster {spec.number}: {checkpoint_path}")
        artifacts.append(
            {
                "spec": spec,
                "model": model,
                "x_scaler": final_x_scaler,
                "y_scaler": final_y_scaler,
            }
        )
        partition_reports.append(
            {
                **_partition_dict(spec, bus),
                "input_dim": x_local.shape[1],
                "output_dim": y_local.shape[1],
                "selected_coefficients": dataclasses.asdict(best_config),
                "selected_epoch": selected_epoch,
                "selected_validation_metrics": selected_metrics,
                "tuning_results": tuning_results,
                "checkpoint": str(checkpoint_path),
            }
        )

    prediction, truth = _predict_global(
        artifacts,
        x,
        y,
        test_idx,
        case,
        args.batch_size or case.batch_size,
        device,
    )
    final_metrics = _evaluate_global(
        prediction,
        truth,
        y[train_idx],
        flow_p,
        flow_q,
        objective,
        test_idx,
        outage_idx,
        case,
        physics,
        args.feasibility_tolerance,
    )
    report = {
        "training_mode": "partitioned",
        "cluster_file": str(
            Path(args.cluster_file)
            if args.cluster_file
            else Path(__file__).with_name(f"{args.case}_clusters.json")
        ),
        "split": {
            "fit": len(fit_idx),
            "validation": len(validation_idx),
            "full_train_refit": len(train_idx),
            "test": len(test_idx),
        },
        "partitions": partition_reports,
        "final_test_metrics": final_metrics,
    }
    report_path = out_dir / f"{case.name}_partitioned_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL PARTITIONED TEST METRICS")
    print(json.dumps(final_metrics, indent=2))
    print(f"Saved metrics: {report_path}")
