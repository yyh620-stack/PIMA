"""Training, validation selection, evaluation metrics, and checkpointing."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path
import random
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from COPF.core import (
    CASES,
    CaseConfig,
    LossConfig,
    PIMALoss,
    PhysicsData,
    Standardizer,
    TwoStageDNN,
    architecture_for,
    describe_dimensions,
)
from COPF.data import (
    load_mat_dataset,
    load_physics_dataset,
    make_loader,
    split_train_validation,
)


def _build_model(case: CaseConfig, arch: Sequence[int]) -> nn.Module:
    return TwoStageDNN(
        input_dim=arch[0],
        state_dim=case.state_dim,
        gen_dim=case.gen_dim,
        stage1_hidden=arch[1:3],
        stage2_hidden=(arch[-2],),
    )


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: PIMALoss,
    device: torch.device,
) -> None:
    model.train()
    sample_count = len(loader.dataset)
    for xb, yb, pfb, qfb, outage_b in loader:
        xb, yb = xb.to(device), yb.to(device)
        pfb, qfb, outage_b = pfb.to(device), qfb.to(device), outage_b.to(device)
        pred, _ = model(xb)
        loss, _, violations = criterion(pred, yb, pfb, qfb, outage_b)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        criterion.update_lambdas(violations, xb.size(0) / sample_count)


def _violation_metrics(
    metrics: Dict[str, float],
    name: str,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    base: float,
    tolerance_pu: float,
) -> None:
    violation = np.maximum(lower - values, 0.0) + np.maximum(values - upper, 0.0)
    violation_pu = violation / base
    metrics[f"eta_{name}_pct"] = float(np.mean(violation_pu <= tolerance_pu) * 100.0)
    metrics[f"{name}_mean_violation"] = float(np.mean(violation))


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


def _paper_delta(errors: np.ndarray, nbus: int) -> float:
    """Average the paper's per-sample L2 error divided by system bus count."""
    return float(np.mean(np.sqrt(np.sum(errors**2, axis=1)) / nbus))


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: PIMALoss,
    physics: PhysicsData,
    device: torch.device,
    feasibility_tolerance: float,
    true_objective: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    model.eval()
    pred_norm_parts, true_norm_parts = [], []
    pred_flow_p_parts, pred_flow_q_parts = [], []
    true_flow_p_parts, true_flow_q_parts = [], []
    with torch.no_grad():
        for xb, yb, pfb, qfb, outage_b in loader:
            xb, yb, outage_b = xb.to(device), yb.to(device), outage_b.to(device)
            pred_norm, _ = model(xb)
            pred = pred_norm * criterion.y_std + criterion.y_mean
            nbus = criterion.case.nbus
            pf_hat, qf_hat = criterion._branch_flow(
                pred[:, :nbus], pred[:, nbus : 2 * nbus], outage_b
            )
            pred_norm_parts.append(pred_norm.cpu().numpy())
            true_norm_parts.append(yb.cpu().numpy())
            pred_flow_p_parts.append((pf_hat * physics.base_mva).cpu().numpy())
            pred_flow_q_parts.append((qf_hat * physics.base_mva).cpu().numpy())
            true_flow_p_parts.append(pfb.numpy())
            true_flow_q_parts.append(qfb.numpy())

    pred_norm = np.concatenate(pred_norm_parts)
    true_norm = np.concatenate(true_norm_parts)
    pred = pred_norm * criterion.y_std.cpu().numpy() + criterion.y_mean.cpu().numpy()
    true = true_norm * criterion.y_std.cpu().numpy() + criterion.y_mean.cpu().numpy()
    pf_error = np.concatenate(pred_flow_p_parts) - np.concatenate(true_flow_p_parts)
    qf_error = np.concatenate(pred_flow_q_parts) - np.concatenate(true_flow_q_parts)
    metrics: Dict[str, float] = {
        "normalized_mse": float(np.mean((pred_norm - true_norm) ** 2)),
        "pflow_mse_pu": float(np.mean((pf_error / physics.base_mva) ** 2)),
        "qflow_mse_pu": float(np.mean((qf_error / physics.base_mva) ** 2)),
    }
    nbus, ngen = criterion.case.nbus, criterion.case.ngen
    vm = pred[:, :nbus]
    va = pred[:, nbus : 2 * nbus]
    pg = pred[:, 2 * nbus : 2 * nbus + ngen]
    qg = pred[:, 2 * nbus + ngen : 2 * nbus + 2 * ngen]
    metrics.update({
        "delta_p_paper_mw": _paper_delta(pg - true[:, 2 * nbus : 2 * nbus + ngen], nbus),
        "delta_q_paper_mvar": _paper_delta(qg - true[:, 2 * nbus + ngen :], nbus),
        "delta_v_paper_pu": _paper_delta(vm - true[:, :nbus], nbus),
        "delta_theta_paper_deg": _paper_delta(
            np.rad2deg(va - true[:, nbus : 2 * nbus]), nbus
        ),
    })
    _violation_metrics(
        metrics, "v", vm, physics.vmin, physics.vmax, 1.0, feasibility_tolerance
    )
    _violation_metrics(
        metrics, "pg", pg, physics.pmin, physics.pmax,
        physics.base_mva, feasibility_tolerance,
    )
    _violation_metrics(
        metrics, "qg", qg, physics.qmin, physics.qmax,
        physics.base_mva, feasibility_tolerance,
    )
    metrics["constraint_feasibility_min_pct"] = min(
        metrics["eta_v_pct"], metrics["eta_pg_pct"], metrics["eta_qg_pct"]
    )
    metrics["mean_constraint_violation_pu"] = float(np.mean([
        metrics["v_mean_violation"],
        metrics["pg_mean_violation"] / physics.base_mva,
        metrics["qg_mean_violation"] / physics.base_mva,
    ]))

    if true_objective is not None and physics.gencost is not None:
        predicted_objective = _generation_cost(pg, physics.gencost[:ngen])
        denominator = np.maximum(np.abs(true_objective), 1e-12)
        gap = 100.0 * (predicted_objective - true_objective) / denominator
        metrics["optimality_gap_mean_pct"] = float(np.mean(gap))
        metrics["optimality_gap_abs_mean_pct"] = float(np.mean(np.abs(gap)))
        metrics["optimality_gap_max_abs_pct"] = float(np.max(np.abs(gap)))
    return metrics


def _validation_quality(metrics: Dict[str, float]) -> float:
    return (
        metrics["normalized_mse"]
        + metrics["pflow_mse_pu"]
        + metrics["qflow_mse_pu"]
    )


def _reported_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Keep only the feasibility and optimality metrics requested for reports."""
    keys = (
        "delta_p_paper_mw",
        "delta_q_paper_mvar",
        "delta_v_paper_pu",
        "delta_theta_paper_deg",
        "eta_v_pct",
        "eta_pg_pct",
        "eta_qg_pct",
        "constraint_feasibility_min_pct",
        "optimality_gap_mean_pct",
        "optimality_gap_abs_mean_pct",
        "optimality_gap_max_abs_pct",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _select_validation_result(
    results: Sequence[Tuple[object, Dict[str, float]]],
    quality_tolerance: float,
    feasibility_target: float,
) -> Tuple[object, Dict[str, float]]:
    """Select by quality first, then feasibility among near-equal results."""
    if not results:
        raise ValueError("No validation results are available for selection")
    if quality_tolerance < 0:
        raise ValueError("selection_quality_tolerance must be non-negative")

    best_quality = min(_validation_quality(metrics) for _, metrics in results)
    quality_limit = best_quality * (1.0 + quality_tolerance)
    candidates = [
        item for item in results
        if _validation_quality(item[1]) <= quality_limit
    ]

    def selection_key(item: Tuple[object, Dict[str, float]]) -> Tuple[float, ...]:
        metrics = item[1]
        feasibility = metrics["constraint_feasibility_min_pct"]
        return (
            0.0 if feasibility >= feasibility_target else 1.0,
            -feasibility,
            metrics["mean_constraint_violation_pu"],
            _validation_quality(metrics),
        )

    return min(candidates, key=selection_key)


def _fit_model(
    args: argparse.Namespace,
    case: CaseConfig,
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
    objective: Optional[np.ndarray] = None,
) -> Tuple[nn.Module, PIMALoss, int, Optional[Dict[str, float]]]:
    set_random_seed(args.seed if args.seed is not None else 0)
    batch_size = args.batch_size or case.batch_size
    fit_loader = make_loader(
        x_norm, y_norm, flow_p, flow_q, outage_idx, fit_idx, batch_size, True
    )
    validation_loader = None
    if validation_idx is not None:
        validation_loader = make_loader(
            x_norm, y_norm, flow_p, flow_q, outage_idx,
            validation_idx, batch_size, False,
        )
    model = _build_model(case, arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr or case.lr)
    criterion = PIMALoss(
        case, y_scaler, physics, device,
        config.cy, config.cv, config.cp, config.cq,
        config.rho_v, config.rho_pg, config.rho_qg,
    )
    validation_history = []
    for epoch in range(1, epochs + 1):
        _train_one_epoch(model, fit_loader, optimizer, criterion, device)
        validation_metrics = None
        if validation_loader is not None:
            validation_objective = (
                None if objective is None else objective[validation_idx]
            )
            validation_metrics = evaluate_metrics(
                model, validation_loader, criterion, physics, device,
                args.feasibility_tolerance, validation_objective,
            )
            validation_history.append((epoch, copy.deepcopy(validation_metrics)))
        if epoch == 1 or epoch % args.log_every == 0 or epoch == epochs:
            if validation_metrics is not None:
                print(
                    f"[{label}] epoch {epoch:04d} "
                    f"eta_V={validation_metrics['eta_v_pct']:.3f}% "
                    f"eta_Pg={validation_metrics['eta_pg_pct']:.3f}% "
                    f"eta_Qg={validation_metrics['eta_qg_pct']:.3f}% "
                    f"gap={validation_metrics.get('optimality_gap_mean_pct', float('nan')):.4f}%"
                )
            else:
                print(f"[{label}] epoch {epoch:04d}/{epochs:04d}")

    best_epoch, best_metrics = epochs, None
    if validation_history:
        selectable_history = [
            item for item in validation_history
            if item[0] > args.selection_warmup
        ]
        if not selectable_history:
            selectable_history = [validation_history[-1]]
        selected, best_metrics = _select_validation_result(
            selectable_history,
            args.selection_quality_tolerance,
            args.feasibility_target,
        )
        best_epoch = int(selected)
    return model, criterion, best_epoch, best_metrics


def train(args: argparse.Namespace) -> None:
    case = CASES[args.case]
    print(json.dumps(describe_dimensions(case), indent=2))
    if args.data is None:
        print("No --data provided; dimension summary printed only.")
        return

    data_path = Path(args.data)
    x, y, train_idx, test_idx = load_mat_dataset(data_path, case, args.contingency)
    flow_p, flow_q, objective, physics = load_physics_dataset(data_path, case)
    arch = architecture_for(case, args.contingency)
    if x.shape[1] != arch[0] or y.shape[1] != arch[-1]:
        raise ValueError(
            f"Dataset dimensions {x.shape[1]}/{y.shape[1]} "
            f"do not match {arch[0]}/{arch[-1]}"
        )
    if args.contingency == "scalar":
        outage_idx = np.rint(x[:, -1] * case.nbranch).astype(np.int64) - 1
    else:
        outage_idx = np.argmax(x[:, -case.nbranch :], axis=1).astype(np.int64)

    seed = args.seed if args.seed is not None else 0
    fit_idx, validation_idx = split_train_validation(
        train_idx, outage_idx, args.validation_ratio, seed
    )
    print(
        f"split: fit={len(fit_idx)} validation={len(validation_idx)} "
        f"test={len(test_idx)} (test is held out until final evaluation)"
    )
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")

    tune_x_scaler = Standardizer.fit(x[fit_idx])
    tune_y_scaler = Standardizer.fit(y[fit_idx])
    tune_x_norm = tune_x_scaler.transform(x)
    tune_y_norm = tune_y_scaler.transform(y)
    candidates = args.loss_candidate or [LossConfig()]
    tuning_results = []
    candidate_choices = []
    for number, candidate in enumerate(candidates, start=1):
        label = f"candidate {number}/{len(candidates)} {dataclasses.asdict(candidate)}"
        _, _, candidate_epoch, metrics = _fit_model(
            args, case, arch, physics, tune_x_norm, tune_y_norm, tune_y_scaler,
            flow_p, flow_q, outage_idx, fit_idx, device, candidate,
            args.epochs, label, validation_idx, objective,
        )
        assert metrics is not None
        tuning_results.append({
            "coefficients": dataclasses.asdict(candidate),
            "best_epoch": candidate_epoch,
            "validation_metrics": _reported_metrics(metrics),
        })
        candidate_choices.append((candidate, candidate_epoch, metrics))

    selected_number, _ = _select_validation_result(
        [(number, item[2]) for number, item in enumerate(candidate_choices)],
        args.selection_quality_tolerance,
        args.feasibility_target,
    )
    best_candidate, selected_epoch, _ = candidate_choices[int(selected_number)]
    print(
        f"selected coefficients: {dataclasses.asdict(best_candidate)}, "
        f"epoch={selected_epoch}"
    )

    final_x_scaler = Standardizer.fit(x[train_idx])
    final_y_scaler = Standardizer.fit(y[train_idx])
    final_x_norm = final_x_scaler.transform(x)
    final_y_norm = final_y_scaler.transform(y)
    model, criterion, _, _ = _fit_model(
        args, case, arch, physics, final_x_norm, final_y_norm, final_y_scaler,
        flow_p, flow_q, outage_idx, train_idx, device, best_candidate,
        selected_epoch, "final refit",
    )
    test_loader = make_loader(
        final_x_norm, final_y_norm, flow_p, flow_q, outage_idx,
        test_idx, args.batch_size or case.batch_size, False,
    )
    test_objective = None if objective is None else objective[test_idx]
    final_metrics = evaluate_metrics(
        model, test_loader, criterion, physics, device,
        args.feasibility_tolerance, test_objective,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{case.name}_{args.contingency}.pt"
    save_checkpoint(
        checkpoint_path, model, case, arch, final_x_scaler, final_y_scaler,
        criterion, best_candidate, selected_epoch, args,
    )
    report = {
        "split": {
            "fit": len(fit_idx),
            "validation": len(validation_idx),
            "full_train_refit": len(train_idx),
            "test": len(test_idx),
        },
        "selection_rule": (
            "quality within tolerance first, then feasibility target and violation"
        ),
        "selection_warmup": args.selection_warmup,
        "selection_quality_tolerance": args.selection_quality_tolerance,
        "selected_coefficients": dataclasses.asdict(best_candidate),
        "selected_epoch": selected_epoch,
        "tuning_results": tuning_results,
        "final_test_metrics": _reported_metrics(final_metrics),
    }
    report_path = out_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL TEST METRICS (test_idx evaluated once)")
    print(json.dumps(_reported_metrics(final_metrics), indent=2))
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {report_path}")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    path: Path,
    model: nn.Module,
    case: CaseConfig,
    arch: Sequence[int],
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    criterion: PIMALoss,
    loss_config: LossConfig,
    selected_epoch: int,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "case": dataclasses.asdict(case),
            "architecture": tuple(arch),
            "contingency": args.contingency,
            "x_scaler": x_scaler.state_dict(),
            "y_scaler": y_scaler.state_dict(),
            "pima_loss": criterion.state_dict(),
            "loss_coefficients": dataclasses.asdict(loss_config),
            "selected_epoch": selected_epoch,
            "validation_ratio": args.validation_ratio,
            "feasibility_target": args.feasibility_target,
            "selection_warmup": args.selection_warmup,
            "selection_quality_tolerance": args.selection_quality_tolerance,
        },
        path,
    )
