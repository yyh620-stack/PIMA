"""Dataset loading, MATPOWER metadata conversion, and data-loader helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from COPF.core import CaseConfig, PhysicsData


def load_mat_dataset(
    path: Path, case: CaseConfig, contingency: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y, train_idx, and test_idx with y=[Vm, Va, Pg, Qg]."""
    try:
        return _load_mat_h5(path, case, contingency)
    except OSError:
        return _load_mat_scipy(path, case, contingency)


def _load_mat_h5(
    path: Path, case: CaseConfig, contingency: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise KeyError("Expected MATLAB struct group '/data'")
        g = f["data"]
        pd = _mat_array(g["X_Pd"], cols=case.nbus)
        qd = _mat_array(g["X_Qd"], cols=case.nbus)
        vm = _mat_array(g["Y_Vm"], cols=case.nbus)
        va = _mat_array(g["Y_Va_rad"], cols=case.nbus)
        pg = _mat_array(g["Y_Pg"], cols=case.ngen)
        qg = _mat_array(g["Y_Qg"], cols=case.ngen)

        if contingency == "onehot":
            xi = _mat_array(g["X_xi"], cols=case.nbranch)
            outage_ids = None
        else:
            xi = None
            outage_ids = _mat_array(g["outage_ids"], cols=1).reshape(-1, 1)

        y = np.concatenate([vm, va, pg, qg], axis=1)
        train_idx = _mat_array(g["train_idx"], cols=1).astype(np.int64).ravel() - 1
        test_idx = _mat_array(g["test_idx"], cols=1).astype(np.int64).ravel() - 1

    return _build_dataset(
        case, contingency, pd, qd, y, xi, outage_ids, train_idx, test_idx
    )


def _load_mat_scipy(
    path: Path, case: CaseConfig, contingency: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.io import loadmat

    data = loadmat(path, squeeze_me=True, struct_as_record=False)["data"]
    pd = np.asarray(data.X_Pd, dtype=np.float32)
    qd = np.asarray(data.X_Qd, dtype=np.float32)
    vm = np.asarray(data.Y_Vm, dtype=np.float32)
    va = np.asarray(data.Y_Va_rad, dtype=np.float32)
    pg = np.asarray(data.Y_Pg, dtype=np.float32)
    qg = np.asarray(data.Y_Qg, dtype=np.float32)

    if contingency == "onehot":
        xi = np.asarray(data.X_xi, dtype=np.float32)
        outage_ids = None
    else:
        xi = None
        outage_ids = np.asarray(data.outage_ids, dtype=np.float32).reshape(-1, 1)

    y = np.concatenate([vm, va, pg, qg], axis=1)
    train_idx = np.asarray(data.train_idx, dtype=np.int64).ravel() - 1
    test_idx = np.asarray(data.test_idx, dtype=np.int64).ravel() - 1
    return _build_dataset(
        case, contingency, pd, qd, y, xi, outage_ids, train_idx, test_idx
    )


def load_physics_dataset(
    path: Path, case: CaseConfig
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], PhysicsData]:
    """Load branch-flow labels and MATPOWER constants for Equation (21)."""
    try:
        import h5py

        with h5py.File(path, "r") as f:
            g = f["data"]
            mpc = g["base_mpc"]
            pf = _mat_array(g["Y_Pf"], cols=case.nbranch)
            qf = _mat_array(g["Y_Qf"], cols=case.nbranch)
            objective = (
                _mat_array(g["Y_obj"], cols=1).ravel() if "Y_obj" in g else None
            )
            base_mva = float(np.asarray(mpc["baseMVA"]).squeeze())
            bus = _mat_array(mpc["bus"], cols=13)
            gen = _mat_array(mpc["gen"], cols=21)
            branch = _mat_array(mpc["branch"], cols=13)
            gencost = _mat_array(mpc["gencost"], cols=7) if "gencost" in mpc else None
    except (ImportError, OSError):
        from scipy.io import loadmat

        data = loadmat(path, squeeze_me=True, struct_as_record=False)["data"]
        mpc = data.base_mpc
        pf = np.asarray(data.Y_Pf, dtype=np.float32)
        qf = np.asarray(data.Y_Qf, dtype=np.float32)
        objective = (
            np.asarray(data.Y_obj, dtype=np.float32).ravel()
            if hasattr(data, "Y_obj")
            else None
        )
        base_mva = float(np.asarray(mpc.baseMVA).squeeze())
        bus = np.asarray(mpc.bus, dtype=np.float32)
        gen = np.asarray(mpc.gen, dtype=np.float32)
        branch = np.asarray(mpc.branch, dtype=np.float32)
        gencost = (
            np.asarray(mpc.gencost, dtype=np.float32)
            if hasattr(mpc, "gencost")
            else None
        )

    physics = _build_physics_data(base_mva, bus, gen, branch, gencost)
    return pf.astype(np.float32), qf.astype(np.float32), objective, physics


def load_case_matrices(
    path: Path, case: CaseConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return MATPOWER bus, generator, and branch matrices."""
    try:
        import h5py

        with h5py.File(path, "r") as f:
            mpc = f["data"]["base_mpc"]
            bus = _mat_array(mpc["bus"], cols=13)
            gen = _mat_array(mpc["gen"], cols=21)
            branch = _mat_array(mpc["branch"], cols=13)
    except (ImportError, OSError):
        from scipy.io import loadmat

        mpc = loadmat(path, squeeze_me=True, struct_as_record=False)["data"].base_mpc
        bus = np.asarray(mpc.bus, dtype=np.float32)
        gen = np.asarray(mpc.gen, dtype=np.float32)
        branch = np.asarray(mpc.branch, dtype=np.float32)

    if bus.shape[0] != case.nbus or gen.shape[0] != case.ngen:
        raise ValueError(
            f"MATPOWER metadata dimensions {bus.shape[0]}/{gen.shape[0]} do not "
            f"match configured case {case.nbus}/{case.ngen}"
        )
    return bus.astype(np.float32), gen.astype(np.float32), branch.astype(np.float32)


def _build_physics_data(
    base_mva: float,
    bus: np.ndarray,
    gen: np.ndarray,
    branch: np.ndarray,
    gencost: Optional[np.ndarray],
) -> PhysicsData:
    bus_i, vmax, vmin = 0, 11, 12
    qmax, qmin, pmax, pmin = 3, 4, 8, 9
    f_bus, t_bus, br_r, br_x, br_b = 0, 1, 2, 3, 4
    tap_col, shift_col, status_col = 8, 9, 10

    bus_rows = {int(number): row for row, number in enumerate(bus[:, bus_i])}
    fbus = np.asarray([bus_rows[int(number)] for number in branch[:, f_bus]])
    tbus = np.asarray([bus_rows[int(number)] for number in branch[:, t_bus]])

    tap = branch[:, tap_col].astype(np.complex64)
    tap[tap == 0] = 1.0
    tap *= np.exp(1j * np.deg2rad(branch[:, shift_col])).astype(np.complex64)
    status = branch[:, status_col]
    series_y = status / (branch[:, br_r] + 1j * branch[:, br_x])
    charging_y = 1j * status * branch[:, br_b] / 2.0
    yff = (series_y + charging_y) / (tap * np.conj(tap))
    yft = -series_y / np.conj(tap)

    return PhysicsData(
        base_mva=base_mva,
        fbus=fbus.astype(np.int64),
        tbus=tbus.astype(np.int64),
        yff=yff.astype(np.complex64),
        yft=yft.astype(np.complex64),
        vmin=bus[:, vmin].astype(np.float32),
        vmax=bus[:, vmax].astype(np.float32),
        pmin=gen[:, pmin].astype(np.float32),
        pmax=gen[:, pmax].astype(np.float32),
        qmin=gen[:, qmin].astype(np.float32),
        qmax=gen[:, qmax].astype(np.float32),
        gencost=None if gencost is None else gencost.astype(np.float32),
    )


def _build_dataset(
    case: CaseConfig,
    contingency: str,
    pd: np.ndarray,
    qd: np.ndarray,
    y: np.ndarray,
    xi: Optional[np.ndarray],
    outage_ids: Optional[np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if contingency == "onehot":
        x = np.concatenate([pd, qd, xi], axis=1)
    else:
        assert outage_ids is not None
        xi = outage_ids / max(case.nbranch, 1)
        x = np.concatenate([pd, qd, xi], axis=1)
    return x.astype(np.float32), y.astype(np.float32), train_idx, test_idx


def _mat_array(dataset, cols: int) -> np.ndarray:
    arr = np.asarray(dataset, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim == 2 and arr.shape[0] == cols and arr.shape[1] != cols:
        arr = arr.T
    return arr


def split_train_validation(
    train_idx: np.ndarray,
    outage_idx: np.ndarray,
    validation_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratify the validation split by outaged branch."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between zero and one")
    rng = np.random.default_rng(seed)
    fit_parts, validation_parts = [], []
    for outage in np.unique(outage_idx[train_idx]):
        group = train_idx[outage_idx[train_idx] == outage].copy()
        rng.shuffle(group)
        if len(group) < 2:
            fit_parts.append(group)
            continue
        validation_size = max(1, min(len(group) - 1, round(len(group) * validation_ratio)))
        validation_parts.append(group[:validation_size])
        fit_parts.append(group[validation_size:])
    fit_idx = np.concatenate(fit_parts)
    validation_idx = np.concatenate(validation_parts)
    rng.shuffle(fit_idx)
    rng.shuffle(validation_idx)
    return fit_idx, validation_idx


def make_loader(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    flow_p: np.ndarray,
    flow_q: np.ndarray,
    outage_idx: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x_norm[indices]),
        torch.from_numpy(y_norm[indices]),
        torch.from_numpy(flow_p[indices]),
        torch.from_numpy(flow_q[indices]),
        torch.from_numpy(outage_idx[indices]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
