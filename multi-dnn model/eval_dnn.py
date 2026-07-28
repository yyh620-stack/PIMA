from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from COPF.core import CASES, TwoStageDNN, architecture_for
from COPF.data import load_mat_dataset

def rmse(a):
    return float(np.sqrt(np.mean(a * a)))

def mae(a):
    return float(np.mean(np.abs(a)))

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--data", required=True)
p.add_argument("--case", default="case118")
p.add_argument("--contingency", default="scalar", choices=("scalar", "onehot"))
p.add_argument("--batch-size", type=int, default=512)
p.add_argument("--device", default="cpu")
args = p.parse_args()

case = CASES[args.case]
ckpt = torch.load(args.ckpt, map_location=args.device)
arch = tuple(ckpt.get("architecture", architecture_for(case, args.contingency)))
model = TwoStageDNN(
    input_dim=arch[0],
    state_dim=case.state_dim,
    gen_dim=case.gen_dim,
    stage1_hidden=arch[1:3],
    stage2_hidden=(arch[-2],),
)

model.load_state_dict(ckpt["model_state_dict"])
model.to(args.device).eval()

x, y, train_idx, test_idx = load_mat_dataset(Path(args.data), case, args.contingency)

xs = ckpt["x_scaler"]
ys = ckpt["y_scaler"]

x_norm = ((x - xs["mean"]) / xs["std"]).astype(np.float32)
y_norm = ((y - ys["mean"]) / ys["std"]).astype(np.float32)

loader = DataLoader(
    TensorDataset(
        torch.from_numpy(x_norm[test_idx]),
        torch.from_numpy(y_norm[test_idx]),
    ),
    batch_size=args.batch_size,
    shuffle=False,
)

preds = []
trues = []

with torch.no_grad():
    for xb, yb in loader:
        pred, _ = model(xb.to(args.device))
        preds.append(pred.cpu().numpy())
        trues.append(yb.numpy())

pred_norm = np.vstack(preds)
true_norm = np.vstack(trues)

err_norm = pred_norm - true_norm
pred = pred_norm * ys["std"] + ys["mean"]
true = true_norm * ys["std"] + ys["mean"]
err = pred - true

slices = {
    "Vm": slice(0, case.nbus),
    "Va_rad": slice(case.nbus, 2 * case.nbus),
    "Pg": slice(2 * case.nbus, 2 * case.nbus + case.ngen),
    "Qg": slice(2 * case.nbus + case.ngen, 2 * case.nbus + 2 * case.ngen),
}

print(f"test samples: {len(test_idx)}")
print(f"normalized_mse: {float(np.mean(err_norm ** 2)):.6e}")

for name, sl in slices.items():
    e = err[:, sl]
    print(
        f"{name:6s} "
        f"RMSE={rmse(e):.6e} "
        f"MAE={mae(e):.6e} "
        f"MAXAE={float(np.max(np.abs(e))):.6e}"
    )
