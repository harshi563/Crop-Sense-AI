"""
trainer.py  —  Training loop with early stopping + metric tracking.
"""
import time, logging, numpy as np
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")


def metrics(y_true, y_pred):
    return {
        "R2":   round(float(r2_score(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "MAPE": round(float(np.mean(np.abs((y_true-y_pred)/(np.abs(y_true)+1e-6)))*100), 2),
    }


class HuberMSE(nn.Module):
    def __init__(self): super().__init__(); self.h=nn.HuberLoss(delta=1.0); self.m=nn.MSELoss()
    def forward(self, p, t): return 0.85*self.h(p,t) + 0.15*self.m(p,t)


def train(model, train_dl, val_dl, device, scaler=None,
          epochs=50, lr=1e-3, patience=12, save_dir=Path("outputs")):
    save_dir = Path(save_dir); save_dir.mkdir(exist_ok=True)
    model    = model.to(device)
    crit     = HuberMSE()
    opt      = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched    = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/20)

    best_r2, wait = -1e9, 0
    history = {"train_loss":[], "val_loss":[], "R2":[], "RMSE":[]}

    for ep in range(1, epochs+1):
        # train
        model.train(); tloss = 0
        for sat, wx, soil, y in train_dl:
            sat,wx,soil,y = [t.to(device) for t in (sat,wx,soil,y)]
            opt.zero_grad()
            loss = crit(model(sat,wx,soil), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tloss += loss.item()*len(y)
        tloss /= len(train_dl.dataset)

        # validate
        model.eval(); vloss = 0; preds, trues = [], []
        with torch.no_grad():
            for sat, wx, soil, y in val_dl:
                sat,wx,soil,y = [t.to(device) for t in (sat,wx,soil,y)]
                p = model(sat,wx,soil)
                vloss += crit(p,y).item()*len(y)
                preds.append(p.cpu().numpy().ravel())
                trues.append(y.cpu().numpy().ravel())
        vloss /= len(val_dl.dataset)
        preds = np.concatenate(preds); trues = np.concatenate(trues)
        if scaler: preds=scaler.inverse_y(preds); trues=scaler.inverse_y(trues)
        m = metrics(trues, preds)

        history["train_loss"].append(round(tloss,5))
        history["val_loss"].append(round(vloss,5))
        history["R2"].append(m["R2"]); history["RMSE"].append(m["RMSE"])

        if m["R2"] > best_r2:
            best_r2 = m["R2"]; wait = 0
            torch.save({"ep":ep,"state":model.state_dict(),"metrics":m},
                       save_dir/"best.pt")
        else:
            wait += 1
            if wait >= patience:
                log.info(f"Early stop at epoch {ep}"); break

        if ep%5==0 or ep<=3:
            log.info(f"Ep {ep:3d}/{epochs} | loss tr={tloss:.4f} va={vloss:.4f} "
                     f"| R²={m['R2']:.4f} RMSE={m['RMSE']:.4f}")
        sched.step()

    ck = torch.load(save_dir/"best.pt", weights_only=False)
    model.load_state_dict(ck["state"])
    log.info(f"Best → ep {ck['ep']} R²={ck['metrics']['R2']} RMSE={ck['metrics']['RMSE']}")
    return history


@torch.no_grad()
def evaluate(model, loader, device, scaler=None):
    model.eval(); preds, trues = [], []
    for sat, wx, soil, y in loader:
        sat,wx,soil,y = [t.to(device) for t in (sat,wx,soil,y)]
        preds.append(model(sat,wx,soil).cpu().numpy().ravel())
        trues.append(y.cpu().numpy().ravel())
    p = np.concatenate(preds); t = np.concatenate(trues)
    if scaler: p=scaler.inverse_y(p); t=scaler.inverse_y(t)
    return metrics(t, p)
