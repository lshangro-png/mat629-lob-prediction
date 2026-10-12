"""Training loop with early stopping on the VALIDATION split (never on test)."""
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_proba(model, loader, device):
    """Returns (softmax probabilities (N, 3), true classes (N,))."""
    model.eval()
    probs, labels = [], []
    for Xb, yb in loader:
        probs.append(torch.softmax(model(Xb.to(device, non_blocking=True)), dim=-1).cpu())
        labels.append(yb)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def _val_metrics(model, loader, criterion, device):
    model.eval()
    loss, preds, labels = 0.0, [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model(Xb)
            loss += criterion(logits, yb).item() * len(yb)
            preds.append(logits.argmax(1).cpu())
            labels.append(yb.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {
        "loss": loss / len(labels),
        "acc": float((preds == labels).mean()),
        # macro F1 over the two directional classes (Down, Up)
        "dir_f1": f1_score(labels, preds, labels=[0, 2], average="macro", zero_division=0),
    }


def fit(model, loaders, class_w, device, epochs=50, lr=1e-4, weight_decay=1e-2,
        patience=10, scheduler="cosine", monitor="loss"):
    """Train on loaders['train']; early-stop and keep the best weights by
    `monitor` ('loss' | 'acc' | 'dir_f1') on loaders['val']."""
    criterion = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler == "cosine":
        sched, per_batch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs), False
    elif scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(loaders["train"]),
            pct_start=0.1, anneal_strategy="cos")
        per_batch = True
    else:
        raise ValueError(scheduler)

    sign = 1.0 if monitor == "loss" else -1.0      # lower-is-better score
    best_score, best_state, bad_epochs = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_dir_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for Xb, yb in loaders["train"]:
            Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if per_batch:
                sched.step()
            running += loss.item() * len(yb)
            n += len(yb)
        if not per_batch:
            sched.step()

        val = _val_metrics(model, loaders["val"], criterion, device)
        history["train_loss"].append(running / n)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        history["val_dir_f1"].append(val["dir_f1"])
        print(f"Ep {epoch:3d}/{epochs} | train {running / n:.4f} | val loss {val['loss']:.4f} "
              f"| val acc {val['acc']:.4f} | val DirF1 {val['dir_f1']:.4f}")

        score = sign * val[monitor]
        if best_state is None or score < best_score:
            best_score, bad_epochs = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stop at epoch {epoch}; best val {monitor} at epoch "
                      f"{history['best_epoch']}: {sign * best_score:.4f}")
                break

    model.load_state_dict(best_state)
    return history
