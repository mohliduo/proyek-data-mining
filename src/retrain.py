"""
Retraining pipeline dengan hyperparameter tuning, class balancing,
threshold optimization, dan PR-AUC sebagai primary metric.

Usage:
    python -m src.retrain
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import seaborn as sns
import torch
import yaml
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metadata"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
for p in (MODELS_DIR, METRICS_DIR, FIG_DIR):
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- utilities
def best_threshold_f1(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Cari threshold yang memaksimalkan F1 pada kurva precision-recall."""
    p, r, t = precision_recall_curve(y_true, y_proba)
    f1 = (2 * p * r) / np.where((p + r) == 0, 1, p + r)
    idx = int(np.argmax(f1[:-1]))  # buang titik terakhir (no-threshold)
    return float(t[idx]), float(f1[idx])


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "pr_auc":    float(average_precision_score(y_true, y_proba)),
        "roc_auc":   float(roc_auc_score(y_true, y_proba)),
        "f1":        float(f1_score(y_true, y_pred)),
        "recall":    float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
    }


def save_confusion_matrix(y_true, y_pred, name: str) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Stroke", "Stroke"],
                yticklabels=["No Stroke", "Stroke"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{name} — Test (tuned threshold)")
    plt.tight_layout()
    out = FIG_DIR / f"cm_{name}_test.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def log_run(name: str, family: str, model, val_m: dict, test_m: dict,
            params_logged: dict, cm_path: Path, log_fn=mlflow.sklearn.log_model):
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("model_family", family)
        mlflow.set_tag("stage", "tuned")
        mlflow.log_params(params_logged)
        for k, v in val_m.items():  mlflow.log_metric(f"val_{k}", v)
        for k, v in test_m.items(): mlflow.log_metric(f"test_{k}", v)
        mlflow.log_artifact(str(cm_path), artifact_path="confusion_matrix")
        if log_fn is not None and model is not None:
            try:
                log_fn(model, name="model")
            except Exception as e:
                print(f"  [warn] log_model gagal untuk {name}: {e}")
        return run.info.run_id


# ---------------------------------------------------------------- training
def main():
    print("=" * 70)
    print("RETRAIN — primary metric: PR-AUC, threshold tuning per model")
    print("=" * 70)

    with open(PROJECT_ROOT / "params.yaml") as f:
        params = yaml.safe_load(f)
    rs = params["dataset"]["random_state"]
    np.random.seed(rs); torch.manual_seed(rs)

    # MLflow
    db = (PROJECT_ROOT / params["mlflow"]["tracking_db"]).resolve()
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    # Data
    z = np.load(DATA_PROCESSED / "tabnet_ready.npz")
    X_train, y_train = z["X_train"], z["y_train"]
    X_val,   y_val   = z["X_val"],   z["y_val"]
    X_test,  y_test  = z["X_test"],  z["y_test"]
    with open(DATA_PROCESSED / "tabnet_meta.json") as f:
        meta = json.load(f)
    cat_idxs, cat_dims = meta["cat_idxs"], meta["cat_dims"]

    n_neg = int((y_train == 0).sum()); n_pos = int((y_train == 1).sum())
    spw = n_neg / max(n_pos, 1)
    print(f"\nTrain: pos={n_pos}, neg={n_neg}  → scale_pos_weight={spw:.3f}")
    print(f"Val: pos={int(y_val.sum())}, Test: pos={int(y_test.sum())}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    leaderboard = []

    # ============================================================ LogReg
    print("[1/4] Logistic Regression — RandomizedSearchCV...")
    lr_dist = {
        "C": np.logspace(-3, 2, 30),
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
        "class_weight": ["balanced", None],
    }
    lr_search = RandomizedSearchCV(
        LogisticRegression(random_state=rs, max_iter=2000),
        lr_dist, n_iter=25, scoring="average_precision",
        cv=cv, n_jobs=-1, random_state=rs, verbose=0,
    )
    lr_search.fit(X_train, y_train)
    lr_best = lr_search.best_estimator_
    proba_val = lr_best.predict_proba(X_val)[:, 1]
    thr, _ = best_threshold_f1(y_val, proba_val)
    val_m = evaluate(y_val, proba_val, thr)
    test_m = evaluate(y_test, lr_best.predict_proba(X_test)[:, 1], thr)
    cm_path = save_confusion_matrix(y_test, (lr_best.predict_proba(X_test)[:, 1] >= thr).astype(int), "logreg")
    rid = log_run("logreg", "linear", lr_best, val_m, test_m,
                  {**lr_search.best_params_, "random_state": rs}, cm_path)
    leaderboard.append({"run_name": "logreg", "family": "linear",
                        "val": val_m, "test": test_m, "run_id": rid,
                        "best_params": lr_search.best_params_})
    print(f"   best: {lr_search.best_params_}")
    print(f"   val PR-AUC={val_m['pr_auc']:.4f}  test PR-AUC={test_m['pr_auc']:.4f}  thr={thr:.3f}\n")

    # ============================================================ Random Forest
    print("[2/4] Random Forest — RandomizedSearchCV...")
    rf_dist = {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [None, 6, 10, 14, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=rs, n_jobs=-1),
        rf_dist, n_iter=20, scoring="average_precision",
        cv=cv, n_jobs=-1, random_state=rs, verbose=0,
    )
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    proba_val = rf_best.predict_proba(X_val)[:, 1]
    thr, _ = best_threshold_f1(y_val, proba_val)
    val_m = evaluate(y_val, proba_val, thr)
    test_m = evaluate(y_test, rf_best.predict_proba(X_test)[:, 1], thr)
    cm_path = save_confusion_matrix(y_test, (rf_best.predict_proba(X_test)[:, 1] >= thr).astype(int), "randomforest")
    rid = log_run("random_forest", "tree-ensemble", rf_best, val_m, test_m,
                  {**rf_search.best_params_, "random_state": rs}, cm_path)
    leaderboard.append({"run_name": "random_forest", "family": "tree-ensemble",
                        "val": val_m, "test": test_m, "run_id": rid,
                        "best_params": rf_search.best_params_})
    print(f"   best: {rf_search.best_params_}")
    print(f"   val PR-AUC={val_m['pr_auc']:.4f}  test PR-AUC={test_m['pr_auc']:.4f}  thr={thr:.3f}\n")

    # ============================================================ XGBoost (+ scale_pos_weight)
    print("[3/4] XGBoost — RandomizedSearchCV (with scale_pos_weight)...")
    xgb_dist = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3],
        "scale_pos_weight": [1.0, spw / 2, spw, spw * 1.5],
    }
    xgb_search = RandomizedSearchCV(
        XGBClassifier(random_state=rs, eval_metric="aucpr",
                      tree_method="hist", n_jobs=1),
        xgb_dist, n_iter=25, scoring="average_precision",
        cv=cv, n_jobs=2, random_state=rs, verbose=0,
    )
    xgb_search.fit(X_train, y_train)
    xgb_best = xgb_search.best_estimator_
    proba_val = xgb_best.predict_proba(X_val)[:, 1]
    thr, _ = best_threshold_f1(y_val, proba_val)
    val_m = evaluate(y_val, proba_val, thr)
    test_m = evaluate(y_test, xgb_best.predict_proba(X_test)[:, 1], thr)
    cm_path = save_confusion_matrix(y_test, (xgb_best.predict_proba(X_test)[:, 1] >= thr).astype(int), "xgboost")
    rid = log_run("xgboost", "gradient-boosting", xgb_best, val_m, test_m,
                  {**xgb_search.best_params_, "random_state": rs},
                  cm_path, log_fn=mlflow.xgboost.log_model)
    leaderboard.append({"run_name": "xgboost", "family": "gradient-boosting",
                        "val": val_m, "test": test_m, "run_id": rid,
                        "best_params": xgb_search.best_params_})
    print(f"   best: {xgb_search.best_params_}")
    print(f"   val PR-AUC={val_m['pr_auc']:.4f}  test PR-AUC={test_m['pr_auc']:.4f}  thr={thr:.3f}\n")

    # ============================================================ TabNet
    print("[4/4] TabNet — retrain dengan params.yaml + threshold tuning...")
    tn_cfg = params["tabnet"]
    weights = {0: 1.0, 1: spw}  # class weighting
    tn = TabNetClassifier(
        n_d=tn_cfg["n_d"], n_a=tn_cfg["n_a"], n_steps=tn_cfg["n_steps"],
        gamma=tn_cfg["gamma"], lambda_sparse=tn_cfg["lambda_sparse"],
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=2,
        optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
        scheduler_params=dict(step_size=20, gamma=0.9),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=rs, verbose=0,
    )
    tn.fit(
        X_train.astype(np.float32), y_train.astype(np.int64),
        eval_set=[(X_val.astype(np.float32), y_val.astype(np.int64))],
        eval_name=["val"], eval_metric=["auc"],
        max_epochs=tn_cfg["max_epochs"], patience=tn_cfg["patience"],
        batch_size=tn_cfg["batch_size"], virtual_batch_size=tn_cfg["virtual_batch_size"],
        weights=weights,
    )
    proba_val = tn.predict_proba(X_val.astype(np.float32))[:, 1]
    thr, _ = best_threshold_f1(y_val, proba_val)
    proba_test = tn.predict_proba(X_test.astype(np.float32))[:, 1]
    val_m = evaluate(y_val, proba_val, thr)
    test_m = evaluate(y_test, proba_test, thr)
    cm_path = save_confusion_matrix(y_test, (proba_test >= thr).astype(int), "tabnet")
    # Simpan TabNet sebagai zip
    tn_save = MODELS_DIR / "tabnet" / "tabnet_model"
    tn_save.parent.mkdir(parents=True, exist_ok=True)
    tn.save_model(str(tn_save))
    with mlflow.start_run(run_name="tabnet") as run:
        mlflow.set_tag("model_family", "deep-tabular")
        mlflow.set_tag("stage", "tuned")
        mlflow.log_params({f"tabnet_{k}": v for k, v in tn_cfg.items()})
        mlflow.log_param("random_state", rs)
        for k, v in val_m.items():  mlflow.log_metric(f"val_{k}", v)
        for k, v in test_m.items(): mlflow.log_metric(f"test_{k}", v)
        mlflow.log_artifact(str(cm_path), artifact_path="confusion_matrix")
        mlflow.log_artifact(f"{tn_save}.zip", artifact_path="tabnet_model")
        rid = run.info.run_id
    leaderboard.append({"run_name": "tabnet", "family": "deep-tabular",
                        "val": val_m, "test": test_m, "run_id": rid,
                        "best_params": tn_cfg})
    print(f"   val PR-AUC={val_m['pr_auc']:.4f}  test PR-AUC={test_m['pr_auc']:.4f}  thr={thr:.3f}\n")

    # ============================================================ leaderboard & best
    leaderboard.sort(key=lambda r: r["val"]["pr_auc"], reverse=True)
    print("=" * 70)
    print("LEADERBOARD (sorted by val PR-AUC)")
    print("=" * 70)
    print(f"{'model':<16} {'val_PR':>8} {'val_F1':>8} {'val_Rec':>8} {'test_PR':>8} {'test_F1':>8} {'test_Rec':>8} {'thr':>6}")
    for r in leaderboard:
        v, t = r["val"], r["test"]
        print(f"{r['run_name']:<16} {v['pr_auc']:>8.4f} {v['f1']:>8.4f} {v['recall']:>8.4f} "
              f"{t['pr_auc']:>8.4f} {t['f1']:>8.4f} {t['recall']:>8.4f} {t['threshold']:>6.3f}")

    best = leaderboard[0]
    print(f"\n★ BEST MODEL: {best['run_name']} ({best['family']})  run_id={best['run_id']}")

    # Save summary
    summary = {
        "experiment": params["mlflow"]["experiment_name"],
        "primary_metric": "val_pr_auc",
        "leaderboard": [
            {"run_name": r["run_name"], "family": r["family"], "run_id": r["run_id"],
             "val": r["val"], "test": r["test"],
             "best_params": {k: (v if not isinstance(v, (np.integer, np.floating)) else v.item())
                             for k, v in r["best_params"].items()}}
            for r in leaderboard
        ],
    }
    with open(METRICS_DIR / "05_modeling_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save best model artifact ke models/best_model/
    import shutil
    target = MODELS_DIR / "best_model"
    if target.exists(): shutil.rmtree(target)
    target.mkdir(parents=True)

    if best["family"] == "deep-tabular":
        shutil.copy(f"{tn_save}.zip", target / "tabnet_model.zip")
        joblib.dump({"threshold": best["test"]["threshold"]}, target / "threshold.joblib")
        artifact_rel = "models/best_model"
    else:
        # cari pkl di mlruns
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        run = client.get_run(best["run_id"])
        local = mlflow.artifacts.download_artifacts(f"runs:/{best['run_id']}/model")
        for f in Path(local).iterdir():
            if f.is_file(): shutil.copy(f, target / f.name)
        joblib.dump({"threshold": best["test"]["threshold"]}, target / "threshold.joblib")
        artifact_rel = "models/best_model"

    cm_best = FIG_DIR / f"cm_{best['run_name']}_test.png"
    if cm_best.exists():
        shutil.copy(cm_best, target / "confusion_matrix_test.png")

    best_meta = {
        "run_id": best["run_id"], "run_name": best["run_name"], "family": best["family"],
        "primary_metric": "val_pr_auc",
        "threshold": best["test"]["threshold"],
        "val_metrics": best["val"], "test_metrics": best["test"],
        "best_params": summary["leaderboard"][0]["best_params"],
        "artifact_local_path": artifact_rel,
        "tabnet_artifact_path": "models/tabnet",
    }
    with open(MODELS_DIR / "best_model.json", "w") as f:
        json.dump(best_meta, f, indent=2, default=str)

    print(f"\n✓ Summary  : {METRICS_DIR / '05_modeling_summary.json'}")
    print(f"✓ Best model artifact: {target}")
    print(f"✓ Best meta: {MODELS_DIR / 'best_model.json'}")


if __name__ == "__main__":
    main()
