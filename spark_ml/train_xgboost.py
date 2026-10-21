"""
spark_ml/train_xgboost.py
============================
Modèle 4 — XGBoost
  • scale_pos_weight pour gérer le déséquilibre NORMAL/ATTACK
  • early_stopping_rounds sur un validation set (anti-overfitting)
  • Hyperparamètres réglés pour éviter la mémorisation des données
  • Feature importance (gain, cover, weight)

Usage :
  python3 train_xgboost.py
  python3 train_xgboost.py --data-dir /chemin/vers/data
"""

import os
import json
import time
import pickle
import argparse
import warnings
import numpy as np
from pathlib import Path
from collections import Counter

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score,
)

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))
MODELS_DIR    = ARTIFACTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME    = "XGBoost"


def load_data():
    X_train = np.load(ARTIFACTS_DIR / "X_train.npy")
    X_test  = np.load(ARTIFACTS_DIR / "X_test.npy")
    y_train = np.load(ARTIFACTS_DIR / "y_train.npy").astype(int)
    y_test  = np.load(ARTIFACTS_DIR / "y_test.npy").astype(int)
    with open(ARTIFACTS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    return X_train, X_test, y_train, y_test, feature_names


# ─────────────────────── ENTRAÎNEMENT ────────────────────────────
def train(X_train, y_train, feature_names):
    print(f"\n{'='*60}")
    print(f"  ENTRAÎNEMENT — {MODEL_NAME}")
    print(f"{'='*60}")
    print(f"  Train shape  : {X_train.shape}")

    counts        = Counter(y_train)
    n_normal      = counts[0]
    n_attack      = counts[1]
    scale_pos_weight = n_normal / n_attack  # compenser le déséquilibre
    print(f"  NORMAL: {n_normal:,}  ATTACK: {n_attack:,}")
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}")

    # Split validation pour early stopping (pas le test set → pas de data leakage)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        stratify=y_train,
        random_state=42,
    )
    print(f"  Train effectif: {X_tr.shape[0]:,}  Val: {X_val.shape[0]:,}")

    model = xgb.XGBClassifier(
        # ── Architecture ──
        n_estimators        = 3000,          # Large plafond — early stopping arrête au bon moment
        max_depth           = 6,             # Clé anti-overfitting (6 est standard)
        min_child_weight    = 5,             # Feuille ≥ 5 exemples pondérés
        gamma               = 0.1,           # Gain minimum pour splitter (pruning)

        # ── Régularisation ──
        subsample           = 0.8,           # 80% des lignes par arbre (bagging)
        colsample_bytree    = 0.8,           # 80% des features par arbre
        reg_alpha           = 0.1,           # L1 regularisation
        reg_lambda          = 1.0,           # L2 regularisation

        # ── Déséquilibre ──
        scale_pos_weight    = scale_pos_weight,

        # ── Apprentissage ──
        learning_rate       = 0.1,
        objective           = "binary:logistic",
        # aucpr (PR curve) → plus sensible au déséquilibre, déclenche early stopping efficacement
        eval_metric         = "aucpr",

        # ── Performance ──
        tree_method         = "hist",        # Rapide même sans GPU
        n_jobs              = -1,
        random_state        = 42,
        verbosity           = 0,
        # early_stopping_rounds dans le constructeur (XGBoost ≥ 2.0)
        early_stopping_rounds = 30,
    )

    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set = [(X_tr, y_tr), (X_val, y_val)],
        verbose  = False,
    )
    elapsed = time.time() - t0

    best_iter  = model.best_iteration
    best_score = model.best_score
    print(f"\n  ✓ Entraînement     : {elapsed:.1f}s")
    print(f"  ✓ Meilleur iter    : {best_iter} / {1000}")
    print(f"  ✓ Meilleur AUC val : {best_score:.4f}")

    if best_iter < 50:
        print("  ⚠ Early stopping très précoce — essayer un learning_rate plus élevé")
    elif best_iter > 900:
        print("  ⚠ N'a pas convergé — augmenter n_estimators")
    else:
        print(f"  ✓ Convergence saine à l'itération {best_iter}")

    # Vérification overfitting : AUC train vs AUC val au meilleur iter
    evals_result = model.evals_result()
    if evals_result:
        metric_key = "aucpr"
        train_score = evals_result["validation_0"][metric_key][best_iter]
        val_score   = evals_result["validation_1"][metric_key][best_iter]
        gap         = train_score - val_score
        print(f"\n  AUCPR train (best iter) : {train_score:.4f}")
        print(f"  AUCPR val   (best iter) : {val_score:.4f}")
        print(f"  Gap train-val           : {gap:.4f}  "
              f"{'✓ Pas d overfitting' if gap < 0.02 else '⚠ Léger overfitting'}")

        # Sauvegarder la courbe pour visualisation
        history = {
            "train_aucpr": evals_result["validation_0"][metric_key],
            "val_aucpr":   evals_result["validation_1"][metric_key],
            "best_iter": best_iter,
        }
        with open(ARTIFACTS_DIR / "xgboost_training_history.json", "w") as f:
            json.dump(history, f)

    return model


# ─────────────────────── ÉVALUATION ──────────────────────────────
def evaluate(model, X_test, y_test, feature_names):
    print(f"\n{'='*60}")
    print(f"  ÉVALUATION SUR TEST SET — {MODEL_NAME}")
    print(f"{'='*60}")

    t0     = time.time()
    y_pred = model.predict(X_test)
    inf_ms = (time.time() - t0) / len(X_test) * 1000

    y_proba = model.predict_proba(X_test)[:, 1]

    prec   = precision_score(y_test, y_pred, zero_division=0)
    rec    = recall_score(y_test, y_pred, zero_division=0)
    f1     = f1_score(y_test, y_pred, zero_division=0)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    auc    = roc_auc_score(y_test, y_proba)
    ap     = average_precision_score(y_test, y_proba)
    cm     = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr    = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n  Précision (Precision) : {prec:.4f}")
    print(f"  Rappel    (Recall)    : {rec:.4f}")
    print(f"  F1 binaire            : {f1:.4f}")
    print(f"  F1 macro              : {f1_mac:.4f}")
    print(f"  AUC-ROC               : {auc:.4f}")
    print(f"  Average Precision     : {ap:.4f}")
    print(f"  Taux Faux Positifs    : {fpr*100:.2f}%")
    print(f"  Taux Faux Négatifs    : {fnr*100:.2f}%")
    print(f"  Temps inférence       : {inf_ms:.4f} ms/sample")
    print(f"\n  Matrice de confusion :")
    print(f"              Prédit NORMAL  Prédit ATTACK")
    print(f"  Réel NORMAL    {tn:>8,}       {fp:>8,}")
    print(f"  Réel ATTACK    {fn:>8,}       {tp:>8,}")
    print(f"\n  Rapport complet :")
    print(classification_report(y_test, y_pred,
                                 target_names=["NORMAL", "ATTACK"],
                                 zero_division=0))

    # ── Feature importance (par gain = impact sur la réduction de l'erreur) ──
    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:15]
    print(f"  Top 15 features (gain) :")
    for i in top_idx:
        bar = "█" * int(importances[i] * 300)
        print(f"    {feature_names[i]:35s} {importances[i]:.4f}  {bar}")

    results = {
        "model": MODEL_NAME,
        "best_iteration": int(model.best_iteration),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1_binary": round(f1, 4),   "f1_macro": round(f1_mac, 4),
        "auc_roc": round(auc, 4),    "avg_precision": round(ap, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "inference_ms_per_sample": round(inf_ms, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                              "fn": int(fn), "tp": int(tp)},
        "top_features": [feature_names[i] for i in top_idx],
    }
    with open(ARTIFACTS_DIR / f"{MODEL_NAME}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Résultats → {ARTIFACTS_DIR}/{MODEL_NAME}_results.json")
    return results


# ─────────────────────────── MAIN ────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data")
    args = parser.parse_args()

    if not (ARTIFACTS_DIR / "X_train.npy").exists():
        import subprocess, sys
        subprocess.run([sys.executable, "preprocessing.py",
                        "--data-dir", args.data_dir], check=True)

    X_train, X_test, y_train, y_test, feature_names = load_data()

    model   = train(X_train, y_train, feature_names)
    results = evaluate(model, X_test, y_test, feature_names)

    model_path = MODELS_DIR / "xgboost.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Modèle sauvegardé → {model_path}")
    print(f"\n✅ {MODEL_NAME} — terminé")
