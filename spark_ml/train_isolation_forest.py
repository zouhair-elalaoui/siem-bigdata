"""
spark_ml/train_isolation_forest.py
=====================================
Modèle 2 — Isolation Forest
  • Entraîné UNIQUEMENT sur le trafic NORMAL (semi-supervisé)
    → détecte les attaques comme des anomalies
  • Seuil de décision calibré sur un split de validation (pas sur le test)
  • Approche réaliste pour un SIEM : on ne connaît pas toujours les attaques à l'avance

Usage :
  python3 train_isolation_forest.py
  python3 train_isolation_forest.py --data-dir /chemin/vers/data
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

from sklearn.ensemble import IsolationForest
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
MODEL_NAME    = "IsolationForest"


def load_data():
    X_train = np.load(ARTIFACTS_DIR / "X_train.npy")
    X_test  = np.load(ARTIFACTS_DIR / "X_test.npy")
    y_train = np.load(ARTIFACTS_DIR / "y_train.npy").astype(int)
    y_test  = np.load(ARTIFACTS_DIR / "y_test.npy").astype(int)
    with open(ARTIFACTS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    return X_train, X_test, y_train, y_test, feature_names


# ─────────────────────── ENTRAÎNEMENT ────────────────────────────
def train(X_train, y_train):
    print(f"\n{'='*60}")
    print(f"  ENTRAÎNEMENT — {MODEL_NAME}")
    print(f"{'='*60}")

    # Stratégie semi-supervisée : entraîner sur le trafic NORMAL uniquement
    normal_mask = (y_train == 0)
    X_normal    = X_train[normal_mask]
    n_attack    = (y_train == 1).sum()
    n_total     = len(y_train)
    contamination = min(n_attack / n_total * 1.2, 0.45)  # légèrement surestimé

    print(f"  Train total     : {X_train.shape}")
    print(f"  Entraîné sur    : {X_normal.shape[0]:,} exemples NORMAUX uniquement")
    print(f"  Contamination   : {contamination:.4f}")

    model = IsolationForest(
        n_estimators  = 200,
        max_samples   = min(256, len(X_normal)),  # valeur standard optimale
        contamination = contamination,
        max_features  = 1.0,
        bootstrap     = False,
        n_jobs        = -1,
        random_state  = 42,
    )

    t0 = time.time()
    model.fit(X_normal)
    elapsed = time.time() - t0
    print(f"  ✓ Entraînement : {elapsed:.1f}s")
    return model


# ─────────────────────── CALIBRATION DU SEUIL ────────────────────
def calibrate_threshold(model, X_train, y_train):
    """
    Cherche le seuil d'anomalie qui maximise le F1 sur un split de validation
    (séparé du test — pas de data leakage).
    """
    print(f"\n[CALIBRATION] Recherche du seuil optimal sur validation set...")

    # Split validation depuis le train (15%)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        stratify=y_train,
        random_state=42,
    )

    # Score d'anomalie : plus élevé = plus anormal
    scores_val = -model.score_samples(X_val)

    # Tester des seuils allant du 5e au 95e percentile
    percentiles = np.linspace(5, 95, 90)
    thresholds  = np.percentile(scores_val, percentiles)

    best_f1, best_thresh = 0.0, np.median(scores_val)

    for thresh in thresholds:
        preds = (scores_val >= thresh).astype(int)
        f1    = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = thresh

    # Afficher la courbe résumée
    print(f"  Seuil optimal : {best_thresh:.4f}  (F1 validation = {best_f1:.4f})")
    return best_thresh


# ─────────────────────── ÉVALUATION ──────────────────────────────
def evaluate(model, X_test, y_test, threshold):
    print(f"\n{'='*60}")
    print(f"  ÉVALUATION SUR TEST SET — {MODEL_NAME}")
    print(f"{'='*60}")

    t0             = time.time()
    scores_test    = -model.score_samples(X_test)  # plus élevé = plus anormal
    y_pred         = (scores_test >= threshold).astype(int)
    inf_ms         = (time.time() - t0) / len(X_test) * 1000

    prec   = precision_score(y_test, y_pred, zero_division=0)
    rec    = recall_score(y_test, y_pred, zero_division=0)
    f1     = f1_score(y_test, y_pred, zero_division=0)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    auc    = roc_auc_score(y_test, scores_test)
    ap     = average_precision_score(y_test, scores_test)
    cm     = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr    = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n  Seuil utilisé         : {threshold:.4f}")
    print(f"  Précision (Precision) : {prec:.4f}")
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

    # Analyse de séparation des distributions de scores
    att_scores    = scores_test[y_test == 1]
    norm_scores   = scores_test[y_test == 0]
    separation    = att_scores.mean() - norm_scores.mean()
    print(f"  Score moyen ATTACK : {att_scores.mean():.4f}  ±{att_scores.std():.4f}")
    print(f"  Score moyen NORMAL : {norm_scores.mean():.4f}  ±{norm_scores.std():.4f}")
    print(f"  Séparation         : {separation:.4f}  "
          f"{'✓ Bonne' if separation > 0.1 else '⚠ Faible (modèle non-supervisé)'}")

    results = {
        "model": MODEL_NAME,
        "threshold": round(threshold, 4),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1_binary": round(f1, 4),   "f1_macro": round(f1_mac, 4),
        "auc_roc": round(auc, 4),    "avg_precision": round(ap, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "inference_ms_per_sample": round(inf_ms, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                              "fn": int(fn), "tp": int(tp)},
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

    model     = train(X_train, y_train)
    threshold = calibrate_threshold(model, X_train, y_train)
    results   = evaluate(model, X_test, y_test, threshold)

    # Sauvegarder modèle + seuil ensemble
    artifact = {"model": model, "threshold": threshold}
    model_path = MODELS_DIR / "isolation_forest.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\n  Modèle + seuil sauvegardés → {model_path}")
    print(f"\n✅ {MODEL_NAME} — terminé")
