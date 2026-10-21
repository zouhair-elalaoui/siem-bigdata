"""
spark_ml/train_random_forest.py
=================================
Modèle 1 — Random Forest
  • class_weight='balanced' pour gérer le déséquilibre
  • Hyperparamètres réglés pour éviter l'overfitting (max_depth, min_samples_leaf)
  • OOB score comme estimation interne gratuite
  • Validation croisée stratifiée sur le train pour vérifier la stabilité
  • Évaluation finale sur le TEST SET officiel UNSW-NB15

Usage :
  python3 train_random_forest.py
  python3 train_random_forest.py --data-dir /chemin/vers/data
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score,
)

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))
MODELS_DIR    = ARTIFACTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME    = "RandomForest"


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
    print(f"  Train shape : {X_train.shape}")
    print(f"  Distribution: {Counter(y_train)}")

    model = RandomForestClassifier(
        n_estimators     = 300,          # Assez d'arbres pour la stabilité
        max_depth        = 25,           # Limite la complexité → anti-overfitting
        min_samples_leaf = 4,            # Chaque feuille ≥ 4 exemples
        min_samples_split= 8,            # Splitter seulement si ≥ 8 exemples
        max_features     = "sqrt",       # Réduire la corrélation entre arbres
        class_weight     = "balanced",   # Gestion automatique du déséquilibre
        oob_score        = True,         # Score out-of-bag (généralisation estimée)
        bootstrap        = True,
        n_jobs           = -1,
        random_state     = 42,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n  ✓ Entraînement : {elapsed:.1f}s")
    print(f"  ✓ OOB Score    : {model.oob_score_:.4f}  "
          f"← estimation sur données non vues pendant l'entraînement")
    return model


# ─────────────────────── VALIDATION CROISÉE ──────────────────────
def cross_validate(model, X_train, y_train, cv=5):
    """
    CV stratifiée sur le jeu d'entraînement.
    Sert à détecter l'overfitting (si score CV << score train → surapprentissage).
    On limite à 30 000 exemples pour la rapidité.
    """
    print(f"\n[CV] Validation croisée ({cv} folds stratifiés)...")
    MAX_SAMPLES = 30_000
    if len(X_train) > MAX_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), MAX_SAMPLES, replace=False)
        Xc, yc = X_train[idx], y_train[idx]
    else:
        Xc, yc = X_train, y_train

    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, Xc, yc, cv=skf, scoring="f1", n_jobs=-1)

    print(f"  F1 par fold : {[f'{s:.4f}' for s in scores]}")
    print(f"  Moyenne     : {scores.mean():.4f}  ±{scores.std():.4f}")
    if scores.std() > 0.03:
        print("  ⚠ Variance élevée entre folds — vérifier la stabilité du modèle")
    else:
        print("  ✓ Variance faible — modèle stable")
    return scores


# ─────────────────────── ÉVALUATION ──────────────────────────────
def evaluate(model, X_test, y_test, feature_names):
    print(f"\n{'='*60}")
    print(f"  ÉVALUATION SUR TEST SET — {MODEL_NAME}")
    print(f"{'='*60}")

    t0    = time.time()
    y_pred = model.predict(X_test)
    inf_ms = (time.time() - t0) / len(X_test) * 1000  # ms par sample

    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Métriques ──
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

    # ── Affichage ──
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

    # ── Top features ──
    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:15]
    print(f"  Top 15 features importantes :")
    for i in top_idx:
        bar = "█" * int(importances[i] * 300)
        print(f"    {feature_names[i]:35s} {importances[i]:.4f}  {bar}")

    # ── Sauvegarde résultats ──
    results = {
        "model": MODEL_NAME,
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


# ─────────────────────── MAIN ────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Sauter la validation croisée (plus rapide)")
    args = parser.parse_args()

    # Lancer preprocessing si les artefacts sont absents
    if not (ARTIFACTS_DIR / "X_train.npy").exists():
        print("[INFO] Artefacts absents → lancement du preprocessing...")
        import subprocess, sys
        subprocess.run([sys.executable, "preprocessing.py",
                        "--data-dir", args.data_dir], check=True)

    X_train, X_test, y_train, y_test, feature_names = load_data()

    model   = train(X_train, y_train)

    if not args.skip_cv:
        cross_validate(model, X_train, y_train)

    results = evaluate(model, X_test, y_test, feature_names)

    # Sauvegarde du modèle
    model_path = MODELS_DIR / "random_forest.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Modèle sauvegardé → {model_path}")
    print(f"\n✅ {MODEL_NAME} — terminé")
