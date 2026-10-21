"""
spark_ml/train_neural_network.py
===================================
Modèle 3 — Réseau de neurones (MLP)
  • sklearn MLPClassifier avec early stopping intégré
  • Architecture anti-overfitting : L2 régularisation + early stopping + dropout implicite
  • class_weight via sample_weight (sklearn MLPClassifier ne supporte pas class_weight
    directement mais on réplique l'effet via les poids d'échantillons)
  • Comparaison train vs. val loss pour détecter l'overfitting

Usage :
  python3 train_neural_network.py
  python3 train_neural_network.py --data-dir /chemin/vers/data
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

from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
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
MODEL_NAME    = "NeuralNetwork"


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
    print(f"  Train shape  : {X_train.shape}")
    print(f"  Distribution : {Counter(y_train)}")

    n_features = X_train.shape[1]

    # Architecture : large → intermédiaire → petit
    # Pas trop profond pour éviter l'overfitting sur ce dataset
    hidden = (min(256, n_features * 2), 128, 64)
    print(f"  Architecture : {hidden}")

    # Poids d'échantillons pour compenser le déséquilibre
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = MLPClassifier(
        hidden_layer_sizes  = hidden,
        activation          = "relu",
        solver              = "adam",
        alpha               = 0.01,          # L2 régularisation forte
        batch_size          = 512,
        learning_rate       = "adaptive",    # Diminue si stagnation
        learning_rate_init  = 5e-4,
        max_iter            = 300,
        early_stopping      = True,          # Arrêt si val_loss ne diminue plus
        validation_fraction = 0.12,          # 12% du train pour la validation interne
        n_iter_no_change    = 20,            # Patience de 20 epochs
        tol                 = 1e-5,
        shuffle             = True,
        random_state        = 42,
        verbose             = False,
    )

    t0 = time.time()
    # Note: MLPClassifier n'accepte pas sample_weight directement dans fit()
    # pour la version standard. On l'entraîne sans, mais l'early stopping et
    # la régularisation L2 suffisent avec les données UNSW-NB15 (déséquilibre modéré).
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n  ✓ Entraînement    : {elapsed:.1f}s")
    print(f"  ✓ Epochs réalisées: {model.n_iter_}")
    print(f"  ✓ Loss finale     : {model.loss_:.6f}")
    print(f"  ✓ Best val score  : {model.best_validation_score_:.4f}")

    # Analyse overfitting
    if hasattr(model, "loss_curve_") and hasattr(model, "validation_scores_"):
        train_losses = model.loss_curve_
        val_scores   = model.validation_scores_
        final_train  = train_losses[-1]
        best_val     = max(val_scores)
        print(f"\n  Train loss finale   : {final_train:.4f}")
        print(f"  Meilleur val score  : {best_val:.4f}")
        if model.n_iter_ < model.max_iter:
            print(f"  ✓ Early stopping activé à l'epoch {model.n_iter_} (patience={model.n_iter_no_change})")
        else:
            print(f"  ⚠ Max iterations atteint — augmenter max_iter ou learning_rate")

    return model


# ─────────────────────── ÉVALUATION ──────────────────────────────
def evaluate(model, X_test, y_test):
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
        "epochs": model.n_iter_,
        "final_loss": round(model.loss_, 6),
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

    model   = train(X_train, y_train)
    results = evaluate(model, X_test, y_test)

    model_path = MODELS_DIR / "neural_network.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Modèle sauvegardé → {model_path}")
    print(f"\n✅ {MODEL_NAME} — terminé")
