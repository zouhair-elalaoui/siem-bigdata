"""
spark_ml/evaluate_models.py
==============================
Comparaison complète de tous les modèles entraînés.
  • Charge les modèles sauvegardés (pas de ré-entraînement)
  • Évalue sur le TEST SET officiel UNSW-NB15
  • Tableau de comparaison : précision, rappel, F1, AUC-ROC, temps d'inférence
  • Recommandation automatique du meilleur modèle pour un SIEM

Usage :
  python3 evaluate_models.py
  python3 evaluate_models.py --data-dir /chemin/vers/data
"""

import os
import json
import time
import pickle
import argparse
import warnings
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score,
)

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))
MODELS_DIR    = ARTIFACTS_DIR / "models"


def load_test_data():
    X_test = np.load(ARTIFACTS_DIR / "X_test.npy")
    y_test = np.load(ARTIFACTS_DIR / "y_test.npy").astype(int)
    with open(ARTIFACTS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    return X_test, y_test, feature_names


# ─────────────────────── CHARGEMENT DES MODÈLES ──────────────────
def load_all_models():
    """Charge tous les modèles présents dans le dossier models/."""
    models = {}

    model_files = {
        "RandomForest":    "random_forest.pkl",
        "IsolationForest": "isolation_forest.pkl",
        "NeuralNetwork":   "neural_network.pkl",
        "XGBoost":         "xgboost.pkl",
    }

    for name, filename in model_files.items():
        path = MODELS_DIR / filename
        if path.exists():
            with open(path, "rb") as f:
                obj = pickle.load(f)
            # L'Isolation Forest est sauvegardé avec son seuil
            if isinstance(obj, dict) and "model" in obj:
                models[name] = obj  # {"model": ..., "threshold": ...}
            else:
                models[name] = {"model": obj, "threshold": None}
            print(f"  ✓ Chargé : {name}")
        else:
            print(f"  ✗ Absent : {name}  (lancer train_{filename.replace('.pkl','')}.py)")

    return models


# ─────────────────────── ÉVALUATION D'UN MODÈLE ──────────────────
def evaluate_one(name, model_obj, X_test, y_test):
    model     = model_obj["model"]
    threshold = model_obj["threshold"]

    t0 = time.time()

    if name == "IsolationForest":
        # Isolation Forest utilise des scores d'anomalie + seuil calibré
        scores    = -model.score_samples(X_test)
        y_pred    = (scores >= threshold).astype(int)
        y_proba   = scores
    else:
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    inf_ms = (time.time() - t0) / len(X_test) * 1000

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

    return {
        "model":     name,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "f1_macro":  f1_mac,
        "auc_roc":   auc,
        "avg_prec":  ap,
        "fpr":       fpr,
        "fnr":       fnr,
        "inf_ms":    inf_ms,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "y_pred":    y_pred,
    }


# ─────────────────────── TABLEAU COMPARATIF ──────────────────────
def print_comparison_table(all_results):
    print(f"\n{'='*100}")
    print(f"  TABLEAU COMPARATIF — TEST SET OFFICIEL UNSW-NB15")
    print(f"{'='*100}")

    header = (f"  {'Modèle':<20} {'Précision':>10} {'Rappel':>10} "
              f"{'F1':>10} {'F1-macro':>10} {'AUC-ROC':>10} "
              f"{'FP%':>8} {'FN%':>8} {'ms/sample':>12}")
    print(header)
    print(f"  {'-'*96}")

    for r in all_results:
        line = (f"  {r['model']:<20} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                f"{r['f1']:>10.4f} {r['f1_macro']:>10.4f} {r['auc_roc']:>10.4f} "
                f"{r['fpr']*100:>7.2f}% {r['fnr']*100:>7.2f}% {r['inf_ms']:>11.4f}")
        print(line)

    print(f"  {'-'*96}")
    print(f"\n  Colonnes : Précision=VP/(VP+FP) | Rappel=VP/(VP+FN) | "
          f"FP%=Faux Positifs | FN%=Faux Négatifs")


# ─────────────────────── RECOMMANDATION ──────────────────────────
def recommend(all_results):
    """
    Dans un SIEM, le rappel est plus important que la précision
    (mieux vaut trop d'alertes que des attaques manquées).
    Mais un F1 élevé assure l'équilibre.
    On recommande sur la base du F1 binaire + AUC.
    """
    print(f"\n{'='*60}")
    print(f"  RECOMMANDATION POUR LE SIEM")
    print(f"{'='*60}")

    # Classer par F1 + AUC combinés
    scored = [(r, 0.5 * r["f1"] + 0.5 * r["auc_roc"]) for r in all_results]
    scored.sort(key=lambda x: x[1], reverse=True)

    best = scored[0][0]
    print(f"\n  🏆 Meilleur modèle global : {best['model']}")
    print(f"     F1={best['f1']:.4f}  AUC={best['auc_roc']:.4f}  "
          f"FP={best['fpr']*100:.1f}%  FN={best['fnr']*100:.1f}%")

    # Meilleur rappel (moins de Faux Négatifs = moins d'attaques manquées)
    best_recall = max(all_results, key=lambda r: r["recall"])
    print(f"\n  🎯 Meilleur rappel (SIEM : minimiser les attaques manquées) :")
    print(f"     {best_recall['model']}  Rappel={best_recall['recall']:.4f}  "
          f"FN={best_recall['fnr']*100:.1f}%")

    # Plus rapide (temps réel)
    fastest = min(all_results, key=lambda r: r["inf_ms"])
    print(f"\n  ⚡ Plus rapide (temps réel Spark Streaming) :")
    print(f"     {fastest['model']}  {fastest['inf_ms']:.4f} ms/sample")

    print(f"\n  Conseil d'architecture :")
    print(f"     → En production : {best['model']} pour la détection principale")
    print(f"     → En parallèle  : IsolationForest pour les anomalies inconnues (zero-day)")
    print(f"     → Seuil d'alerte : rappel ≥ 0.90 est recommandé pour un SIEM")


# ─────────────────────── RAPPORT DÉTAILLÉ ────────────────────────
def print_detailed_reports(all_results, X_test, y_test, models):
    print(f"\n{'='*60}")
    print(f"  RAPPORTS DÉTAILLÉS PAR MODÈLE")
    print(f"{'='*60}")

    for r in all_results:
        print(f"\n  ── {r['model']} ──")
        print(f"     Matrice de confusion :")
        print(f"                 Prédit NORMAL  Prédit ATTACK")
        print(f"     Réel NORMAL    {r['tn']:>8,}       {r['fp']:>8,}")
        print(f"     Réel ATTACK    {r['fn']:>8,}       {r['tp']:>8,}")
        print(f"\n     Rapport de classification :")
        print(classification_report(y_test, r["y_pred"],
                                     target_names=["NORMAL", "ATTACK"],
                                     zero_division=0,
                                     indent=5))


# ─────────────────────── SAUVEGARDE JSON ─────────────────────────
def save_comparison(all_results):
    export = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k != "y_pred"}
        row["precision"] = round(row["precision"], 4)
        row["recall"]    = round(row["recall"], 4)
        row["f1"]        = round(row["f1"], 4)
        row["f1_macro"]  = round(row["f1_macro"], 4)
        row["auc_roc"]   = round(row["auc_roc"], 4)
        row["avg_prec"]  = round(row["avg_prec"], 4)
        row["fpr"]       = round(row["fpr"], 4)
        row["fnr"]       = round(row["fnr"], 4)
        row["inf_ms"]    = round(row["inf_ms"], 4)
        export.append(row)
    path = ARTIFACTS_DIR / "comparison_results.json"
    with open(path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\n  Comparaison sauvegardée → {path}")


# ─────────────────────────── MAIN ────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--detailed", action="store_true",
                        help="Afficher les rapports détaillés par modèle")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ÉVALUATION COMPARATIVE — UNSW-NB15")
    print(f"{'='*60}")

    # Preprocessing si nécessaire
    if not (ARTIFACTS_DIR / "X_test.npy").exists():
        import subprocess, sys
        subprocess.run([sys.executable, "preprocessing.py",
                        "--data-dir", args.data_dir], check=True)

    X_test, y_test, feature_names = load_test_data()
    print(f"\n  Test set : {X_test.shape}  |  Labels : {dict(zip(*np.unique(y_test, return_counts=True)))}")

    print(f"\n[LOAD] Chargement des modèles...")
    models = load_all_models()

    if not models:
        print("\n❌ Aucun modèle trouvé. Lancez d'abord les scripts train_*.py")
        exit(1)

    # Évaluation de chaque modèle
    print(f"\n[EVAL] Évaluation de {len(models)} modèle(s) sur le test set...")
    all_results = []
    for name, model_obj in models.items():
        print(f"  → {name}...")
        result = evaluate_one(name, model_obj, X_test, y_test)
        all_results.append(result)

    # Affichages
    print_comparison_table(all_results)
    recommend(all_results)

    if args.detailed:
        print_detailed_reports(all_results, X_test, y_test, models)

    save_comparison(all_results)

    print(f"\n✅ evaluate_models.py — terminé")
