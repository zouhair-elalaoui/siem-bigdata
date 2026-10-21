"""
spark_ml/preprocessing.py
===========================
Prétraitement du dataset UNSW-NB15 pour la détection d'intrusions.

Ce script :
  1. Charge les fichiers train/test officiels (split déjà fourni par les auteurs)
  2. Nettoie et encode les colonnes catégorielles (proto, service, state)
  3. Normalise avec RobustScaler (résistant aux outliers)
  4. Sauvegarde tous les artefacts (scaler, encodeurs, noms de features)
     pour réutilisation dans Spark et dans les scripts de modèles

Usage :
  python3 preprocessing.py
  python3 preprocessing.py --data-dir /chemin/vers/data
"""

import os
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")

# ─────────────────────────── CHEMINS ─────────────────────────────
DATA_DIR      = Path(os.getenv("DATA_DIR", "../data"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE  = DATA_DIR / "UNSW_NB15_testing-set.csv"

# Colonnes à supprimer (identifiants + cible secondaire)
DROP_COLS   = ["id", "attack_cat"]
TARGET_COL  = "label"
# Colonnes catégorielles à encoder
CAT_COLS    = ["proto", "service", "state"]


# ─────────────────────────── CHARGEMENT ──────────────────────────
def load_data(data_dir: Path = DATA_DIR):
    train_path = data_dir / "UNSW_NB15_training-set.csv"
    test_path  = data_dir / "UNSW_NB15_testing-set.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Fichiers introuvables dans {data_dir}.\n"
            "Attendu : UNSW_NB15_training-set.csv et UNSW_NB15_testing-set.csv"
        )

    print(f"[LOAD] Chargement train : {train_path}")
    train = pd.read_csv(train_path)
    train.columns = train.columns.str.strip()

    print(f"[LOAD] Chargement test  : {test_path}")
    test  = pd.read_csv(test_path)
    test.columns = test.columns.str.strip()

    print(f"  Train : {train.shape}  |  Test : {test.shape}")

    # Afficher la distribution des classes
    for name, df in [("Train", train), ("Test", test)]:
        counts = df[TARGET_COL].value_counts().to_dict()
        total  = len(df)
        print(f"  {name} labels → "
              f"NORMAL: {counts.get(0,0):,} ({counts.get(0,0)/total*100:.1f}%)  "
              f"ATTACK: {counts.get(1,0):,} ({counts.get(1,0)/total*100:.1f}%)")

    return train, test


# ─────────────────────────── NETTOYAGE ───────────────────────────
def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    """
    Applique le pipeline complet de prétraitement.
    Retourne X_train, X_test, y_train, y_test (numpy arrays)
    ainsi que la liste des noms de features finaux.
    """
    print("\n[PREP] Démarrage du prétraitement...")

    # ── 1. Séparer cibles ──
    y_train = train[TARGET_COL].values.astype(int)
    y_test  = test[TARGET_COL].values.astype(int)

    # ── 2. Supprimer colonnes inutiles ──
    drop_train = [c for c in DROP_COLS if c in train.columns]
    drop_test  = [c for c in DROP_COLS if c in test.columns]
    train = train.drop(columns=drop_train + [TARGET_COL])
    test  = test.drop(columns=drop_test  + [TARGET_COL])
    print(f"  → Colonnes supprimées : {drop_train}")

    # ── 3. Encoder les colonnes catégorielles ──
    # On fit les encodeurs sur le train UNIQUEMENT
    encoders = {}
    for col in CAT_COLS:
        if col not in train.columns:
            continue
        le = LabelEncoder()
        le.fit(train[col].astype(str))

        # Gérer les valeurs inconnues dans le test (remplacer par la valeur la plus fréquente)
        most_common = train[col].mode()[0]
        test_col_safe = test[col].astype(str).apply(
            lambda v: v if v in le.classes_ else most_common
        )
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test_col_safe)
        encoders[col] = le
        print(f"  → '{col}' encodé : {len(le.classes_)} classes")

    # Sauvegarder les encodeurs
    with open(ARTIFACTS_DIR / "cat_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    # ── 4. Vérifier qu'il ne reste que du numérique ──
    non_numeric = train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  ⚠ Colonnes non numériques restantes supprimées : {non_numeric}")
        train = train.drop(columns=non_numeric)
        test  = test.drop(columns=[c for c in non_numeric if c in test.columns])

    # ── 5. Aligner les colonnes train/test (sécurité) ──
    common_cols = [c for c in train.columns if c in test.columns]
    train = train[common_cols]
    test  = test[common_cols]

    # ── 6. Supprimer colonnes à variance nulle (fit sur train) ──
    n_before = train.shape[1]
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(train)
    mask = vt.get_support()
    train = train.loc[:, mask]
    test  = test.loc[:, mask]
    removed = n_before - train.shape[1]
    if removed:
        print(f"  → {removed} colonne(s) à variance nulle supprimée(s)")

    feature_names = train.columns.tolist()
    print(f"  → Features finales : {len(feature_names)}")

    # ── 7. Conversion en float32 (économie mémoire) ──
    X_train = train.values.astype(np.float32)
    X_test  = test.values.astype(np.float32)

    # ── 8. Clip des outliers extrêmes (±10 sigma, fit sur train) ──
    means = X_train.mean(axis=0)
    stds  = X_train.std(axis=0)
    stds[stds == 0] = 1.0   # éviter division par zéro
    X_train = np.clip(X_train, means - 10 * stds, means + 10 * stds)
    X_test  = np.clip(X_test,  means - 10 * stds, means + 10 * stds)

    # ── 9. Normalisation RobustScaler (résistant aux outliers) ──
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print(f"  → RobustScaler appliqué")

    # ── 10. Sauvegarder les artefacts ──
    with open(ARTIFACTS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(ARTIFACTS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(ARTIFACTS_DIR / "clip_params.json", "w") as f:
        json.dump({"means": means.tolist(), "stds": stds.tolist()}, f)

    np.save(ARTIFACTS_DIR / "X_train.npy", X_train)
    np.save(ARTIFACTS_DIR / "X_test.npy",  X_test)
    np.save(ARTIFACTS_DIR / "y_train.npy", y_train)
    np.save(ARTIFACTS_DIR / "y_test.npy",  y_test)

    print(f"\n[PREP] ✓ Terminé")
    print(f"  X_train : {X_train.shape}  y_train : {Counter(y_train)}")
    print(f"  X_test  : {X_test.shape}   y_test  : {Counter(y_test)}")
    print(f"  Artefacts → {ARTIFACTS_DIR}/")

    return X_train, X_test, y_train, y_test, feature_names


# ─────────────────────── CHARGEMENT ARTEFACTS ────────────────────
def load_artifacts():
    """Charge les données prétraitées depuis les fichiers .npy (évite de tout refaire)."""
    X_train = np.load(ARTIFACTS_DIR / "X_train.npy")
    X_test  = np.load(ARTIFACTS_DIR / "X_test.npy")
    y_train = np.load(ARTIFACTS_DIR / "y_train.npy")
    y_test  = np.load(ARTIFACTS_DIR / "y_test.npy")
    with open(ARTIFACTS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    return X_train, X_test, y_train, y_test, feature_names


def artifacts_exist():
    return all((ARTIFACTS_DIR / f).exists()
               for f in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"])


# ─────────────────────────── MAIN ────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--force", action="store_true",
                        help="Refaire le prétraitement même si les artefacts existent")
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)

    if not args.force and artifacts_exist():
        print("[PREP] Artefacts déjà présents. Utilisez --force pour recalculer.")
    else:
        train, test = load_data(DATA_DIR)
        preprocess(train, test)

    print("\n✅ preprocessing.py — OK")
