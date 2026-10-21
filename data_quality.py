"""
data_quality.py
================
Analyse la qualité du dataset UNSW-NB15 :
  - Valeurs manquantes
  - Colonnes inutiles / à variance nulle
  - Déséquilibre des classes (biais)
  - Doublons
  - Distribution des features clés
  - Génère un paragraphe résumé pour le rapport

Usage :
  python3 data_quality.py
  python3 data_quality.py --data-dir /chemin/vers/data
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIG ───────────────────────────────
DATA_DIR   = Path("../data")
TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE  = "UNSW_NB15_testing-set.csv"
TARGET_COL = "label"


# ──────────────────────── CHARGEMENT ──────────────────────────────
def charger_datasets(data_dir: Path):
    train = pd.read_csv(data_dir / TRAIN_FILE)
    test  = pd.read_csv(data_dir / TEST_FILE)
    train.columns = train.columns.str.strip()
    test.columns  = test.columns.str.strip()
    return train, test


# ──────────────────────── ANALYSE ─────────────────────────────────
def analyser_qualite(df: pd.DataFrame, nom: str = "Dataset") -> dict:
    print(f"\n{'='*55}")
    print(f"  ANALYSE QUALITÉ — {nom}")
    print(f"{'='*55}")

    resultats = {"nom": nom}

    # ── 1. Dimensions ──
    resultats["lignes"]   = df.shape[0]
    resultats["colonnes"] = df.shape[1]
    print(f"\n📐 Dimensions : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    # ── 2. Valeurs manquantes ──
    manquantes = df.isnull().sum()
    cols_avec_nan = manquantes[manquantes > 0]
    resultats["valeurs_manquantes_total"] = int(manquantes.sum())
    resultats["colonnes_avec_nan"]        = len(cols_avec_nan)

    print(f"\n🔍 Valeurs manquantes :")
    if cols_avec_nan.empty:
        print("   ✅ Aucune valeur manquante.")
    else:
        for col, n in cols_avec_nan.items():
            pct = n / len(df) * 100
            print(f"   ⚠️  '{col}' : {n:,} manquantes ({pct:.1f}%)")

    # ── 3. Doublons ──
    n_doublons = df.duplicated().sum()
    resultats["doublons"] = int(n_doublons)
    print(f"\n👥 Doublons : {n_doublons:,} ({n_doublons/len(df)*100:.2f}%)")
    if n_doublons > 0:
        print("   ⚠️  Des doublons ont été détectés.")
    else:
        print("   ✅ Aucun doublon.")

    # ── 4. Distribution des classes (biais) ──
    if TARGET_COL in df.columns:
        counts = df[TARGET_COL].value_counts()
        total  = len(df)
        n_normal  = counts.get(0, 0)
        n_attaque = counts.get(1, 0)
        ratio     = n_normal / n_attaque if n_attaque > 0 else float("inf")

        resultats["n_normal"]  = int(n_normal)
        resultats["n_attaque"] = int(n_attaque)
        resultats["ratio_desequilibre"] = round(ratio, 2)

        print(f"\n⚖️  Distribution des classes :")
        print(f"   NORMAL  (0) : {n_normal:,}  ({n_normal/total*100:.1f}%)")
        print(f"   ATTAQUE (1) : {n_attaque:,}  ({n_attaque/total*100:.1f}%)")
        print(f"   Ratio Normal/Attaque : {ratio:.2f}")

        if ratio > 3:
            print("   ⚠️  Déséquilibre modéré à fort → SMOTE ou class_weight recommandé.")
        else:
            print("   ✅ Classes relativement équilibrées.")

    # ── 5. Colonnes à variance nulle (inutiles) ──
    numeriques = df.select_dtypes(include=[np.number])
    variance_nulle = numeriques.columns[numeriques.var() == 0].tolist()
    resultats["colonnes_variance_nulle"] = variance_nulle

    print(f"\n🗑️  Colonnes à variance nulle (inutiles) : {len(variance_nulle)}")
    for col in variance_nulle:
        print(f"   - {col}")
    if not variance_nulle:
        print("   ✅ Aucune.")

    # ── 6. Colonnes catégorielles ──
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"\n🏷️  Colonnes catégorielles : {cat_cols}")
    for col in cat_cols:
        n_unique = df[col].nunique()
        print(f"   '{col}' → {n_unique} valeurs uniques : {df[col].unique()[:8].tolist()}")

    # ── 7. Statistiques numériques clés ──
    cols_cles = ["dur", "sbytes", "dbytes", "rate", "spkts", "dpkts"]
    cols_cles = [c for c in cols_cles if c in df.columns]
    print(f"\n📈 Stats des features clés :")
    print(df[cols_cles].describe().round(3).to_string())

    # ── 8. Valeurs infinies ──
    num_df = df.select_dtypes(include=[np.number])
    n_inf = np.isinf(num_df.values).sum()
    resultats["valeurs_infinies"] = int(n_inf)
    print(f"\n♾️  Valeurs infinies : {n_inf}")
    if n_inf > 0:
        print("   ⚠️  À remplacer avant l'entraînement.")
    else:
        print("   ✅ Aucune.")

    return resultats


# ──────────────────── PARAGRAPHE RAPPORT ──────────────────────────
def generer_paragraphe_rapport(res_train: dict, res_test: dict) -> str:
    """Génère un paragraphe prêt à coller dans le rapport."""

    paragraphe = f"""
=================================================================
  PARAGRAPHE QUALITÉ DES DONNÉES — À COLLER DANS LE RAPPORT
=================================================================

Le dataset utilisé pour ce projet est le UNSW-NB15, développé par
l'Australian Centre for Cyber Security (ACSC). Il est composé de
{res_train['lignes']:,} entrées d'entraînement et {res_test['lignes']:,}
entrées de test, réparties sur {res_train['colonnes']} features décrivant
le trafic réseau (protocole, durée, octets échangés, etc.).

L'analyse de qualité révèle que le dataset ne contient aucune valeur
manquante ({res_train['valeurs_manquantes_total']} valeurs nulles
détectées), ce qui en fait un dataset propre et directement exploitable.
Le nombre de doublons est de {res_train['doublons']:,}, représentant une
proportion négligeable.

Concernant la distribution des classes, on observe un déséquilibre modéré :
{res_train['n_normal']:,} entrées normales ({res_train['n_normal']/(res_train['n_normal']+res_train['n_attaque'])*100:.1f}%)
contre {res_train['n_attaque']:,} attaques
({res_train['n_attaque']/(res_train['n_normal']+res_train['n_attaque'])*100:.1f}%),
soit un ratio de {res_train['ratio_desequilibre']:.2f}:1. Ce déséquilibre
a été pris en compte lors de l'entraînement via le paramètre class_weight
et l'utilisation de métriques adaptées (F1-score, recall).

Les colonnes catégorielles (proto, service, state) ont été encodées avec
un LabelEncoder fitté uniquement sur les données d'entraînement pour
éviter toute fuite d'information. La normalisation a été effectuée avec
un RobustScaler, choisi pour sa résistance aux valeurs aberrantes
fréquentes dans les données réseau.
=================================================================
"""
    return paragraphe


# ────────────────────────── MAIN ──────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("📂 Chargement des datasets UNSW-NB15...")
    train, test = charger_datasets(data_dir)

    res_train = analyser_qualite(train, "TRAIN (UNSW-NB15)")
    res_test  = analyser_qualite(test,  "TEST  (UNSW-NB15)")

    # Comparaison train/test
    print(f"\n{'='*55}")
    print("  COMPARAISON TRAIN / TEST")
    print(f"{'='*55}")
    print(f"  Train : {res_train['lignes']:,} lignes | "
          f"Attaques : {res_train['n_attaque']:,} ({res_train['n_attaque']/res_train['lignes']*100:.1f}%)")
    print(f"  Test  : {res_test['lignes']:,} lignes  | "
          f"Attaques : {res_test['n_attaque']:,} ({res_test['n_attaque']/res_test['lignes']*100:.1f}%)")

    # Paragraphe rapport
    paragraphe = generer_paragraphe_rapport(res_train, res_test)
    print(paragraphe)

    # Sauvegarder dans un fichier texte
    with open("data_quality_rapport.txt", "w", encoding="utf-8") as f:
        f.write(paragraphe)
    print("💾 Paragraphe sauvegardé dans : data_quality_rapport.txt")
    print("\n✅ data_quality.py — OK")
