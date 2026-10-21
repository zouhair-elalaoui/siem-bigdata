"""
spark_ml/elasticsearch_sender.py
==================================
Lit depuis Kafka (flux-reseau), applique les 4 modèles IA,
calcule le vote majoritaire, et envoie :

  ┌─ Elasticsearch ──────────────────────────────────────────┐
  │  index siem-alertes : alertes ATTACK uniquement          │
  └──────────────────────────────────────────────────────────┘
  ┌─ MongoDB ────────────────────────────────────────────────┐
  │  collection logs_bruts : TOUS les flux (ATTACK + NORMAL) │
  │  collection alertes    : backup des ATTACK               │
  └──────────────────────────────────────────────────────────┘

Modèles : Random Forest · XGBoost · Neural Network · Isolation Forest
Vote majoritaire : ATTACK si >= 2 modèles sur 4 votent ATTACK
Sévérité : HIGH (4/4) · MEDIUM (3/4) · LOW (2/4)

Lancement :
  python3 elasticsearch_sender.py
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────── CONFIG SPARK ────────────────────────────────
os.environ['PYSPARK_SUBMIT_ARGS'] = (
    '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 '
    'pyspark-shell'
)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType, DoubleType
)

# ─────────────────── CONFIG ───────────────────────────────────────
ARTIFACTS_DIR    = Path("./artifacts")
MODELS_DIR       = ARTIFACTS_DIR / "models"
KAFKA_BROKER     = "localhost:9092"
KAFKA_TOPIC      = "flux-reseau"
ES_HOST          = "localhost"
ES_PORT          = 9200
ES_INDEX         = "siem-alertes"
N_VOTE_THRESHOLD = 2

# ─────────────────── SCHÉMA JSON UNSW-NB15 ───────────────────────
SCHEMA = StructType([
    StructField("id",                LongType(),   True),
    StructField("dur",               DoubleType(), True),
    StructField("proto",             StringType(), True),
    StructField("service",           StringType(), True),
    StructField("state",             StringType(), True),
    StructField("spkts",             LongType(),   True),
    StructField("dpkts",             LongType(),   True),
    StructField("sbytes",            LongType(),   True),
    StructField("dbytes",            LongType(),   True),
    StructField("rate",              DoubleType(), True),
    StructField("sttl",              LongType(),   True),
    StructField("dttl",              LongType(),   True),
    StructField("sload",             DoubleType(), True),
    StructField("dload",             DoubleType(), True),
    StructField("sloss",             LongType(),   True),
    StructField("dloss",             LongType(),   True),
    StructField("sinpkt",            DoubleType(), True),
    StructField("dinpkt",            DoubleType(), True),
    StructField("sjit",              DoubleType(), True),
    StructField("djit",              DoubleType(), True),
    StructField("swin",              LongType(),   True),
    StructField("stcpb",             LongType(),   True),
    StructField("dtcpb",             LongType(),   True),
    StructField("dwin",              LongType(),   True),
    StructField("tcprtt",            DoubleType(), True),
    StructField("synack",            DoubleType(), True),
    StructField("ackdat",            DoubleType(), True),
    StructField("smean",             LongType(),   True),
    StructField("dmean",             LongType(),   True),
    StructField("trans_depth",       LongType(),   True),
    StructField("response_body_len", LongType(),   True),
    StructField("ct_srv_src",        LongType(),   True),
    StructField("ct_state_ttl",      LongType(),   True),
    StructField("ct_dst_ltm",        LongType(),   True),
    StructField("ct_src_dport_ltm",  LongType(),   True),
    StructField("ct_dst_sport_ltm",  LongType(),   True),
    StructField("ct_dst_src_ltm",    LongType(),   True),
    StructField("is_ftp_login",      LongType(),   True),
    StructField("ct_ftp_cmd",        LongType(),   True),
    StructField("ct_flw_http_mthd",  LongType(),   True),
    StructField("ct_src_ltm",        LongType(),   True),
    StructField("ct_srv_dst",        LongType(),   True),
    StructField("is_sm_ips_ports",   LongType(),   True),
    StructField("attack_cat",        StringType(), True),
    StructField("label",             LongType(),   True),
])


# ─────────────────── CHARGEMENT ARTEFACTS ────────────────────────
def load_artifacts():
    with open(ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(ARTIFACTS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(ARTIFACTS_DIR / "cat_encoders.pkl", "rb") as f:
        cat_encoders = pickle.load(f)
    with open(ARTIFACTS_DIR / "clip_params.json") as f:
        clip_params = json.load(f)
    return scaler, feature_names, cat_encoders, clip_params


# ─────────────────── CHARGEMENT MODÈLES ──────────────────────────
def load_models():
    model_files = [
        ("RandomForest",    "random_forest.pkl"),
        ("XGBoost",         "xgboost.pkl"),
        ("NeuralNetwork",   "neural_network.pkl"),
        ("IsolationForest", "isolation_forest.pkl"),
    ]
    models = {}
    for name, fname in model_files:
        path = MODELS_DIR / fname
        if path.exists():
            with open(path, "rb") as f:
                obj = pickle.load(f)
            models[name] = obj if isinstance(obj, dict) and "model" in obj \
                           else {"model": obj, "threshold": None}
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} absent ({fname} introuvable dans {MODELS_DIR})")
    return models


# ─────────────────── PRÉTRAITEMENT ───────────────────────────────
def preprocess(pdf, scaler, feature_names, cat_encoders, clip_params):
    df = pdf.copy()
    df = df.drop(columns=["id", "attack_cat", "label"], errors="ignore")

    for col in ["proto", "service", "state"]:
        le = cat_encoders.get(col)
        if le is None or col not in df.columns:
            continue
        most_common = le.classes_[0]
        df[col] = df[col].astype(str).apply(
            lambda v: v if v in le.classes_ else most_common
        )
        df[col] = le.transform(df[col])

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_names]

    means = np.array(clip_params["means"])
    stds  = np.array(clip_params["stds"])
    stds[stds == 0] = 1.0
    X = np.clip(df.values.astype(np.float32),
                means - 10 * stds, means + 10 * stds)
    return scaler.transform(X)


# ─────────────────── PRÉDICTION INDIVIDUELLE ─────────────────────
def predict(X, models):
    results = {}
    for name, obj in models.items():
        model, threshold = obj["model"], obj.get("threshold")
        try:
            if name == "IsolationForest":
                scores = -model.score_samples(X)
                preds  = (scores >= threshold).astype(int)
                score_min, score_max = scores.min(), scores.max()
                denom  = score_max - score_min if score_max > score_min else 1.0
                probas = (scores - score_min) / denom
            else:
                preds  = model.predict(X)
                probas = model.predict_proba(X)[:, 1]
            results[name] = {"preds": preds.astype(int), "probas": probas}
        except Exception as e:
            print(f"  ⚠ {name} erreur de prédiction : {e}")
    return results


# ─────────────────── VOTE MAJORITAIRE ────────────────────────────
def majority_vote(results, threshold=N_VOTE_THRESHOLD):
    n_models  = len(results)
    if n_models == 0:
        raise ValueError("Aucun modèle disponible pour le vote.")

    n_samples = len(next(iter(results.values()))["preds"])
    votes     = np.zeros(n_samples, dtype=int)
    for name, res in results.items():
        votes += res["preds"]

    vote_final = (votes >= threshold).astype(int)

    def severity(v):
        if v == n_models:       return "HIGH"
        elif v >= n_models - 1: return "MEDIUM"
        elif v >= threshold:    return "LOW"
        else:                   return "NORMAL"

    severite = [severity(int(v)) for v in votes]

    supervised = [n for n in results if n != "IsolationForest"]
    if supervised:
        probas_stack = np.stack([results[n]["probas"] for n in supervised], axis=1)
        confidence   = probas_stack.mean(axis=1)
    else:
        confidence = votes / n_models

    return votes, vote_final, severite, confidence


# ─────────────────── ELASTICSEARCH ───────────────────────────────
def init_es():
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(
            [{"host": ES_HOST, "port": ES_PORT, "scheme": "http"}],
            request_timeout=10,
        )
        if not es.ping():
            print(f"  ⚠ Elasticsearch non joignable sur {ES_HOST}:{ES_PORT}")
            return None

        if not es.indices.exists(index=ES_INDEX):
            es.indices.create(index=ES_INDEX, body={
                "mappings": {"properties": {
                    "timestamp":   {"type": "date"},
                    "verdict":     {"type": "keyword"},
                    "vote_final":  {"type": "integer"},
                    "votes":       {"type": "integer"},
                    "n_models":    {"type": "integer"},
                    "severite":    {"type": "keyword"},
                    "confidence":  {"type": "float"},
                    "rf_pred":     {"type": "integer"},
                    "rf_proba":    {"type": "float"},
                    "xgb_pred":    {"type": "integer"},
                    "xgb_proba":   {"type": "float"},
                    "nn_pred":     {"type": "integer"},
                    "nn_proba":    {"type": "float"},
                    "if_pred":     {"type": "integer"},
                    "if_score":    {"type": "float"},
                    "proto":       {"type": "keyword"},
                    "service":     {"type": "keyword"},
                    "state":       {"type": "keyword"},
                    "attack_cat":  {"type": "keyword"},
                    "real_label":  {"type": "integer"},
                    "dur":         {"type": "float"},
                    "sbytes":      {"type": "long"},
                    "dbytes":      {"type": "long"},
                    "rate":        {"type": "float"},
                }}
            })
            print(f"  ✓ Index '{ES_INDEX}' créé")
        else:
            print(f"  ✓ Index '{ES_INDEX}' existant")
        return es

    except ImportError:
        print("  ⚠ elasticsearch non installé → pip install elasticsearch")
        return None
    except Exception as e:
        print(f"  ⚠ Erreur ES : {e}")
        return None


def send_to_es(es, docs):
    if not es or not docs:
        return
    try:
        from elasticsearch.helpers import bulk
        bulk(es, [{"_index": ES_INDEX, "_source": d} for d in docs],
             raise_on_error=False)
    except Exception as e:
        print(f"  ⚠ Erreur envoi ES : {e}")


# ─────────────────── MONGODB ─────────────────────────────────────
def init_mongo():
    """
    Initialise MongoDB via mongodb_config.py (situé dans le dossier parent).
    Retourne (client, db) ou (None, None) si indisponible.
    """
    try:
        # mongodb_config.py est dans le dossier parent de spark_ml/
        parent = str(Path(__file__).resolve().parent.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)

        from mongodb_config import init_mongodb
        client, db = init_mongodb()
        return client, db

    except ConnectionError as e:
        print(f"  ⚠ MongoDB indisponible : {e}")
        return None, None
    except ImportError:
        print("  ⚠ mongodb_config.py introuvable — MongoDB désactivé")
        return None, None
    except Exception as e:
        print(f"  ⚠ Erreur MongoDB init : {e}")
        return None, None


def send_to_mongo(db, pdf, vote_final, votes, severite, confidence,
                  results, MODEL_ES_KEYS):
    """
    Envoie vers MongoDB :
      1. logs_bruts  — TOUS les flux du batch (ATTACK + NORMAL)
      2. alertes     — backup des flux ATTACK uniquement

    C'est l'appel à sauvegarder_batch_logs() et sauvegarder_alerte()
    de mongodb_config.py qui fait le vrai travail.
    """
    if db is None:
        return

    try:
        from mongodb_config import (
            sauvegarder_batch_logs,
            sauvegarder_alerte,
        )

        now = datetime.utcnow().isoformat()
        n   = len(pdf)

        # ── 1. logs_bruts : TOUS les flux (ATTACK + NORMAL) ──────
        logs_bruts = []
        for i in range(n):
            row = pdf.iloc[i]
            logs_bruts.append({
                "flow_id":    int(row.get("id", i)),
                "dur":        float(row.get("dur", 0)),
                "proto":      str(row.get("proto", "")),
                "service":    str(row.get("service", "")),
                "state":      str(row.get("state", "")),
                "spkts":      int(row.get("spkts", 0)),
                "dpkts":      int(row.get("dpkts", 0)),
                "sbytes":     int(row.get("sbytes", 0)),
                "dbytes":     int(row.get("dbytes", 0)),
                "rate":       float(row.get("rate", 0)),
                "attack_cat": str(row.get("attack_cat", "Unknown")),
                "label":      int(row.get("label", -1)),
                # Résultat du pipeline ML
                "vote_final": int(vote_final[i]),
                "votes":      int(votes[i]),
                "verdict":    "ATTACK" if vote_final[i] == 1 else "NORMAL",
                "severite":   severite[i],
                "confidence": round(float(confidence[i]), 4),
                "timestamp":  now,
            })

        n_inserted = sauvegarder_batch_logs(db, logs_bruts, topic=KAFKA_TOPIC)
        print(f"  ✓ {n_inserted} flux bruts → MongoDB [logs_bruts]")

        # ── 2. alertes : backup ATTACK uniquement ─────────────────
        n_attack_backup = 0
        for i in range(n):
            if vote_final[i] == 0:
                continue
            row   = pdf.iloc[i]
            alerte = {
                "flow_id":    int(row.get("id", i)),
                "timestamp":  now,
                "verdict":    "ATTACK",
                "vote_final": int(vote_final[i]),
                "votes":      int(votes[i]),
                "severite":   severite[i],
                "confidence": round(float(confidence[i]), 4),
                "proto":      str(row.get("proto", "")),
                "service":    str(row.get("service", "")),
                "attack_cat": str(row.get("attack_cat", "Unknown")),
                "real_label": int(row.get("label", -1)),
                "sbytes":     int(row.get("sbytes", 0)),
                "dbytes":     int(row.get("dbytes", 0)),
            }
            # Prédictions individuelles par modèle
            for name, (pred_key, score_key) in MODEL_ES_KEYS.items():
                if name in results:
                    alerte[pred_key]  = int(results[name]["preds"][i])
                    alerte[score_key] = round(float(results[name]["probas"][i]), 4)

            sauvegarder_alerte(db, alerte)
            n_attack_backup += 1

        if n_attack_backup > 0:
            print(f"  ✓ {n_attack_backup} alertes backup → MongoDB [alertes]")

    except Exception as e:
        print(f"  ⚠ Erreur insertion MongoDB : {e}")


# ─────────────────── TRAITEMENT PAR BATCH ────────────────────────
def make_processor(models, scaler, feature_names, cat_encoders,
                   clip_params, es, db):

    MODEL_ES_KEYS = {
        "RandomForest":    ("rf_pred",  "rf_proba"),
        "XGBoost":         ("xgb_pred", "xgb_proba"),
        "NeuralNetwork":   ("nn_pred",  "nn_proba"),
        "IsolationForest": ("if_pred",  "if_score"),
    }

    def process(batch_df, batch_id):
        pdf = batch_df.toPandas()
        if pdf.empty:
            return

        n = len(pdf)
        print(f"\n[Batch {batch_id}] {n} flux reçus — {datetime.utcnow().strftime('%H:%M:%S')}")

        # 1. Prétraitement
        try:
            X = preprocess(pdf, scaler, feature_names, cat_encoders, clip_params)
        except Exception as e:
            print(f"  ✗ Prétraitement échoué : {e}")
            return

        # 2. Prédictions individuelles
        results = predict(X, models)
        if not results:
            print("  ✗ Aucun modèle n'a pu prédire.")
            return

        # 3. Vote majoritaire
        votes, vote_final, severite, confidence = majority_vote(results)

        # 4. Résumé console
        n_attack = int(vote_final.sum())
        n_normal = n - n_attack
        print(f"  Vote ({len(results)} modèles, seuil={N_VOTE_THRESHOLD}) → "
              f"NORMAL: {n_normal}  ATTACK: {n_attack}")
        for name, res in results.items():
            n_att = int(res["preds"].sum())
            avg   = float(np.mean(res["probas"]))
            print(f"     {name:<20} ATTACK={n_att:4d}  avg_proba={avg:.3f}")

        sev_counts = {s: severite.count(s) for s in ["HIGH", "MEDIUM", "LOW"]}
        print(f"  Sévérité → HIGH:{sev_counts['HIGH']}  "
              f"MEDIUM:{sev_counts['MEDIUM']}  LOW:{sev_counts['LOW']}")

        # 5. Construction documents Elasticsearch (ATTACK uniquement)
        docs = []
        for i in range(n):
            if vote_final[i] == 0:
                continue
            row = pdf.iloc[i]
            doc = {
                "timestamp":  datetime.utcnow().isoformat(),
                "flow_id":    int(row.get("id", i)),
                "vote_final": int(vote_final[i]),
                "votes":      int(votes[i]),
                "n_models":   len(results),
                "verdict":    "ATTACK",
                "severite":   severite[i],
                "confidence": round(float(confidence[i]), 4),
                "proto":      str(row.get("proto", "")),
                "service":    str(row.get("service", "")),
                "state":      str(row.get("state", "")),
                "attack_cat": str(row.get("attack_cat", "Unknown")),
                "real_label": int(row.get("label", -1)),
                "dur":        float(row.get("dur", 0)),
                "sbytes":     int(row.get("sbytes", 0)),
                "dbytes":     int(row.get("dbytes", 0)),
                "rate":       float(row.get("rate", 0)),
            }
            for name, (pred_key, score_key) in MODEL_ES_KEYS.items():
                if name in results:
                    doc[pred_key]  = int(results[name]["preds"][i])
                    doc[score_key] = round(float(results[name]["probas"][i]), 4)
            docs.append(doc)

        # 6. Envoi Elasticsearch — alertes ATTACK
        if docs:
            send_to_es(es, docs)
            print(f"  ✓ {len(docs)} alertes → Elasticsearch [{ES_INDEX}]")
        else:
            print("  ✓ Aucune attaque détectée dans ce batch.")

        # 7. Envoi MongoDB — logs_bruts (tout) + alertes backup (ATTACK)
        send_to_mongo(db, pdf, vote_final, votes, severite,
                      confidence, results, MODEL_ES_KEYS)

    return process


# ─────────────────────────── MAIN ────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  SIEM — Spark + 4 Modèles IA + Vote Majoritaire")
    print("         Elasticsearch  +  MongoDB")
    print("=" * 60)
    print(f"  Kafka  : {KAFKA_BROKER}  topic: {KAFKA_TOPIC}")
    print(f"  ES     : {ES_HOST}:{ES_PORT}  index: {ES_INDEX}")
    print(f"  Seuil vote ATTACK : {N_VOTE_THRESHOLD}/4 modèles")
    print("=" * 60)

    # Vérification artefacts obligatoires
    for fname in ["scaler.pkl", "feature_names.json",
                  "cat_encoders.pkl", "clip_params.json"]:
        if not (ARTIFACTS_DIR / fname).exists():
            print(f"\n❌ Artefact manquant : {fname}")
            print("   → Lancez d'abord : python3 preprocessing.py")
            exit(1)

    print("\n[INIT] Chargement des artefacts...")
    scaler, feature_names, cat_encoders, clip_params = load_artifacts()

    print("[INIT] Chargement des modèles...")
    models = load_models()
    if not models:
        print("\n❌ Aucun modèle trouvé. Lancez les scripts train_*.py d'abord.")
        exit(1)
    print(f"\n[INIT] {len(models)} modèle(s) : {', '.join(models.keys())}")

    print("\n[INIT] Connexion Elasticsearch...")
    es = init_es()
    if es is None:
        print("  [INFO] ES indisponible — alertes non indexées")

    # ── Connexion MongoDB (NOUVEAU) ───────────────────────────────
    print("\n[INIT] Connexion MongoDB...")
    mongo_client, db = init_mongo()
    if db is None:
        print("  [INFO] MongoDB indisponible — stockage brut désactivé")
    else:
        print(f"  ✓ MongoDB connecté → siem_db")

    # Spark Session
    spark = SparkSession.builder \
        .appName("SIEM-ML-Elasticsearch-MongoDB") \
        .master("local[*]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print("\n[SPARK] Session démarrée")

    # Lecture Kafka
    raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .option("maxOffsetsPerTrigger", 1000) \
        .load()

    # Parsing JSON
    parsed = raw.select(
        F.from_json(F.col("value").cast("string"), SCHEMA).alias("d")
    ).select("d.*")

    # Pipeline IA + ES + MongoDB
    processor = make_processor(
        models, scaler, feature_names, cat_encoders, clip_params, es, db
    )

    query = parsed.writeStream \
        .foreachBatch(processor) \
        .trigger(processingTime="5 seconds") \
        .option("checkpointLocation", "./checkpoint_es") \
        .start()

    print(f"\n[SPARK] Pipeline actif — topic '{KAFKA_TOPIC}'")
    print(f"        ES     → alertes ATTACK uniquement [{ES_INDEX}]")
    print(f"        MongoDB→ tous les flux bruts [logs_bruts]")
    print(f"                 backup alertes ATTACK [alertes]")
    print(f"        Trigger toutes les 5s | Ctrl+C pour arrêter\n")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n[STOP] Arrêt demandé...")
        query.stop()
        spark.stop()
        if mongo_client:
            mongo_client.close()
            print("[STOP] Connexion MongoDB fermée proprement.")
        print("[STOP] Terminé.")
