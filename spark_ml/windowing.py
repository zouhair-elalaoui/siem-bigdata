"""
spark_ml/windowing.py
======================
Agrégation par fenêtres temporelles glissantes (Spark Structured Streaming).

Ce module complète elasticsearch_sender.py : là où le sender traite chaque
flux individuellement (prédiction par ligne), windowing.py agrège les alertes
déjà produites et calcule des statistiques par période de temps.

Deux pipelines tournent en parallèle :

  Pipeline A — Fenêtre glissante 1 min / pas 30 s
  ─────────────────────────────────────────────────
  Pour chaque fenêtre de 60 secondes (avance toutes les 30 s) :
    - Nombre total de flux reçus
    - Nombre d'ATTACK détectées (vote majoritaire)
    - Taux d'attaque (%)
    - Décompte par sévérité (HIGH / MEDIUM / LOW)
    - Décompte par protocole
    - Décompte par catégorie d'attaque
    - Score de menace global de la fenêtre

  Pipeline B — Fenêtre tumbling 5 min (non chevauchante)
  ────────────────────────────────────────────────────────
  Toutes les 5 minutes, snapshot consolidé envoyé dans ES
  (index : siem-window-stats) pour les dashboards Kibana.

Prérequis :
  - Kafka qui tourne avec le topic "flux-reseau"
  - elasticsearch_sender.py exécuté en parallèle (ou standalone)
  - python3 preprocessing.py + train_*.py déjà lancés

Lancement (en parallèle du sender) :
  python3 windowing.py

Architecture :
  Kafka (flux-reseau)
       │
       ├──► elasticsearch_sender.py  (prédiction par ligne → index siem-alertes)
       │
       └──► windowing.py             (agrégation temporelle → index siem-window-stats)
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

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
    StringType, LongType, DoubleType, IntegerType
)

# ─────────────────── CONFIG ───────────────────────────────────────
ARTIFACTS_DIR = Path("./artifacts")
MODELS_DIR    = ARTIFACTS_DIR / "models"
KAFKA_BROKER  = "localhost:9092"
KAFKA_TOPIC   = "flux-reseau"
ES_HOST       = "localhost"
ES_PORT       = 9200
ES_INDEX_WIN  = "siem-window-stats"   # index des stats de fenêtres

# Paramètres des fenêtres
SLIDING_WINDOW_DURATION = "1 minute"   # durée de la fenêtre glissante
SLIDING_WINDOW_SLIDE    = "30 seconds" # pas de glissement
TUMBLING_WINDOW_DURATION = "5 minutes" # fenêtre tumbling (non chevauchante)

# Seuil d'alerte : si taux_attaque > ALERT_THRESHOLD → log WARNING
ALERT_THRESHOLD_PCT = 30.0             # 30 % de flux classés ATTACK → alerte

# Vote majoritaire : seuil identique à elasticsearch_sender.py
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
    StructField("sttl",             LongType(),   True),
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


# ─────────────────── CHARGEMENT ARTEFACTS + MODÈLES ──────────────
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
            print(f"  ✗ {name} absent")
    return models


# ─────────────────── PRÉTRAITEMENT ───────────────────────────────
def preprocess(pdf, scaler, feature_names, cat_encoders, clip_params):
    df = pdf.copy()
    df = df.drop(columns=["id", "attack_cat", "label",
                           "window_start", "window_end",
                           "kafka_timestamp"], errors="ignore")

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


# ─────────────────── PRÉDICTION + VOTE ───────────────────────────
def predict_and_vote(X, models, threshold=N_VOTE_THRESHOLD):
    """
    Retourne (vote_final, votes, confidence, probas_rf) pour chaque flux.
    vote_final : array int  — 1=ATTACK, 0=NORMAL
    votes      : array int  — nombre de modèles votant ATTACK
    confidence : array float — moyenne probas supervisés
    """
    results = {}
    for name, obj in models.items():
        model, thr = obj["model"], obj.get("threshold")
        try:
            if name == "IsolationForest":
                scores = -model.score_samples(X)
                preds  = (scores >= thr).astype(int)
                s_min, s_max = scores.min(), scores.max()
                denom  = s_max - s_min if s_max > s_min else 1.0
                probas = (scores - s_min) / denom
            else:
                preds  = model.predict(X)
                probas = model.predict_proba(X)[:, 1]
            results[name] = {"preds": preds.astype(int), "probas": probas}
        except Exception as e:
            print(f"  ⚠ {name} : {e}")

    if not results:
        return None, None, None

    n_samples = len(next(iter(results.values()))["preds"])
    votes = np.zeros(n_samples, dtype=int)
    for res in results.values():
        votes += res["preds"]

    vote_final = (votes >= threshold).astype(int)

    supervised = [n for n in results if n != "IsolationForest"]
    if supervised:
        stack = np.stack([results[n]["probas"] for n in supervised], axis=1)
        confidence = stack.mean(axis=1)
    else:
        confidence = votes / len(results)

    return vote_final, votes, confidence


def severity_label(v, n_models):
    if v == n_models:       return "HIGH"
    elif v >= n_models - 1: return "MEDIUM"
    elif v >= N_VOTE_THRESHOLD: return "LOW"
    return "NORMAL"


# ─────────────────── ELASTICSEARCH ───────────────────────────────
def init_es():
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(
            [{"host": ES_HOST, "port": ES_PORT, "scheme": "http"}],
            request_timeout=10,
        )
        if not es.ping():
            print(f"  ⚠ ES non joignable sur {ES_HOST}:{ES_PORT}")
            return None

        if not es.indices.exists(index=ES_INDEX_WIN):
            es.indices.create(index=ES_INDEX_WIN, body={
                "mappings": {"properties": {
                    # Fenêtre temporelle
                    "window_start":       {"type": "date"},
                    "window_end":         {"type": "date"},
                    "window_type":        {"type": "keyword"},  # sliding / tumbling
                    "window_duration_s":  {"type": "integer"},
                    # Comptages globaux
                    "total_flux":         {"type": "long"},
                    "total_attack":       {"type": "long"},
                    "total_normal":       {"type": "long"},
                    "taux_attaque_pct":   {"type": "float"},
                    "avg_confidence":     {"type": "float"},
                    "threat_score":       {"type": "float"},  # score 0-100
                    "alerte_critique":    {"type": "boolean"},
                    # Sévérité
                    "high_count":         {"type": "long"},
                    "medium_count":       {"type": "long"},
                    "low_count":          {"type": "long"},
                    # Top protocoles
                    "top_proto":          {"type": "keyword"},
                    "proto_counts":       {"type": "object"},
                    # Top catégories d'attaque
                    "top_attack_cat":     {"type": "keyword"},
                    "attack_cat_counts":  {"type": "object"},
                    # Stats réseau agrégées
                    "avg_dur":            {"type": "float"},
                    "avg_rate":           {"type": "float"},
                    "total_sbytes":       {"type": "long"},
                    "total_dbytes":       {"type": "long"},
                }}
            })
            print(f"  ✓ Index '{ES_INDEX_WIN}' créé")
        else:
            print(f"  ✓ Index '{ES_INDEX_WIN}' existant")
        return es

    except ImportError:
        print("  ⚠ pip install elasticsearch")
        return None
    except Exception as e:
        print(f"  ⚠ Erreur ES : {e}")
        return None


def send_to_es(es, docs):
    if not es or not docs:
        return
    try:
        from elasticsearch.helpers import bulk
        bulk(es, [{"_index": ES_INDEX_WIN, "_source": d} for d in docs],
             raise_on_error=False)
    except Exception as e:
        print(f"  ⚠ Envoi ES : {e}")


# ─────────────────── PROCESSOR FENÊTRE GLISSANTE ─────────────────
def make_sliding_processor(models, scaler, feature_names,
                            cat_encoders, clip_params, es,
                            window_duration_s=60):
    """
    foreachBatch handler pour la fenêtre glissante (1 min / 30 s).
    Chaque batch contient les flux de la fenêtre courante.
    On prédit sur tous les flux, on agrège, on envoie dans ES.
    """

    def process(batch_df, batch_id):
        pdf = batch_df.toPandas()
        if pdf.empty:
            return

        n = len(pdf)
        now = datetime.now(timezone.utc)
        win_start = now.replace(second=0, microsecond=0)

        print(f"\n[Sliding {batch_id}] {n} flux — fenêtre {SLIDING_WINDOW_DURATION}")

        # ── Prédictions ──────────────────────────────────────────
        try:
            X = preprocess(pdf, scaler, feature_names, cat_encoders, clip_params)
        except Exception as e:
            print(f"  ✗ Prétraitement : {e}")
            return

        vote_final, votes, confidence = predict_and_vote(X, models)
        if vote_final is None:
            return

        n_models = len(models)
        severities = [severity_label(int(v), n_models) for v in votes]

        # ── Agrégation globale ───────────────────────────────────
        n_attack = int(vote_final.sum())
        n_normal = n - n_attack
        taux     = round(100.0 * n_attack / n if n > 0 else 0.0, 2)
        avg_conf = round(float(confidence[vote_final == 1].mean())
                         if n_attack > 0 else 0.0, 4)

        # Score de menace 0-100 : pondère le taux et la sévérité
        high_c   = severities.count("HIGH")
        med_c    = severities.count("MEDIUM")
        low_c    = severities.count("LOW")
        threat   = round(min(100.0,
                    taux * 0.5 +
                    (high_c * 3.0 + med_c * 1.5 + low_c * 0.5) / max(n, 1) * 50), 2)

        alerte_critique = taux >= ALERT_THRESHOLD_PCT or high_c > 0

        # ── Décomptes par protocole ──────────────────────────────
        pdf["_vote"] = vote_final
        attack_rows  = pdf[pdf["_vote"] == 1]

        proto_counts = {}
        if "proto" in attack_rows.columns:
            proto_counts = attack_rows["proto"].value_counts().to_dict()
        top_proto = max(proto_counts, key=proto_counts.get) \
                    if proto_counts else "N/A"

        # ── Décomptes par catégorie d'attaque ────────────────────
        cat_counts = {}
        if "attack_cat" in attack_rows.columns:
            cat_counts = attack_rows["attack_cat"].fillna("Unknown") \
                                    .value_counts().to_dict()
        top_cat = max(cat_counts, key=cat_counts.get) \
                  if cat_counts else "N/A"

        # ── Stats réseau ─────────────────────────────────────────
        avg_dur   = float(pdf["dur"].mean())   if "dur"    in pdf.columns else 0.0
        avg_rate  = float(pdf["rate"].mean())  if "rate"   in pdf.columns else 0.0
        tot_sb    = int(pdf["sbytes"].sum())   if "sbytes" in pdf.columns else 0
        tot_db    = int(pdf["dbytes"].sum())   if "dbytes" in pdf.columns else 0

        # ── Log console ──────────────────────────────────────────
        status = "🔴 CRITIQUE" if alerte_critique else "🟢 Normal"
        print(f"  {status}  taux={taux}%  threat={threat}/100  "
              f"HIGH={high_c} MED={med_c} LOW={low_c}")
        print(f"  top_proto={top_proto}  top_cat={top_cat}  "
              f"avg_conf={avg_conf}")
        if alerte_critique:
            print(f"  ⚠ ALERTE — taux d'attaque dépasse {ALERT_THRESHOLD_PCT}% "
                  f"ou présence d'attaques HIGH dans la fenêtre !")

        # ── Document Elasticsearch ───────────────────────────────
        doc = {
            "window_start":      win_start.isoformat(),
            "window_end":        now.isoformat(),
            "window_type":       "sliding",
            "window_duration_s": window_duration_s,
            "total_flux":        n,
            "total_attack":      n_attack,
            "total_normal":      n_normal,
            "taux_attaque_pct":  taux,
            "avg_confidence":    avg_conf,
            "threat_score":      threat,
            "alerte_critique":   alerte_critique,
            "high_count":        high_c,
            "medium_count":      med_c,
            "low_count":         low_c,
            "top_proto":         top_proto,
            "proto_counts":      proto_counts,
            "top_attack_cat":    top_cat,
            "attack_cat_counts": cat_counts,
            "avg_dur":           round(avg_dur,  4),
            "avg_rate":          round(avg_rate, 4),
            "total_sbytes":      tot_sb,
            "total_dbytes":      tot_db,
        }

        send_to_es(es, [doc])
        print(f"  ✓ Stats fenêtre → ES [{ES_INDEX_WIN}]")

    return process


# ─────────────────── PROCESSOR FENÊTRE TUMBLING ──────────────────
def make_tumbling_processor(models, scaler, feature_names,
                             cat_encoders, clip_params, es,
                             window_duration_s=300):
    """
    foreachBatch handler pour la fenêtre tumbling (5 min, non chevauchante).
    Snapshot consolidé toutes les 5 minutes — idéal pour dashboards Kibana.
    Même logique que la sliding, mais window_type='tumbling'.
    """

    def process(batch_df, batch_id):
        pdf = batch_df.toPandas()
        if pdf.empty:
            return

        n   = len(pdf)
        now = datetime.now(timezone.utc)
        print(f"\n[Tumbling {batch_id}] {n} flux — snapshot {TUMBLING_WINDOW_DURATION}")

        try:
            X = preprocess(pdf, scaler, feature_names, cat_encoders, clip_params)
        except Exception as e:
            print(f"  ✗ Prétraitement : {e}")
            return

        vote_final, votes, confidence = predict_and_vote(X, models)
        if vote_final is None:
            return

        n_models   = len(models)
        severities = [severity_label(int(v), n_models) for v in votes]
        n_attack   = int(vote_final.sum())
        n_normal   = n - n_attack
        taux       = round(100.0 * n_attack / n if n > 0 else 0.0, 2)
        avg_conf   = round(float(confidence[vote_final == 1].mean())
                           if n_attack > 0 else 0.0, 4)
        high_c     = severities.count("HIGH")
        med_c      = severities.count("MEDIUM")
        low_c      = severities.count("LOW")
        threat     = round(min(100.0,
                        taux * 0.5 +
                        (high_c * 3.0 + med_c * 1.5 + low_c * 0.5) / max(n, 1) * 50), 2)

        pdf["_vote"] = vote_final
        attack_rows  = pdf[pdf["_vote"] == 1]

        proto_counts = {}
        if "proto" in attack_rows.columns:
            proto_counts = attack_rows["proto"].value_counts().to_dict()
        top_proto = max(proto_counts, key=proto_counts.get) if proto_counts else "N/A"

        cat_counts = {}
        if "attack_cat" in attack_rows.columns:
            cat_counts = attack_rows["attack_cat"].fillna("Unknown") \
                                    .value_counts().to_dict()
        top_cat = max(cat_counts, key=cat_counts.get) if cat_counts else "N/A"

        avg_dur  = float(pdf["dur"].mean())   if "dur"    in pdf.columns else 0.0
        avg_rate = float(pdf["rate"].mean())  if "rate"   in pdf.columns else 0.0
        tot_sb   = int(pdf["sbytes"].sum())   if "sbytes" in pdf.columns else 0
        tot_db   = int(pdf["dbytes"].sum())   if "dbytes" in pdf.columns else 0

        print(f"  Snapshot 5min — taux={taux}%  threat={threat}/100  "
              f"HIGH={high_c} MED={med_c} LOW={low_c}")

        doc = {
            "window_start":      (now.replace(minute=(now.minute // 5) * 5,
                                              second=0, microsecond=0)).isoformat(),
            "window_end":        now.isoformat(),
            "window_type":       "tumbling",
            "window_duration_s": window_duration_s,
            "total_flux":        n,
            "total_attack":      n_attack,
            "total_normal":      n_normal,
            "taux_attaque_pct":  taux,
            "avg_confidence":    avg_conf,
            "threat_score":      threat,
            "alerte_critique":   taux >= ALERT_THRESHOLD_PCT or high_c > 0,
            "high_count":        high_c,
            "medium_count":      med_c,
            "low_count":         low_c,
            "top_proto":         top_proto,
            "proto_counts":      proto_counts,
            "top_attack_cat":    top_cat,
            "attack_cat_counts": cat_counts,
            "avg_dur":           round(avg_dur,  4),
            "avg_rate":          round(avg_rate, 4),
            "total_sbytes":      tot_sb,
            "total_dbytes":      tot_db,
        }

        send_to_es(es, [doc])
        print(f"  ✓ Snapshot tumbling → ES [{ES_INDEX_WIN}]")

    return process


# ─────────────────────────── MAIN ────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  SIEM — Spark Windowing (Sliding + Tumbling)")
    print("=" * 60)
    print(f"  Kafka     : {KAFKA_BROKER}  topic: {KAFKA_TOPIC}")
    print(f"  ES        : {ES_HOST}:{ES_PORT}  index: {ES_INDEX_WIN}")
    print(f"  Sliding   : {SLIDING_WINDOW_DURATION} / slide {SLIDING_WINDOW_SLIDE}")
    print(f"  Tumbling  : {TUMBLING_WINDOW_DURATION}")
    print(f"  Alerte si : taux > {ALERT_THRESHOLD_PCT}%  ou  HIGH > 0")
    print("=" * 60)

    # Vérification artefacts
    for fname in ["scaler.pkl", "feature_names.json",
                  "cat_encoders.pkl", "clip_params.json"]:
        if not (ARTIFACTS_DIR / fname).exists():
            print(f"\n❌ Artefact manquant : {fname}")
            print("   → Lancez d'abord : python3 preprocessing.py")
            exit(1)

    print("\n[INIT] Chargement artefacts...")
    scaler, feature_names, cat_encoders, clip_params = load_artifacts()

    print("[INIT] Chargement modèles...")
    models = load_models()
    if not models:
        print("\n❌ Aucun modèle trouvé.")
        exit(1)

    print(f"\n[INIT] {len(models)} modèle(s) : {', '.join(models.keys())}")

    print("\n[INIT] Connexion Elasticsearch...")
    es = init_es()
    if es is None:
        print("  [INFO] Mode console uniquement")

    # ── Spark Session ─────────────────────────────────────────────
    spark = SparkSession.builder \
        .appName("SIEM-Windowing") \
        .master("local[*]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print("\n[SPARK] Session démarrée")

    # ── Lecture Kafka commune ─────────────────────────────────────
    raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .option("maxOffsetsPerTrigger", 2000) \
        .load()

    # Parse JSON + conserve le timestamp Kafka pour le windowing Spark natif
    parsed = raw.select(
        F.from_json(F.col("value").cast("string"), SCHEMA).alias("d"),
        F.col("timestamp").alias("kafka_timestamp")          # ← timestamp Kafka
    ).select("d.*", "kafka_timestamp")

    # ─────────────────────────────────────────────────────────────
    # PIPELINE A — Fenêtre glissante (foreachBatch, trigger 30 s)
    # On utilise foreachBatch car on a besoin de pandas/sklearn.
    # Spark ne peut pas appliquer des modèles sklearn directement
    # dans les opérations de streaming natives.
    # ─────────────────────────────────────────────────────────────
    sliding_processor = make_sliding_processor(
        models, scaler, feature_names, cat_encoders, clip_params,
        es, window_duration_s=60
    )

    query_sliding = parsed.writeStream \
        .foreachBatch(sliding_processor) \
        .trigger(processingTime=SLIDING_WINDOW_SLIDE) \
        .option("checkpointLocation", "./checkpoint_sliding") \
        .start()

    print(f"\n[PIPELINE A] Fenêtre glissante active "
          f"({SLIDING_WINDOW_DURATION} / {SLIDING_WINDOW_SLIDE})")

    # ─────────────────────────────────────────────────────────────
    # PIPELINE B — Fenêtre tumbling (foreachBatch, trigger 5 min)
    # ─────────────────────────────────────────────────────────────
    tumbling_processor = make_tumbling_processor(
        models, scaler, feature_names, cat_encoders, clip_params,
        es, window_duration_s=300
    )

    query_tumbling = parsed.writeStream \
        .foreachBatch(tumbling_processor) \
        .trigger(processingTime=TUMBLING_WINDOW_DURATION) \
        .option("checkpointLocation", "./checkpoint_tumbling") \
        .start()

    print(f"[PIPELINE B] Fenêtre tumbling active ({TUMBLING_WINDOW_DURATION})")
    print(f"\n  Ctrl+C pour arrêter les deux pipelines\n")

    # ── Attente des deux queries ──────────────────────────────────
    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        print("\n[STOP] Arrêt demandé...")
        query_sliding.stop()
        query_tumbling.stop()
        spark.stop()
        print("[STOP] Terminé proprement.")
