"""
mongodb_config.py
==================
Configure MongoDB pour le SIEM :
  - Stocke les logs bruts complets reçus depuis Kafka
  - Fournit des fonctions réutilisables pour P3 (Spark streaming)

Usage standalone :
  python3 mongodb_config.py

Utilisation depuis P3 :
  from mongodb_config import init_mongodb, sauvegarder_log_brut
"""

from datetime import datetime
from pymongo import MongoClient, ASCENDING, errors as mongo_errors

# ─────────────────────────── CONFIG ───────────────────────────────
MONGO_HOST = "localhost"
MONGO_PORT = 27017
MONGO_DB   = "siem_db"

COLLECTIONS = {
    "logs_bruts":   "logs_bruts",       # tous les logs reçus de Kafka (raw)
    "alertes":      "alertes",          # doublons des alertes ES (backup)
    "stats":        "stats_sessions",   # statistiques agrégées
}


# ──────────────────────── CONNEXION ───────────────────────────────
def init_mongodb() -> tuple:
    """
    Initialise MongoDB et retourne (client, db).
    Crée les collections et index si nécessaire.
    """
    client = MongoClient(MONGO_HOST, MONGO_PORT, serverSelectionTimeoutMS=5000)

    # Vérifier la connexion
    try:
        client.server_info()
    except mongo_errors.ServerSelectionTimeoutError:
        raise ConnectionError(
            f"❌ Impossible de joindre MongoDB sur {MONGO_HOST}:{MONGO_PORT}\n"
            "   Vérifiez que Docker tourne : docker compose up -d"
        )

    db = client[MONGO_DB]
    print(f"✅ Connecté à MongoDB ({MONGO_HOST}:{MONGO_PORT}) → DB: '{MONGO_DB}'")

    _creer_index(db)
    return client, db


def _creer_index(db) -> None:
    """Crée les index MongoDB pour optimiser les requêtes."""
    # Index sur timestamp pour les logs bruts
    db[COLLECTIONS["logs_bruts"]].create_index(
        [("timestamp", ASCENDING)], name="idx_timestamp"
    )
    # Index sur label pour filtrer les attaques
    db[COLLECTIONS["logs_bruts"]].create_index(
        [("label", ASCENDING)], name="idx_label"
    )
    # Index sur proto+service pour les stats réseau
    db[COLLECTIONS["logs_bruts"]].create_index(
        [("proto", ASCENDING), ("service", ASCENDING)], name="idx_proto_service"
    )
    print("✅ Index MongoDB créés.")


# ──────────────────────── LOGS BRUTS ──────────────────────────────
def sauvegarder_log_brut(db, log: dict, topic: str = "flux-reseau") -> str:
    """
    Sauvegarde un log brut complet reçu depuis Kafka.
    Retourne l'ID du document inséré.

    Paramètre `log` : dictionnaire avec tous les champs UNSW-NB15
    Paramètre `topic` : nom du topic Kafka source
    """
    document = {
        **log,
        "kafka_topic":  topic,
        "timestamp":    log.get("timestamp", datetime.utcnow().isoformat()),
        "ingere_le":    datetime.utcnow(),
    }

    result = db[COLLECTIONS["logs_bruts"]].insert_one(document)
    return str(result.inserted_id)


def sauvegarder_batch_logs(db, logs: list, topic: str = "flux-reseau") -> int:
    """
    Sauvegarde une liste de logs bruts en une seule opération.
    Retourne le nombre de documents insérés.
    """
    if not logs:
        return 0

    documents = [
        {
            **log,
            "kafka_topic": topic,
            "timestamp":   log.get("timestamp", datetime.utcnow().isoformat()),
            "ingere_le":   datetime.utcnow(),
        }
        for log in logs
    ]

    result = db[COLLECTIONS["logs_bruts"]].insert_many(documents)
    print(f"✅ {len(result.inserted_ids)} log(s) brut(s) sauvegardé(s) dans MongoDB")
    return len(result.inserted_ids)


# ──────────────────────── ALERTES (BACKUP) ────────────────────────
def sauvegarder_alerte(db, alerte: dict) -> str:
    """
    Sauvegarde une alerte (backup MongoDB en plus d'Elasticsearch).
    Retourne l'ID du document inséré.
    """
    alerte["sauvegarde_le"] = datetime.utcnow()
    result = db[COLLECTIONS["alertes"]].insert_one(alerte)
    return str(result.inserted_id)


# ──────────────────────── REQUÊTES UTILES ─────────────────────────
def compter_attaques(db) -> dict:
    """Retourne le nombre d'attaques vs normal dans les logs bruts."""
    total    = db[COLLECTIONS["logs_bruts"]].count_documents({})
    attaques = db[COLLECTIONS["logs_bruts"]].count_documents({"label": 1})
    normaux  = db[COLLECTIONS["logs_bruts"]].count_documents({"label": 0})
    return {"total": total, "attaques": attaques, "normaux": normaux}


def logs_recents(db, n: int = 10) -> list:
    """Retourne les n derniers logs insérés."""
    cursor = db[COLLECTIONS["logs_bruts"]].find(
        {}, {"_id": 0}
    ).sort("ingere_le", -1).limit(n)
    return list(cursor)


# ────────────────────────── MAIN ──────────────────────────────────
if __name__ == "__main__":
    client, db = init_mongodb()

    # Test : insertion d'un log brut factice
    log_test = {
        "id":       999,
        "dur":      0.5,
        "proto":    "tcp",
        "service":  "http",
        "state":    "FIN",
        "spkts":    10,
        "dpkts":    8,
        "sbytes":   1500,
        "dbytes":   300,
        "label":    1,
        "attack_cat": "Exploits",
        "timestamp": datetime.utcnow().isoformat(),
    }

    doc_id = sauvegarder_log_brut(db, log_test, topic="flux-reseau")
    print(f"\n🗂️  Log test inséré → ID : {doc_id}")

    # Stats
    stats = compter_attaques(db)
    print(f"📊 Stats MongoDB : {stats}")

    client.close()
    print("\n✅ mongodb_config.py — OK")
