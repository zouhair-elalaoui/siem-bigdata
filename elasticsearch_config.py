"""
elasticsearch_config.py
========================
Configure Elasticsearch pour le SIEM :
  - Crée l'index "alertes-securite" avec le mapping approprié
  - Fournit des fonctions réutilisables pour P3 (Spark streaming)

Usage standalone :
  python3 elasticsearch_config.py

Utilisation depuis P3 :
  from elasticsearch_config import envoyer_alerte, init_elasticsearch
"""

import json
from datetime import datetime
from elasticsearch import Elasticsearch, exceptions as es_exceptions

# ─────────────────────────── CONFIG ───────────────────────────────
ES_HOST  = "localhost"
ES_PORT  = 9200
ES_INDEX = "alertes-securite"

# ──────────────────────── MAPPING INDEX ───────────────────────────
MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "timestamp":      {"type": "date"},
            "ip_src":         {"type": "ip"},
            "ip_dst":         {"type": "ip"},
            "port_src":       {"type": "integer"},
            "port_dst":       {"type": "integer"},
            "proto":          {"type": "keyword"},
            "service":        {"type": "keyword"},
            "state":          {"type": "keyword"},
            "label":          {"type": "integer"},        # 0 = normal, 1 = attaque
            "attack_cat":     {"type": "keyword"},        # type d'attaque
            "prediction_rf":  {"type": "integer"},        # Random Forest
            "prediction_if":  {"type": "integer"},        # Isolation Forest
            "prediction_nn":  {"type": "integer"},        # Neural Network
            "vote_final":     {"type": "integer"},        # vote majoritaire
            "score_rf":       {"type": "float"},          # probabilité RF
            "score_nn":       {"type": "float"},          # probabilité NN
            "duree":          {"type": "float"},
            "sbytes":         {"type": "long"},
            "dbytes":         {"type": "long"},
            "severite":       {"type": "keyword"},        # LOW / MEDIUM / HIGH
            "source":         {"type": "keyword"},        # kafka topic
        }
    }
}


# ──────────────────────── CONNEXION ───────────────────────────────
def init_elasticsearch() -> Elasticsearch:
    """Initialise et retourne le client Elasticsearch."""
    es = Elasticsearch(f"http://{ES_HOST}:{ES_PORT}")

    if not es.ping():
        raise ConnectionError(
            f"❌ Impossible de joindre Elasticsearch sur {ES_HOST}:{ES_PORT}\n"
            "   Vérifiez que Docker tourne : docker compose up -d"
        )
    print(f"✅ Connecté à Elasticsearch ({ES_HOST}:{ES_PORT})")
    return es


# ──────────────────────── CRÉATION INDEX ──────────────────────────
def creer_index(es: Elasticsearch, index: str = ES_INDEX) -> None:
    """Crée l'index Elasticsearch s'il n'existe pas déjà."""
    if es.indices.exists(index=index):
        print(f"ℹ️  Index '{index}' déjà existant.")
        return

    es.indices.create(index=index, body=MAPPING)
    print(f"✅ Index '{index}' créé avec succès.")


# ──────────────────────── ENVOI ALERTE ────────────────────────────
def envoyer_alerte(es: Elasticsearch, alerte: dict, index: str = ES_INDEX) -> bool:
    """
    Indexe une alerte dans Elasticsearch.

    Paramètres attendus dans `alerte` (envoyé par P3) :
      - prediction_rf, prediction_if, prediction_nn (int : 0 ou 1)
      - score_rf, score_nn (float)
      - toutes les features réseau (proto, service, state, sbytes, ...)

    Retourne True si succès, False sinon.
    """
    # Ajouter timestamp si absent
    if "timestamp" not in alerte:
        alerte["timestamp"] = datetime.utcnow().isoformat()

    # Calcul vote majoritaire
    votes = [
        alerte.get("prediction_rf", 0),
        alerte.get("prediction_if", 0),
        alerte.get("prediction_nn", 0),
    ]
    alerte["vote_final"] = 1 if sum(votes) >= 2 else 0

    # Calcul sévérité
    score = alerte.get("score_rf", alerte.get("score_nn", 0.0))
    if alerte["vote_final"] == 0:
        alerte["severite"] = "NORMAL"
    elif score >= 0.85:
        alerte["severite"] = "HIGH"
    elif score >= 0.60:
        alerte["severite"] = "MEDIUM"
    else:
        alerte["severite"] = "LOW"

    try:
        es.index(index=index, document=alerte)
        return True
    except es_exceptions.ElasticsearchException as e:
        print(f"❌ Erreur Elasticsearch : {e}")
        return False


def envoyer_batch(es: Elasticsearch, alertes: list, index: str = ES_INDEX) -> int:
    """
    Envoie une liste d'alertes en bulk.
    Retourne le nombre d'alertes envoyées avec succès.
    """
    from elasticsearch.helpers import bulk

    actions = [
        {
            "_index": index,
            "_source": {
                **a,
                "timestamp": a.get("timestamp", datetime.utcnow().isoformat()),
                "vote_final": 1 if sum([
                    a.get("prediction_rf", 0),
                    a.get("prediction_if", 0),
                    a.get("prediction_nn", 0)
                ]) >= 2 else 0
            }
        }
        for a in alertes
    ]

    success, errors = bulk(es, actions, raise_on_error=False)
    if errors:
        print(f"⚠️  {len(errors)} erreur(s) lors du bulk insert")
    print(f"✅ {success} alerte(s) indexée(s) dans '{index}'")
    return success


# ────────────────────────── MAIN ──────────────────────────────────
if __name__ == "__main__":
    es = init_elasticsearch()
    creer_index(es)

    # Test : envoi d'une alerte factice
    test_alerte = {
        "timestamp":      datetime.utcnow().isoformat(),
        "proto":          "tcp",
        "service":        "http",
        "state":          "FIN",
        "sbytes":         1500,
        "dbytes":         300,
        "duree":          0.5,
        "prediction_rf":  1,
        "prediction_if":  1,
        "prediction_nn":  0,
        "score_rf":       0.91,
        "score_nn":       0.43,
        "attack_cat":     "Exploits",
        "source":         "flux-reseau"
    }

    ok = envoyer_alerte(es, test_alerte)
    if ok:
        print(f"\n🎯 Alerte test envoyée → sévérité : {test_alerte['severite']}")

    # Vérifier le contenu de l'index
    count = es.count(index=ES_INDEX)
    print(f"📊 Documents dans '{ES_INDEX}' : {count['count']}")
