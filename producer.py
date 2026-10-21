import pandas as pd
from kafka import KafkaProducer
import json
import time

# Configuration du producer Kafka
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# =====================================================
# 1. Envoi du dataset UNSW-NB15 vers le topic logs-systeme
# =====================================================
print("Envoi logs système...")

df_logs = pd.read_csv('data/OpenSSH_2k_log_structured.csv')

for _, row in df_logs.iterrows():
    producer.send('logs-systeme', value=row.to_dict())
    time.sleep(0.0001)

print(f"{len(df_logs)} logs envoyés !")

# =====================================================
# 2. Envoi du même dataset vers le topic flux-reseau
# =====================================================
print("Envoi flux réseau...")

df_network = pd.read_csv('data/UNSW_NB15_testing-set.csv')

for _, row in df_network.iterrows():
    producer.send('flux-reseau', value=row.to_dict())
    time.sleep(0.0001)

print(f"{len(df_network)} flux réseau envoyés !")

# =====================================================
# 3. Finalisation
# =====================================================
producer.flush()
print("Toutes les données ont été envoyées vers Kafka !")
