from kafka import KafkaProducer, KafkaConsumer
import json

# Test envoi d'un message dans chaque topic
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topics = ['logs-systeme', 'flux-reseau', 'threat-intel']

for topic in topics:
    producer.send(topic, {'test': f'Kafka topic {topic} fonctionne !'})
    print(f'Message envoyé dans {topic}')

producer.flush()
print('Kafka est prêt !')
