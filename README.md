# Les commandes à executer en ordre pour avoir une compatibilité avec le projet
```bash

sudo apt update
sudo apt install -y git docker.io docker-compose-v2

sudo systemctl start docker
sudo systemctl enable docker

sudo usermod -aG docker $USER

git --version
docker --version
docker compose version

cd ~/projet-siem-bigdata-F

sudo apt update
sudo apt install -y python3.13-venv python3.13-full

python3 -m venv venv
source venv/bin/activate

sudo apt update
sudo apt install -y python3-pip

python3 -m pip install --upgrade pip setuptools wheel --break-system-packages

pip3 install pyspark==3.5.0
python3 -c "import pyspark; print(pyspark.__version__)"

sudo apt update
sudo apt install -y default-jdk

echo 'export JAVA_HOME=/usr/lib/jvm/default-java' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

java -version
echo $JAVA_HOME

pip install --upgrade pip

pip install kafka-python pyspark pandas numpy scikit-learn xgboost tensorflow elasticsearch pymongo paramiko requests matplotlib joblib

pip3 install kaggle --break-system-packages

newgrp docker

docker compose up -d

docker compose ps
docker compose up -d kafka
sleep 60
docker compose ps

python3 kafka_config.py

pip install elasticsearch==8.13.0 --break-system-packages
python3 elasticsearch_config.py
python3 mongodb_config.py
python3 data_quality.py --data-dir ./data

```

# POUR TESTER LE PROJET POUR LA PREMIÈRE FOIS

```bash

newgrp docker
docker compose up -d

docker compose up -d kafka
sleep 60
docker compose ps


cd spark_ml
source venv/bin/activate

python3 preprocessing.py
python3 train_random_forest.py
python3 train_isolation_forest.py
python3 train_neural_network.py
python3 train_xgboost.py
python3 evaluate_models.py
```

```bash

cd spark_ml
source venv/bin/activate
python3 elasticsearch_sender.py
```

SI le ficher marche et s'arrete sans message erreur donc cela marche, alors ouvre un nouveau terminal et fait :

```bash

cd ~/projet-siem-bigdata-F
source venv/bin/activate
python3 producer.py
```
Si le fichier commence l'execution retourne vers l'autre terminal, apres quelque min tu vas voir les données s'affichent et classées.
cela peut prendre 20min.

# RMQ : POUR RÉINITIALISER LES BASE DE DONNÉE
## Pour elastic
```bash
curl -X DELETE "http://localhost:9200/alertes-securite"
curl -X DELETE "http://localhost:9200/siem-alertes"
curl -X DELETE "http://localhost:9200/siem-window-stats"
```
## Pour Mongodb
```bash
docker exec -it projet-siem-bigdata-f-mongodb-1 mongo
```
dans l'interface de MONGOdb :

```bash

use siem_db

show collections

db.logs_bruts.deleteMany({})

db.alertes.deleteMany({})
```

# SI N'EST PAS LA PREMIÈRE FOIS :
même étapes juste on n'éxecute pas l'entrainement des modèles.

# CONSULTER MONGODB
```bash
docker exec -it projet-siem-bigdata-f-mongodb-1 mongo
```
dans l'interface de MONGOdb :

```bash

use siem_db

show collections

db.logs_bruts.countDocuments({})

db.alertes.countDocuments({})
```

# VOIR LES DASHBOARDS SUR ELASTIC
Elasticsearch : http://localhost:9200⁠
1- Ouvre Kibana
2- Stack management>Data views>créer data views
3- Name : SIEM Alertes
   Index pattern : siem-alertes
4- save
5- Va dans Stack Management → Saved Objects
6- Clique sur Import
7- Sélectionne "Dashboards.ndjson"



# LES AUTRES LOCAL HOSTS
Kibana : http://localhost:5601⁠
Spark UI : http://localhost:8080⁠


