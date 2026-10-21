## Execution

Follow these steps to run the SIEM Big Data project on Ubuntu. Open three terminal windows and keep them running while the pipeline is active.

### 1. Terminal 1 — Environment Setup and Main Pipeline

Navigate to the project directory:

```bash
cd ~/projet-siem-bigdata-F
```

#### Java 17

Install Java 17 and select it as the default version:

```bash
sudo apt update
sudo apt install -y openjdk-17-jdk
sudo update-alternatives --config java

export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"

java -version
```

#### Python Environment

Create the Python 3.11 environment and install the project dependencies using `uv`:

```bash
uv python pin 3.11
uv venv --python 3.11 .venv311
uv pip install --python .venv311/bin/python -r requirements.txt
```

If `.venv311` already exists, reuse it instead of recreating it.

#### Docker Services

Install Docker, Docker Compose, and the required utilities:

```bash
sudo apt install -y git docker.io docker-compose-v2 util-linux-extra
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Apply the new Docker group membership in the current terminal:

```bash
newgrp docker
```

Start the services and check their status:

```bash
docker compose up -d
docker compose ps

docker compose up -d kafka
sleep 60
docker compose ps
```

#### Initialize the Components

Activate the virtual environment and run the component configuration scripts:

```bash
source .venv311/bin/activate

python kafka_config.py
python elasticsearch_config.py
python mongodb_config.py
python data_quality.py --data-dir ./data
```

Launch the main SIEM pipeline:

```bash
cd spark_ml
python elasticsearch_sender.py
```

Keep this terminal running.

### 2. Terminal 2 — Real-Time Windowing Analysis

Open a second terminal:

```bash
cd ~/projet-siem-bigdata-F
source .venv311/bin/activate
cd spark_ml
```

Select Java 17 if multiple Java versions are installed, then set the environment variables for this terminal:

```bash
sudo update-alternatives --config java

export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"

java -version
```

Run the windowing analysis:

```bash
python windowing.py
```

Keep this terminal running.

### 3. Terminal 3 — Network Traffic Producer

Open a third terminal:

```bash
cd ~/projet-siem-bigdata-F
source .venv311/bin/activate
python3 producer.py
```

Keep this terminal running while generating traffic.

### Execution Order

1. **Terminal 1:** Start Docker services, initialize Kafka/Elasticsearch/MongoDB, check data quality, and launch `elasticsearch_sender.py`.
2. **Terminal 2:** Launch `windowing.py` for sliding and tumbling window analysis.
3. **Terminal 3:** Run `producer.py` to send network traffic to Kafka.

All three terminals should remain open while the pipeline is running.
