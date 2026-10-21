# Real-Time SIEM Platform for Network Anomaly Detection

## Project Overview

This project implements a **Security Information and Event Management (SIEM) platform** designed to detect network intrusions and analyze potential security threats in real time using Big Data technologies and machine learning.

The platform processes network traffic streams, identifies suspicious activity, and generates alerts that can be monitored and analyzed through a centralized architecture.

## How It Works

The platform combines distributed stream processing, machine learning, and data storage to build a real-time threat detection pipeline.

1. **Data ingestion — Apache Kafka**  
   Kafka receives and distributes network traffic data through streaming topics.

2. **Stream processing — Apache Spark Structured Streaming**  
   Spark processes incoming data streams and prepares network flow records for analysis.

3. **Machine learning — Ensemble-based detection**  
   Four models analyze network activity:
   - Random Forest
   - Isolation Forest
   - Multilayer Perceptron (MLP)
   - XGBoost

   Their predictions are combined using a majority-voting strategy to support the detection of potentially malicious traffic and anomalous behavior.

4. **Alerting and visualization — Elasticsearch and Kibana**  
   Detected threats are sent to Elasticsearch for indexing and can be explored through Kibana dashboards.

5. **Data storage — MongoDB**  
   MongoDB stores raw log data for further investigation and analysis.

## Results

Among the four evaluated models, **XGBoost achieved the best overall performance**, with an F1-score of **91.65%** and an ROC-AUC of **98.36%** on the UNSW-NB15 test set. These results highlight its ability to distinguish between normal and malicious network traffic while maintaining a strong balance between precision and recall.

The model also reported an inference latency of **0.0029 ms per sample**, indicating promising potential for fast network traffic classification.

## Technology Stack

- **Programming:** Python
- **Stream ingestion:** Apache Kafka
- **Stream processing:** Apache Spark Structured Streaming
- **Machine learning:** Scikit-learn, XGBoost, MLP Neural Network
- **Search and alert indexing:** Elasticsearch
- **Visualization:** Kibana
- **Log storage:** MongoDB
- **Deployment and services:** Docker and Docker Compose

## Execution
**Prerequisites**
- Ubuntu
- `uv` for Python environment and dependency management
- Python 3.11
- Java 17
- Docker and Docker Compose
  
Follow the [**Execution_Guide**](Execution_Guide.md) to configure the environment and run the platform.


> **Important:** Keep the existing execution commands and terminal-by-terminal instructions in this section. They cover environment setup, service startup, data quality checks, Elasticsearch configuration, Spark streaming, and the Kafka producer. Follow the execution order specified in the existing instructions.

## Project Team

Developed as an academic Big Data and Artificial Intelligence project at the **National School of Applied Sciences of Tetouan (ENSA Tetouan), Morocco**.

**Team members**
- SOULAMA Doubie Judicael Noé
- EL ALAOUI SALEH Zouhair
- PROBI Nouhaila
- EL JABRI Hafsa
- SIMBRE Achraf

**Supervisor:** Pr. SASSI Imad

**Academic year:** 2025–2026

## Full Project Report

For the complete academic report, including the methodology, architecture, implementation details, and extended evaluation, see:

[**Read the full project report (PDF)**](Rapport.pdf)


