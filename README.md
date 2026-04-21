# Federated SLM Text Classification

This repository contains the code, notebooks, experimental logs, and result files for the paper:

**A Federated Learning Approach Based on Fine-Tuning Small Language Models for Text Classification**

## Overview

This project investigates federated text classification using small language models (SLMs) under non-IID data distributions. The study focuses on the trade-off between classification performance and deployment efficiency by comparing:

- Traditional machine learning baselines
- Centralized Transformer baselines
- Federated small-language-model settings
- A larger federated baseline based on BERT-base

The main lightweight backbones are **TinyBERT** and **DistilBERT**, combined with **LoRA-based parameter-efficient fine-tuning** under **FedAvg**.

## Datasets

The experiments are conducted on three benchmark text classification datasets:

- **AG News**: 4-class topic classification
- **IMDB**: binary sentiment classification
- **SST-2**: sentence-level sentiment classification from GLUE

## Compared Methods

### Traditional baselines
- TF-IDF with Logistic Regression
- TF-IDF with Support Vector Machine

### Centralized Transformer baselines
- Centralized TinyBERT with full fine-tuning
- Centralized TinyBERT with LoRA
- Centralized DistilBERT with full fine-tuning
- Centralized DistilBERT with LoRA

### Federated baselines
- Federated TinyBERT with LoRA
- Federated DistilBERT with LoRA
- Federated BERT-base with full fine-tuning

## Repository Structure

```text
federated-slm-text-classification/
├── bertbase/
│   ├── ag news/
│   ├── imbd/
│   └── SST2/
├── dis/
│   ├── AG NEWS/
│   ├── imdb/
│   └── SST2/
├── tiny/
│   ├── AG NEW/
│   ├── IMDB/
│   └── SST2/
├── classical_results.csv
├── TF_IDF_+_Logistic_Regression_và_TF_IDF_+_SVM_chạy_trên_cả_AG_News,_IMDB,_SST_2,.ipynb
└── README.md
