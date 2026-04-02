# EdgeMamba-3

**Edge-Centric State Space Model for Graph Learning**

Yajat Kapur · 22BBS0110 · CBS1904 Capstone

## Overview

EdgeMamba-3 is a linear-complexity graph learning architecture that combines:
- **Line Graph Transformation**: Converts edges to nodes for edge-centric processing
- **LTAS (Linear Topology-Aware Serialization)**: O(L log L) learned ordering via GATConv
- **Bidirectional Mamba-3 Encoder**: SSM with trapezoidal discretization, complex-valued states, and MIMO rank-R

Evaluated on:
- **LRGB**: Peptides-func (classification), Peptides-struct (regression)
- **RelBench**: rel-hm user-churn (classification), rel-amazon user-ltv (regression)

## Quick Start

```bash
# Setup environment
conda create -n edgemamba3 python=3.10 -y
conda activate edgemamba3

# Install dependencies
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Install Mamba-3 from source
git clone https://github.com/state-spaces/mamba.git
cd mamba && pip install -e ".[causal-conv1d]" && cd ..

# Install project
pip install -e .

# Validate environment
python scripts/validate_env.py

# Run tests
pytest tests/ -v

# Run experiments
python scripts/train_lrgb.py --config configs/lrgb_peptides_func.yaml
```

## Docker

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

## Project Structure

```
edgemamba3/
├── data/           # Data loaders (LRGB, RelBench)
├── models/         # Core model modules
├── baselines/      # Ablation baselines
├── train/          # Training infrastructure
├── ablations/      # Ablation study runner
├── configs/        # Experiment configs (YAML)
├── scripts/        # Entry points and utilities
├── tests/          # Unit tests
└── docker/         # Docker setup
```
