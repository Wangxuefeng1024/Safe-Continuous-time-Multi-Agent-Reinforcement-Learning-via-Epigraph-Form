# Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form

This repository contains the official implementation of the ICLR 2026 paper:

**Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form**

📄 Paper (OpenReview):  
https://openreview.net/forum?id=U6z5Y1htbe

---

## Overview

This codebase implements an **Epigraph-based Continuous-Time Multi-Agent Reinforcement Learning (EPI)** framework for safety-constrained multi-agent systems.  
The proposed method reformulates the continuous-time constrained HJB equation using an epigraph form, enabling stable and scalable learning without manual penalty tuning.

The implementation focuses on:
- continuous-time multi-agent reinforcement learning,
- safety constraint enforcement via epigraph reformulation,
- value gradient iteration (VGI),

---

## Repository Structure

The repository is organized as follows:

```
EPI_continuous_marl/
├── algo/
│   ├── epigraph_pinn/
│   │   ├── epi_pinn_agent.py   # Core implementation of the EPI algorithm
│   │   ├── network.py          # Neural network definitions (policy, value, dynamics, cost)
│   │   └── __pycache__/
│   ├── main.py                 # Off-the-shelf training script (entry point)
│   ├── memory.py               # Replay buffer utilities
│   ├── normalized_env.py       # Environment normalization
│   ├── random_process.py       # Exploration noise utilities
│   └── utils.py                # Common helper functions
├── continuous_env/              # Continuous-time environments
├── requirements.txt
└── README.md
```

---

## Code Description

### `main.py`
`main.py` serves as an **off-the-shelf training script**.  
It handles:
- argument parsing,
- training loops,
- logging and checkpointing.

This file is the main entry point for running experiments.

---

### `epi_pinn_agent.py`
This file contains the **core implementation of the proposed Epigraph-based algorithm (EPI)**, including:
- epigraph-form value learning,
- safety (constraint) critic training,
- continuous-time Bellman/HJB residuals,
- value gradient iteration (VGI),
- policy optimization via Hamiltonian minimization.


---

## Installation

### Install Dependencies

We recommend using a virtual environment.

```bash
pip install -r requirements.txt
```

---

## Running an Example

All experiments are launched through `main.py`.

### Example Command

```bash
python main.py --algo epi --scenario formation --seed 113
```

## Reference

If you find this repository useful, please refer to our paper:

**Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form**  
ICLR 2026  

```bash
@article{wang2026safe,
  title={Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form},
  author={Wang, Xuefeng and Zhang, Lei and Pu, Henglin and Li, Husheng and Qureshi, Ahmed H},
  journal={arXiv preprint arXiv:2602.17078},
  year={2026}
}
```
---
