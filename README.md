# CDR: Customizable Density Ratios for Preference Annotation

Official repository for the paper: "CDR: Customizable Density Ratios of Strong-over-weak LLMs for Preference Annotation"

[![arXiv](https://img.shields.io/badge/arXiv-2411.02481-b31b1b.svg)](https://arxiv.org/abs/2411.02481)

## Installation

# Create and activate conda environment

```bash
conda create -n cdr python=3.10
conda activate cdr
Install dependencies
pip install vllm
Clone and install repository
git clone <repo link>
cd CDR-preference-annotation
pip install -e .
```


## Using CDR Reward

### Model Setup
The following models are required with their respective GPU requirements:

| Model Role | Model Name | GPU Requirements | Port |
|------------|------------|------------------|------|
| Policy | meta-llama/Meta-Llama-3-8B-Instruct | 2 GPUs | 8021 |
| Strong | NousResearch/Nous-Hermes-2-Mistral-7B-DPO | 3 GPUs | 8022 |
| Weak | teknium/OpenHermes-2.5-Mistral-7B | 4 GPUs | 8023 |

### Quick Start
1. Launch the vLLM serving pipeline:

```bash
bash launch_vllm_pipeline.sh
```
> Note: You can customize the policy, strong, or weak models by modifying the `launch_vllm_pipeline.sh` script.

2. Follow the examples in `demo.ipynb` to implement the reward function.

## Released Models

We provide several models fine-tuned from Llama-3-8b-instruct using different reward functions:

### Available Checkpoints
- **Best CDR Model** (DPO-vs-SFT reward choice): [Download](dummy-link)
- **CDR Model** (SFT-vs-base reward choice): [Download](dummy-link)
- **ArmoRM Model** (State-of-the-art trained classifier reward): [Download](dummy-link)

## Citation

If you find this work useful, please cite our paper:

bibtex
@article{xu2023cdr,
title={CDR: Customizable Density Ratios of Strong-over-weak LLMs for Preference Annotation},
author={Xu, Guangxuan and Xu, Kai and Sudalairaj, Shivchander and Wang, Hao and Srivastava, Akash},
journal={arXiv preprint arXiv:2411.02481},
year={2023}
}

