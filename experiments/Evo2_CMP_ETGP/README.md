# Finetune Evo2 on Contact map prediction and ETGP tasks

## Setup

We follow the original Evo2 environment setup

### Requirements

Evo 2 is based on [StripedHyena 2](https://github.com/Zymrael/vortex) which requires python>=3.11. Evo 2 uses [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) FP8 for some layers which requires an H100 (or other GPU with compute capability ≥8.9). We are actively investigating ways to avoid this requirement.

### Installation

To install Evo 2 for inference or generation, please clone and install from GitHub. We recommend using a new conda environment with python>=3.11.

```bash
git clone --recurse-submodules git@github.com:ArcInstitute/evo2.git
cd evo2
pip install .
```

## Finetuning

## Contact Map Prediction Task

### Finetuning: Taking cell type HFF as an example

```markdown
python test/filetune_contact_map.py
```

### Inference

```markdown
python test/evaluate_contact_map.py
```

## ETGP Task 

### Finetuning

```markdown
python test/finetune_ETGP.py
```

### Inference

```markdown
python test/evaluate_ETGP.py
```