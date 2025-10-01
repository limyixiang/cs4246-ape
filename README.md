# CS4246-APE

This repository provides a reference implementation of Automatic Prompt Engineer (APE) inspired by the paper [Automatic Prompt Engineer](https://arxiv.org/abs/2211.01910). The library exposes a modular search pipeline that can be driven by Hugging Face models or any custom language model backend.

## Features

- Meta-prompt based generation of instruction candidates
- Evaluation of candidates against validation examples using an arbitrary language model
- Optional self-refinement step powered by a separate critic model
- Command line interface for running the search over JSONL datasets
- Lightweight test suite with deterministic dummy models

## Installation

```bash
pip install -r requirements.txt
```

or install the minimum dependencies manually:

```bash
pip install transformers
```

## Usage

1. Prepare two JSONL files containing training and evaluation examples. Each line should have the keys `input` and `output`:

```json
{"input": "5 6", "output": "11"}
```

2. Run the APE search pipeline:

```bash
python scripts/run_ape.py \
  --train path/to/train.jsonl \
  --eval path/to/eval.jsonl \
  --generator-model google/flan-t5-base \
  --task-model google/flan-t5-base \
  --evaluator-model google/flan-t5-base
```

The script prints the top instructions along with their validation scores.

## Testing

Run the unit tests with:

```bash
pytest
```
