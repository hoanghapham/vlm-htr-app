---
title: HTR with VLM
sdk: gradio
sdk_version: 5.30.0
python_version: 3.10.13
suggested_hardware: t4-small
app_file: src/gradio_ui/main.py
models:
    - nazounoryuu/florence_base__mixed__page__line_od
    - nazounoryuu/florence_base__mixed__line_bbox__ocr
tags:
    - htr
    - ocr
    - text_detection
    - vlm
    - historical
    - swedish
    - handwritten
    - manuscripts
preload_from_hub:
    - nazounoryuu/florence_base__mixed__page__line_od
    - nazounoryuu/florence_base__mixed__line_bbox__ocr
---

# HTR on Swedish Historical Manuscripts using Visual Language Model

## Installation guide

Run the following commands:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment with uv
uv venv --python 3.11

# Activate the virtual environment
source .venv/bin/activate

# Install packages (Recommended)
uv sync

# Or install using uv pip
uv pip install -r requirements.txt
```

## Run locally

```bash
# Run using gradio
gradio src/app/main.py

# Run using python
python src/app/main.py
```

## Deploy on HuggingFace Space
(To be updated)

## Run with Docker
(To be updated)
