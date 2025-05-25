---
title: HTR on Swedish Historical Manuscripts using Visual Language Model
sdk: gradio
sdk_version: 5.30.0
python_version: 3.10.13
suggested_hardware: t4-small
app_file: app/main.py
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

(TODO: Update README)