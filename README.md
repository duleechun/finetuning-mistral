# Finetune-Mistral
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Model: Mistral-7B](https://img.shields.io/badge/Model-Mistral--7B-red)
![Status: Training](https://img.shields.io/badge/Status-Actively_Training-brightgreen)
![Last Updated](https://img.shields.io/github/last-commit/duleechun/finetuning-mistral)

This project fine-tunes the Mistral-7B model using Alpaca-style JSONL data formatted from real-world chat logs.  
Includes scripts for QLoRA-based optimization, training with 4-bit quantization, and environment setup.

## Structure

- `alpaca_data.jsonl` – Training data (Alpaca format)
- `run_qlora_finetune.py` – Main training script
- `qlora-env/` – Environment setup for QLoRA
- `finetune-*` – Model-specific tuning directories

## Goals

- Efficient fine-tuning on 8GB VRAM
- Compatibility with local inference frameworks like `llama.cpp`, `transformers`, and `Ollama`
- Optional integration into RAG systems

## Usage

Coming soon...

## Author

Abdulkadir Mohamed Haji  
📧 ahaji14@wgu.edu  
🔗 [github.com/duleechun](https://github.com/duleechun)
