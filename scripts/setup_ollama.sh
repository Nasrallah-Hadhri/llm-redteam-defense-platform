#!/usr/bin/env bash
# Pull required local target models via Ollama.
set -e

echo "Pulling local target models for red-teaming platform..."
ollama pull llama3:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b

echo "Setup complete. All models ready."
