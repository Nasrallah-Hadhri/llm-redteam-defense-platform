@echo off
REM Pull required local target models via Ollama on Windows.
echo Pulling local target models for red-teaming platform...

ollama pull llama3:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b

echo Setup complete. All models ready.
pause
