FROM ubuntu:22.04

# Установка Ollama
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://ollama.com/install.sh | sh

# Открытие порта
EXPOSE 11434

# Запуск Ollama
CMD ["ollama", "serve"]
