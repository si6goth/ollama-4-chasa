"""
Задание 5: Интеграция Ollama с Python (библиотека ollama)
Подключается к локальному серверу Ollama и отправляет запрос на генерацию текста.
"""

import ollama


def generate_text(prompt: str, model: str = "llama3.2") -> str:
    """Генерация текста через Ollama."""
    response = ollama.generate(model=model, prompt=prompt)
    return response["response"]


if __name__ == "__main__":
    prompt = "Напиши функцию на Python для вычисления факториала"
    print(f"Запрос: {prompt}\n")
    result = generate_text(prompt)
    print("Ответ:")
    print(result)
