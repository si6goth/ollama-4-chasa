"""
Задание 6: Потоковая передача ответов (Streaming)
Использует потоковый режим для вывода текста в реальном времени.
"""

import time
import ollama


def stream_response(prompt: str, model: str = "llama3.2") -> None:
    """Потоковая генерация ответа."""
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        time.sleep(0.01)  # небольшая задержка для визуализации
    print()


if __name__ == "__main__":
    prompt = "Расскажи короткую историю про космос"
    print(f"Запрос: {prompt}\nОтвет:\n")
    stream_response(prompt)
