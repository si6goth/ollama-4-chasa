"""
Задание 7: Структурированный вывод (JSON-режим)
Запрашивает у модели данные в формате JSON и парсит ответ.
"""

import json
import ollama


def get_json_response(prompt: str, model: str = "llama3.2") -> dict:
    """Получение структурированного JSON-ответа от модели."""
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": 'Отвечай только в формате JSON. Всегда используй ключи: "name", "age", "city"',
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw_content = response["message"]["content"]
    try:
        data = json.loads(raw_content)
        return data
    except json.JSONDecodeError as e:
        print(f"Ошибка парсинга JSON: {e}")
        print(f"Полученный текст: {raw_content}")
        return {}


if __name__ == "__main__":
    prompt = "Представь вымышленного человека"
    print(f"Запрос: {prompt}\n")
    result = get_json_response(prompt)
    if result:
        print("Распарсенные данные:")
        print(f"  Имя: {result.get('name', 'N/A')}")
        print(f"  Возраст: {result.get('age', 'N/A')}")
        print(f"  Город: {result.get('city', 'N/A')}")
