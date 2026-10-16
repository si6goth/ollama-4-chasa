"""
Задание 8: Вызов инструментов (Tool Calling)
Использует механизм вызова внешних инструментов через Ollama.
"""

import ollama


def get_weather(city: str) -> str:
    """Имитация получения погоды."""
    weathers = {
        "москва": "В Москве +20°C, ясно",
        "санкт-петербург": "В Санкт-Петербурге +15°C, облачно",
        "новосибирск": "В Новосибирске +10°C, дождь",
    }
    return weathers.get(city.lower(), f"Неизвестный город: {city}")


if __name__ == "__main__":
    response = ollama.chat(
        model="qwen3",
        messages=[{"role": "user", "content": "Какая сейчас погода в Москве?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Получить погоду в городе",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
    )
    print("Ответ модели:")
    print(response)
