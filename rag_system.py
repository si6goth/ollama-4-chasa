"""
Задание 9: Создание простой RAG-системы
Система дополненной генерации с локальной LLM и векторным поиском.
"""

import json
import ollama
from sentence_transformers import SentenceTransformer, util


class SimpleRAG:
    """Простая RAG-система с эмбеддингами и Ollama."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", llm_model: str = "llama3.2"):
        self.embedder = SentenceTransformer(model_name)
        self.llm_model = llm_model
        self.documents = []
        self.embeddings = None

    def add_documents(self, texts: list[str]) -> None:
        """Добавление документов и создание эмбеддингов."""
        self.documents = texts
        self.embeddings = self.embedder.encode(texts, convert_to_tensor=True)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Поиск наиболее релевантных документов."""
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_indices = scores.topk(top_k)
        return [self.documents[idx.item()] for idx in top_indices.indices]

    def answer(self, query: str, top_k: int = 3) -> str:
        """Генерация ответа на основе найденных документов."""
        relevant_docs = self.search(query, top_k)
        context = "\n\n".join(relevant_docs)
        prompt = (
            f"Ответь на вопрос, используя следующую информацию:\n\n"
            f"{context}\n\n"
            f"Вопрос: {query}\n\n"
            f"Ответ:"
        )
        response = ollama.generate(model=self.llm_model, prompt=prompt)
        return response["response"]


if __name__ == "__main__":
    # Пример документов
    docs = [
        "Python — высокоуровневый язык программирования общего назначения. "
        "Он поддерживает структурное, объектное и функциональное программирование.",
        "JavaScript — язык программирования, который обычно внедряется как часть веб-страниц. "
        "Он обеспечивает динамическое поведение для интерактивных сайтов.",
        "Docker — платформа для разработки, доставки и запуска приложений в контейнерах. "
        "Контейнеры позволяют отделять приложения от инфраструктуры.",
        "Ollama — инструмент для запуска больших языковых моделей локально. "
        "Он поддерживает такие модели, как Llama, Mistral и Qwen.",
        "RAG (Retrieval-Augmented Generation) — подход, сочетающий поиск информации "
        "и генерацию текста языковой моделью для более точных ответов.",
    ]

    rag = SimpleRAG()
    rag.add_documents(docs)

    query = "Что такое Ollama?"
    print(f"Запрос: {query}\n")
    answer = rag.answer(query)
    print("Ответ:")
    print(answer)
