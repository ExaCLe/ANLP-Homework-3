import ollama
import numpy as np
from tqdm import tqdm

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"


class VectorDB:
    def __init__(self, embedding_model_str: str):
        self.embedding_model_str = embedding_model_str
        self.documents = []

    def add_document(self, document: str):
        embedding = self.embed_text(document)
        self.documents.append(VectorDBEntry(text=document, embedding=embedding))

    def add_documents(self, documents: list[str]):
        for document in documents:
            self.add_document(document)

    def embed_text(self, text: str) -> np.ndarray:
        return np.array(
            ollama.embed(model=EMBEDDING_MODEL, input=text)["embeddings"][0]
        )

    def retrieve_top_n(self, query: str, n: int = 5) -> list[str]:
        query_embedding = self.embed_text(query)
        print("Computing similarities...")
        similarities: list[tuple[str, float]] = [
            (entry.text, cosine_similarity(query_embedding, entry.embedding))
            for entry in tqdm(self.documents)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in similarities[:n]]


class VectorDBEntry:
    def __init__(self, text: str, embedding: np.ndarray):
        self.text = text
        self.embedding = embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)  # pyrefly: ignore
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return np.dot(a, b) / (a_norm * b_norm)  # pyrefly: ignore


def load_cats_dataset() -> list[str]:
    dataset: list[str] = []
    with open("cat-facts.txt", "r") as file:
        for line in file:
            line = line.strip()
            if line:
                dataset.append(line)

    return dataset


class LLM:
    def __init__(self, model_str: str):
        self.model_str = model_str

    def generate(self, prompt: str):
        response = ollama.chat(
            model=self.model_str,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        for chunk in response:
            yield chunk.message.content


def main():
    # put the dataset into the vector database
    vector_db = VectorDB(embedding_model_str=EMBEDDING_MODEL)
    dataset = load_cats_dataset()
    vector_db.add_documents(dataset)
    llm = LLM(model_str=LANGUAGE_MODEL)

    while True:
        query = input("Enter a query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        results = vector_db.retrieve_top_n(query, n=10)
        results_str = "\n".join(results)
        instruction_prompt = f"""You are a helpful chatbot.
        Use only the following pieces of context to answer the question. Don't make up any new information:
        {results_str}"""
        prompt = f"{instruction_prompt}\n\nQuestion: {query}\nAnswer:"
        print(f"Prompt: {prompt}")
        print("Generating response...")
        for chunk in llm.generate(prompt):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
