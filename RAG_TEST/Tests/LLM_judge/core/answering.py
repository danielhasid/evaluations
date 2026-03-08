from typing import Dict, List

import openai

from .dataset_answering import generate_answers_for_dataset as shared_generate_answers_for_dataset
from .env import require_openai_api_key


# ─────────────────────────────────────
# CHROMA RAG CHAIN (ready to use - not active yet)
# ─────────────────────────────────────
# HOW TO ACTIVATE:
#   1. Uncomment the imports and ChromaRAGAnswerGenerator class below.
#   2. In evaluation_center.py (or wherever the generator is created), swap:
#        generator = OpenAIAnswerGenerator(...)
#      with:
#        generator = ChromaRAGAnswerGenerator(...)
#   3. Make sure persist_directory and collection_name match your Chroma DB.
#   4. Also activate the retrieval_context extraction in dataset_answering.py (see comment there).
#
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
#
# class ChromaRAGAnswerGenerator:
#     def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2):
#         llm = ChatOpenAI(model=model_name, temperature=temperature)
#         embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#         vectorstore = Chroma(
#             collection_name="rag_docs",      # must match collection name used at index time
#             persist_directory="./chroma_db", # path to your persisted Chroma DB on disk
#             embedding_function=embeddings,
#         )
#         retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", "Answer only from context. If missing, say 'I don't know.'"),
#             ("human", "Context:\n{context}\n\nQuestion: {input}")
#         ])
#         self.chain = create_retrieval_chain(
#             retriever,
#             create_stuff_documents_chain(llm, prompt)
#         )
#
#     def generate(self, messages: list) -> str:
#         # Extract question text from last message (matches OpenAIAnswerGenerator interface)
#         question = messages[-1]["content"].split("\n\n", 1)[-1]
#         result = self.chain.invoke({"input": question})
#         return result["answer"]


class OpenAIAnswerGenerator:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7):
        api_key = require_openai_api_key()
        self.model_name = model_name
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()


def generate_answer(question: str, answer_generator: OpenAIAnswerGenerator) -> str:
    messages = [{
        "role": "user",
        "content": f"Answer the following question concisely and accurately:\n\n{question}",
    }]
    return answer_generator.generate(messages)


def generate_answers_for_dataset(qa_pairs: list, answer_generator: OpenAIAnswerGenerator) -> list:
    """Backward-compatible passthrough to shared dataset answer generation."""
    return shared_generate_answers_for_dataset(qa_pairs, answer_generator=answer_generator)
