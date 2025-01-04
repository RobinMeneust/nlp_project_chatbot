from abc import ABC

from nlp_chat_bot.rag.abstract_rag import AbstractRAG
from langchain.prompts import ChatPromptTemplate

class AbstractQueryTranslationRAG(AbstractRAG, ABC):
    def __init__(self, dataset_path, embedding_function, vector_store_path, llm, splitter=None, late_chunking=False):
        super().__init__(dataset_path, embedding_function, vector_store_path, splitter, llm, late_chunking)
        template = """You are an AI language model assistant. Your task is to generate three 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {question}"""
        self._prompt_perspectives = ChatPromptTemplate.from_template(template)

    def _retrieve_docs_multiple_questions(self, initial_question, k):
        questions_gen_prompt = self._prompt_perspectives.invoke({"question": initial_question})
        questions = self.llm.invoke(questions_gen_prompt)

        docs = {}

        for question in questions.content.split("\n"):
            docs[question], scores = zip(*self._vector_store.similarity_search_with_score(question, k))
            for doc, score in zip(docs[question], scores):
                doc.metadata["score"] = score

        return docs
