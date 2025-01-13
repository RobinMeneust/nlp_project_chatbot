from abc import ABC

from nlp_chat_bot.rag.abstract_rag import AbstractRAG
from langchain.prompts import ChatPromptTemplate

class AbstractQueryTranslationRAG(AbstractRAG, ABC):
    """Abstract class for the Query Translation RAG model

    Attributes:
        _prompt_perspectives (ChatPromptTemplate): The prompt to generate multiple perspectives on the user question
    """
    def __init__(self, vector_store, llm=None):
        """Initializes the AbstractQueryTranslationRAG object

        Args:
            vector_store (VectorStore): The vector store to use
            llm (object): The language model to use
        """
        super().__init__(vector_store, llm)
        template = """You are an AI language model assistant. Your task is to generate three 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {question}"""
        self._prompt_perspectives = ChatPromptTemplate.from_template(template)

    def _retrieve_docs_multiple_questions(self, initial_question, k):
        """Retrieves documents for multiple questions generated from the initial question

        Args:
            initial_question (str): The initial question
            k (int): The number of documents to retrieve

        Returns:
            dict: A dictionary containing the documents and their scores
        """
        questions_gen_prompt = self._prompt_perspectives.invoke({"question": initial_question})
        questions = self.llm.invoke(questions_gen_prompt)

        questions = questions.content.split("\n")
        if len(questions) > k:
            questions = questions[:k]
        questions.append(initial_question)

        docs = {}

        for question in questions:
            docs[question], scores = zip(*self._vector_store.similarity_search_with_score(question, k))
            for doc, score in zip(docs[question], scores):
                doc.metadata["score"] = score

        return docs
