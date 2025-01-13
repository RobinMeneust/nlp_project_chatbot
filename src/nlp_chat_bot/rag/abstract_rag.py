import gc
from abc import ABC, abstractmethod

import torch
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain import hub
from langgraph.graph import START, StateGraph


class State(TypedDict):
    """State dictionary for the RAG model

    Attributes:
        question (str): The question to ask
        context (List[Document]): The context documents
        answer (str): The answer to the question
        chat_history (List[str]): The chat history
    """
    question: str
    context: List[Document]
    answer: str
    chat_history: List[str]

class AbstractRAG(ABC):
    """Abstract class for the RAG model

    Attributes:
        prompt (str): The prompt to use
        llm (object): The language model to use
        _vector_store (VectorStore): The vector store to use
        _graph (StateGraph): The state graph used with Langchain to use the RAG model
    """
    def __init__(self, vector_store, llm=None):
        """Initializes the AbstractRAG object

        Args:
            vector_store (VectorStore): The vector store to use
            llm (object): The language model to use
        """
        if llm is None:
            print("WARNING: No LLM model provided. Only retrieval can be performed.")
        self.prompt = hub.pull("rlm/rag-prompt")
        self.llm = llm
        self._vector_store = vector_store
        self._graph = None


    def reset_db(self):
        """Resets the vector store (delete all documents)"""
        self._vector_store.reset_collection()

    def get_num_docs(self):
        """Returns the number of documents in the vector store

        Returns:
            int: The number of documents
        """
        return len(self._vector_store.get()['documents'])

    @abstractmethod
    def retrieve(self, state: State, k: int = 3):
        """Retrieves documents for the given state (i.e. question, and eventually context)

        Args:
            state (State): The state to use
            k (int): The number of documents to retrieve

        Raises:
            NotImplementedError: If the method is not implemented (because it's abstract)
        """
        raise NotImplementedError()

    @abstractmethod
    def generate(self, state: State):
        """Generates a response for the given state (i.e. question and context (docs...))

        Args:
            state (State): The state to use

        Raises:
            NotImplementedError: If the method is not implemented (because it's abstract)
        """
        raise NotImplementedError()
    
    def invoke(self, query: State):
        """Invokes the RAG model with the given query

        Args:
            query (State): The query to use

        Returns:
            State: The response from the RAG model

        Raises:
            ValueError: If no LLM model was provided to this RAG object
        """
        if not self.llm:
            raise ValueError("No LLM model provided at RAG object initialization. Cannot generate response.")

        return self._graph.invoke(query)
    
    def _compile(self):
        """Compiles the state graph for the RAG model"""
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        self._graph = graph_builder.compile()

    def __del__(self):
        """Deletes the LLM model (free VRAM)"""
        del self.llm
