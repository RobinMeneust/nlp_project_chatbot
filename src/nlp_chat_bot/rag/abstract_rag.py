from abc import ABC, abstractmethod

from nlp_chat_bot.doc_loader.document_loader import DocumentLoader
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain import hub
from langgraph.graph import START, StateGraph

from nlp_chat_bot.vector_store.chroma_vector_store_builder import ChromaVectorStoreBuilder


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class AbstractRAG(ABC):
    def __init__(self, dataset_path, embedding_function, vector_store_path, splitter=None, llm=None, late_chunking=False):
        if llm is None:
            print("WARNING: No LLM model provided. Only retrieval can be performed.")
        self._is_late_chunking = late_chunking
        self.document_loader = DocumentLoader()
        self.splitter = splitter
        self.prompt = hub.pull("rlm/rag-prompt")
        self.llm = llm
        self._vector_store = ChromaVectorStoreBuilder(dataset_path, embedding_function, vector_store_path, splitter, late_chunking).build()
        self._graph = None


    def reset_db(self):
        self._vector_store.reset_collection()

    def get_num_docs(self):
        return len(self._vector_store.get()['documents'])

    @abstractmethod
    def retrieve(self, state: State, k: int = 3):
        raise NotImplementedError()

    @abstractmethod
    def generate(self, state: State):
        raise NotImplementedError()
    
    def invoke(self, query: State):
        if not self.llm:
            raise ValueError("No LLM model provided at RAG object initialization. Cannot generate response.")

        return self._graph.invoke(query)
    
    def _compile(self):
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        self._graph = graph_builder.compile()
