from nlp_chat_bot.doc_loader.document_loader import DocumentLoader
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain import hub
from langgraph.graph import START, StateGraph


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class RAG:
    def __init__(self, dataset_path, vector_store, splitter=None, llm=None):
        print("WARNING: No LLM model provided. Only retrieval can be performed.")
        self.document_loader = DocumentLoader()
        self.splitter = splitter
        self.vector_store = vector_store
        self.prompt = hub.pull("rlm/rag-prompt")
        self.llm = llm
        self.graph = self._compile()
        self._store_documents(dataset_path)
    
    def _store_documents(self, data_path):
        docs = self.document_loader.load(data_path)
        if self.splitter is not None:
            docs = self.splitter.split_documents(docs)
        
        if not docs or len(docs) == 0:
            raise ValueError("No document found")
        
        self.vector_store.add_documents(documents=docs)
    
    def retrieve(self, state: State, k: int = 3):
        docs, scores = zip(*self.vector_store.similarity_search_with_score(state["question"], k))
        for doc, score in zip(docs, scores):
            doc.metadata["score"] = score
        
        return {"context": docs}
    
    def generate(self, state: State):
        if not self.llm:
            raise ValueError("No LLM model provided at RAG object initialization. Cannot generate response.")
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}
    
    def invoke(self, query: State):
        if not self.llm:
            raise ValueError("No LLM model provided at RAG object initialization. Cannot generate response.")
        return self.graph.invoke(query)
    
    def _compile(self):
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()
