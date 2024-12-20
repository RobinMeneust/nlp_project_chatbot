from nlp_chat_bot.doc_loader.document_loader import DocumentLoader
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain import hub
from langgraph.graph import START, StateGraph


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class RAG():
    def __init__(self, dataset_path, splitter, vector_store, llm=None):
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
    

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma  import Chroma
from nlp_chat_bot.model.minilm import MiniLM

if __name__ == "__main__":
    dataset_path = "data"
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  # chunk size (characters)
        chunk_overlap=10,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    
    minilm = MiniLM(model_download_path="models")
    vector_store = Chroma(embedding_function=minilm)
    rag = RAG(dataset_path, splitter, vector_store)
    print("LENGTH", len(vector_store.get()['documents']))
    docs_retrieved = rag.retrieve(state = {"question": "What is the acronym AIA?", "context": []})
    
    print("Num docs:", len(docs_retrieved["context"]))
    
    for i in range(len(docs_retrieved["context"])):
        doc = docs_retrieved["context"][i]
        print("\n\n", "#"*30,"\n")
        print(f"doc {i}: (score: {doc.metadata['score']})")
        print(doc.page_content)
    print(docs_retrieved["context"][0].page_content)    
    
    # state = {"question": "What are some ways to reduce stress?", "context": []}
    # response = rag.invoke(state)
    # print(response)