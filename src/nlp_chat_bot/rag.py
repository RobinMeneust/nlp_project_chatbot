import chromadb
from langchain_chroma import Chroma

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
    def __init__(self, dataset_path, embedding_function=None, splitter=None, llm=None, late_chunking=False, vector_store=None):
        if llm is None:
            print("WARNING: No LLM model provided. Only retrieval can be performed.")
        self._is_late_chunking = late_chunking
        self.document_loader = DocumentLoader()
        self.embedding_function = embedding_function
        self.splitter = splitter
        self.prompt = hub.pull("rlm/rag-prompt")
        self.llm = llm
        if vector_store is None:
            if embedding_function is None:
                raise ValueError("Either provide a vector store or an embedding function")
            self.vector_store = self._init_vector_store(dataset_path)
        else:
            self.vector_store = vector_store
        self.graph = self._compile()

    def _init_vector_store(self, data_path):
        docs = self.document_loader.load(data_path)

        if not docs or len(docs) == 0:
            raise ValueError("No document found")

        chroma_client = chromadb.Client()

        # if collection exist, delete it
        try:
            chroma_client.delete_collection("chatbot_docs_collection")
            print("The collection already existed and was deleted")
        except:
            pass

        collection = chroma_client.create_collection(name="chatbot_docs_collection")

        if self._is_late_chunking:
            # 1st split so that the model can handle the input
            splitter_max_tokens = self.embedding_function.get_splitter_max_tokens()
            docs = splitter_max_tokens.split_documents(docs)

            print(f"Embedding and storing {len(docs)} documents...")

            # 2nd split (late chunking)
            splitter = self.embedding_function.get_splitter()
            split_docs = splitter.split_documents(docs)
            embeddings = self.embedding_function.embed_documents([d.page_content for d in docs])

            collection.add(documents=[d.page_content for d in split_docs], embeddings=embeddings, ids=[f"{i}" for i in range(len(split_docs))])
        else:
            if self.splitter is not None:
                docs = self.splitter.split_documents(docs)
            print(f"Embedding and storing {len(docs)} chunks...")
            embeddings = self.embedding_function.embed_documents([d.page_content for d in docs])
            collection.add(documents=[d.page_content for d in docs], embeddings=embeddings, ids=[f"{i}" for i in range(len(docs))])
            # collection.add(documents=[d.page_content for d in docs], ids=[f"{i}" for i in range(len(docs))])

        return Chroma(client=chroma_client, collection_name="chatbot_docs_collection", embedding_function=self.embedding_function)

    def get_num_docs(self):
        return len(self.vector_store.get()['documents'])

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
