from nlp_chat_bot.rag.abstract_rag import AbstractRAG
from nlp_chat_bot.rag.abstract_rag import State

class ClassicRAG(AbstractRAG):
    def __init__(self, dataset_path, embedding_function, vector_store_path, splitter=None, llm=None, late_chunking=False):
        super().__init__(dataset_path, embedding_function, vector_store_path, splitter, llm, late_chunking)
        self._compile()

    def retrieve(self, state: State, k: int = 3):
        docs, scores = zip(*self._vector_store.similarity_search_with_score(state["question"], k))
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
    

