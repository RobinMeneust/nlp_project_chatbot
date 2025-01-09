from nlp_chat_bot.rag.abstract_rag import AbstractRAG
from nlp_chat_bot.rag.abstract_rag import State

class ClassicRAG(AbstractRAG):
    def __init__(self, vector_store, llm=None):
        super().__init__(vector_store, llm)
        self._compile()

    def retrieve(self, state: State, k: int = 3):
        try:
            docs, scores = zip(*self._vector_store.similarity_search_with_score(state["question"], k))
            for doc, score in zip(docs, scores):
                doc.metadata["score"] = score

            return {"context": docs}
        except Exception as e:
            print("Error in retrieving documents: ", e)
            return {"context": []}

    def generate(self, state: State):
        if not self.llm:
            raise ValueError("No LLM model provided at RAG object initialization. Cannot generate response.")
        if "context" in state and len(state["context"]) > 0:
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            context = docs_content
        else:
            context = ""
        if "chat_history" in state:
            context += "\n\nPrevious chat messages: " + "\n".join([item["role"] + ": " + item["content"] for item in state["chat_history"]])
        prompt = self.prompt.invoke({"question": state["question"], "context": context})
        response = self.llm.invoke(prompt)
        return {"answer": response.content}
    

