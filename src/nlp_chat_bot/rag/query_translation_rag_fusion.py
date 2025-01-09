from nlp_chat_bot.rag.abstract_query_translation_rag import AbstractQueryTranslationRAG
from nlp_chat_bot.rag.abstract_rag import State

class QueryTranslationRAGFusion(AbstractQueryTranslationRAG):
    def __init__(self, vector_store, llm=None):
        super().__init__(vector_store, llm)
        self._compile()

    def retrieve(self, state: State, k: int = 3):
        try:
            docs_per_question = self._retrieve_docs_multiple_questions(state["question"], k)
            docs = []
            for question, doc_list in docs_per_question.items():
                docs.extend(doc_list)

            docs.sort(key=lambda x: x.metadata["score"], reverse=True)
            docs = docs[:k]

            return {"context": docs}

        except Exception as e:
            print("Error in retrieving documents: ", e)
            return {"context": []}

    def generate(self, state: State):
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
    

