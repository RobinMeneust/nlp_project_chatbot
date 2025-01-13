from nlp_chat_bot.rag.abstract_query_translation_rag import AbstractQueryTranslationRAG
from nlp_chat_bot.rag.abstract_rag import State

class QueryTranslationRAGFusion(AbstractQueryTranslationRAG):
    """Class for the Query Translation RAG Fusion model

    Attributes:
        prompt (str): The prompt to use
        llm (object): The language model to use
        _vector_store (VectorStore): The vector store to use
        _graph (StateGraph): The state graph used with Langchain to use the RAG model
    """
    def __init__(self, vector_store, llm=None):
        """Initializes the QueryTranslationRAGFusion object

        Args:
            vector_store (VectorStore): The vector store to use
            llm (object): The language model to use
        """
        super().__init__(vector_store, llm)
        self._compile()

    def retrieve(self, state: State, k: int = 3):
        """Retrieves documents for the given state (i.e. question, and eventually context)

        Args:
            state (State): The state to use
            k (int): The number of documents to retrieve

        Returns:
            State: New state (i.e. with the documents and their scores)
        """
        try:
            docs_per_question = self._retrieve_docs_multiple_questions(state["question"], k)
            docs = []
            for question, doc_list in docs_per_question.items():
                docs.extend(doc_list)

            docs.sort(key=lambda x: x.metadata["score"], reverse=True)
            docs = docs[:k]

            return {"context": docs}

        except Exception:
            return {"context": []}

    def generate(self, state: State):
        """Generates a response for the given state (i.e. question and context (docs...))

        Args:
            state (State): The state to use

        Returns:
            State: New state (i.e. with the answer)
        """
        print("generating response (Fusion)")
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
