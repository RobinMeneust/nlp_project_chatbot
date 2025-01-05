from langchain_core.prompts import ChatPromptTemplate

from nlp_chat_bot.rag.abstract_query_translation_rag import AbstractQueryTranslationRAG
from nlp_chat_bot.rag.abstract_rag import AbstractRAG
from nlp_chat_bot.rag.abstract_rag import State

class QueryTranslationRAGDecomposition(AbstractQueryTranslationRAG):
    def __init__(self, dataset_path, embedding_function, vector_store_path, llm, splitter=None, late_chunking=False, update_docs=True, document_loader=None):
        super().__init__(dataset_path, embedding_function, vector_store_path, llm, splitter, late_chunking, update_docs, document_loader)
        template = "Here is a set of Q+A pairs:\n\n{context}\n\nUse these to synthesize an answer to the question: {question}\n"
        self._template_qa_pairs = ChatPromptTemplate.from_template(template)
        self._compile()

    def retrieve(self, state: State, k: int = 3):
        docs_per_question = self._retrieve_docs_multiple_questions(state["question"], k)
        return {"context": docs_per_question}

    def generate(self, state: State):
        docs_per_question = state["context"]

        context_qa = ""
        for question, docs in docs_per_question.items():
            docs_content = "\n\n".join(doc.page_content for doc in docs)
            prompt = self.prompt.invoke({"question": question, "context": docs_content})
            response = self.llm.invoke(prompt)
            context_qa += f"Q: {question}\nA: {response.content}\n\n"

        state["context"] = []
        for docs in docs_per_question.values():
            for doc in docs:
                if doc not in state["context"]:
                    state["context"].append(doc)

        context = context_qa
        if "chat_history" in state:
            context += "\n\nPrevious chat messages: " + "\n".join([item["role"] + ": " + item["content"] for item in state["chat_history"]])
        prompt = self.prompt.invoke({"question": state["question"], "context": context})

        prompt += self._template_qa_pairs.invoke({"question": state["question"], "context": context})
        response = self.llm.invoke(prompt)
        return {"answer": response.content}
    

