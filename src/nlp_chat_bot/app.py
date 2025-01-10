import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from nlp_chat_bot.model.embedding.minilm import MiniLM
from nlp_chat_bot.rag.classic_rag import ClassicRAG
from nlp_chat_bot.model.embedding.late_chunking_embedding import LateChunkingEmbedding
from dotenv import load_dotenv

from nlp_chat_bot.rag.query_translation_rag_decomposition import QueryTranslationRAGDecomposition
from nlp_chat_bot.rag.query_translation_rag_fusion import QueryTranslationRAGFusion
from nlp_chat_bot.vector_store.naive_chunking_chroma_vector_store_builder import NaiveChunkingChromaVectorStoreBuilder
from nlp_chat_bot.vector_store.late_chunking_chroma_vector_store_builder import LateChunkingChromaVectorStoreBuilder


class ChatBotApp:
    def __init__(self, chatbot_name="chatbot_name", update_docs=False, rag_model="none", is_late_chunking=True):
        rags_models = {
            "none": ClassicRAG,
            "decomposition": QueryTranslationRAGDecomposition,
            "fusion": QueryTranslationRAGFusion
        }
        self._chatbot_name = chatbot_name
        self._update_docs = update_docs
        self._rag_model_class = rags_models[rag_model]
        self._is_late_chunking = is_late_chunking
        self.history_max_length = 5

        self._rag = self._init_rag()

    def _init_rag(self):
        print("Initializing RAG...")
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        # get current file parent
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_download_path = os.path.join(root_path, "models")
        dataset_path = os.path.join(root_path, "data")
        vector_store_path = os.path.join(root_path, "chromadb")

        load_dotenv()

        splitter = None
        document_loader = None

        if self._is_late_chunking:
            embedding_function = LateChunkingEmbedding(model_download_path)
            vector_store = LateChunkingChromaVectorStoreBuilder(dataset_path,
                                                                embedding_function,
                                                                vector_store_path,
                                                                splitter,
                                                                document_loader).build(self._update_docs)
        else:
            embedding_function = MiniLM(model_download_path)
            vector_store = NaiveChunkingChromaVectorStoreBuilder(dataset_path,
                                                                embedding_function,
                                                                vector_store_path,
                                                                splitter,
                                                                document_loader).build(self._update_docs)

        llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

        return self._rag_model_class(
            vector_store=vector_store,
            llm=llm_gemini,
        )

    def get_name(self):
        return self._chatbot_name

    def invoke(self, query, chat_history=None):
        if chat_history is not None:
            query["chat_history"] = chat_history[-self.history_max_length:]
            return self._rag.invoke(query=query)
        return self._rag.invoke(query=query)

    def clear_documents(self):
        self._rag.reset_db()