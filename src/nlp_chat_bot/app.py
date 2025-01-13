import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from nlp_chat_bot.model.embedding.minilm import MiniLM
from nlp_chat_bot.model.llm.gemma import Gemma
from nlp_chat_bot.model.llm.mistral import Mistral
from nlp_chat_bot.rag.classic_rag import ClassicRAG
from nlp_chat_bot.model.embedding.late_chunking_embedding import LateChunkingEmbedding
from dotenv import load_dotenv

from nlp_chat_bot.rag.query_translation_rag_decomposition import QueryTranslationRAGDecomposition
from nlp_chat_bot.rag.query_translation_rag_fusion import QueryTranslationRAGFusion
from nlp_chat_bot.vector_store.naive_chunking_chroma_vector_store_builder import NaiveChunkingChromaVectorStoreBuilder
from nlp_chat_bot.vector_store.late_chunking_chroma_vector_store_builder import LateChunkingChromaVectorStoreBuilder


class ChatBotApp:
    def __init__(self, chatbot_name="My Assistant", update_docs=False, rag_mode="none", is_late_chunking=True, llm_model_name="gemini-1.5-flash"):
        print(f"ChatBotApp(chatbot_name={chatbot_name}, update_docs={update_docs}, rag_mode={rag_mode}, is_late_chunking={is_late_chunking}, llm_model_name={llm_model_name})")
        rags_models = {
            "none": ClassicRAG,
            "decomposition": QueryTranslationRAGDecomposition,
            "fusion": QueryTranslationRAGFusion
        }
        self._chatbot_name = chatbot_name
        self._update_docs = update_docs

        if rag_mode not in rags_models:
            raise Exception(f"Unknown RAG mode: {rag_mode}")
        self._rag_model_class = rags_models[rag_mode]

        self._is_late_chunking = is_late_chunking
        self.history_max_length = 5
        self._llm_model_name = llm_model_name

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

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=0 # chunk overlap (characters)
        )

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


        if self._llm_model_name == "gemini-1.5-flash":
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        elif self._llm_model_name == "gemma-2b":
            llm = Gemma(model_download_path)
        elif self._llm_model_name == "mistral-7b":
            llm = Mistral(model_download_path)
        else:
            raise Exception(f"Unknown LLM model name: {self._llm_model_name}")

        return self._rag_model_class(
            vector_store=vector_store,
            llm=llm,
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