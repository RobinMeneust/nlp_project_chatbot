from langchain_google_genai import ChatGoogleGenerativeAI
import os

from nlp_chat_bot.model.minilm import MiniLM
from nlp_chat_bot.rag import RAG
from nlp_chat_bot.model.late_chunking_embedding import LateChunkingEmbedding
from dotenv import load_dotenv

class ChatBotApp:
    def __init__(self, chatbot_name="chatbot_name"):
        self._chatbot_name = chatbot_name
        self._rag = self._init_rag()

    def _init_rag(self):
        print("Initializing RAG...")
        # get current file parent
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_download_path = os.path.join(root_path, "models")
        dataset_path = os.path.join(root_path, "data")
        vector_store_path = os.path.join(root_path, "chromadb")

        load_dotenv()

        embedding_function = LateChunkingEmbedding(model_download_path)
        # embedding_function = MiniLM(model_download_path)
        llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        return RAG(dataset_path, embedding_function, vector_store_path, late_chunking=True, llm=llm_gemini)

    def get_name(self):
        return self._chatbot_name

    def invoke(self, query):
        return self._rag.invoke(query=query)