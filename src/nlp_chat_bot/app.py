# based on https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nlp_chat_bot.model.minilm import MiniLM
from nlp_chat_bot.rag import RAG
import getpass
import os
from nlp_chat_bot.rag import RAG
from nlp_chat_bot.model.late_chunking_embedding import LateChunkingEmbedding
from dotenv import load_dotenv

class ChatBotApp:
    def __init__(self):
        self._rag = self._init_rag()
        self._start_app()

    def _init_rag(self):
        # get current file parent
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_download_path = os.path.join(root_path, "models")
        dataset_path = os.path.join(root_path, "data")
        print("Initializing RAG...")

        load_dotenv()
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        embedding_function = LateChunkingEmbedding(model_download_path)

        llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        return RAG(dataset_path, embedding_function, late_chunking=True, llm=llm_gemini)

    def _start_app(self):
        print("Starting App...")
        chatbot_name = "chatbot_name"

        # App Initialization
        st.title(chatbot_name)

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("How can I help you?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            generated_answer = self._rag.invoke(query={"question": prompt})["answer"]

            response = f"{chatbot_name}: {generated_answer}"

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        # self._rag.invoke(query={"question":"What is the the article 93 of GRPD?"})

if __name__ == "__main__":
    ChatBotApp()