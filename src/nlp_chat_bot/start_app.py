import time

import streamlit as st
from numpy.lib.user_array import container

from nlp_chat_bot.app import ChatBotApp

# It should not be put in a if __name__ = "main" block (https://stackoverflow.com/questions/58787589/why-does-my-streamlit-application-open-multiple-times


if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = "none"

if "chunking_mode" not in st.session_state:
    st.session_state.is_late_chunking = True

if "app" not in st.session_state:
    with st.spinner('Loading documents...'):
        st.session_state.app = ChatBotApp()

if "is_chat_history_on" not in st.session_state:
    st.session_state.is_chat_history_on = False

# App Initialization
st.title(st.session_state.app.get_name())

def update_rag():
    with st.spinner('Loading...'):
        st.session_state.app = ChatBotApp(
            rag_mode=st.session_state["rag_mode"],
            is_late_chunking=st.session_state["is_late_chunking"],
            llm_model_name=st.session_state["llm_model"]
        )

def update_docs():
    with st.spinner('Loading documents...'):
        st.session_state.app = ChatBotApp(
            rag_mode=st.session_state["rag_mode"],
            is_late_chunking=st.session_state["is_late_chunking"],
            update_docs=True,
            llm_model_name=st.session_state["llm_model"]
        )

def clear_docs():
    with st.spinner('Clearing documents...'):
        st.session_state.app.clear_documents()

# Options
with st.sidebar:
    st.header("Settings")
    st.divider()

    st.subheader("Enable chat history")
    with st.expander(label="Show/Hide", expanded=True):
        st.session_state["is_chat_history_on"] = st.checkbox("Enable history", st.session_state["is_chat_history_on"])
    st.divider()

    st.subheader("Choose a query translation method (classic means \"no translation\")")
    with st.expander(label="Show/Hide", expanded=True):
        st.session_state["rag_mode"] = st.radio("Query translation method", ["none", "decomposition", "fusion"], index=0, on_change=update_rag)
    st.divider()

    st.subheader("Choose chunking mode")
    with st.expander(label="Show/Hide", expanded=True):
        st.checkbox("Enable late chunking", st.session_state["is_late_chunking"], on_change=update_rag)
    st.divider()

    st.subheader("Choose LLM model")
    with st.expander(label="Show/Hide", expanded=True):
        st.session_state["llm_model"] = st.radio("LLM model", ["gemini-1.5-flash", "gemma-2b", "mistral-7b"], index=0, on_change=update_rag)
    st.divider()

    st.subheader("Update documents from data folder")
    st.button("Update docs", on_click=update_docs)
    st.write("Note: Documents are stored for the current embeddings function settings. If it changes you need to re-update it so that the vector database for the new embedding function is updated.")
    st.divider()

    st.subheader("Clear documents storage")
    st.button("Clear documents", on_click=clear_docs)
    st.write("WARNING: All documents will have to be re-loaded")
    st.divider()

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

    try:
        if st.session_state["is_chat_history_on"]:
            output = st.session_state.app.invoke(query={"question": prompt}, chat_history=st.session_state.messages)
        else:
            output = st.session_state.app.invoke(query={"question": prompt})
        answer = output["answer"]
    except Exception as e:
        answer = f"Sorry. An error occurred"
        st.exception(e)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})


