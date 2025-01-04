import streamlit as st
from nlp_chat_bot.app import ChatBotApp

# It should not be put in a if __name__ = "main" block (https://stackoverflow.com/questions/58787589/why-does-my-streamlit-application-open-multiple-times

if "app" not in st.session_state:
    with st.spinner('Loading documents...'):
        st.session_state.app = ChatBotApp()

# App Initialization
st.title(st.session_state.app.get_name())

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
        response = st.session_state.app.invoke(query={"question": prompt})["answer"]
    except Exception as e:
        response = f"Sorry. An error occurred"
        st.exception(e)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})