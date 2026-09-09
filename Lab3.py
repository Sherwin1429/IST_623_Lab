import streamlit as st
from openai import OpenAI

st.title("Lab 3 - Streaming Chatbot with Memory")

st.write(
    "Ask me a question! I will remember our recent conversation."
)

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

system_prompt = {
    "role": "system",
    "content": (
        "You are a helpful chatbot. "
        "Explain everything using simple language that a 10-year-old can understand. "
        "When answering a new question, give a clear answer and end with exactly: "
        "'Do you want more info?' "
        "If the user says yes, give more information about the previous topic and "
        "again end with exactly: 'Do you want more info?' "
        "If the user says no, do not give more information. Instead say: "
        "'Okay! What can I help you with?'"
    )
}

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What can I help you with?"):

    user_message = {
        "role": "user",
        "content": prompt
    }

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get only the last two complete user/assistant exchanges
    previous_messages = st.session_state.messages[:-1]

    exchanges = []
    current_exchange = []

    for message in previous_messages:
        if message["role"] == "user":
            if current_exchange:
                exchanges.append(current_exchange)

            current_exchange = [message]

        elif message["role"] == "assistant" and current_exchange:
            current_exchange.append(message)
            exchanges.append(current_exchange)
            current_exchange = []

    # Keep only the last two completed exchanges
    recent_messages = []

    for exchange in exchanges[-2:]:
        recent_messages.extend(exchange)

    # Add the current user message
    recent_messages.append(user_message)

    messages_for_api = [
        system_prompt,
        *recent_messages
    ]

    with st.chat_message("assistant"):

        stream = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages_for_api,
            stream=True
        )

        response = st.write_stream(stream)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )