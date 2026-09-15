import sys

# Fix SQLite version for ChromaDB
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from openai import OpenAI
import chromadb
from pypdf import PdfReader
import os


# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("Lab 4 - RAG Course Information Chatbot")

st.write(
    "Ask me questions about the Syracuse University courses "
    "in the provided syllabus documents."
)


# --------------------------------------------------
# OpenAI Client
# --------------------------------------------------

openai_api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(
    api_key=openai_api_key
)


# --------------------------------------------------
# Function to Read PDF
# --------------------------------------------------

def read_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# --------------------------------------------------
# Function to Create Embedding
# --------------------------------------------------

def create_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# --------------------------------------------------
# Create Vector Database
# --------------------------------------------------

if "Lab4_VectorDB" not in st.session_state:

    with st.spinner("Creating vector database..."):

        chroma_client = chromadb.Client()

        collection = chroma_client.get_or_create_collection(
    name="Lab4Collection"
)

        pdf_folder = "Lab4_Data"

        pdf_files = [
            file
            for file in os.listdir(pdf_folder)
            if file.lower().endswith(".pdf")
        ]

        for index, filename in enumerate(pdf_files):

            file_path = os.path.join(
                pdf_folder,
                filename
            )

            document_text = read_pdf(file_path)

            embedding = create_embedding(
                document_text
            )

            collection.add(
                ids=[str(index)],
                documents=[document_text],
                embeddings=[embedding],
                metadatas=[
                    {
                        "filename": filename
                    }
                ]
            )

        st.session_state.Lab4_VectorDB = collection


# --------------------------------------------------
# Get Vector Database
# --------------------------------------------------

collection = st.session_state.Lab4_VectorDB


# --------------------------------------------------
# Chat Memory
# --------------------------------------------------

if "lab4_messages" not in st.session_state:
    st.session_state.lab4_messages = []


# Display previous messages
for message in st.session_state.lab4_messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Chat Input
# --------------------------------------------------

if prompt := st.chat_input(
    "Ask a question about the courses..."
):

    # Save user message
    user_message = {
        "role": "user",
        "content": prompt
    }

    st.session_state.lab4_messages.append(
        user_message
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)


    # --------------------------------------------------
    # RAG Retrieval
    # --------------------------------------------------

    query_embedding = create_embedding(
        prompt
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_documents = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]


    # Combine retrieved documents
    rag_context = ""

    for document, metadata in zip(
        retrieved_documents,
        retrieved_metadata
    ):

        rag_context += (
            f"\n\nSource: {metadata['filename']}\n"
            f"{document}"
        )


    # --------------------------------------------------
    # System Prompt
    # --------------------------------------------------

    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful course information chatbot. "
            "Answer questions using simple and clear language. "
            "Use the retrieved course syllabus information when it "
            "contains information relevant to the user's question. "
            "When you use information from the retrieved documents, "
            "clearly say 'Based on the retrieved course documents:' "
            "before your answer. "
            "If the retrieved documents do not contain enough "
            "information to answer the question, clearly say that "
            "the answer was not found in the provided course documents. "
            "Do not add unrelated information or offer additional services "
            "unless the user asks for them."
            "\n\nRetrieved course information:\n"
            + rag_context
        )
    }


    # --------------------------------------------------
    # Conversation Memory
    # --------------------------------------------------

    previous_messages = (
        st.session_state.lab4_messages[:-1]
    )

    exchanges = []
    current_exchange = []

    for message in previous_messages:

        if message["role"] == "user":

            if current_exchange:
                exchanges.append(
                    current_exchange
                )

            current_exchange = [message]

        elif (
            message["role"] == "assistant"
            and current_exchange
        ):

            current_exchange.append(message)

            exchanges.append(
                current_exchange
            )

            current_exchange = []


    # Keep last two completed exchanges
    recent_messages = []

    for exchange in exchanges[-2:]:
        recent_messages.extend(exchange)


    # Add current question
    recent_messages.append(
        user_message
    )


    # --------------------------------------------------
    # Send to OpenAI
    # --------------------------------------------------

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

        response = st.write_stream(
            stream
        )


    # Save assistant response
    st.session_state.lab4_messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )