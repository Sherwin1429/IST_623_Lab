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
# Function to Split Text into Chunks
# --------------------------------------------------

def split_text(text, chunk_size=3000, overlap=300):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


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

        # Delete old collection so we can rebuild it
        # using chunks instead of entire PDFs
        try:
            chroma_client.delete_collection(
                name="Lab4Collection"
            )
        except Exception:
            pass

        collection = chroma_client.create_collection(
            name="Lab4Collection"
        )

        pdf_folder = "Lab4_Data"

        pdf_files = [
            file
            for file in os.listdir(pdf_folder)
            if file.lower().endswith(".pdf")
        ]

        document_id = 0

        for filename in pdf_files:

            file_path = os.path.join(
                pdf_folder,
                filename
            )

            document_text = read_pdf(file_path)

            # Split PDF into smaller chunks
            chunks = split_text(document_text)

            for chunk_number, chunk in enumerate(chunks):

                embedding = create_embedding(chunk)

                collection.add(
                    ids=[str(document_id)],
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[
                        {
                            "filename": filename,
                            "chunk": chunk_number
                        }
                    ]
                )

                document_id += 1

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

    user_message = {
        "role": "user",
        "content": prompt
    }

    st.session_state.lab4_messages.append(
        user_message
    )

    with st.chat_message("user"):
        st.markdown(prompt)


    # --------------------------------------------------
    # RAG Retrieval
    # --------------------------------------------------

    query_embedding = create_embedding(prompt)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    retrieved_documents = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]


    # --------------------------------------------------
    # Prefer Course Mentioned in Question
    # --------------------------------------------------

    course_numbers = [
        "195",
        "256",
        "314",
        "343",
        "387",
        "418",
        "488"
    ]

    mentioned_course = None

    for course_number in course_numbers:

        if course_number in prompt:
            mentioned_course = course_number
            break


    # If user mentioned a specific course,
    # directly retrieve chunks from that syllabus
    if mentioned_course:

        course_results = collection.get()

        matching_documents = []
        matching_metadata = []

        for document, metadata in zip(
            course_results["documents"],
            course_results["metadatas"]
        ):

            if mentioned_course in metadata["filename"]:

                matching_documents.append(document)
                matching_metadata.append(metadata)

        if matching_documents:

            retrieved_documents = matching_documents
            retrieved_metadata = matching_metadata


    # --------------------------------------------------
    # Build RAG Context
    # --------------------------------------------------

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

        "Answer using only the retrieved course syllabus information. "

        "When the retrieved documents contain the answer, begin with "
        "'Based on the retrieved course documents:' and answer the question. "

        "If the retrieved documents contain only part of the answer, explain "
        "what information was found and clearly state what information was "
        "not found. "

        "If the answer cannot be found in the retrieved documents, say "
        "'I could not find this information in the provided course documents.' "

        "Do not invent information. "
        "Do not suggest websites, Blackboard, email addresses, instructors, "
        "uploads, or other sources unless the user specifically asks for them. "
        "Do not offer additional services at the end of the answer."

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


    recent_messages = []

    for exchange in exchanges[-2:]:
        recent_messages.extend(exchange)


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

        response = st.write_stream(stream)


    # Save assistant response
    st.session_state.lab4_messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )