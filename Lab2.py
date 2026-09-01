import streamlit as st
from openai import OpenAI
import fitz

# Page Title
st.title("Lab 2 - Document Summarizer")

st.write(
    "Upload a document and choose your summary options from the sidebar."
)



# Sidebar Options
language = st.sidebar.selectbox(
    "Choose a language",
    [
        "English",
        "Spanish",
        "French",
        "German",
        "Italian"
    ]
)

summary_type = st.sidebar.selectbox(
    "Choose a summary type",
    [
        "Summarize the document in 100 words",
        "Summarize the document in 2 connecting paragraphs",
        "Summarize the document in 5 bullet points"
    ]
)

model_choice = st.sidebar.selectbox(
    "Choose a model",
    [
        "gpt-5-nano",
        "gpt-5-mini"
    ]
)


# OpenAI API Key
openai_api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=openai_api_key)




# File Uploader
uploaded_file = st.file_uploader(
    "Upload a document (.txt, .md, or .pdf)",
    type=("txt", "md", "pdf")
)


# Process Uploaded Document
if uploaded_file:

    # Read PDF
    if uploaded_file.type == "application/pdf":

        pdf_document = fitz.open(
            stream=uploaded_file.read(),
            filetype="pdf"
        )

        document = ""

        for page in pdf_document:
            document += page.get_text()

        pdf_document.close()

    # Read TXT or MD
    else:
        document = uploaded_file.read().decode("utf-8")


    # Instructions for OpenAI
    messages = [
        {
            "role": "system",
            "content": (
                "You are a document summarization assistant. "
                f"Provide the summary in {language}."
            )
        },
        {
            "role": "user",
            "content": (
                f"{summary_type}.\n\n"
                f"Document:\n{document}"
            )
        }
    ]


    
    # Generate Summary

    stream = client.chat.completions.create(
        model=model_choice,
        messages=messages,
        stream=True,
    )


    # Display Summary
    st.subheader("Summary")

    st.write_stream(stream)