import streamlit as st
import os
from streamlit_chat import message
import numpy as np
import pandas as pd
from io import StringIO
import io
import PyPDF2
import pymupdf
import tempfile
import base64
from tqdm.auto import tqdm
import math
from transformers import pipeline

from collections import Counter
import nltk
from nltk.corpus import stopwords


from sentence_transformers import SentenceTransformer
import torch
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# if device != 'cuda':
#     st.markdown(f"you are using {device}. This is much slower than using "
#     "a CUDA-enabled GPU. If on colab you can change this by "
#     "clicking Runtime > change runtime type > GPU.")

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
def display_title():
    selected_value = st.session_state["value"]

    st.header(f'Vedic Scriptures: {selected_value} :blue[book] :books:')

question = "ask anything about scriptures"
def open_chat():
    question = st.session_state["faq"]

    

if "value" not in st.session_state:
    st.session_state["value"] = None

if "faq" not in st.session_state:
    st.session_state["faq"] = None

st.divider()

def highlight_pdf(file_path, text_to_highlight, page_numbers):
    # Create a temporary file to save the modified PDF
    # temp_pdf_path = "temp_highlighted_pdf.pdf"
    # Create a temporary file to save the modified PDF
    # with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    #     temp_pdf_path = temp_file.name

    # Open the original PDF
    doc = pymupdf.open(file_path)

    pages_to_display = [doc.load_page(page_number) for page_number in page_numbers]

    print("pages_to_display") 
    print(pages_to_display)

    # Tokenize the text into words
    words = text_to_highlight.split()

   

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.lower() not in stop_words]
    
    print(words)
    
    # Count the frequency of each word
    word_counts = Counter(words)

    # Get the top N most frequent words
    # top_words = [word for word, _ in word_counts.most_common(5)]

    # Iterate over each page in the PDF
    for page in pages_to_display:
        
        # Highlight the specified words on the canvas
        for word in words:
            highlight_rect = page.search_for(word, quads=True)
            # Highlight the text
            # highlight_rect = pymupdf.Rect(word)
        # highlight_annot = page.add_highlight_annot(highlight_rect)
        # highlight_annot.set_colors({"stroke": pymupdf.utils.getColor("yellow")})
        # highlight_annot.update()
            page.add_highlight_annot(highlight_rect)
    
        # Create a new document with only the specified pages
    new_doc = pymupdf.open()
    for page in pages_to_display:
        new_doc.insert_pdf(doc, from_page=page.number, to_page=page.number)

    # Save the modified PDF
    # Save the document to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_pdf_path = temp_file.name
        new_doc.save(temp_pdf_path)
    
    print(temp_pdf_path)

    # new_doc.save("example_highlighted.pdf")

    return temp_pdf_path

file_path = "../Transformers/Bhagavad-Gita-As-It-Is.pdf"
text_to_highlight = ""
sources = []

# Function to display PDF in Streamlit
def display_highlighted_pdf(file_path, text_to_highlight, sources):
    # pdf_path = "../Transformers/Bhagavad-Gita-As-It-Is.pdf"
    # sources = [7,8]
    # response_text = "I offer my respectful obeisances unto the lotus feet of my spiritual master and unto the feet of all Vaiñëavas. I offer my respectful"
    
    pdf_path = highlight_pdf(file_path=file_path, text_to_highlight=text_to_highlight, page_numbers=sources)

    with open(pdf_path, "rb") as file:
        pdf_bytes = file.read()
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Creating a Index(Pinecone Vector Database)
import os
# import pinecone


def get_faiss_semantic_index():
    import pickle

    # File path to the pickle file
    file_path = "./HuggingFaceEmbeddings.pkl"

    # Load embeddings from the pickle file
    with open(file_path, "rb") as f:
        index = pickle.load(f)

    print("Embeddings loaded successfully.")
    return index

# def promt_engineer(text):
PROMPT_TEMPLATE = """
Instructions:
--------------------------------------------------------
you're a vedic scriptures AI expert. you shouldnot answer to any other domain specific question.
You 1000 Dollars rewards for Before answering questions always try to map the question related to the TITLE > CHAPTER > TEXT > PURPORT.
You 1000 Dollars rewards Must provide the Chapter Number and Text number in this format chapter <no> : Text <no>
You 1000 Dollars rewards Must provide the Title of the chapter. you also provide source path from where youre answering the question.
You 1000 Dollars penality for the relevant questions to answer.
Please dont answer from the public sources strictly answer from the context.
If the question is not related to the context replay with question doesnot belongs to vedic scriptures or Vedic literature.
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
    # # Load the summarization pipeline with the specified model
    # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # # Generate the prompt
    # prompt = prompt_template.format(text=text)

    # # Generate the summary
    # summary = summarizer(prompt, max_length=1024, min_length=50)[0]["summary_text"]
    
    # with st.sidebar:
    #     st.divider()
    #     st.markdown("*:red[Text Summary Generation]* from above Top 5 **:green[similarity search results]**.")
    #     st.write(summary)
    #     st.divider()

def chat_actions():
    
    index = get_faiss_semantic_index()

    st.session_state["chat_history"].append(
        {"role": "user", "content": st.session_state["chat_input"]},
    )

    # query_embedding = model.encode(st.session_state["chat_input"])
    query = st.session_state["chat_input"]
    docs = index.similarity_search(query, k=2)
    for doc in docs:
        print("\n")
        print(str(doc.metadata["page"]+1) + ":", doc.page_content)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])

    sources = [doc.metadata.get("page", None) for doc in docs]
    

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": f"{response_text}",
        },  # This can be replaced with your chat response logic
    )
        # break;
    # Example usage
    file_path = "../Transformers/Bhagavad-Gita-As-It-Is.pdf"
    text_to_highlight = context_text.strip()
    display_highlighted_pdf(file_path, response_text, sources)

with st.sidebar:
    option = st.selectbox(
    "Select Your Favorite Scriptures",
    ("Bhagvatgeetha", "Bhagavatham", "Ramayanam"),
    # index=None,
    # placeholder="Select scriptures...",
    key="value",
    on_change=display_title
    )

    st.write("You selected:", option)

    faq = st.selectbox(
    "Select Your Favorite Scriptures",
    ("Why does atheism exist even when all questions are answered in Bhagavad Gita?", 
     "Why don’t all souls surrender to Lord Krishna, although he has demonstrated that everyone is part and parcel of Him, and all can be liberated from all sufferings by surrendering to Him?",
     "Why do souls misuse their independence by rebelling against Lord Krishna?"),
    # index=None,
    # placeholder="Select scriptures...",
    key="faq",
    on_change=open_chat
    )
    st.write("You selected:", faq)
    

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.chat_input(question, on_submit=chat_actions, key="chat_input")

    for i in st.session_state["chat_history"]:
        with st.chat_message(name=i["role"]):
            st.write(i["content"])





