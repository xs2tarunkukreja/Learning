import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.community.chat_models import ChatOpenAI

OPENAI_API_KEY = "sk-kdhjsdhkjdhsdlkhdskjdshkds"

# Upload PDF Files
st.header("My First ChatBot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions.", type="pdf")

# Extract Text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

# Break Text into Chunks with LangChain
text_splitter = RecursiveCharacterTextSplitter(
    separators="\n",
    chunk_size=1000,
    chunk_overlap=150, # It help to keep context from previous chunk.
    length_function=len
)
chunks = text_splitter(text)


# Generate Embedding
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Creating Vector Store - FAISS
vector_store = FAISS.from_texts(chunks, embeddings)
"""
    It is doing three things
        Generate Embedding using OpenAI.
        Initialize Vector Store
        Store Chunk and Embedding.
"""

# Get User ask question
user_question = st.text_input("Type your question here")

# Do Similarity Search
if user_question:
    match = vector_store.similarity_search(user_question)


# Output Result
# chain > take question, get relevant document > pass it to LLM
    llm = ChatOpenAI(
        openai_api_key = OPENAI_API_KEY,
        temperature = 0,# Lower value means not random. 
        max_tokens = 1000,
        model_name="gpt-3.5-turbo"
    )

    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(question=user_question, input_document=match)

    st.write(response)