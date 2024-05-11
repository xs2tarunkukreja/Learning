# Setting up the environment and key
Building our own ChatBot using ChatGPT API, Langchain, Streamlit

Objective - Create a chatbot that hook up to our personal documents and provides QnA ability.
    ChatGPT was trained on Public Data.

## Architecture

PDFs > Chunks > OpenAI (Embeddings) > 'Vector Store" >
    We provide pdfs in chunks to Open AI. It generate embedding and save in Vector Store.

User > Query > Semantic Search > "Vector Store" > Ranked Result > LLM (Open AI) > Result > User --(Cycle)
    User ask query that converted to embedding that will be semantic search to vector store... it provide ranked result.

Pre-requistee
    IDE - PyCharm or Jupyter
    Open AI - API Key
        https://platform.openai.com/api-keys

# Creating Chatbot - Part 1
chatbot.py 
    Python - 3.11
    pip install streamlit pypdf2 langchain faiss-cpu

    streamlit - UI
    pypdf2 - read PDF file
    langchain - Interface to using OpenAPI services.
    faiss-cpu - vector store to store embeddings.

    1. Create UI so user can upload PDFs.
        streamlit run chatbot.py
    2. Extract text and convert to chunk.

# Creating Chatbot - Part 2
    3. To generate embedding through OpenAI (You may use some other model.)
    4. Store in Vector Space. 

# Creating Chatbot - Part 3
    5. Need a way user ask a question
    6. Semantic Search
    7. Pass Ranked result to Open AI.
    8. Show result to user.