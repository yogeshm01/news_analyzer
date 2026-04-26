import os
import streamlit as st
import pickle
import time

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ================== UI ==================
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# ================== GROQ API KEY ==================
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    # Check if it's set in environment (for local testing)
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    else:
        st.sidebar.error("⚠️ GROQ_API_KEY not found! Please set it in Streamlit Secrets.")
        st.stop()

# ================== INPUT URLs ==================
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"

main_placeholder = st.empty()

# ================== LLM (GROQ) ==================
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",  # faster & free-friendly
    temperature=0.7
)

# ================== PROCESS URLs ==================
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)

    main_placeholder.text("Data Loading...Started...✅")
    data = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    main_placeholder.text("Text Splitting...Started...✅")
    docs = text_splitter.split_documents(data)

    # ================== FREE EMBEDDINGS ==================
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    main_placeholder.text("Building Embeddings...⏳")
    vectorstore = FAISS.from_documents(docs, embeddings)

    time.sleep(2)

    # Save FAISS index
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    main_placeholder.text("Processing Done...✅")

# ================== QUERY ==================
query = main_placeholder.text_input("Ask a Question: ")

if query:
    if os.path.exists(file_path):

        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=retriever
        )

        result = chain.invoke({"question": query})

        # ================== OUTPUT ==================
        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)