"""Shared helpers for the free / self-hosted SmartHire stack.

This module replaces the previous *paid* dependencies with free, local ones:
  - OpenAI chat model      -> Ollama (local LLM, e.g. llama3.1)
  - OpenAI embeddings      -> HuggingFace sentence-transformers (local)
  - AWS OpenSearch indices -> local CSV-backed document store (./data/local_store)

Vector search still uses FAISS, which was already free.
"""
import os
import ast

import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# --- Configuration (override any of these via environment variables) --------
# Local LLM served by Ollama (https://ollama.com). One-time setup:
#   brew install ollama && ollama serve
#   ollama pull llama3.1
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Free, local embedding model. Downloaded automatically on first use (~80MB).
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Local document store directory (replaces AWS OpenSearch indices).
LOCAL_STORE_DIR = os.environ.get("LOCAL_STORE_DIR", "./data/local_store")

# Where uploaded resumes live locally (replaces the AWS S3 bucket).
RESUME_DIR = os.environ.get("RESUME_DIR", "./data/resumes")

# Reusable RAG prompt (inlined so we don't depend on the LangChain hub / network).
_RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are an assistant for question-answering tasks. Use the following "
    "pieces of retrieved context to answer the question. If you don't know "
    "the answer, just say that you don't know.\n"
    "Question: {question}\n"
    "Context: {context}\n"
    "Answer:"
)


def get_llm():
    """Return the chat LLM. Free local default: Ollama.

    To use another free provider instead, swap the return line, e.g.:
      from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini free tier
      return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
      from langchain_groq import ChatGroq                        # Groq free tier
      return ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    """
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)


def get_embeddings():
    """Return a free, local embedding model (HuggingFace sentence-transformers)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_specific_api_key(filename='credential.txt', key_name='smtp_user'):
    """
    Load a specific value from a credential file containing a Python dict literal.

    With the free stack the only secrets needed are the SMTP login used to send
    job-recommendation emails (key names: smtp_user, smtp_password).

    :param filename: The name of the file to read.
    :param key_name: The key to retrieve (e.g. smtp_user, smtp_password).
    :return: The value, or None if not found.
    """
    try:
        with open(filename, 'r') as file:
            file_contents = file.read()
            credentials_dict = ast.literal_eval(file_contents)
            return credentials_dict.get(key_name)
    except FileNotFoundError:
        print("The credentials file was not found.")
    except SyntaxError as e:
        print(f"Syntax error in the credentials file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


def rag_invoke(loader):
    """Build a retrieval-augmented chain over the documents from `loader`.

    Uses local FAISS + HuggingFace embeddings + Ollama (all free).

    :param loader: a LangChain document loader (e.g. PyPDFLoader, CSVLoader).
    :return: a runnable rag_chain; call rag_chain.invoke("<question>").
    """
    pages = loader.load_and_split()
    faiss_index = FAISS.from_documents(pages, get_embeddings())
    retriever = faiss_index.as_retriever()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | _RAG_PROMPT
        | get_llm()
        | StrOutputParser()
    )
    return rag_chain


# --- Local document store (replaces AWS OpenSearch) -------------------------
def _index_path(index_name):
    return os.path.join(LOCAL_STORE_DIR, f"{index_name}.csv")


def save_documents(index_name, data_df, dedupe_on=None, mode='append'):
    """Persist a DataFrame to the local store under `index_name`.

    :param index_name: logical store name, e.g. "jobs" or "resumes".
    :param data_df: DataFrame of documents to store.
    :param dedupe_on: optional column to drop duplicates on (keeps latest).
    :param mode: 'append' (default) merges with existing data; 'overwrite' replaces it.
    """
    os.makedirs(LOCAL_STORE_DIR, exist_ok=True)
    path = _index_path(index_name)

    if mode == 'append' and os.path.exists(path):
        existing = pd.read_csv(path)
        data_df = pd.concat([existing, data_df], ignore_index=True)

    if dedupe_on and dedupe_on in data_df.columns:
        data_df = data_df.drop_duplicates(subset=[dedupe_on], keep='last')

    data_df.to_csv(path, index=False)
    print(f"Saved {len(data_df)} docs to local store '{index_name}' ({path}).")


def load_documents(index_name):
    """Load all documents previously stored under `index_name`."""
    path = _index_path(index_name)
    if not os.path.exists(path):
        print(f"No local store found for '{index_name}' ({path}).")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} docs from local store '{index_name}'.")
    return df
