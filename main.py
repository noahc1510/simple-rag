import os
from core.document_loader import HtmlLoader
from core.text_splitter import RecursiveCharacterTextSplitter
from core.vectorstore import ChromaEngine
from core.retriever import Retriever

from langchain_openai import OpenAIEmbeddings

import dotenv
dotenv.load_dotenv()

# Document Loader
loader = HtmlLoader()
docs = "https://lilianweng.github.io/posts/2023-06-23-agent/"


# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)


# Set Embedding Engine
emb = OpenAIEmbeddings(
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY")
)
vectorstore = ChromaEngine(embedding_function=emb)

# Add Document
vectorstore.add_documents(all_splits)

# Retrie
retriever = Retriever(vectorstore)
retrieved_docs = retriever.query("What are the approaches to Task Decomposition?")

pass

