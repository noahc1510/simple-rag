import os
from simplerag.document_loader import HtmlLoader
from simplerag.text_splitter import RecursiveCharacterTextSplitter
from simplerag.kb_services import ChromaService
from simplerag.retriever import Retriever
from simplerag.reranker import BgeReranker

from langchain_openai import OpenAIEmbeddings

import dotenv
dotenv.load_dotenv()

# Document Loader
loader = HtmlLoader()
docs = loader.run("https://lilianweng.github.io/posts/2023-06-23-agent/")


# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)


# Set Embedding Engine
emb = OpenAIEmbeddings(
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)
vectorstore = ChromaService(name="example_collections", embedding_function=emb)

# Add Document
vectorstore.add_documents(all_splits)

# Retrieve
retriever = Retriever(kb_svc=vectorstore)
retrieved_docs = retriever.query("What are the approaches to Task Decomposition?")

reranker = BgeReranker()
reranked_docs = reranker.compress_documents(retrieved_docs, "What are the approaches to Task Decomposition?")

pass

