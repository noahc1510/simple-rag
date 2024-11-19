from chromadb import Embeddings
from base import VectorStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class ChromaEngine(VectorStore):
    def __init__(self, embedding_function:Embeddings, persist_dirctory="chroma_langchain_db") -> None:
        if embedding_function == None:
            embedding_function = OpenAIEmbeddings()

        self.vectorstore = Chroma(
            collection_name="example_collection",
            embedding_function=embedding_function,
            persist_directory="./chroma_langchain_db",
        )
        super().__init__()
    
    def add_documents(self,documents:list[Document], **kwargs):
        self.vectorstore.add_documents(documents, **kwargs)