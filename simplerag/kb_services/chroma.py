from chromadb import Embeddings
from .base import KBService
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class ChromaService(KBService):
    def __init__(self,name, embedding_function:Embeddings=None, persist_directory="data/chroma_langchain_db") -> None:
        if embedding_function == None:
            embedding_function = OpenAIEmbeddings()

        self.vectorstore = Chroma(
            collection_name=name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
        )
        super().__init__()
    
    def add_documents(self,documents:list[Document], **kwargs):
        self.vectorstore.add_documents(documents, **kwargs)