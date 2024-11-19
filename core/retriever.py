from vectorstore.base import VectorStore
from langchain_core.vectorstores import VectorStoreRetriever

class Retriever():
    def __init__(self, vectorstore:VectorStore) -> None:
        self.vectorstore = vectorstore

    def query(self, input_query: str, top_k=6):
        # TODO: 在这里重新设计retriever和langchain解耦
        self.retriever:VectorStoreRetriever = self.vectorstore.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        return self.retriever.invoke(input_query)