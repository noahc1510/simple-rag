from .kb_services import KBService
from langchain_core.vectorstores import VectorStoreRetriever

class Retriever():
    def __init__(self, kb_svc:KBService) -> None:
        self.vectorstore = kb_svc.vectorstore

    def query(self, input_query: str, top_k=10):
        # TODO: 在这里重新设计retriever和langchain解耦
        self.retriever:VectorStoreRetriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        return self.retriever.invoke(input_query)
        