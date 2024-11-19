from langchain_core.vectorstores import VectorStore as LangchainVectorStore

class VectorStore():
    def __init__(self) -> None:
        if self.vectorstore is None:
            self.vectorstore:LangchainVectorStore = LangchainVectorStore()
        pass