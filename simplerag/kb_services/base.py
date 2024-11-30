from langchain_core.vectorstores import VectorStore as LangchainVectorStore

class KBService():
    def __init__(self) -> None:
        if self.vectorstore is None:
            self.vectorstore:LangchainVectorStore = LangchainVectorStore()
        pass