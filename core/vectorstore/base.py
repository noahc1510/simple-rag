from langchain_core.vectorstores import VectorStore

class VectorStore():
    def __init__(self) -> None:
        self.vectorstore:VectorStore = VectorStore()
        pass