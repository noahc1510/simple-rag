from unittest import loader
from fastapi import FastAPI, UploadFile
from langchain_openai import OpenAIEmbeddings
from core.text_splitter import RecursiveCharacterTextSplitter
import requests
from sympy import EX
import uvicorn
from pydantic import BaseModel
import os

from core.kb_services import ChromaService
from core.retriever import Retriever

import dotenv
dotenv.load_dotenv()

app = FastAPI()

# TODO: Configurable
emb = OpenAIEmbeddings(
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)
loader_map = {
    "html": "core.document_loader.HtmlLoader",
    "pdf": "core.document_loader.PDFLoader",
}

@app.get("/")
def read_root():
    return {"healthy": True}


class QueryRequest(BaseModel):
    query: str
    kb_name: str
    top_k: int = 6

class AddDocumentRequest(BaseModel):
    url: str
    kb_name: str
    # emedding_model: str = "text-embedding-ada-002"


@app.post("/upload")
def upload(file: UploadFile, kb_name: str):
    try:
        if not os.path.exists("data/orig_content"):
            os.makedirs("data/orig_content")
        open(f'data/orig_content/{file.filename}', 'wb').write(file.file.read())
        vectorstore = ChromaService(
            name=kb_name,
            embedding_function=emb
        )
        file_extension = file.filename.split(".")[-1]
        loader_class_path = loader_map.get(file_extension)
        if not loader_class_path:
            return {"status": "error: Unsupported file type"}
        
        module_name, class_name = loader_class_path.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        loader_class = getattr(module, class_name)
        loader = loader_class()
        docs = loader.run(f'data/orig_content/{kb_name}/{file.filename}')

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)

        vectorstore.add_documents(all_splits)
        return {"status": "success", "filename": file.filename, "kb_name": kb_name, "length": len(file.file.read())}
    except Exception as e:
        return {"error": f"error: {e}"}


# def add_document(request: AddDocumentRequest):
#     try:
#         response = requests.get(request.url)
#         content_type = response.headers.get('Content-Type')
#         if 'text/html' in content_type:
#             file_extension = 'html'
#         else:
#             file_extension = request.url.split('.')[-1]
#             file = response.content
            
#             open(f'data/orig_content/{file.filename}', 'wb').write(file.content)
            

#         vectorstore = ChromaService(
#             name=request.kb_name,
#             embedding_function=emb
#         )
#         # vectorstore.add_documents()
#         return {"status": "success"}
#     except Exception as e:
#         return {"status": f"error: {e}"}

@app.post("/query")
def query(request: QueryRequest):
    vectorstore = ChromaService(
        name=request.kb_name,
        embedding_function=emb
    )
    retriever = Retriever(kb_svc=vectorstore)
    retrieved_docs = retriever.query(request.query, top_k=request.top_k)
    return {"docs": retrieved_docs}




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)