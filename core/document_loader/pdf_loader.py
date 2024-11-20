from typing import List, Optional

from .base import DocumentLoader
from .utils import OCR
from langchain_core.documents import Document


class PDFLoader(DocumentLoader):
    def __init__(self):
        super().__init__()

    def run(self, pdf_path, ocr_function: Optional[OCR] = None)->List[Document]:
        if isinstance(ocr_function, OCR):
            pass
        else:
            # 不使用OCR, 直接读取PDF
            import fitz  # PyMuPDF

            def read_pdf_text_with_pymupdf(pdf_path):
                document = fitz.open(pdf_path)
                text = ""
                for page_num in range(len(document)):
                    page = document.load_page(page_num)
                    text += page.get_text()
                return text

            text = read_pdf_text_with_pymupdf(pdf_path)
            # print(text)
            return [Document(page_content=text, metadata={"pdf_path": pdf_path})]


if __name__ == "__main__":
    loader = PDFLoader()
    loader.run("test.pdf", None)
    pass
