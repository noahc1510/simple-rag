from core.document_loader import PDFLoader

if __name__ == "__main__":
    path = "/Users/noah/Downloads/report.pdf"
    loader = PDFLoader()
    ret = loader.run(pdf_path=path)
    pass