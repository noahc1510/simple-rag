from .base import OCR
from rapidocr_paddle import RapidOCR

class PaddleOCR(OCR):
    def __init__(self, **kwargs) -> None:
        self.engine = RapidOCR()
        super().__init__(**kwargs)
    
    def run(self, img):
        return self.engine(img)