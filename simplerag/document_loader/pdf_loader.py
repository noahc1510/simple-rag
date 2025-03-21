from typing import List, Optional

from .base import DocumentLoader
from .utils import OCR
from langchain_core.documents import Document
from tqdm import tqdm

import cv2
from PIL import Image


class PDFLoader(DocumentLoader):
    def __init__(self):
        super().__init__()

    def run(self, pdf_path, ocr_function: Optional[OCR] = None)->List[Document]:
        def rotate_img(img, angle):
            """
            img   --image
            angle --rotation angle
            return--rotated img
            """

            h, w = img.shape[:2]
            rotate_center = (w / 2, h / 2)
            # 获取旋转矩阵
            # 参数1为旋转中心点;
            # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
            # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
            M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
            # 计算图像新边界
            new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
            new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
            # 调整旋转矩阵以考虑平移
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

            rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
            return rotated_img

        if ocr_function is not None:
            import fitz  # pyMuPDF里面的fitz包，不要与pip install fitz混淆
            import numpy as np

            doc = fitz.open(pdf_path)
            resp = ""

            b_unit = tqdm(
                total=doc.page_count, desc="RapidOCRPDFLoader context page index: 0"
            )
            for i, page in enumerate(doc):
                b_unit.set_description(
                    "RapidOCRPDFLoader context page index: {}".format(i)
                )
                b_unit.refresh()
                text = page.get_text("")
                resp += text + "\n"

                img_list = page.get_image_info(xrefs=True)
                for img in img_list:
                    if xref := img.get("xref"):
                        bbox = img["bbox"]
                        # 检查图片尺寸是否超过设定的阈值
                        if (bbox[2] - bbox[0]) / (page.rect.width) < 0.6 or (bbox[3] - bbox[1]) / (
                            page.rect.height
                        ) < 0.6:
                            continue
                        pix = fitz.Pixmap(doc, xref)
                        samples = pix.samples
                        if int(page.rotation) != 0:  # 如果Page有旋转角度，则旋转图片
                            img_array = np.frombuffer(
                                pix.samples, dtype=np.uint8
                            ).reshape(pix.height, pix.width, -1)
                            tmp_img = Image.fromarray(img_array)
                            ori_img = cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR)
                            rot_img = rotate_img(img=ori_img, angle=360 - page.rotation)
                            img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                        else:
                            img_array = np.frombuffer(
                                pix.samples, dtype=np.uint8
                            ).reshape(pix.height, pix.width, -1)

                        result, _ = ocr_function.run(img_array)
                        if result:
                            ocr_result = [line[1] for line in result]
                            resp += "\n".join(ocr_result)

                # 更新进度
                b_unit.update(1)
            print(resp)
            return [Document(page_content=resp, metadata={"pdf_path": pdf_path})]
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
            print("text:", text)
            if text:
                return [Document(page_content=text, metadata={"pdf_path": pdf_path})]
            else:
                return []


if __name__ == "__main__":
    loader = PDFLoader()
    loader.run("test.pdf", None)
    pass
