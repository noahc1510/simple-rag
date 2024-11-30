from .base import DocumentLoader
import bs4
from langchain_community.document_loaders import WebBaseLoader


class HtmlLoader(DocumentLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, url):
        # Only keep post title, headers, and content from the full HTML.
        bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs={"parse_only": bs4_strainer},
        )
        docs = loader.load()

        len(docs[0].page_content)
        return docs

if __name__ == "__main__":
    loader = HtmlLoader()
    ret = loader.run("https://lilianweng.github.io/posts/2023-06-23-agent/")
    print(ret[0].page_content)
    pass