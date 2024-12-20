from langchain.document_loaders import CSVLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader


class DocumentLoader():      
    def __init__(self) -> None:
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.csv': CSVLoader,
            '.html': UnstructuredHTMLLoader,
            '.txt': TextLoader
        }

    def load(self, data_path):
        docs = []
        for file_type, loader_cls in self.loaders.items():
            loader = DirectoryLoader(
                path=data_path,
                glob=f"**/*{file_type}",
                loader_cls=loader_cls,
                show_progress=True
            )
            docs += loader.load()
        return docs