from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader


class DocumentLoader():
    """ DocumentLoader class to load documents from a directory

    Attributes:
        loaders (dict): A dictionary of file extensions and their respective loader classes
        loaders_kwargs (dict): A dictionary of file extensions and their respective loader kwargs
    """
    def __init__(self) -> None:
        """ Initialize DocumentLoader"""
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.csv': CSVLoader,
            '.html': UnstructuredHTMLLoader,
            '.txt': TextLoader
        }

        self.loaders_kwargs = {
            '.pdf': {},
            '.csv': {"encoding": "utf-8"},
            '.html': {},
            '.txt': {}
        }

    def load(self, data_path):
        """ Load documents from a directory and only process the files with the extensions specified in the loaders attribute

        Args:
            data_path (str): Path to the directory containing the documents

        Returns:
            list: A list of loaded documents
        """
        docs = []
        for file_type, loader_cls in self.loaders.items():
            loader = DirectoryLoader(
                path=data_path,
                glob=f"**/*{file_type}",
                loader_cls=loader_cls,
                show_progress=True,
                loader_kwargs=self.loaders_kwargs[file_type]
            )
            docs += loader.load()
        return docs