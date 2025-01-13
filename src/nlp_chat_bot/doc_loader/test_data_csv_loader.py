from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import DirectoryLoader


class TestDataCSVLoader():
    """ TestDataCSVLoader class to load documents from the given directory. Tailored for the test set specified in the README.md that is used in the tests notebooks"""

    def load(self, data_path):
        """ Load documents from a directory containing CSV files

        Args:
            data_path (str): Path to the directory containing the CSV files

        Returns:
            list: A list of loaded documents
        """
        docs = []
        loader = DirectoryLoader(
            path=data_path,
            glob=f"**/*.csv",
            loader_cls=CSVLoader,
            show_progress=True,
            loader_kwargs={"encoding":"utf-8", "content_columns":["passage"], "metadata_columns":["id"]}
        )
        docs += loader.load()
        return docs
