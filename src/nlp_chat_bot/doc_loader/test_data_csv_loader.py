from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import DirectoryLoader


class TestDataCSVLoader():
    def load(self, data_path):
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
