import os
import uuid
from abc import ABC, abstractmethod
from gc import collect

import chromadb
import langchain_chroma
from langchain_chroma import Chroma
from tqdm import tqdm

from nlp_chat_bot.doc_loader.document_loader import DocumentLoader


class AbstractChromaVectorStoreBuilder(ABC):
    def __init__(self, data_path, embedding_function, vector_store_path, splitter=None, document_loader=None, batch_size=1000):
        if document_loader is None:
            self._document_loader = DocumentLoader()
        else:
            self._document_loader = document_loader
        self._embedding_function = embedding_function
        self._vector_store_path = str(os.path.join(vector_store_path, self._embedding_function.get_id()))
        self._splitter = splitter
        self._data_path = data_path
        self._collection_name = f"chatbot_docs_collection"
        self._batch_size = batch_size if batch_size is not None and batch_size > 0 else 10000

    def build(self, update_docs=True, reset=False) -> langchain_chroma.vectorstores.Chroma:
        return self._init_vector_store(self._data_path, update_docs, reset)

    def _init_vector_store(self, data_path, update_docs, reset):
        chroma_client = chromadb.PersistentClient(self._vector_store_path)

        if reset:
            try:
                chroma_client.delete_collection(self._collection_name)
            except ValueError:
                pass # it does not exist yet

        if not update_docs:
            return Chroma(client=chroma_client, collection_name=self._collection_name, embedding_function=self._embedding_function)


        docs = self._document_loader.load(data_path)
        if not docs or len(docs) == 0:
            raise ValueError("No document found")


        collection = chroma_client.get_or_create_collection(name=self._collection_name)


        if self._splitter is not None:
            docs = self._splitter.split_documents(docs)

        self._load_docs(collection, docs)
        print("Documents are now loaded")

        return Chroma(client=chroma_client, collection_name=self._collection_name, embedding_function=self._embedding_function)

    @abstractmethod
    def _load_docs(self, collection, docs):
        pass