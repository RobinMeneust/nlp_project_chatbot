import uuid
from gc import collect

import chromadb
import langchain_chroma
from langchain_chroma import Chroma
from tqdm import tqdm

from nlp_chat_bot.doc_loader.document_loader import DocumentLoader


class ChromaVectorStoreBuilder:
    def __init__(self, data_path, embedding_function, vector_store_path, splitter=None, is_late_chunking=False):
        self._document_loader = DocumentLoader()
        self._is_late_chunking = is_late_chunking
        self._embedding_function = embedding_function
        self._vector_store_path = vector_store_path
        self._splitter = splitter
        self._data_path = data_path
        self._collection_name = f"chatbot_docs_collection_{self._embedding_function.get_id()}"

    def build(self) -> langchain_chroma.vectorstores.Chroma:
        return self._init_vector_store(self._data_path)

    def _add_unique_to_collection(self, collection, docs, embeddings):
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in docs]
        existing_ids = set(collection.get()["ids"])
        added_ids = set()
        added_docs = []
        added_embeddings = []


        for i in range(len(ids)):
            if ids[i] not in added_ids and ids[i] not in existing_ids:
                try:
                    added_docs.append(docs[i].page_content)
                    added_embeddings.append(embeddings[i])
                    added_ids.add(ids[i])
                except ValueError:
                    # It already exists
                    pass

        added_ids = list(added_ids)

        # print(f"added_ids: {len(added_ids)}, added_docs: {len(added_docs)}, added_embeddings: {len(added_embeddings)}, existing_ids: {len(existing_ids)}")

        if len(added_docs) > 0:
            collection.add(
                documents=added_docs,
                embeddings=added_embeddings,
                ids=added_ids
           )

    def _filter_existing_docs(self, collection, docs, late_chunking_splitter=None):
        if late_chunking_splitter is not None:
            filtered_docs = []
            for doc in docs:
                docs_chunks = late_chunking_splitter.split_documents([doc])
                filtered_chunks = self._filter_existing_docs(collection, docs_chunks)
                if len(filtered_chunks) > 0:
                    # Some chunks are new
                    filtered_docs.append(doc)
            return filtered_docs
        else:
            existing_ids = set(collection.get()["ids"])
            ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in docs]

            filtered_docs = []
            for i in range(len(ids)):
                if ids[i] not in existing_ids:
                    filtered_docs.append(docs[i])

        return filtered_docs

    def _init_vector_store(self, data_path):
        docs = self._document_loader.load(data_path)

        if not docs or len(docs) == 0:
            raise ValueError("No document found")

        chroma_client = chromadb.PersistentClient(self._vector_store_path)
        collection = chroma_client.get_or_create_collection(name=self._collection_name)

        if self._is_late_chunking:
            # 1st split so that the model can handle the input
            splitter_max_tokens = self._embedding_function.get_splitter_max_tokens()
            docs = splitter_max_tokens.split_documents(docs)
            docs = self._filter_existing_docs(collection, docs, self._embedding_function.get_splitter())

            print(f"Embedding and storing {len(docs)} new documents...")
            for d in tqdm(docs):
                # 2nd split (late chunking)
                splitter = self._embedding_function.get_splitter()
                split_docs = splitter.split_documents([d])
                embeddings = self._embedding_function.embed_documents([d.page_content])
                self._add_unique_to_collection(collection, split_docs, embeddings)
        else:
            if self._splitter is not None:
                docs = self._splitter.split_documents(docs)
            docs = self._filter_existing_docs(collection, docs)
            print(f"Embedding and storing {len(docs)} new documents...")
            for d in tqdm(docs):
                embeddings = self._embedding_function.embed_documents([d.page_content])
                self._add_unique_to_collection(collection, [d], embeddings)

        return Chroma(client=chroma_client, collection_name=self._collection_name, embedding_function=self._embedding_function)
