from concurrent.futures import ThreadPoolExecutor
import time
import uuid
from gc import collect

import chromadb
import langchain_chroma
from langchain_chroma import Chroma
from tqdm import tqdm

from nlp_chat_bot.doc_loader.document_loader import DocumentLoader
from nlp_chat_bot.vector_store.abstract_chroma_vector_store_builder import AbstractChromaVectorStoreBuilder


class LateChunkingChromaVectorStoreBuilder(AbstractChromaVectorStoreBuilder):
    """Class for the Late Chunking Chroma Vector Store Builder

    Attributes:
        _document_loader (DocumentLoader): The document loader to use
        _embedding_function (object): The embedding function to use
        _vector_store_path (str): The path to the vector store
        _splitter (object): The splitter to use
        _data_path (str): The path to the data
        _collection_name (str): The name of the collection
        _batch_size (int): The batch size to use
    """
    def __init__(self, data_path, embedding_function, vector_store_path, splitter=None, document_loader=None):
        """Initializes the LateChunkingChromaVectorStoreBuilder object

        Args:
            data_path (str): The path to the data
            embedding_function (object): The embedding function to use
            vector_store_path (str): The path to the vector store
            splitter (object): The splitter to use
            document_loader (DocumentLoader): The document loader to use. If None, a default one will be used
        """
        super().__init__(data_path, embedding_function, vector_store_path, splitter, document_loader, batch_size=100)
        if self._splitter is None:
            raise ValueError("Late chunking requires a splitter, because it's then equivalent to naive chunking so you should use NaiveChunkingChromaVectorStoreBuilder instead")

        if splitter._chunk_overlap > 0:
            raise ValueError("Late chunking requires a splitter with no overlap")

    def _filter_existing_docs(self, collection, docs, chunks):
        """Filters the existing documents (i.e. only keeps the ones that are not in the collection)

        Args:
            collection (Chroma): The collection to use
            docs (list): The documents to filter
            chunks (list): The chunks to filter

        Returns:
            list: The filtered documents
            list: The filtered chunks
        """
        existing_ids = set(collection.get()["ids"])

        if len(existing_ids) == 0:
            return docs, chunks

        filtered_docs = []
        filtered_chunks = []
        for i in tqdm(range(len(docs)), desc="Filtering existing documents"):
            ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, d.page_content)) for d in chunks[i]]
            is_duplicate = True
            for c_id in ids:
                # if at least one chunk is not in the collection, then the whole document is considered as not in the collection
                if c_id not in existing_ids:
                    is_duplicate = False
                    break

            if not is_duplicate:
                filtered_docs.append(docs[i])
                filtered_chunks.append(chunks[i])

        return filtered_docs, filtered_chunks

    def _add_unique_to_collection(self, collection, docs, chunks):
        """Adds unique documents to the collection

        Args:
            collection (Chroma): The collection to use
            docs (list): The documents to add
            chunks (list): The chunks to add

        """
        if len(docs) == 0:
            return

        docs_ids_unordered_set = set()

        docs_ids = []
        docs_contents = []
        old_chunks = chunks
        chunks = []

        for i in range(len(docs)):
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, docs[i].page_content))
            if doc_id not in docs_ids_unordered_set:
                docs_ids_unordered_set.add(doc_id)

                docs_ids.append(doc_id)
                docs_contents.append(docs[i].page_content)
                chunks.append(old_chunks[i])

        chunks_ids = []
        chunks_contents = []
        chunks_metadatas = []

        for i in range(len(chunks)):
            doc_id = docs_ids[i]
            for j in range(len(chunks[i])):
                chunk_id = f"{doc_id}_{uuid.uuid5(uuid.NAMESPACE_DNS, chunks[i][j].page_content)}"
                if chunk_id not in docs_ids_unordered_set:
                    docs_ids_unordered_set.add(chunk_id)

                    chunks_ids.append(chunk_id)
                    chunks_contents.append(chunks[i][j].page_content)
                    chunks_metadatas.append(chunks[i][j].metadata)

        chunks_embeddings = self._embedding_function.embed_documents(docs_contents, chunks)

        # print(f"Storing {sum([len(c) for c in chunks])} total chunks len, {len(chunks_contents)} chunks with {len(chunks_embeddings)} embeddings")



        collection.upsert(
            documents=chunks_contents,
            embeddings=chunks_embeddings,
            ids=chunks_ids,
            metadatas=chunks_metadatas
        )


    def _load_docs(self, collection, docs):
        """Loads the documents to the collection

        Args:
            collection (object): The collection object
            docs (list): The list of documents
        """
        # 1st split so that the model can handle the input
        splitter_max_tokens = self._embedding_function.get_splitter_max_tokens()
        docs = splitter_max_tokens.split_documents(docs)

        # 2nd split (considering late chunking)
        chunks = [self._splitter.split_documents([doc]) for doc in docs]
        docs, chunks = self._filter_existing_docs(collection, docs, chunks)

        i = 0
        desc = f"Storing {len(docs)} documents embeddings (batch size is {self._batch_size})" if self._batch_size > 1 else f"Storing {len(docs)} documents embeddings"
        with tqdm(total=len(docs), desc=desc) as pbar:
            while i < len(docs):
                end_index = i + self._batch_size
                batch_docs = docs[i:end_index]
                if len(batch_docs) == 0:
                    continue

                self._add_unique_to_collection(collection, batch_docs, chunks[i:end_index])

                i = end_index
                pbar.update(self._batch_size)

            if i < len(docs):
                self._add_unique_to_collection(collection, docs[i:], chunks[i:])
                pbar.update(len(docs) - i)


