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
    def __init__(self, data_path, embedding_function, vector_store_path, splitter=None, document_loader=None):
        super().__init__(data_path, embedding_function, vector_store_path, splitter, document_loader, batch_size=1)

    def _filter_existing_docs(self, collection, docs, chunks):
        existing_ids = set(collection.get()["ids"])
        filtered_docs = []
        filtered_chunks = []
        for i in tqdm(range(len(docs)), desc="Filtering existing documents"):
            ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, d.page_content)) for d in chunks[i]]
            is_not_duplicate = False
            for c_id in ids:
                # if at least one chunk is not in the collection, then the whole document is considered as not in the collection
                if c_id in existing_ids:
                    is_not_duplicate = True
                    break

            if is_not_duplicate:
                filtered_docs.append(docs[i])
                filtered_chunks.append(chunks[i])

        return filtered_docs, filtered_chunks

    def _add_unique_to_collection(self, collection, docs, chunks):
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

        chunks_embeddings = self._embedding_function.embed_documents(docs_contents)

        collection.upsert(
            documents=chunks_contents,
            embeddings=chunks_embeddings,
            ids=chunks_ids,
            metadatas=chunks_metadatas
        )

    def _load_docs(self, collection, docs):
        # 1st split so that the model can handle the input
        splitter_max_tokens = self._embedding_function.get_splitter_max_tokens()
        docs = splitter_max_tokens.split_documents(docs)

        start = time.time()
        splitter = self._embedding_function.get_splitter()
        chunks = [splitter.split_documents([doc]) for doc in docs]
        docs, chunks = self._filter_existing_docs(collection, docs, chunks)
        print(f"Filtering existing documents took {time.time() - start} seconds")

        i = 0
        desc = f"Storing documents embeddings (batch size is {self._batch_size})" if self._batch_size > 1 else "Storing documents embeddings"
        with tqdm(total=len(docs), desc=desc) as pbar:
            while i < len(docs):
                batch_docs = docs[i:i + self._batch_size]
                if len(batch_docs) == 0:
                    continue

                # 2nd split (late chunking)
                self._add_unique_to_collection(collection, batch_docs, chunks[i:])

                i += self._batch_size
                pbar.update(self._batch_size)

            if i < len(docs):
                self._add_unique_to_collection(collection, docs[i:], chunks[i:])
                pbar.update(len(docs) - i)


