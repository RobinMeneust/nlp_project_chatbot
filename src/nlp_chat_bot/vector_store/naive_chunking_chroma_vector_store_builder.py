import uuid
from tqdm import tqdm
from nlp_chat_bot.vector_store.abstract_chroma_vector_store_builder import AbstractChromaVectorStoreBuilder


class NaiveChunkingChromaVectorStoreBuilder(AbstractChromaVectorStoreBuilder):
    """Class for the Naive Chunking Chroma Vector Store Builder

    Attributes:
        _document_loader (DocumentLoader): The document loader to use
        _embedding_function (object): The embedding function to use
        _vector_store_path (str): The path to the vector store
        _splitter (object): The splitter to use
        _data_path (str): The path to the data
    """
    def __init__(self, data_path, embedding_function, vector_store_path, splitter=None, document_loader=None):
        """Initializes the NaiveChunkingChromaVectorStoreBuilder object

        Args:
            data_path (str): The path to the data
            embedding_function (object): The embedding function to use
            vector_store_path (str): The path to the vector store
            splitter (object): The splitter to use
            document_loader (DocumentLoader): The document loader to use. If None, a default one will be used
        """
        super().__init__(data_path, embedding_function, vector_store_path, splitter, document_loader)

    def _filter_existing_docs(self, collection, docs):
        """Filters the existing documents (i.e. only keeps the ones that are not in the collection)

        Args:
            collection (Chroma): The collection to use
            docs (list): The documents to filter

        Returns:
            list: The filtered documents
        """
        existing_ids = set(collection.get()["ids"])

        if len(existing_ids) == 0:
            return docs

        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in docs]

        filtered_docs = []
        for i in range(len(ids)):
            if ids[i] not in existing_ids:
                filtered_docs.append(docs[i])

        return filtered_docs

    def _add_unique_to_collection(self, collection, docs):
        """Adds unique documents to the collection

        Args:
            collection (Chroma): The collection to use
            docs (list): The documents to add
        """
        if len(docs) == 0:
            return

        docs_ids_unordered_set = set()

        docs_ids = []
        docs_contents = []
        docs_metadatas = []

        for i in range(len(docs)):
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, docs[i].page_content))
            if doc_id not in docs_ids_unordered_set:
                docs_ids_unordered_set.add(doc_id)

                docs_ids.append(doc_id)
                docs_contents.append(docs[i].page_content)
                docs_metadatas.append(docs[i].metadata)

        # for text in docs_contents:
        #     docs_embeddings.append(self._embedding_function.embed_documents([text])[0])
        docs_embeddings = self._embedding_function.embed_documents(docs_contents)

        collection.upsert(
            documents=docs_contents,
            embeddings=docs_embeddings,
            ids=docs_ids,
            metadatas=docs_metadatas
       )

    def _load_docs(self, collection, docs):
        """Loads the documents to the collection

        Args:
            collection (object): The collection object
            docs (list): The list of documents
        """
        if self._splitter is not None:
            docs = self._splitter.split_documents(docs)
        docs = self._filter_existing_docs(collection, docs)

        i = 0
        desc = f"Storing documents embeddings (batch size is {self._batch_size})" if self._batch_size > 1 else "Storing documents embeddings"
        with tqdm(total=len(docs), desc=desc) as pbar:
            while i < len(docs):
                batch_docs = docs[i:i + self._batch_size]
                if len(batch_docs) == 0:
                    continue
                # embeddings = self._embedding_function.embed_documents([d.page_content for d in batch_docs])
                self._add_unique_to_collection(collection, batch_docs)
                i += self._batch_size
                pbar.update(self._batch_size)

            if i < len(docs):
                self._add_unique_to_collection(collection, docs[i:])
                pbar.update(len(docs) - i)
