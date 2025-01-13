from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class MiniLM:
    """ This class is used to embed documents using the MiniLM model

    Attributes:
        model (SentenceTransformer): The model used to embed the documents
    """
    def __init__(self, model_download_path):
        """Initializes the MiniLM object

        Args:
            model_download_path (str): The path to download the model
        """
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=model_download_path)

    def embed_documents(self, docs, verbose=False):
        """Embeds a list of documents

        Args:
            docs (list): A list of documents to embed
            verbose (bool): Whether to show a progress bar

        Returns:
            list: A list of embeddings
        """
        output = []

        if verbose:
            docs = tqdm(docs)

        for d in docs:
            output.append(self.model.encode(d).tolist())
        return output

    def embed_query(self, query):
        """Embeds a query

        Args:
            query (str): The query to embed

        Returns:
            list: The embedding of the query
        """
        return self.model.encode(query) #self.model.encode(query).tolist()

    def get_id(self):
        """Returns the id of the model (used for saving the embeddings in different vector stores)

        Returns:
            str: The id of the model
        """
        return "minilm"