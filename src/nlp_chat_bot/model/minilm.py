from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class MiniLM:
    def __init__(self, model_download_path):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=model_download_path)

    def embed_documents(self, docs):
        output = []
        for d in docs:
            output.append(self.model.encode(d).tolist())
        return output

    def embed_query(self, query):
        return self.model.encode(query) #self.model.encode(query).tolist()

    def get_id(self):
        return "minilm"