from sentence_transformers import SentenceTransformer

class MiniLM:
    def __init__(self, model_download_path):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=model_download_path)

    def embed_documents(self, docs):
        return [self.model.encode(d).tolist() for d in docs]

    def embed_query(self, query):
        return self.model.encode(query) #self.model.encode(query).tolist()
