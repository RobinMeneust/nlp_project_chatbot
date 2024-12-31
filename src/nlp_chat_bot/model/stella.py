import os
from sentence_transformers import SentenceTransformer

class Stella:
    def __init__(self, model_download_path):
        self.model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", cache_folder=model_download_path, trust_remote_code=True).cuda()

    def embed_documents(self, docs):
        return [self.model.encode(d).tolist() for d in docs]

    def embed_query(self, query):
        return self.model.encode(query)

