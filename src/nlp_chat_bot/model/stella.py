import os
from sentence_transformers import SentenceTransformer
from pathlib import Path

class Stella:
    def __init__(self, model_download_path):
        self.model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", cache_folder=model_download_path, trust_remote_code=True).cuda()

    def embed_documents(self, docs):
        return [self.model.encode(d).tolist() for d in docs]

    def embed_query(self, query):
        return self.model.encode(query)


# This model supports two prompts: "s2p_query" and "s2s_query" for sentence-to-passage and sentence-to-sentence tasks, respectively.
# They are defined in `config_sentence_transformers.json`

queries = [
    "What are some ways to reduce stress?",
    "What are the benefits of drinking green tea?",
]
# docs do not need any prompts
docs = [
    "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
    "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
]


model_download_path = os.path.join(Path(__file__).resolve().parent.parent.parent.parent, "models")

model = Stella(model_download_path)
query_embeddings = model.embed_query(queries)
doc_embeddings = model.embed_documents(docs)
print(query_embeddings.shape, doc_embeddings.shape)
