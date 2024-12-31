import numpy as np
from transformers import AutoModel
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


class LateChunkingEmbedding:
    def __init__(self, model_download_path, chunk_size):
        self._model = AutoModel.from_pretrained("dunzhang/stella_en_1.5B_v5", cache_dir=model_download_path, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained("dunzhang/stella_en_1.5B_v5", cache_dir=model_download_path, trust_remote_code=True)
        self._encode_model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", cache_folder=model_download_path, trust_remote_code=True)
        self._chunk_size = chunk_size # TODO not used here

    # based on https://github.com/jina-ai/late-chunking
    def _chunk(self, doc):
        tokens = self._tokenizer(doc, return_tensors="pt", return_offsets_mapping = True)
        end_of_sentence_tokens = [self._tokenizer.convert_tokens_to_ids(c) for c in ['.', '!', '?']]
        offsets = tokens['offset_mapping'][0]
        token_ids = tokens['input_ids'][0]

        # list of tuples (chunk_id, start_position)
        # List of (token index, token pos in text + 1) where token is a punctuation mark
        chunks_pos = []
        for i in range(len(offsets)):
            token_id = token_ids[i]
            start = offsets[i][0]

            if token_id in end_of_sentence_tokens and token_ids[i] != token_ids[i + 1]:
                # if it's the last token, or it's an end of sentence token and the next one is not one (to avoid "..." case)
                chunks_pos.append((i, int(start + 1)))

        # print(offsets)
        # print(chunks_pos)
        chunks_pos = [(1, 0)] + chunks_pos
        chunks = []

        # print(chunks_pos)

        # Chunks (text)
        for i in range(len(chunks_pos)-1):
            chunks.append(doc[chunks_pos[i][1]:chunks_pos[i+1][1]])

        # print(chunks)
        span_annotations = []

        # Start and end positions of the chunks (in tokens not characters)
        for i in range(len(chunks_pos)-1):
            span_annotations.append((chunks_pos[i][0], chunks_pos[i+1][0]))

        # print(span_annotations)
        del tokens['offset_mapping']
        return tokens, chunks, span_annotations

    def _late_chunking(self, tokens_embeddings, span_annotations):
        # https://colab.research.google.com/drive/15vNZb6AsU7byjYoaEtXuNu567JWNzXOz?usp=sharing
        chunks_embeddings = []
        for embeddings, annotations in zip(tokens_embeddings, span_annotations):
            pooled_embeddings = [
                embeddings[start:end].sum(dim=0) / (end - start)
                for start, end in annotations
                if (end - start) >= 1
            ]
            pooled_embeddings = [
                embedding.detach().cpu().numpy() for embedding in pooled_embeddings
            ]
            chunks_embeddings.append(pooled_embeddings)

        return chunks_embeddings

    def embed_documents(self, docs):
        docs_embeddings = []
        for d in docs:
            inputs, _, span_annotations = self._chunk(d)
            outputs = self._model(**inputs)
            embeddings = self._late_chunking(outputs[0], [span_annotations])[0]
            docs_embeddings += embeddings

        return docs_embeddings

    def embed_query(self, query):
        return self._encode_model.encode(query)


import os
if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)

    model_download_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))) + "/models"
    chunk_size = 1000
    embedding_function = LateChunkingEmbedding(model_download_path, chunk_size)
    docs = ["This is a test document. It has two sentences. The second sentence is this one."]
    print(len(embedding_function.embed_documents(docs)[0]), ":", embedding_function.embed_documents(docs))
    print(len(embedding_function.embed_query("What is the acronym AIA?")), ":", embedding_function.embed_query("What is the acronym AIA?"))
    print("Done")