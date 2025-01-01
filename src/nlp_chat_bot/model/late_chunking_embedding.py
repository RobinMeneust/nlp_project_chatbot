import gc

from langchain_text_splitters import CharacterTextSplitter
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer
import torch

class LateChunkingEmbedding:
    def __init__(self, model_download_path, chunk_size=100, chunk_overlap=0, device=torch.device('cuda')):
        self._model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-small-en", cache_dir=model_download_path, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-small-en", cache_dir=model_download_path, trust_remote_code=True)
        self._model.to(device)
        self._device = device
        self._max_tokens = 8192
        self._splitter = CharacterTextSplitter.from_huggingface_tokenizer(
            self._tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._splitter_max_tokens = CharacterTextSplitter.from_huggingface_tokenizer(
            self._tokenizer, chunk_size=self._max_tokens, chunk_overlap=0
        )

    def get_splitter(self):
        return self._splitter

    def get_splitter_max_tokens(self):
        return self._splitter_max_tokens

    # based on https://github.com/jina-ai/late-chunking
    def _chunk(self, doc):
        tokens = self._tokenizer(doc, return_tensors="pt", return_offsets_mapping = True, truncation=True, max_length=self._max_tokens, padding="max_length")
        offsets = tokens['offset_mapping'][0]

        texts = self._splitter.split_text(doc)

        end_char_indices = [len(texts[0])]
        for i in range(1, len(texts)):
            end_char_indices.append(end_char_indices[-1] + len(texts[i]))

        chunks_pos = [(1,0)]
        chunk_index = 0
        for i in range(1, len(offsets)):
            if offsets[i][1] >= end_char_indices[chunk_index]:
                if i < len(offsets) - 1:
                    chunks_pos.append((i, int(offsets[i+1][0])))
                else:
                    chunks_pos.append((i, len(offsets)-1))
                chunk_index += 1
            if chunk_index == len(end_char_indices):
                break



        # list of tuples (chunk_id, start_position)
        # List of (token index, token pos in text + 1) where token is a punctuation mark
        # chunks_pos = []
        # for i in range(len(offsets)):
        #     token_id = token_ids[i]
        #     start = offsets[i][0]
        #
        #     if token_id in end_of_sentence_tokens and token_ids[i] != token_ids[i + 1]:
        #         # if it's the last token, or it's an end of sentence token and the next one is not one (to avoid "..." case)
        #         chunks_pos.append((i, int(start + 1)))

        # print(offsets)
        # print(chunks_pos)
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
        for d in tqdm(docs):
            inputs, _, span_annotations = self._chunk(d)
            inputs.to(self._device)
            # print(len(d), len(inputs['input_ids'][0]),"\n")
            outputs = self._model(**inputs)
            embeddings = self._late_chunking(outputs[0], [span_annotations])[0]
            docs_embeddings += embeddings

            # print(f"embeddings in late_chunking_embedding.py: {len(docs_embeddings)} (+{len(embeddings)})")

            del inputs, outputs, embeddings, span_annotations
            torch.cuda.empty_cache()
            gc.collect()

        return docs_embeddings

    def embed_query(self, query):
        return self._model.encode(query)

# import os
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# if __name__ == "__main__":
#     current_file_path = os.path.abspath(__file__)
#
#     model_download_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))) + "/models"
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=100,  # chunk size (characters)
#         chunk_overlap=10,  # chunk overlap (characters)
#         add_start_index=True,  # track index in original document
#     )
#     embedding_function = LateChunkingEmbedding(model_download_path, splitter)
#     docs = ["This is a test document. It has two sentences. The second sentence is this one."]
#     print(len(embedding_function.embed_documents(docs)[0]), ":", embedding_function.embed_documents(docs))
#     print(len(embedding_function.embed_query("What is the acronym AIA?")), ":", embedding_function.embed_query("What is the acronym AIA?"))
#     print("Done")