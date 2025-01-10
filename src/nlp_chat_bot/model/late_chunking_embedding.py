import gc

import numpy as np
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
    def _chunk(self, doc, chunk_texts=None):
        tokens = self._tokenizer(doc, return_tensors="pt", return_offsets_mapping = True, truncation=True, max_length=self._max_tokens, padding="max_length")
        offsets = tokens['offset_mapping'][0]

        if chunk_texts is not None:
            texts = chunk_texts
        else:
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

        # print(chunks)
        span_annotations = []

        # Start and end positions of the chunks (in tokens not characters)
        for i in range(len(chunks_pos)-1):
            span_annotations.append((chunks_pos[i][0], chunks_pos[i+1][0]))

        # print(span_annotations)
        del tokens['offset_mapping']

        return tokens, span_annotations

    def _late_chunking(self, tokens_embeddings, span_annotations):
        # https://colab.research.google.com/drive/15vNZb6AsU7byjYoaEtXuNu567JWNzXOz?usp=sharing
        chunks_embeddings = []
        for embeddings, annotations in zip(tokens_embeddings, span_annotations):
            # Compute pooled embeddings for each span
            pooled_embeddings = [
                embeddings[start:end].mean(dim=0).detach().cpu().numpy()
                for start, end in annotations
                if (end - start) >= 1
            ]
            chunks_embeddings.append(pooled_embeddings)

        return chunks_embeddings


    def embed_documents(self, docs, chunks=None):
        docs_embeddings = []

        for i in range(len(docs)):
            doc = docs[i]
            if chunks is None:
                chunk_texts = None
            else:
                chunk_texts = [c.page_content for c in chunks[i]]
            inputs, span_annotations = self._chunk(doc, chunk_texts)
            # inputs = inputs.to(self._device)
            for k, v in inputs.items():
                inputs[k] = v.pin_memory().to(self._device, non_blocking=True)
            outputs = self._model(**inputs)
            embeddings = self._late_chunking(outputs[0], [span_annotations])[0]
            docs_embeddings += embeddings

            del inputs, outputs, embeddings, span_annotations
            # torch.cuda.empty_cache()
            # gc.collect()

        return docs_embeddings

    def embed_query(self, query):
        return self._model.encode(query)

    def get_id(self):
        return "late_chunking_embedding_jinaai"