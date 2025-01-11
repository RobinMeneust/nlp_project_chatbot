import gc

import numpy as np
from langchain_text_splitters import CharacterTextSplitter
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer
import torch

class LateChunkingEmbedding:
    def __init__(self, model_download_path, device=torch.device('cuda')):
        self._model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-small-en", cache_dir=model_download_path, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-small-en", cache_dir=model_download_path, trust_remote_code=True)
        self._model.to(device)
        self._device = device
        self._max_tokens = 8192
        self._splitter_max_tokens = CharacterTextSplitter.from_huggingface_tokenizer(
            self._tokenizer, chunk_size=self._max_tokens, chunk_overlap=0
        )

    def get_splitter_max_tokens(self):
        return self._splitter_max_tokens

    # based on https://github.com/jina-ai/late-chunking
    def _chunk(self, doc, chunk_texts):
        tokens = self._tokenizer(doc, return_tensors="pt", return_offsets_mapping = True, truncation=True, max_length=self._max_tokens, padding="max_length")
        offsets = tokens['offset_mapping'][0]

        end_char_indices = [len(chunk_texts[0])]
        for i in range(1, len(chunk_texts)):
            end_char_indices.append(end_char_indices[-1] + len(chunk_texts[i]))

        span_annotations = []

        chunk_index = 0 # used for the stopping criterion

        # Start and end positions of the chunks (in tokens not characters)
        start_idx = 1
        end_idx = 1
        while end_idx < len(offsets):
            if offsets[end_idx][1] >= end_char_indices[chunk_index]:
                span_annotations.append((start_idx, end_idx))
                start_idx = end_idx
                chunk_index += 1

            if chunk_index == len(end_char_indices):
                break
            end_idx += 1

        end_idx -= 1
        if start_idx < end_idx:
            span_annotations.append((start_idx, end_idx))

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


    def embed_documents(self, docs, chunks):
        docs_embeddings = []

        for i in range(len(docs)):
            doc = docs[i]
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