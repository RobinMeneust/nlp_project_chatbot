import gc

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
        self._splitter_max_tokens = RecursiveCharacterTextSplitter(
            chunk_size = self._max_tokens*3, chunk_overlap=0 # token split didn't work so we use character split
        )

    def get_splitter_max_tokens(self):
        return self._splitter_max_tokens

    # based on https://github.com/jina-ai/late-chunking
    def _chunk(self, doc, chunk_texts):
        tokens = self._tokenizer(doc, return_tensors="pt", return_offsets_mapping = True)
        if len(tokens['input_ids'][0]) > self._max_tokens:
            raise Exception(f"Document is too long: {len(tokens['input_ids'][0])} tokens")
        end_char_indices = [len(chunk_texts[0])]
        for i in range(1, len(chunk_texts)):
            end_char_indices.append((end_char_indices[-1] + len(chunk_texts[i])))

        offsets = tokens['offset_mapping'][0]



        span_annotations = []

        chunk_index = 0 # used for the stopping criterion

        # Start and end positions of the chunks (in tokens not characters)
        start_token_idx = 1
        end_token_idx = 1

        last_token_idx = len(offsets)-1

        while chunk_index < len(end_char_indices):
            # if chunk_index == len(end_char_indices) - 1:
            #     print(f"current :{offsets[end_idx][1]}, target: {end_char_indices[chunk_index]}, max len: {len(offsets)}")

            # if padding or end chunk character reached
            if offsets[end_token_idx][1] >= end_char_indices[chunk_index]:
                # print("chunk idx: ", chunk_index, "end_idx: ", end_idx, "end_char_indices[chunk_index]: ", end_char_indices[chunk_index], "len end_char_indices: ", len(end_char_indices), "offsets[end_idx][1]:", offsets[end_idx][1])
                span_annotations.append((start_token_idx, end_token_idx))

                if offsets[end_token_idx][1] == end_char_indices[chunk_index]:
                    # If the end of the chunk is exactly at the end of the token
                    end_token_idx = end_token_idx
                else:
                    # Otherwise we keep the previous token (because the end character is in the middle of the token)
                    end_token_idx = end_token_idx - 1

                start_token_idx = end_token_idx

                chunk_index += 1
            elif end_token_idx == last_token_idx:
                # If we reached the end of the tokens
                span_annotations.append((start_token_idx, end_token_idx))
                break
            end_token_idx += 1

        # print(span_annotations)
        del tokens['offset_mapping']

        if len(span_annotations) != len(chunk_texts):
            exception_txt = f"""Number of chunks and span annotations do not match
            \tNumber of tokens: {len(tokens['input_ids'][0])}
            \tTotal number of characters in chunk_texts: {sum([len(txt) for txt in chunk_texts])}
            \tNumber of characters in chunk_texts: {[len(txt) for txt in chunk_texts]}
            \tLasts offsets values: {offsets[-2], offsets[-1]}
            \tNumber of chunks: {len(chunk_texts)}
            \tlength of end_char_indices: {len(end_char_indices)}
            \tNumber of span annotations: {len(span_annotations)}
            \tspan_annotations: {span_annotations}
            \tend_char_indices: {end_char_indices}
            \tlen offsets: {len([offsets[i] for i in range(len(offsets)) if offsets[i][0] != 0 or i == 0])}
            """
            raise Exception(exception_txt)

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