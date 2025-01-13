import gc
import os

import torch
from langchain_community.llms import LlamaCpp
from huggingface_hub import hf_hub_download
from langchain_core.messages import AIMessage


class Gemma:
    def __init__(self, model_download_path):
        model_file = hf_hub_download(
            repo_id="MaziyarPanahi/gemma-2-2b-it-GGUF",
            filename="gemma-2-2b-it.IQ1_M.gguf",
            cache_dir=os.path.join(model_download_path, "gemma-2-2b-it-GGUF")
        )

        self._llm = LlamaCpp(
            model_path=model_file,
            temperature=0.7,
            n_ctx=0, # 0 means we use the model's value
            max_tokens=300,
            top_p=0.85,
            n_gpu_layers=-1,  # nombre de couches à chargers sur le GPU
            # verbose=False
        )

    def invoke(self, prompt):
        if isinstance(prompt, str):
            return AIMessage(self._llm(prompt))
        return AIMessage(self._llm(prompt.messages[0].content))

    def __del__(self):
        self._llm = None
        gc.collect()
        torch.cuda.empty_cache()




# if __name__ == "__main__":
#     root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.path.dirname(__file__))))))
#     model_download_path = os.path.join(root_path, "models")
#     gemma = Gemma(model_download_path)
#     print(gemma.invoke("Hello, how are you?"))