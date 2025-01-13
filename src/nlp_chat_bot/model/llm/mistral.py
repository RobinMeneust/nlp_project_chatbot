import gc
import os

import torch
from langchain_community.llms import LlamaCpp
from huggingface_hub import hf_hub_download
from langchain_core.messages import AIMessage


class Mistral:
    """ This class is a wrapper around the Mistral model (MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/Mistral-7B-Instruct-v0.3.Q8_0.gguf) to embed queries and documents

    Attributes:
        _llm (LlamaCpp): The Mistral model
    """
    def __init__(self, model_download_path):
        """Initializes the Mistral object

        Args:
            model_download_path (str): The path to download the model
        """
        model_file = hf_hub_download(
            repo_id="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
            filename="Mistral-7B-Instruct-v0.3.Q8_0.gguf",
            cache_dir=os.path.join(model_download_path, "Mistral-7B-Instruct-v0.3-GGUF")
        )

        self._llm = LlamaCpp(
            model_path=model_file,
            temperature=0.7,
            max_tokens=300,
            n_ctx=0,  # 0 means we use the model's value
            top_p=0.85,
            n_gpu_layers=-1,  # nombre de couches à chargers sur le GPU
            # verbose=False
        )

    def invoke(self, prompt):
        """Invokes the model with the given prompt

        Args:
            prompt (str): The prompt to use

        Returns:
            AIMessage: The response from the model
        """
        if isinstance(prompt, str):
            return AIMessage(self._llm(prompt))
        return AIMessage(self._llm(prompt.messages[0].content))

    def __del__(self):
        """Deletes the Mistral model (free VRAM)"""
        self._llm = None
        gc.collect()
        torch.cuda.empty_cache()