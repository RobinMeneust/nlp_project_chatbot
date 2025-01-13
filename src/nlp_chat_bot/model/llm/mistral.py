import os
from langchain_community.llms import LlamaCpp
from huggingface_hub import hf_hub_download
from langchain_core.messages import AIMessage


class Mistral:
    def __init__(self, model_download_path):
        model_file = hf_hub_download(
            repo_id="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
            filename="Mistral-7B-Instruct-v0.3.Q8_0.gguf",
            cache_dir=os.path.join(model_download_path, "Mistral-7B-Instruct-v0.3-GGUF")
        )

        self._llm = LlamaCpp(
            model_path=model_file,
            temperature=0.7,
            max_tokens=300,
            top_p=0.85,
            n_gpu_layers=-1,  # nombre de couches à chargers sur le GPU
            verbose=False
        )

    def invoke(self, prompt):
        return AIMessage(self._llm(prompt.messages[0].content))