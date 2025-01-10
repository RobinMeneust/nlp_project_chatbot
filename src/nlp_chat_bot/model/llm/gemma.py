import os
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
            # temperature=0.,
            max_tokens=200,
            top_p=1,
            n_gpu_layers=-1,  # nombre de couches à chargers sur le GPU
            verbose=False
        )

    def invoke(self, prompt):
        return AIMessage(self._llm(prompt.messages[0].content))

# if __name__ == "__main__":
#     root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.path.dirname(__file__))))))
#     model_download_path = os.path.join(root_path, "models")
#     gemma = Gemma(model_download_path)
#     print(gemma.invoke("Hello, how are you?"))