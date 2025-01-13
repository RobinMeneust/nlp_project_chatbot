# nlp_project_chatbot

## Install

`conda env create -f environment.yaml`
`pip install -e .`

Check install (first activate the environment): `python -c "import torch;print(torch.cuda.is_available())"`

### Issues with LlamaCpp

We use LlamaCpp to load local LLMs models. If it doesn't use your GPU you can try the following steps.

#### On Linux

`CMAKE_ARGS=-DGGML_CUDA=on; FORCE_CMAKE=1; pip install -r requirements.txt`

#### On Windows

Try [this](https://python.langchain.com/docs/integrations/llms/llamacpp/) and [this](https://pypi.org/project/llama-cpp-python/) (check pre-built wheel, e.g. `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/124`)
If ou still have issues, try also [this](https://www.reddit.com/r/LocalLLaMA/comments/14jq3ih/lamacpppython_with_gpu_acceleration_on_windows/) : install cl.exe from Visual Studio Build Tools, and add the path to the PATH environment variable. And add the following variables :

In our case (on Windows 11), the following commands worked:

First we set the following environment variables (warning: the command is different in PowerShell and in cmd to set vars):
```
LLAMA_CUBLAS = "1"
FORCE_CMAKE = "1"
CMAKE_ARGS="-DLLAMA_CUBLAS=on" 
CMAKE_ARGS="-DGGML_CUDA=on"
```
and run `pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir`

## API keys (e.g. Gemini)

Create a `.env` file in the root directory with the following content:

```
GOOGLE_API_KEY=...
```
In `.env` change GOOGLE_API_KEY=... with your own key (https://aistudio.google.com/u/1/apikey).

## Datasets

We tried with a personal PDF file (a project report) so that the data can't be known by the LLM model used.
But we also used the first 1012 lines of [MPST: Movie Plot Synopses with Tags](https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags?resource=download) and [RAG Mini BioASQ](https://huggingface.co/datasets/enelpol/rag-mini-bioasq) (labelled).

Just place your documents (.pdf, .csv, .html or .txt) in the `data` folder.

## Usage

`python -m streamlit run src/nlp_chat_bot/start_app.py`

## Possible Issues and Fixes for Streamlit

- If the app runs the same code multiple times even though you started it once : don't open multiple tabs of the app in the same browser, it will cause issues: (https://stackoverflow.com/questions/76474732/why-is-the-code-executed-multiple-times-whenever-streamlit-is-started)
- If you get Process finished with exit code -1073741819 (0xC0000005) in PyCharm, consider running the app from the terminal instead (using the command above), it fixed it for us. Maybe checking "Run with Python Console" might also fix it.
