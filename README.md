# nlp_project_chatbot

## Install

`conda env create -f environment.yaml`
`pip install -e .`

Check install (first activate the environment): `python -c "import torch;print(torch.cuda.is_available())"`

Create a `.env` file in the root directory with the following content:

```
GOOGLE_API_KEY=...
```
In `.env` change GOOGLE_API_KEY=... with your own key (https://aistudio.google.com/u/1/apikey).

## Usage

`python -m streamlit run src/nlp_chat_bot/app.py`
