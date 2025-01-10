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

## Datasets

We tried with a personal PDF file (a project report) so that the data can't be known by the LLM model used.
But we also used the first 1012 lines of [MPST: Movie Plot Synopses with Tags](https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags?resource=download)

## Usage

`python -m streamlit run src/nlp_chat_bot/start_app.py`

## Possible Issues and Fixes for Streamlit

- If the app runs the same code multiple times even though you started it once : don't open multiple tabs of the app in the same browser, it will cause issues: (https://stackoverflow.com/questions/76474732/why-is-the-code-executed-multiple-times-whenever-streamlit-is-started)
- If you get Process finished with exit code -1073741819 (0xC0000005) in PyCharm, consider running the app from the terminal instead (using the command above), it fixed it for us. Maybe checking "Run with Python Console" might also fix it.

## Dev

For the `profiler.py` script, install gprof2dot using `pip install gprof2dot`. Other requirements such as Graphviz or dot might be required