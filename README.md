# AI Summarization
Repo showcasing AI summarization tool.

## Summary
This repo was created to summarize Wikilingua dataset 

## Setup

### Installing Dependencies

Install following dependencies (on macOS):

- Python packages: run `pip3 install -r requirements.txt`
- Download `mistral-7b-v0.1.Q2_K.gguf` model from Hugging Face [TheBloke/Mistral-7B-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF) repo into local `models` directory.
- Download `llama-2-7b-Q2_K.gguf` model from Hugging Face [TheBloke/Llama-2-7B-GGUF]([https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF(https://huggingface.co/TheBloke/Llama-2-7B-GGUF/tree/main)) repo into local `models` directory.

## Running
Simply run 'python summarization_script.py'

### Jupyter Notebook

An attempt was made to replicate these experiments on a free tier Google Collab notebook for access to Nvidia GPU, but in the interest of time could not be completed.
## Details

### Workflow

Depending on the document size, this tool works in following modes:
1. In the simple case, if the whole document can fit into model's context window then summarizartion is based on adding relevant summarization prompt.
2. In case of large documents, document processed using "map-reduce" pattern:
  1. The document is first split into smaller chunks using `RecursiveCharacterTextSplitter`` which tries to respect paragraph and sentence boundarions.
  2. Each chunk is summarized separately (map step).
  3. Chunk summarization are summarized again to give final summary (reduce step).

### Local processing
All processing is done locally on the user's machine.
- Quantified Mistral model (`models/mistral-7b-instruct-v0.2.Q2_K.gguf`)
- Quantified Llama model (`models/llama-2-7b.Q2_K.gguf`)
