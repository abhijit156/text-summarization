import json

import os
import pickle
import random
from urllib.parse import urlparse

from langchain.llms import LlamaCpp
from langchain.document_loaders import TextLoader
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llmlingua import PromptCompressor



############################################################################
### Importing data
############################################################################


documents = "data/english.pkl" # Insert file path here


def load_content():
    """Randomly subset 100 documents from the data and return as a dict"""

    if documents:
        with open(documents, "rb") as file:
            data = pickle.load(file)
        
        # Get all keys in the data
        keys = list(data.keys())
        # Randomly subset 100 documents
        random_keys = random.sample(keys, k=10)
        
        subset_documents = {}
        subset_docnames = []
        # Iterate over the randomly selected keys and store the corresponding text in the dictionary

        for key in random_keys:
            subset_documents[key] = data[key]

            #Also store the document names
            parsed_url = urlparse(key)
            path = parsed_url.path
            subset_docnames.append(path.split("/")[-1])

    return subset_documents, subset_docnames



############################################################################
# Style of summarization: Prompt to the summarization API
############################################################################

# Numbered List style
STYLE="Return one-two line summary of given text"
PROMPT_TRIGGER="SUMMARY"

## Other prompt styles tried -
# "Return a summary in bulleted list format which captures the core ideas in text given"
# "Return bulleted list summarizing the core ideas of given text"

VERBOSE=True


############################################################################
# Read in Model file, set context window, and max answer tokens
############################################################################
# Load 2 bit model. Using the smallest, most lossy model for speed. 
MODEL_FILE="models/llama-2-7b.Q2_K.gguf"
## Load 3 bit model.
# MODEL_FILE="models/llama-2-7b.Q3_K_M.gguf"
# # Load small 3 bit model. 
# MODEL_FILE="models/mistral-7b-instruct-v0.2.Q3_K_S.gguf"
# Load medium 3 bit model. 
# MODEL_FILE="models/mistral-7b-instruct-v0.2.Q3_K_M.gguf"
# Load 6 bit model. 
# MODEL_FILE="models/mistral-7b-instruct-v0.2.Q6_K.gguf"

MODEL_CONTEXT_WINDOW = 128

# Maximal length of model's output, in tokens.
MAX_ANSWER_TOKENS = 128

llm = LlamaCpp(
    model_path=MODEL_FILE, #Model file
    n_ctx=MODEL_CONTEXT_WINDOW, #Context window
    max_tokens=MAX_ANSWER_TOKENS, #Max tokens in output
    temperature=0, #reduce entropy of output
    verbose=VERBOSE,
    n_batch=512,
    n_gpu_layers=1,
)


CHUNK_SIZE=1000
CHUNK_OVERLAP=50

############################################################################
# Load summarization chain
############################################################################
combine_prompt_template = """
Summarize the following text delimited by triple backquotes.
{style}

```{content}```

{trigger}:
"""

map_prompt_template = """
Write a concise summary of the following:
{text}

"""

def summarize_base(llm, content):
    """Summarize whole content at once. The content needs to fit into model's context window."""

    prompt = PromptTemplate.from_template(
        combine_prompt_template
    ).partial(
        style=STYLE,
        trigger=PROMPT_TRIGGER
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=VERBOSE)
    output = chain.run(content)

    return output

def summarize_map_reduce(llm, content):
    """Summarize content potentially larger that model's context window using map-reduce approach."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    split_docs = text_splitter.create_documents([content])
    print(f"Map-Reduce content splits ({len(split_docs)} splits): {[len(sd.page_content) for sd in split_docs]}")

    map_prompt = PromptTemplate.from_template(
        map_prompt_template
    )
    
    combine_prompt = PromptTemplate.from_template(
        combine_prompt_template
    ).partial(
        style=STYLE,
        trigger=PROMPT_TRIGGER
    )

    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        combine_document_variable_name="content",
        verbose=VERBOSE,
    )

    output = chain.run(split_docs)
    return output

############################################################################
# Load content and summarize
############################################################################
documents,docnames = load_content()

#Record time
import time
start = time.time()
# for title,document in documents.items():
for document in documents:
    content_tokens = llm.get_num_tokens(document)
    print(f"Content length: {len(document)} chars, {content_tokens} tokens.")
    # print("Content sample:\n" + document[:200] + "\n\n")

    # Extracting only the summary from the document
    if documents[document]:    
        dict = documents[document]
        doc = list(dict.values())[0]['summary']

    # Keep part of context window for models output.
        base_threshold = 0.75*MODEL_CONTEXT_WINDOW

        if (content_tokens < base_threshold):
            print("Using summarizer: base")
            summary = summarize_base(llm, doc)
        else:
            print("Using summarizer: map-reduce")
            summary = summarize_map_reduce(llm, doc)


        print(f"Content length: {len(summary)} chars, {llm.get_num_tokens(summary)} tokens.")
        print("Summary:\n" + summary + "\n\n")
#Calculate time
end = time.time()

print(f"Runtime of text-summarization program is {end - start} seconds")
print(f"Avg. runtime per document is {(end - start)/len(documents)} seconds")
# print(f"LLM hyperparameters: {llm}")

