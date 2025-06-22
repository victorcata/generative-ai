import numpy as np
from db_utils import get_cosmos_container, get_embeddings_client, create_embeddings
from data_loader import load_markdown_files
from text_utils import split_text
from vector_search import build_search_index
from chatbot import chatbot
from tests import run_tests
from dotenv import load_dotenv
load_dotenv()

container = get_cosmos_container()
client = get_embeddings_client()

print('Loading data...')
data_paths = ["data/frameworks.md",
              "data/own_framework.md", "data/perceptron.md"]
df = load_markdown_files(data_paths)

print('Splitting text into chunks...')
df['chunks'] = df['text'].apply(
    lambda x: split_text(x, 400, 300))

df = df.explode('chunks')

print('Creating embeddings...')
embeddings = []
for chunk in df['chunks']:
    embeddings.append(create_embeddings(client, chunk))

df['embeddings'] = embeddings
embeddings = df['embeddings'].to_list()

print('Building search index...')
nbrs = build_search_index(embeddings, df)

# Your text question
# print('Processing question...')
# question = "what is a perceptron?"
# response = chatbot(client, df, nbrs, question)
# print(response)

run_tests(client, df, nbrs)
