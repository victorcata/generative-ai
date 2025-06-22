import os
from azure.cosmos import CosmosClient
from openai import AzureOpenAI


def get_cosmos_container():
    """
    Get the Cosmos DB container client.
    :return: Cosmos DB container client.
    """

    url = os.getenv('COSMOS_DB_ENDPOINT')
    key = os.getenv('COSMOS_DB_KEY')
    if not url or not key:
        raise ValueError(
            "COSMOS_DB_ENDPOINT and COSMOS_DB_KEY environment variables must be set")

    client = CosmosClient(url, credential=key)

    database_name = os.getenv('COSMOS_DB_DATABASE') or ''
    database = client.get_database_client(database_name)

    return database.get_container_client('data')


def get_embeddings_client():
    """
    Get the Azure OpenAI client for creating embeddings.
    :return: AzureOpenAI client instance.
    """

    return AzureOpenAI(
        api_key=os.environ['AZURE_OPENAI_API_KEY'],
        api_version="2023-12-01-preview",
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT']
    )


def create_embeddings(client, text, model=None):
    """
    Create embeddings for the given text using the specified model.
    :param client: AzureOpenAI client instance.
    :param text: Text to create embeddings for.
    :param model: Optional model name to use for embeddings.
    :return: Embedding vector.
    """

    if model is None:
        model = os.environ['AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT']

    return client.embeddings.create(
        input=text, model=model).data[0].embedding
