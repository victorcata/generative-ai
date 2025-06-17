import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ['AZURE_OPENAI_API_KEY']
openai.api_type = 'azure'
openai.api_version = '2023-05-15'
openai.api_base = os.environ['AZURE_OPENAI_ENDPOINT']
deployment_name = os.environ['AZURE_OPENAI_DEPLOYMENT']
