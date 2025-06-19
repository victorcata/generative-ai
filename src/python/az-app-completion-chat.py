import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# validate data inside .env file

client = AzureOpenAI(
    api_key=os.environ['AZURE_FOUNDRY_API_KEY'],
    azure_endpoint=os.environ['AZURE_FOUNDRY_ENDPOINT'],
    api_version=os.environ['AZURE_FOUNDRY_API_VERSION']
)

# Select the General Purpose curie model for text
model = os.environ['AZURE_FOUNDRY_GPT_DEPLOYMENT']

# Create your first prompt
text_prompt = "Should oxford commas always be used?"

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": text_prompt},])

print(response.choices[0].message.content)
