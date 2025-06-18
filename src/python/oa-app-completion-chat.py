## Simple OpenAI Chat Application using Python

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY","")
assert API_KEY, "ERROR: OpenAI Key is missing"

client = OpenAI(
    api_key=API_KEY
    )

model = "gpt-3.5-turbo"

text_prompt = "Should oxford commas always be used?"

response = client.chat.completions.create(
  model=model,
  messages = [{"role":"system", "content":"You are a helpful assistant."},
               {"role":"user","content":text_prompt},])

print(response.choices[0].message.content)