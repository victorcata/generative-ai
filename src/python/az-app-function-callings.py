import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import json

load_dotenv()

client = AzureOpenAI(
    api_key=os.environ['AZURE_FOUNDRY_API_KEY'],
    azure_endpoint=os.environ['AZURE_FOUNDRY_ENDPOINT'],
    api_version=os.environ['AZURE_FOUNDRY_API_VERSION']
)

deployment = os.environ['AZURE_FOUNDRY_GPT_DEPLOYMENT']


student_1_description = "Emily Johnson is a sophomore majoring in computer science at Duke University. She has a 3.7 GPA. Emily is an active member of the university's Chess Club and Debate Team. She hopes to pursue a career in software engineering after graduating."
student_2_description = "Michael Lee is a sophomore majoring in computer science at Stanford University. He has a 3.8 GPA. Michael is known for his programming skills and is an active member of the university's Robotics Club. He hopes to pursue a career in artificial intelligence after finishing his studies."

prompt1 = f'''
Please extract the following information from the given text and return it as a JSON object:
name
major
school
grades
club
This is the body of text to extract the information from:
{student_1_description}
'''

prompt2 = f'''
Please extract the following information from the given text and return it as a JSON object:
name
major
school
grades
club
This is the body of text to extract the information from:
{student_2_description}
'''

# response from prompt one
openai_response1 = client.chat.completions.create(
    model=deployment,
    messages=[{'role': 'user', 'content': prompt1}]
)
print(openai_response1.choices[0].message.content)

# response from prompt two
openai_response2 = client.chat.completions.create(
    model=deployment,
    messages=[{'role': 'user', 'content': prompt2}]
)
print(openai_response2.choices[0].message.content)

json_response1 = json.loads(openai_response1.choices[0].message.content)
print(json_response1)
