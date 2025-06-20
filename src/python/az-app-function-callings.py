import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import json
import requests

load_dotenv()

client = AzureOpenAI(
    api_key=os.environ['AZURE_FOUNDRY_API_KEY'],
    azure_endpoint=os.environ['AZURE_FOUNDRY_ENDPOINT'],
    api_version=os.environ['AZURE_FOUNDRY_API_VERSION']
)

deployment = os.environ['AZURE_FOUNDRY_GPT_DEPLOYMENT']

# student_1_description = "Emily Johnson is a sophomore majoring in computer science at Duke University. She has a 3.7 GPA. Emily is an active member of the university's Chess Club and Debate Team. She hopes to pursue a career in software engineering after graduating."
# student_2_description = "Michael Lee is a sophomore majoring in computer science at Stanford University. He has a 3.8 GPA. Michael is known for his programming skills and is an active member of the university's Robotics Club. He hopes to pursue a career in artificial intelligence after finishing his studies."

# prompt1 = f'''
# Please extract the following information from the given text and return it as a JSON object:
# Return ONLY a valid JSON object, with no markdown, no code block, and no explanation.
# name
# major
# school
# grades
# club
# This is the body of text to extract the information from:
# {student_1_description}
# '''

# prompt2 = f'''
# Please extract the following information from the given text and return it as a JSON object:
# Return ONLY a valid JSON object, with no markdown, no code block, and no explanation.
# name
# major
# school
# grades
# club
# This is the body of text to extract the information from:
# {student_2_description}
# '''

# # response from prompt one
# openai_response1 = client.chat.completions.create(
#     model=deployment,
#     messages=[{'role': 'user', 'content': prompt1}]
# )

# # response from prompt two
# openai_response2 = client.chat.completions.create(
#     model=deployment,
#     messages=[{'role': 'user', 'content': prompt2}]
# )

# json_response1 = json.loads(openai_response1.choices[0].message.content or '')
# json_response2 = json.loads(openai_response2.choices[0].message.content or '')

messages = [{"role": "user",
             "content": "Find me a good course for a beginner student to learn Azure."}]

functions = [{
    "name": "search_courses",
    "description": "Retrieves courses from the search index based on the parameters provided",
    "parameters": {
        "type": "object",
        "properties": {
             "role": {
                 "type": "string",
                 "description": "The role of the learner (i.e. developer, data scientist, student, etc.)"
             },
             "product": {
                 "type": "string",
                 "description": "The product that the lesson is covering (i.e. Azure, Power BI, etc.)"
             },
             "level": {
                 "type": "string",
                 "description": "The level of experience the learner has prior to taking the course (i.e. beginner, intermediate, advanced)"
             }
             },
        "required": [
            "role"
        ]
    }
}]

response = client.chat.completions.create(model=deployment,
                                          messages=messages,
                                          functions=functions,
                                          function_call="auto")

print(response.choices[0].message)

response_message = response.choices[0].message


def search_courses(role, product, level):
    url = "https://learn.microsoft.com/api/catalog/"
    params = {
        "role": role,
        "product": product,
        "level": level
    }
    response = requests.get(url, params=params)
    modules = response.json()["modules"]
    results = []
    for module in modules[:5]:
        title = module["title"]
        url = module["url"]
        results.append({"title": title, "url": url})
    return str(results)


# Check if the model wants to call a function
if response_message.function_call.name:
    print("Recommended Function call:")
    print(response_message.function_call.name)
    print()

    # Call the function.
    function_name = response_message.function_call.name

    available_functions = {
        "search_courses": search_courses,
    }
    function_to_call = available_functions[function_name]

    function_args = json.loads(response_message.function_call.arguments)
    function_response = function_to_call(**function_args)

    print("Output of function call:")
    print(function_response)
    print(type(function_response))

    # Add the assistant response and function response to the messages
    messages.append(  # adding assistant response to messages
        {
            "role": response_message.role,
            "function_call": {
                "name": function_name,
                "arguments": response_message.function_call.arguments,
            },
            "content": None
        }
    )
    messages.append(  # adding function response to messages
        {
            "role": "function",
            "name": function_name,
            "content": function_response,
        }
    )

print("Messages in next request:")
print(messages)
print()

second_response = client.chat.completions.create(
    messages=messages,
    model=deployment,
    function_call="auto",
    functions=functions,
    temperature=0
)

print(second_response.choices[0].message)
