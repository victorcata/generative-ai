import openai
from openai_config import deployment_name

# Recipe generation
num_recipes = input("How many recipes would you like to see? ")
ingredients = input("List of ingredients (for example, chicken, potatoes, and carrots): ")
filter = input("Filter out these ingredients: ")

prompt = f"Show me {num_recipes} recipes for a dish with the following ingredients: {ingredients}. Per recipe, list all the ingredients used, no {filter}"

messages = [{"role": "user", "content": prompt}]

completion = openai.chat.completions.create(
    model=deployment_name, messages=messages)

print(completion.choices[0].message.content)

# Shopping list
old_prompt_result = completion.choices[0].message.content
prompt = "Produce a shopping list for the generated recipes and please don't include ingredients that I already have."

new_prompt = f"{old_prompt_result} {prompt}"
messages = [{"role": "user", "content": new_prompt}]
completion = openai.Completion.create(engine=deployment_name, messages=messages, max_tokens=1200)

print("Shopping list:")
print(completion.choices[0].message.content)