from openai import OpenAI
import dotenv

dotenv.load_dotenv()

client = OpenAI()

# Uploads the training data file
ft_file = client.files.create(
    file=open("./training-data.jsonl", "rb"),
    purpose="fine-tune"
)

# Create a fine-tuning job using the uploaded file
# ft_filejob = client.fine_tuning.jobs.create(
#     training_file=ft_file.id,
#     model="gpt-3.5-turbo"
# )

job_id = 'ftjob-5XMJNCwBKFe9sBtEIzKkAOo9'

# Retrieve the state of a fine-tune
response = client.fine_tuning.jobs.retrieve(job_id)
print("Job ID:", response.id)
print("Status:", response.status)
print("Trained Tokens:", response.trained_tokens)

# Track events for the fine-tuning job
response = client.fine_tuning.jobs.list_events(job_id)
events = response.data
events.reverse()

for event in events:
    print(event.message)

# Retrieve the identity of the fine-tuned model once ready
response = client.fine_tuning.jobs.retrieve(job_id)
fine_tuned_model_id = response.fine_tuned_model or ''

if not fine_tuned_model_id:
    print("Fine-tuned model is not ready yet.")
else:
    print("Fine-tuned Model ID:", fine_tuned_model_id)
    completion = client.chat.completions.create(
        model=fine_tuned_model_id,
        messages=[
            {"role": "system", "content": "You are Elle, a factual chatbot that answers questions about elements in the periodic table with a limerick"},
            {"role": "user", "content": "Tell me about Strontium"},
        ]
    )
    print(completion.choices[0].message)
