import dotenv
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from prompt import make_prompt

dotenv.load_dotenv()

# DATASET

print('Loading dataset...')
huggingface_dataset_name = 'knkarthick/Dialogsum'
dataset = load_dataset(huggingface_dataset_name)

example_indices = [40, 200]
dash_line = '-'.join('' for x in range(100))


# MODEL
print('Loading model...')
model_name = 'google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

print('Running inference ...')
generation_config = GenerationConfig(
    max_new_tokens=50,
    do_sample=True,
    temperature=.7)

for i, index in enumerate(example_indices):
    # INFERENCE - NO PROMPT ENGINEERING
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    inputs = tokenizer(dialogue, return_tensors='pt')
    output = tokenizer.decode(model.generate(
        inputs['input_ids'], generation_config=generation_config)[0], skip_special_tokens=True)

    print(dash_line)
    print(f"Example {i + 1} from the dataset:")
    print(dash_line)
    print(f"Summary: {dialogue}")
    print(dash_line)
    print(f"Summary: {summary}")
    print(dash_line)
    print(f"Model response (no prompt eng): {output}")

    # INFERENCE - WITH PROMPT ENGINEERING - ZERO SHOT
    prompt = f"""
    Dialogue:
    {dialogue}

    What was going on?
    """
    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(model.generate(
        inputs['input_ids'], generation_config=generation_config)[0], skip_special_tokens=True)

    print(f"Model response (prompt zero-shot): {output}")

    # INFERENCE - ONE SHOT
    example_indices_full = [40]
    example_index_to_summarize = [200]
    one_shot_prompt = make_prompt(
        example_indices_full, example_index_to_summarize, dataset)

    inputs = tokenizer(one_shot_prompt, return_tensors='pt')
    output = tokenizer.decode(model.generate(
        inputs['input_ids'], generation_config=generation_config)[0], skip_special_tokens=True)

    print(f"Model response (one-shot): {output}")

    # INFERENCE - FEW SHOT
    example_indices_full = [40, 80]
    example_index_to_summarize = 200
    few_shot_prompt = make_prompt(
        example_indices_full, example_index_to_summarize, dataset)

    inputs = tokenizer(few_shot_prompt, return_tensors='pt')
    output = tokenizer.decode(model.generate(
        inputs['input_ids'], generation_config=generation_config)[0], skip_special_tokens=True)

    print(f"Model response (few-shot): {output}")
    print()
