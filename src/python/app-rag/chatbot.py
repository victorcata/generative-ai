from db_utils import create_embeddings
import numpy as np
import os
from openai import OpenAI


def chatbot(client, df, nbrs, user_input):
    API_KEY = os.getenv("OPENAI_API_KEY", "")
    assert API_KEY, "ERROR: OpenAI Key is missing"

    openai_client = OpenAI(
        api_key=API_KEY
    )

    query_vector = create_embeddings(client, user_input)
    distances, indices = nbrs.kneighbors(np.array([query_vector]))

    history = []
    for index in indices[0]:
        history.append(df['chunks'].iloc[index])

    history.append(user_input)

    messages = [
        {"role": "system", "content": "You are an AI assiatant that helps with AI questions."},
        {"role": "user", "content": history[-1]}
    ]

    # use chat completion to generate a response
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=800,
        messages=messages
    )

    return response.choices[0].message
