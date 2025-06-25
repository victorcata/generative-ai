def make_prompt(example_indices_full, example_index_to_summarize, dataset):
    """
    Create a prompt for the model based on the dialogue and summary from the dataset.

    Args:
        example_indices_full (list): List of indices for examples in the dataset.
        example_index_to_summarize (int): Index of the example to summarize.

    Returns:
        str: Formatted prompt string.
    """
    prompt = ''
    for index in example_indices_full:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']

        prompt += f"""
        Dialogue:
        {dialogue}

        What was going on?
        {summary}
        """
    dialogue = dataset['test'][example_index_to_summarize]['dialogue']
    prompt += f"""
    Dialogue:
    {dialogue}
    What was going on?
    """

    return prompt
