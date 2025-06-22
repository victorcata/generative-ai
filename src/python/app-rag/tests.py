from sklearn.metrics import average_precision_score
from chatbot import chatbot

test_cases = [
    {
        "query": "What is a perceptron?",
        "relevant_responses": ["A perceptron is a type of artificial neuron.", "It's a binary classifier used in machine learning."],
        "irrelevant_responses": ["A perceptron is a type of fruit.", "It's a type of car."]
    },
    {
        "query": "What is machine learning?",
        "relevant_responses": ["Machine learning is a method of data analysis that automates analytical model building.", "It's a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."],
        "irrelevant_responses": ["Machine learning is a type of fruit.", "It's a type of car."]
    },
    {
        "query": "What is deep learning?",
        "relevant_responses": ["Deep learning is a subset of machine learning in artificial intelligence (AI) that has networks capable of learning unsupervised from data that is unstructured or unlabeled.", "It's a type of machine learning."],
        "irrelevant_responses": ["Deep learning is a type of fruit.", "It's a type of car."]
    },
    {
        "query": "What is a neural network?",
        "relevant_responses": ["A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.", "It's a type of machine learning."],
        "irrelevant_responses": ["A neural network is a type of fruit.", "It's a type of car."]
    }
]


def run_tests(client, df, nbrs):
    total_average_precision = 0

    for test_case in test_cases:
        query = test_case["query"]
        relevant_responses = test_case["relevant_responses"]
        irrelevant_responses = test_case["irrelevant_responses"]

        response = chatbot(client, df, nbrs, query)

        all_responses = relevant_responses + irrelevant_responses
        true_labels = [1] * len(relevant_responses) + \
            [0] * len(irrelevant_responses)

        predicted_scores = [1 if resp ==
                            response else 0 for resp in all_responses]

        average_precision = average_precision_score(
            true_labels, predicted_scores)

        total_average_precision += average_precision

    # Calculate the mean average precision
    mean_average_precision = total_average_precision / len(test_cases)
    print(f"Mean Average Precision: {mean_average_precision:.4f}")
