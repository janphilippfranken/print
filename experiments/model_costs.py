# Taken from https://github.com/princeton-nlp/SWE-bench/blob/main/inference/run_api.py
import logging
import os 

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The cost per token for each model input.
MODEL_COST_PER_INPUT = {
    'gpt-3.5-turbo-16k-0613': 0.0000015,
    'gpt-3.5-turbo-16k': 0.0000015,
    'gpt-4-0613': 0.00003,
    'gpt-4': 0.00003,
    'gpt-4-32k-0613': 0.00006,
    'gpt-4-32k': 0.00006,
    'gpt-35-turbo-0613': 0.0000015,
    'gpt-35-turbo': 0.0000015,
}

# The cost per token for each model output.
MODEL_COST_PER_OUTPUT = {
    'gpt-3.5-turbo-16k-0613': 0.000002,
    'gpt-3.5-turbo-16k': 0.000002,
    'gpt-4-0613': 0.0006,
    'gpt-4': 0.0006,
    'gpt-4-32k-0613': 0.00012,
    'gpt-4-32k': 0.00012,
    'gpt-35-turbo-0613': 0.000002,
    'gpt-35-turbo': 0.000002,
}

def calc_cost(response):
    """
    Calculates the cost of a response from the openai API.

    Args:
    response (openai.ChatCompletion): The response from the API.

    Returns:
    float: The cost of the response.
    """
    model_name = response.model
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = (
        MODEL_COST_PER_INPUT[model_name] * input_tokens
        + MODEL_COST_PER_OUTPUT[model_name] * output_tokens
    )
    logger.info(f'input_tokens={input_tokens}, output_tokens={output_tokens}, cost={cost:.2f}')
    return cost