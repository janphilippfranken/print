from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains.llm import LLMChain

from printlm.chat_models.crfm import crfmChatLLM

from helpers import extract_code, format_response

from experiments.poc.task import utility_class

llm = crfmChatLLM(model_name="openai/gpt-4-0613", crfm_api_key="")

def meta_utility(improve_str: str):

    """
    Evaluates the algorithm in improve_str to improve the algorithm in algorithm_str, 
    according to some downstream utility function. This meta-utility function can only be called n times.
    """
    n_tests = 3
    expected_utility = 0
    for _ in range(n_tests):
        try: 
            exec(improve_str, globals())
        except:
            continue
        improved_algorithm_str = improve_algorithm(algorithm_str, utility_class, llm)
        expected_utility += utility_class.func(improved_algorithm_str) / n_tests
    return expected_utility