from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains.llm import LLMChain

from helpers import format_response, extract_code


def choose_print_statement(algorithm_str, utility_str, language_model):
    """Let the language model decide where and what to print for debugging."""
    
    expertise = "Decide where and what to print in the given solution to help with debugging. You can only insert one print statement and only print one variable using the print statement."

    message = f"""Choose a new location and content (i.e., variable to print) for a print statement in the following solution to assist with debugging:

```python
{algorithm_str}
```

You will use the return of the print statement to improve the above solution. Upon improvement, you will be evaluated based on the following utility:

```python
{utility_str}
```

Response format:
Line: <int>
Variable: <variable_name>
""" 
    system_prompt = SystemMessagePromptTemplate.from_template(expertise)
    human_prompt = HumanMessagePromptTemplate.from_template(message)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = LLMChain(llm=language_model, prompt=chat_prompt)
    response = chain.run(stop=['System:'])  # this should return location and content for print statement
    response = format_response(response, variables=['Line', 'Variable'])
    return int(response['Line']), response['Variable']