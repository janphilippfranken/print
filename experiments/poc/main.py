import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import os

from helpers import extract_code

# llm class
from print.chat_models.azure import AsyncAzureChatLLM

# taks
from print.task.tasks import TASKS

# get llm api class
def get_llm(
    args: DictConfig,         
    is_azure: bool,
) -> AsyncAzureChatLLM:
    if is_azure:
        llm = AsyncAzureChatLLM(**args.model.azure_api)
        return llm
    else:
        print(f"ERROR: {args.model.api} is not a valid API (yet)")





# run 
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:
    
    # get llm(s)
    args.model.azure_api.api_key = os.getenv("OPENAI_API_KEY")
    is_azure = args.api_name

    task = TASKS[args.task.task_id]
    print(task.task_id)
    print(task.initial_solution)
    # raise NotImplementedError

if __name__ == '__main__':
    main()