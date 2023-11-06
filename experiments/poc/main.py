import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import os

from experiments.apps.data import get_apps_dataset
 
from helpers import extract_code

# llm class
from printlm.chat_models.azure import AzureChatLLM

# prompts 
# from printlm.prompts.code.prompts import CODE_PROMPTS
# from printlm.prompts.repair.prompts import REPAIR_PROMPTS
# from printlm.prompts.feedback.prompts import FEEDBACK_PROMPTS
# from printlm.prompts.task.prompts import TASK_PROMPTS

# get llm api class
def get_llm(
    args: DictConfig,         
    is_azure: bool,
) -> AzureChatLLM:
    if is_azure:
        llm = AzureChatLLM(**args.model.azure_api)
        return llm
    else:
        print(f"ERROR: {args.model.api} is not a valid API (yet)")
        
# run 
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:
    
    # get llm(s)
    args.model.azure_api.api_key = os.getenv("OPENAI_API_KEY")
    is_azure = args.api_name
    
    raise NotImplementedError

if __name__ == '__main__':
    main()